from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .profiles import estimate_idw


class UnsupportedProfile(ValueError):
    pass


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _statistics(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_mean": 0.0, f"{prefix}_p95": 0.0, f"{prefix}_max": 0.0}
    return {
        f"{prefix}_mean": sum(values) / len(values),
        f"{prefix}_p95": _quantile(values, 0.95),
        f"{prefix}_max": max(values),
    }


class ProfileEstimator:
    def __init__(
        self,
        profiles_path: str | Path,
        signatures_path: str | Path,
        allow_extrapolation: bool = False,
        allow_incomplete_plans: bool = False,
        neighbors: int = 4,
    ) -> None:
        with Path(profiles_path).open("r", encoding="utf-8", newline="") as handle:
            profile_rows = list(csv.DictReader(handle))
        with Path(signatures_path).open("r", encoding="utf-8", newline="") as handle:
            signature_rows = list(csv.DictReader(handle))
        self.profiles: dict[str, list[dict[str, float | str]]] = {}
        for row in profile_rows:
            converted: dict[str, float | str] = dict(row)
            for field_name in (
                "batch",
                "prefill_len",
                "decode_len",
                "context_mid",
                "prefill_time_s",
                "decode_s_per_iteration",
            ):
                converted[field_name] = float(row[field_name])
            self.profiles.setdefault(row["plan_id"], []).append(converted)
        self.signatures = {row["plan_id"]: row for row in signature_rows}
        self.allow_extrapolation = allow_extrapolation
        self.allow_incomplete_plans = allow_incomplete_plans
        self.neighbors = neighbors

    def plan_ids(self, model_id: str, hardware_key: str) -> list[str]:
        return sorted(
            plan_id
            for plan_id, signature in self.signatures.items()
            if signature["model_id"] == model_id
            and signature["hardware_key"] == hardware_key
            and (
                self.allow_incomplete_plans
                or signature["placement_complete"].casefold() == "true"
            )
        )

    def signature(self, plan_id: str) -> dict[str, float]:
        row = self.signatures[plan_id]
        return {
            "plan_profile_count": float(row["profile_count"]),
            "plan_mean_prefill_s_per_input_token": float(
                row["mean_prefill_s_per_input_token"]
            ),
            "plan_mean_decode_s_per_iteration": float(
                row["mean_decode_s_per_iteration"]
            ),
            "plan_mean_output_token_throughput": float(
                row["mean_output_token_throughput"]
            ),
            "plan_mean_total_time_s": float(row["mean_total_time_s"]),
        }

    def robust_plan(self, model_id: str, hardware_key: str) -> str:
        candidates = self.plan_ids(model_id, hardware_key)
        if not candidates:
            raise UnsupportedProfile(
                f"No plans for model={model_id}, hardware={hardware_key}"
            )
        return min(
            candidates,
            key=lambda plan_id: (
                -self.signature(plan_id)["plan_profile_count"],
                -self.signature(plan_id)["plan_mean_output_token_throughput"],
                plan_id,
            ),
        )

    def _check_envelope(
        self, plan_id: str, batch: float, length: float, phase: str
    ) -> None:
        if self.allow_extrapolation:
            return
        points = self.profiles[plan_id]
        batches = [float(row["batch"]) for row in points]
        if not min(batches) <= batch <= max(batches):
            raise UnsupportedProfile(
                f"batch={batch} outside profile range for {plan_id}"
            )
        if phase == "prefill":
            lower = min(float(row["prefill_len"]) for row in points)
            upper = max(float(row["prefill_len"]) for row in points)
        else:
            lower = min(float(row["prefill_len"]) for row in points)
            upper = max(
                float(row["prefill_len"]) + float(row["decode_len"]) for row in points
            )
        if not lower <= length <= upper:
            raise UnsupportedProfile(
                f"{phase} length={length} outside profile range for {plan_id}"
            )

    def prefill_time(self, plan_id: str, batch: int, prompt_len: int) -> float:
        self._check_envelope(plan_id, float(batch), float(prompt_len), "prefill")
        references = [
            (
                (float(row["batch"]), float(row["prefill_len"])),
                float(row["prefill_time_s"]),
            )
            for row in self.profiles[plan_id]
        ]
        return estimate_idw(
            (float(batch), float(prompt_len)), references, neighbors=self.neighbors
        )

    def decode_step_time(self, plan_id: str, batch: int, context_len: int) -> float:
        self._check_envelope(plan_id, float(batch), float(context_len), "decode")
        references = [
            (
                (float(row["batch"]), float(row["context_mid"])),
                float(row["decode_s_per_iteration"]),
            )
            for row in self.profiles[plan_id]
        ]
        return estimate_idw(
            (float(batch), float(context_len)), references, neighbors=self.neighbors
        )

    def shape_distance(self, plan_id: str, batch: int, context_len: int) -> float:
        points = self.profiles[plan_id]
        batch_values = [math.log1p(float(row["batch"])) for row in points]
        context_values = [math.log1p(float(row["context_mid"])) for row in points]
        batch_range = max(batch_values) - min(batch_values) or 1.0
        context_range = max(context_values) - min(context_values) or 1.0
        query_batch = math.log1p(batch)
        query_context = math.log1p(context_len)
        return min(
            math.sqrt(
                ((query_batch - math.log1p(float(row["batch"]))) / batch_range) ** 2
                + (
                    (query_context - math.log1p(float(row["context_mid"])))
                    / context_range
                )
                ** 2
            )
            for row in points
        )


@dataclass
class RequestState:
    trace_id: str
    family: str
    request_id: str
    model_id: str
    arrival_time: float
    prompt_len: int
    max_tokens: int
    output_len: int
    generated: int = 0
    prefill_start: float | None = None
    completion_time: float | None = None

    @property
    def context_len(self) -> int:
        return self.prompt_len + self.generated


@dataclass
class ReplayState:
    requests: list[RequestState]
    max_batch: int = 16
    now: float = 0.0
    next_arrival: int = 0
    waiting: list[RequestState] = field(default_factory=list)
    active: list[RequestState] = field(default_factory=list)
    completed: list[RequestState] = field(default_factory=list)

    def clone(self) -> "ReplayState":
        return copy.deepcopy(self)

    @property
    def done(self) -> bool:
        return (
            self.next_arrival >= len(self.requests)
            and not self.waiting
            and not self.active
        )

    def _add_arrivals(self) -> None:
        while (
            self.next_arrival < len(self.requests)
            and self.requests[self.next_arrival].arrival_time <= self.now
        ):
            self.waiting.append(self.requests[self.next_arrival])
            self.next_arrival += 1

    def prepare_epoch(self) -> None:
        self._add_arrivals()
        if not self.waiting and not self.active and self.next_arrival < len(self.requests):
            self.now = max(self.now, self.requests[self.next_arrival].arrival_time)
            self._add_arrivals()

    def online_features(self) -> dict[str, float]:
        self.prepare_epoch()
        waiting_prompts = [float(request.prompt_len) for request in self.waiting]
        active_contexts = [float(request.context_len) for request in self.active]
        waiting_ages = [self.now - request.arrival_time for request in self.waiting]
        max_tokens = [
            float(request.max_tokens) for request in [*self.waiting, *self.active]
        ]
        generated = [float(request.generated) for request in self.active]
        seen_arrivals = [
            request.arrival_time
            for request in self.requests[: self.next_arrival]
            if request.arrival_time <= self.now
        ]
        features: dict[str, float] = {
            "time_s": self.now,
            "waiting_count": float(len(self.waiting)),
            "active_count": float(len(self.active)),
            "total_waiting_prompt_tokens": sum(waiting_prompts),
            "total_active_context_tokens": sum(active_contexts),
            "arrival_rate_1s": sum(time >= self.now - 1.0 for time in seen_arrivals),
            "arrival_rate_10s": sum(time >= self.now - 10.0 for time in seen_arrivals)
            / 10.0,
            "arrival_rate_60s": sum(time >= self.now - 60.0 for time in seen_arrivals)
            / 60.0,
        }
        features.update(_statistics(waiting_prompts, "waiting_prompt"))
        features.update(_statistics(active_contexts, "active_context"))
        features.update(_statistics(waiting_ages, "waiting_age_s"))
        features.update(_statistics(max_tokens, "max_tokens"))
        features.update(_statistics(generated, "generated_tokens"))
        return features

    def apply_action(
        self,
        estimator: ProfileEstimator,
        plan_id: str,
        horizon: int,
        stop_on_profile_boundary: bool = False,
    ) -> dict[str, float]:
        self.prepare_epoch()
        start_time = self.now
        generated_tokens = 0
        executed_decode_steps = 0
        profile_boundary_interrupted = 0
        completed_latencies: list[float] = []
        queue_waits: list[float] = []

        available_slots = self.max_batch - len(self.active)
        if available_slots > 0 and self.waiting:
            selected = self.waiting[:available_slots]
            del self.waiting[:available_slots]
            prefill_start = self.now
            prefill_time = estimator.prefill_time(
                plan_id,
                batch=len(selected),
                prompt_len=max(request.prompt_len for request in selected),
            )
            self.now += prefill_time
            for request in selected:
                request.prefill_start = prefill_start
                queue_waits.append(prefill_start - request.arrival_time)
            self.active.extend(selected)
            self._add_arrivals()

        for _ in range(horizon):
            if not self.active:
                break
            try:
                step_time = estimator.decode_step_time(
                    plan_id,
                    batch=len(self.active),
                    context_len=max(request.context_len for request in self.active),
                )
            except UnsupportedProfile:
                if not stop_on_profile_boundary or executed_decode_steps == 0:
                    raise
                profile_boundary_interrupted = 1
                break
            self.now += step_time
            executed_decode_steps += 1
            finished: list[RequestState] = []
            for request in self.active:
                request.generated += 1
                generated_tokens += 1
                if request.generated >= request.output_len:
                    request.completion_time = self.now
                    completed_latencies.append(self.now - request.arrival_time)
                    finished.append(request)
            if finished:
                self.active = [request for request in self.active if request not in finished]
                self.completed.extend(finished)
            self._add_arrivals()

        duration = self.now - start_time
        return {
            "duration_s": duration,
            "generated_tokens": float(generated_tokens),
            "executed_decode_steps": float(executed_decode_steps),
            "completed_requests": float(len(completed_latencies)),
            "token_throughput": generated_tokens / duration if duration > 0 else 0.0,
            "p95_request_latency_s": _quantile(completed_latencies, 0.95),
            "p95_queue_wait_s": _quantile(queue_waits, 0.95),
            "profile_boundary_interrupted": float(profile_boundary_interrupted),
        }


def request_states(records: list[dict[str, Any]]) -> list[RequestState]:
    return [
        RequestState(
            trace_id=str(row["trace_id"]),
            family=str(row["family"]),
            request_id=str(row["request_id"]),
            model_id=str(row["model_id"]),
            arrival_time=float(row["arrival_time"]),
            prompt_len=int(row["prompt_len"]),
            max_tokens=int(row["max_tokens"]),
            output_len=int(row["output_len"]),
        )
        for row in sorted(records, key=lambda row: (float(row["arrival_time"]), row["request_id"]))
    ]
