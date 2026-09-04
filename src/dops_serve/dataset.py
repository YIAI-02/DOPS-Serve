from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .replay import ProfileEstimator, ReplayState, UnsupportedProfile, request_states


STATE_FEATURES = [
    "waiting_count",
    "active_count",
    "total_waiting_prompt_tokens",
    "total_active_context_tokens",
    "arrival_rate_1s",
    "arrival_rate_10s",
    "arrival_rate_60s",
    "waiting_prompt_mean",
    "waiting_prompt_p95",
    "waiting_prompt_max",
    "active_context_mean",
    "active_context_p95",
    "active_context_max",
    "waiting_age_s_mean",
    "waiting_age_s_p95",
    "waiting_age_s_max",
    "max_tokens_mean",
    "max_tokens_p95",
    "max_tokens_max",
    "generated_tokens_mean",
    "generated_tokens_p95",
    "generated_tokens_max",
]

PLAN_FEATURES = [
    "plan_profile_count",
    "plan_mean_prefill_s_per_input_token",
    "plan_mean_decode_s_per_iteration",
    "plan_mean_output_token_throughput",
    "plan_mean_total_time_s",
    "plan_shape_distance",
]

DATASET_FIELDS = [
    "state_id",
    "trace_id",
    "family",
    "split",
    "model_id",
    "hardware_key",
    "decision_index",
    "time_s",
    "plan_id",
    "horizon",
    *STATE_FEATURES,
    *PLAN_FEATURES,
    "label_token_throughput",
    "label_duration_s",
    "label_generated_tokens",
    "label_completed_requests",
    "label_p95_request_latency_s",
    "label_p95_queue_wait_s",
]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _trace_splits(traces: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    if any(
        row.get("split") not in (None, "")
        for rows in traces.values()
        for row in rows
    ):
        result: dict[str, str] = {}
        for trace_id, rows in traces.items():
            values = {row.get("split") for row in rows}
            if len(values) != 1 or next(iter(values)) not in {
                "train",
                "val",
                "test",
            }:
                raise ValueError(f"Trace {trace_id} has a missing or inconsistent split")
            result[trace_id] = str(next(iter(values)))
        return result

    by_family: dict[str, list[str]] = defaultdict(list)
    for trace_id, rows in traces.items():
        by_family[str(rows[0]["family"])].append(trace_id)
    result = {}
    for family, trace_ids in sorted(by_family.items()):
        ordered = sorted(trace_ids)
        if len(ordered) < 3:
            for trace_id in ordered:
                result[trace_id] = "train"
            continue
        train_end = max(1, int(len(ordered) * 0.7))
        validation_end = max(train_end + 1, int(len(ordered) * 0.85))
        validation_end = min(validation_end, len(ordered) - 1)
        for index, trace_id in enumerate(ordered):
            if index < train_end:
                result[trace_id] = "train"
            elif index < validation_end:
                result[trace_id] = "val"
            else:
                result[trace_id] = "test"
    return result


def build_dataset(
    workload_path: str | Path,
    profiles_path: str | Path,
    signatures_path: str | Path,
    output_path: str | Path,
    hardware_key: str,
    horizons: list[int] | None = None,
    behavior_horizon: int = 64,
    max_batch: int = 16,
    max_decisions_per_trace: int = 1000,
    allow_extrapolation: bool = False,
    allow_incomplete_plans: bool = False,
) -> dict[str, int]:
    actions = horizons or [32, 64, 128]
    workload_rows = _read_jsonl(Path(workload_path))
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in workload_rows:
        traces[str(row["trace_id"])].append(row)
    splits = _trace_splits(traces)
    estimator = ProfileEstimator(
        profiles_path,
        signatures_path,
        allow_extrapolation=allow_extrapolation,
        allow_incomplete_plans=allow_incomplete_plans,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    excluded_path = destination.with_name(destination.stem + "_excluded.csv")
    dataset_count = 0
    state_count = 0
    excluded_count = 0
    with destination.open("w", encoding="utf-8", newline="") as output_handle, (
        excluded_path.open("w", encoding="utf-8", newline="")
    ) as excluded_handle:
        writer = csv.DictWriter(output_handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        excluded_writer = csv.DictWriter(
            excluded_handle,
            fieldnames=["trace_id", "state_id", "plan_id", "horizon", "reason"],
        )
        excluded_writer.writeheader()

        for trace_id, records in sorted(traces.items()):
            model_ids = {str(row["model_id"]) for row in records}
            if len(model_ids) != 1:
                raise ValueError(f"Trace {trace_id} contains multiple models")
            model_id = next(iter(model_ids))
            plan_ids = estimator.plan_ids(model_id, hardware_key)
            if not plan_ids:
                excluded_writer.writerow(
                    {
                        "trace_id": trace_id,
                        "state_id": "",
                        "plan_id": "",
                        "horizon": "",
                        "reason": f"no plan for model={model_id}, hardware={hardware_key}",
                    }
                )
                excluded_count += 1
                continue
            behavior_plan = estimator.robust_plan(model_id, hardware_key)
            state = ReplayState(request_states(records), max_batch=max_batch)
            decision_index = 0
            while not state.done and decision_index < max_decisions_per_trace:
                state.prepare_epoch()
                features = state.online_features()
                state_id = f"{trace_id}:d{decision_index:06d}"
                target_batch = max(
                    1, min(max_batch, len(state.active) or len(state.waiting) or 1)
                )
                target_context = int(
                    max(
                        [request.context_len for request in state.active]
                        or [request.prompt_len for request in state.waiting]
                        or [1]
                    )
                )
                valid_actions = 0
                for plan_id in plan_ids:
                    signature = estimator.signature(plan_id)
                    shape_distance = estimator.shape_distance(
                        plan_id, target_batch, target_context
                    )
                    for horizon in actions:
                        candidate = state.clone()
                        try:
                            metrics = candidate.apply_action(estimator, plan_id, horizon)
                        except UnsupportedProfile as error:
                            excluded_writer.writerow(
                                {
                                    "trace_id": trace_id,
                                    "state_id": state_id,
                                    "plan_id": plan_id,
                                    "horizon": horizon,
                                    "reason": str(error),
                                }
                            )
                            excluded_count += 1
                            continue
                        row: dict[str, Any] = {
                            "state_id": state_id,
                            "trace_id": trace_id,
                            "family": records[0]["family"],
                            "split": splits[trace_id],
                            "model_id": model_id,
                            "hardware_key": hardware_key,
                            "decision_index": decision_index,
                            "time_s": features["time_s"],
                            "plan_id": plan_id,
                            "horizon": horizon,
                            **{name: features[name] for name in STATE_FEATURES},
                            **signature,
                            "plan_shape_distance": shape_distance,
                            "label_token_throughput": metrics["token_throughput"],
                            "label_duration_s": metrics["duration_s"],
                            "label_generated_tokens": metrics["generated_tokens"],
                            "label_completed_requests": metrics["completed_requests"],
                            "label_p95_request_latency_s": metrics[
                                "p95_request_latency_s"
                            ],
                            "label_p95_queue_wait_s": metrics["p95_queue_wait_s"],
                        }
                        writer.writerow(row)
                        dataset_count += 1
                        valid_actions += 1

                if valid_actions == 0:
                    break
                try:
                    state.apply_action(estimator, behavior_plan, behavior_horizon)
                except UnsupportedProfile as error:
                    excluded_writer.writerow(
                        {
                            "trace_id": trace_id,
                            "state_id": state_id,
                            "plan_id": behavior_plan,
                            "horizon": behavior_horizon,
                            "reason": f"behavior action failed: {error}",
                        }
                    )
                    excluded_count += 1
                    break
                state_count += 1
                decision_index += 1

    return {
        "traces": len(traces),
        "states": state_count,
        "rows": dataset_count,
        "excluded": excluded_count,
    }
