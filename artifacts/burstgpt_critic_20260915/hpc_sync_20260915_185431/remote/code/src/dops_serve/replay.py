from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PlacementRequest:
    request_id: str
    arrival_time: float
    prompt_len: int
    max_tokens: int
    output_len: int
    prefill_start: float | None = None
    token_times: list[float] = field(default_factory=list)
    completion_time: float | None = None
    violated: bool = False


class PlacementReplay:
    """FIFO continuous batching for exact fixed-placement DAG execution."""

    def __init__(self, records, evaluator, *, slo_s=None, max_batch=16, decode_window=32):
        self.requests = sorted([
            PlacementRequest(**{key: row[key] for key in ("request_id", "arrival_time", "prompt_len", "max_tokens", "output_len")})
            for row in records
        ], key=lambda r: r.arrival_time)
        for request in self.requests:
            if not 128 <= request.prompt_len <= 2048 or request.max_tokens != 1024 or not 1 <= request.output_len <= 1024:
                raise ValueError(f"{request.request_id}: outside placement v1 serving scope")
        self.evaluator = evaluator
        self.snapshot = evaluator.initial.clone()
        self.slo_s = slo_s
        self.max_batch = max_batch
        self.decode_window = decode_window
        self.next_arrival = 0
        self.waiting = deque()
        self.active = []
        self.completed = []
        self.plans = None
        self.total_decode_iterations = 0
        self.windows = []
        self.step_results = []
        self.previous_boundary = (0.0, 0)

    @property
    def done(self):
        return self.next_arrival == len(self.requests) and not self.waiting and not self.active

    def _arrivals(self):
        while self.next_arrival < len(self.requests) and self.requests[self.next_arrival].arrival_time <= self.snapshot.now:
            self.waiting.append(self.requests[self.next_arrival])
            self.next_arrival += 1

    def _advance(self, end, counts):
        start = self.snapshot.now
        # Throughput training has no queue/deadline integral to maintain.
        for request in self.requests if self.slo_s is not None else ():
            if request.arrival_time >= end:
                continue
            wait_end = min(end, request.prefill_start if request.prefill_start is not None else end)
            counts["waiting_s"] += max(0.0, wait_end - max(start, request.arrival_time))
            lifetime_end = min(end, request.completion_time if request.completion_time is not None else end)
            deadline = request.arrival_time + self.slo_s
            if lifetime_end > deadline and not request.violated:
                request.violated = True
                counts["new_violations"] += 1
            counts["overdue_s"] += max(0.0, lifetime_end - max(start, deadline))
        self.snapshot.now = end
        self._arrivals()

    def observation(self):
        def visible(request):
            return {
                "request_id": request.request_id, "prompt_len": request.prompt_len,
                "max_tokens": request.max_tokens, "generated": len(request.token_times),
                "age_s": self.snapshot.now - request.arrival_time,
                "kv_len": self.snapshot.kv_written_tokens.get(request.request_id, 0),
            }
        scheduler = self.snapshot.scheduler
        return {
            "now": self.snapshot.now,
            "waiting": [visible(r) for r in self.waiting],
            "active": [visible(r) for r in self.active],
            "device_available_s": dict(scheduler.avail),
            "kv_reserved_bytes_by_pim": dict(scheduler.label.kv_bytes_by_pim),
            "kv_actual_bytes_by_pim": {p: sum(self.snapshot.kv_written_tokens.values()) * self.evaluator.kv_bytes_per_token_per_pim for p in ("PIM0", "PIM1")},
            "weight_cache_bytes": {d: cache.used for d, cache in scheduler.buffer.device_cache.items()},
            "weight_cache_capacity_bytes": {d: cache.capacity for d, cache in scheduler.buffer.device_cache.items()},
            "weight_cache_keys": {d: list(cache.items) for d, cache in scheduler.buffer.device_cache.items()},
            "recent_arrivals_1s": sum(self.snapshot.now - 1.0 < r.arrival_time <= self.snapshot.now for r in self.requests[:self.next_arrival]),
            "recent_arrivals_10s": sum(self.snapshot.now - 10.0 < r.arrival_time <= self.snapshot.now for r in self.requests[:self.next_arrival]) / 10.0,
            "recent_arrivals_60s": sum(self.snapshot.now - 60.0 < r.arrival_time <= self.snapshot.now for r in self.requests[:self.next_arrival]) / 60.0,
            "queue_growth_per_s": (len(self.waiting) - self.previous_boundary[1]) / (self.snapshot.now - self.previous_boundary[0]) if self.snapshot.now > self.previous_boundary[0] else 0.0,
            "recent_service_tokens_per_s": self.windows[-1]["output_tokens"] / self.windows[-1]["duration_s"] if self.windows else 0.0,
            "decode_iterations": self.total_decode_iterations,
        }

    def _execute(self, phase, requests, counts):
        from .placement_evaluator import RequestShape

        shapes = [RequestShape(r.request_id, r.prompt_len if phase == "prefill" else 1, self.snapshot.kv_written_tokens[r.request_id]) for r in requests]
        result = self.evaluator.evaluate(self.snapshot, self.plans[phase], phase, shapes)
        self._advance(result.snapshot.now, counts)
        self.snapshot = result.snapshot
        self.step_results.append({
            "phase": phase, "request_ids": [r.request_id for r in requests],
            "duration_s": result.duration_s, "host_evaluator_s": result.host_evaluator_s,
            "communication_bytes": sum(event["bytes"] for event in result.comm_events),
            "communication_s_sum": sum(event["duration"] for event in result.comm_events),
            "peak_runtime_bytes": result.peak_runtime_bytes,
        })
        for request in requests:
            request.token_times.append(self.snapshot.now)
            counts["output_tokens"] += 1
            if len(request.token_times) == request.output_len:
                request.completion_time = self.snapshot.now
                self.completed.append(request)
                self.evaluator.release(self.snapshot, request.request_id)
            else:
                if phase == "prefill":
                    self.active.append(request)
        self.active = [r for r in self.active if r.completion_time is None]

    def run_window(self, decide, reward_coefficients):
        if self.done:
            raise ValueError("trace is already complete")
        start = self.snapshot.now
        counts = {"output_tokens": 0, "waiting_s": 0.0, "new_violations": 0, "overdue_s": 0.0, "decode_iterations": 0}
        self._arrivals()
        if not self.active and not self.waiting:
            self._advance(self.requests[self.next_arrival].arrival_time, counts)
        decision_start = time.perf_counter()
        plans = decide(self.observation(), self.plans)
        decision_s = time.perf_counter() - decision_start
        self.previous_boundary = (self.snapshot.now, len(self.waiting))
        switches = int(self.plans is not None and plans != self.plans)
        self.plans = plans
        self._advance(self.snapshot.now + decision_s, counts)
        while counts["decode_iterations"] < self.decode_window and not self.done:
            self._arrivals()
            admitted = []
            while self.waiting and len(self.active) + len(admitted) < self.max_batch:
                request = self.waiting[0]
                if not self.evaluator.can_reserve(self.snapshot, request.prompt_len, request.max_tokens):
                    break
                self.waiting.popleft()
                self.evaluator.reserve(self.snapshot, request.request_id, request.prompt_len, request.max_tokens)
                request.prefill_start = self.snapshot.now
                admitted.append(request)
            if admitted:
                self._execute("prefill", admitted, counts)
            if self.active:
                self._execute("decode", list(self.active), counts)
                counts["decode_iterations"] += 1
                self.total_decode_iterations += 1
            elif self.waiting:
                if not admitted:
                    raise ValueError(f"FIFO head {self.waiting[0].request_id} cannot fit even in an empty batch")
            elif self.next_arrival < len(self.requests):
                self._advance(self.requests[self.next_arrival].arrival_time, counts)
        # A rollout bootstrap must observe the same boundary as the next
        # policy call. Charge intervening idle time to the finished window.
        if not self.done and not self.active and not self.waiting:
            self._advance(self.requests[self.next_arrival].arrival_time, counts)
        duration = self.snapshot.now - start
        c = reward_coefficients
        token_reward, time_reward = c["a_N"] * counts["output_tokens"], -c["a_T"] * duration
        reward = token_reward + time_reward
        if self.slo_s is not None:
            reward -= c["lambda_q"] * counts["waiting_s"] + c["lambda_l"] * (counts["new_violations"] + counts["overdue_s"] / self.slo_s)
        else:
            for key in ("waiting_s", "new_violations", "overdue_s"):
                del counts[key]
        result = {**counts, "start_s": start, "end_s": self.snapshot.now, "duration_s": duration, "decision_s": decision_s, "placement_switches": switches, "reward": reward, "token_reward": token_reward, "time_reward": time_reward, "terminal": self.done}
        self.windows.append(result)
        return result

    def request_metrics(self):
        rows = []
        for request in self.completed:
            first, finish = request.token_times[0], request.token_times[-1]
            rows.append({
                "request_id": request.request_id, "arrival_time": request.arrival_time,
                "prefill_start": request.prefill_start, "first_token_time": first,
                "token_times": request.token_times, "completion_time": finish,
                "queue_wait_s": request.prefill_start - request.arrival_time,
                "ttft_s": first - request.arrival_time, "e2e_s": finish - request.arrival_time,
                "tpot_s": (finish - first) / (len(request.token_times) - 1) if len(request.token_times) > 1 else None,
                "tbt_s": [b - a for a, b in zip(request.token_times, request.token_times[1:])],
            })
            if self.slo_s is not None:
                rows[-1]["e2e_violated"] = finish - request.arrival_time > self.slo_s
        return rows


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


class BurstPlacementReplay:
    """Continuous FIFO serving; stop only at actual phase plan decisions."""

    def __init__(self, records, evaluator, decode_feature, max_batch=16):
        self.requests = [PlacementRequest(**{k: row[k] for k in
                         ("request_id", "arrival_time", "prompt_len", "max_tokens", "output_len")})
                         for row in records]
        self.evaluator, self.decode_feature = evaluator, decode_feature
        self.snapshot = evaluator.initial.clone()
        self.snapshot.now = self.requests[0].arrival_time
        self.start_time = self.snapshot.now
        self.max_batch = max_batch
        self.waiting, self.active, self.completed = deque(), [], []
        self.next_arrival = 0
        self.plans = {"prefill": None, "decode": None}
        self.decode_counter, self.must_decode = 0, False
        self.pending = None
        self.phase_steps = {"prefill": 0, "decode": 0}
        self.plan_count = 0
        self.output_tokens = 0
        self.decision_s = self.simulator_s = 0.0
        self.peak_runtime_bytes, self.peak_activation_bytes = {}, {}
        self.prepare()

    @property
    def done(self):
        return (self.next_arrival == len(self.requests) and not self.waiting
                and not self.active and self.pending is None)

    def _arrivals(self):
        while self.next_arrival < len(self.requests) and self.requests[self.next_arrival].arrival_time <= self.snapshot.now:
            self.waiting.append(self.requests[self.next_arrival])
            self.next_arrival += 1

    def _visible(self, request):
        return dict(request_id=request.request_id, prompt_len=request.prompt_len,
                    max_tokens=request.max_tokens, generated=len(request.token_times),
                    kv_len=self.snapshot.kv_written_tokens.get(request.request_id, 0),
                    age_s=self.snapshot.now - request.arrival_time)

    def inputs(self):
        from .placement_dataset import serving_policy_inputs
        phase, requests = self.pending
        return serving_policy_inputs(self.snapshot.scheduler, self.evaluator.compiler,
            self.evaluator.shape, phase, [self._visible(r) for r in requests],
            [self._visible(r) for r in self.waiting], len(self.snapshot.kv_reserved_tokens),
            self.decode_feature, self.snapshot.now, self.plans["prefill"], self.plans["decode"])

    def shapes(self):
        from .placement_evaluator import RequestShape
        phase, requests = self.pending
        return [RequestShape(r.request_id, r.prompt_len if phase == "prefill" else 1,
                             self.snapshot.kv_written_tokens[r.request_id]) for r in requests]

    def prepare(self):
        if self.pending is not None:
            return
        while True:
            self._arrivals()
            if self.must_decode and self.active:
                phase, requests = "decode", list(self.active)
                self.must_decode = False
            else:
                admitted = []
                while self.waiting and len(self.active) + len(admitted) < self.max_batch:
                    r = self.waiting[0]
                    if not self.evaluator.can_reserve(self.snapshot, r.prompt_len, r.max_tokens):
                        break
                    self.waiting.popleft()
                    self.evaluator.reserve(self.snapshot, r.request_id, r.prompt_len, r.max_tokens)
                    r.prefill_start = self.snapshot.now
                    admitted.append(r)
                if admitted:
                    phase, requests = "prefill", admitted
                    self.decode_counter = 0
                elif self.active:
                    phase, requests = "decode", list(self.active)
                elif self.waiting:
                    raise ValueError(f"FIFO request {self.waiting[0].request_id} cannot fit an empty batch")
                elif self.next_arrival < len(self.requests):
                    self.snapshot.now = self.requests[self.next_arrival].arrival_time
                    continue
                else:
                    return
            self.pending = (phase, requests)
            if (phase == "prefill" or self.plans["decode"] is None
                    or self.decode_counter < 2 or self.decode_counter % 2 == 0):
                return
            self._execute(self.plans["decode"])

    def _execute(self, plan):
        phase, requests = self.pending
        result = self.evaluator.evaluate(self.snapshot, plan, phase, self.shapes(), inplace=True)
        self.snapshot = result.snapshot
        self.simulator_s += result.host_evaluator_s
        for key, value in result.peak_runtime_bytes.items():
            self.peak_runtime_bytes[key] = max(self.peak_runtime_bytes.get(key, 0), value)
        for key, value in self.snapshot.scheduler.peak_activation_bytes.items():
            self.peak_activation_bytes[key] = max(self.peak_activation_bytes.get(key, 0), value)
        changed = False
        for r in requests:
            r.token_times.append(self.snapshot.now)
            self.output_tokens += 1
            if len(r.token_times) == r.output_len:
                r.completion_time = self.snapshot.now
                self.completed.append(r)
                self.evaluator.release(self.snapshot, r.request_id)
                changed = True
            elif phase == "prefill":
                self.active.append(r)
        self.active = [r for r in self.active if r.completion_time is None]
        self.plans[phase] = plan
        self.phase_steps[phase] += 1
        self.pending = None
        if phase == "prefill":
            self.decode_counter = 0
            self.must_decode = bool(self.active)
        else:
            self.decode_counter = 0 if changed else self.decode_counter + 1

    def step(self, plan):
        start = self.snapshot.now
        self.plan_count += 1
        self._execute(plan)
        self.prepare()  # Includes plan reuse, idle time and the next actual admission.
        return self.snapshot.now - start

    def metrics(self):
        elapsed = self.snapshot.now - self.start_time
        ttft = [r.token_times[0] - r.arrival_time for r in self.completed]
        e2e = [r.completion_time - r.arrival_time for r in self.completed]
        waits = [r.prefill_start - r.arrival_time for r in self.completed]
        tbt = [b - a for r in self.completed for a, b in zip(r.token_times, r.token_times[1:])]
        return dict(requests=len(self.requests), completed=len(self.completed), output_tokens=self.output_tokens,
            simulated_s=elapsed, throughput=self.output_tokens / elapsed, plans=self.plan_count,
            phase_steps=dict(self.phase_steps), decision_s=self.decision_s, simulator_s=self.simulator_s,
            peak_runtime_bytes=self.peak_runtime_bytes, peak_activation_bytes=self.peak_activation_bytes,
            ttft_mean_s=sum(ttft)/len(ttft), ttft_p95_s=_quantile(ttft, .95),
            e2e_mean_s=sum(e2e)/len(e2e), e2e_p95_s=_quantile(e2e, .95),
            queue_mean_s=sum(waits)/len(waits), tbt_p95_s=_quantile(tbt, .95))
