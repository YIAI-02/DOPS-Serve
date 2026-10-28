"""Continuous BurstGPT FIFO replay at actual plan decision boundaries."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class PlacementRequest:
    '''
    Request 在代码中的存在形式
    '''
    request_id: str
    arrival_time: float
    prompt_len: int
    max_tokens: int
    output_len: int
    prefill_start: float | None = None
    token_times: list[float] = field(default_factory=list)
    completion_time: float | None = None
    violated: bool = False


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
    """Continuous FIFO serving; stop only at actual phase plan decisions.
    连续请求服务环境
    """
    def __init__(self, records, evaluator, decode_feature, max_batch=16):
        self.requests = [PlacementRequest(**{k: row[k] for k in
                         ("request_id", "arrival_time", "prompt_len", "max_tokens", "output_len")})
                         for row in records]
        self.evaluator, self.decode_feature = evaluator, decode_feature
        self.snapshot = evaluator.initial.clone()
        self.snapshot.now = self.requests[0].arrival_time
        self.start_time = self.snapshot.now
        self.max_batch = max_batch
        self.waiting, self.active, self.completed = deque(), [], [] #请求容器
        self.next_arrival = 0
        self.plans = {"prefill": None, "decode": None} #初始化调度状态
        self.decode_counter, self.must_decode = 0, False
        self.pending = None
        self.phase_steps = {"prefill": 0, "decode": 0}
        self.plan_count = 0
        self.output_tokens = 0
        self.decision_s = self.simulator_s = 0.0
        self.peak_runtime_bytes, self.peak_activation_bytes = {}, {}
        self.prepare() #决定下一步处理谁、执行什么阶段

    @property
    def done(self):
        return (self.next_arrival == len(self.requests) and not self.waiting
                and not self.active and self.pending is None)

    def _arrivals(self):
        while self.next_arrival < len(self.requests) and self.requests[self.next_arrival].arrival_time <= self.snapshot.now:
            self.waiting.append(self.requests[self.next_arrival])
            self.next_arrival += 1

    def _visible(self, request):
        '''
        筛选策略可以看到的请求信息
        '''
        return dict(request_id=request.request_id, prompt_len=request.prompt_len,
                    max_tokens=request.max_tokens, generated=len(request.token_times),
                    kv_len=self.snapshot.kv_written_tokens.get(request.request_id, 0),
                    age_s=self.snapshot.now - request.arrival_time)

    def inputs(self):
        '''
        生成 actor 所需的 56 维 state、35 维 node 等输入
        '''
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
            self._arrivals() #先处理到达的请求
            if self.must_decode and self.active:
                #完成一批 prefill 后，如果还有活跃请求，先让它们前进一步 decode，避免连续接收新 prefill 而迟迟不推进输出。
                phase, requests = "decode", list(self.active)
                self.must_decode = False
            else:
                admitted = [] #尝试从等待队列接纳新请求
                while self.waiting and len(self.active) + len(admitted) < self.max_batch:#不超过batch
                    r = self.waiting[0]
                    if not self.evaluator.can_reserve(self.snapshot, r.prompt_len, r.max_tokens): #kv cache可以容纳
                        break
                    self.waiting.popleft()
                    self.evaluator.reserve(self.snapshot, r.request_id, r.prompt_len, r.max_tokens)
                    r.prefill_start = self.snapshot.now
                    admitted.append(r)
                if admitted: #接纳了新请求：这一步只对新请求做 prefill
                    phase, requests = "prefill", admitted
                    self.decode_counter = 0
                elif self.active: #没接纳新请求，但有活跃请求：对活跃请求做 decode
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
        '''
        执行 actor 刚生成的 plan
            → 更新请求和模拟状态
            → 处理到达及准入
            → 必要时复用 decode plan
            → 必要时跳过空闲时间
            → 停在下一个需要 actor 的位置
        '''
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
