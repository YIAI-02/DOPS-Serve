"""支持固定 64+64 请求和混合工况的 Gymnasium 环境；策略只选择算子放置。"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .execution import (
    DECODE_LENGTH, OPERATORS, PROMPT_LENGTH,
    PrefixExecution, SharedPlacement, cold_snapshot, full_replay, make_evaluator, request_shape,
    segment_count, segment_tokens, workload_key,
)


class PlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evaluator=None, normalization_s=None, variant="A", *,
                 workloads=None, normalizations=None):
        super().__init__()
        self.evaluator = evaluator if evaluator is not None else make_evaluator()
        self.placement = SharedPlacement(self.evaluator)
        self.workload_features = workloads is not None
        self.workloads = workloads if workloads is not None else [{"prefill_len": PROMPT_LENGTH, "batch": 1}]
        self.known_decode = "decode_len" in self.workloads[0]
        self.normalizations = normalizations if normalizations is not None else {
            workload_key(w): (float(normalization_s) if normalization_s is not None else
                              full_replay(self.evaluator, [[0] * 9] * segment_count(w.get("decode_len", DECODE_LENGTH)),
                                          w["prefill_len"], w["batch"], w.get("decode_len", DECODE_LENGTH)).now)
            for w in self.workloads
        }
        self.workload = self.workloads[0]
        self.normalization_s = self.normalizations[workload_key(self.workload)]
        self.variant = variant
        self.segment_actions = variant in ("C", "D")
        self.enhanced = variant in ("B", "D")
        self.action_space = spaces.MultiBinary(9) if self.segment_actions else spaces.Discrete(2)
        # 3 个设备绝对 timeline / C，2 维阶段 one-hot，decode 进度，9 维当前算子。
        base_dimensions = {"A": 15, "B": 23, "C": 6, "D": 15}[variant]
        dimensions = base_dimensions + 2 * self.workload_features + int(self.known_decode)
        low = np.zeros(dimensions, dtype=np.float32)
        high = np.ones(dimensions, dtype=np.float32)
        high[:1 if self.enhanced else 3] = np.inf
        if variant == "B":
            low[15:base_dimensions] = -1
        if self.workload_features:
            high[base_dimensions:] = np.inf
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)
        # 只读取段开始的整类缓存；逐算子执行中的部分分片加载由动作前缀区分。
        self.weight_operators = tuple(op for op in OPERATORS if op not in ("QK", "SV"))
        self.weight_groups = []
        for op in self.weight_operators:
            nodes = [n for n in self.evaluator.compiler.dag.nodes.values() if n.name.upper() == op]
            self.weight_groups.append([
                {n.weight_id for n in nodes},
                *[{n.weight_id for n in nodes
                   if int(n.attrs.get("head_shard", n.attrs.get("ffn_shard", 0))) == shard}
                  for shard in (0, 1)],
            ])

    @property
    def phase(self):
        return "prefill" if self.segment == 0 else "decode"

    @property
    def decode_length(self):
        return self.workload.get("decode_len", DECODE_LENGTH)

    @property
    def episode_steps(self):
        return segment_count(self.decode_length) * (1 if self.segment_actions else len(OPERATORS))

    def _observation(self):
        if self.enhanced:
            obs = [np.log1p(next(iter(self.snapshot.kv_written_tokens.values()))), *self.segment_cache]
            if self.variant == "B":
                obs.extend(self.choices.get(op, -1) for op in OPERATORS[:8])
        else:
            obs = np.zeros(6 if self.segment_actions else 15, dtype=np.float32)
            obs[:3] = [self.snapshot.scheduler.avail[d] / self.normalization_s
                       for d in self.evaluator.compiler.devices]
            obs[3 + int(self.segment > 0)] = 1.0
            obs[5] = self.decode_completed / self.decode_length
            if not self.segment_actions and self.steps < self.episode_steps:
                obs[6 + self.operator_index] = 1.0
        if self.workload_features:
            obs = [*obs, np.log1p(self.workload["prefill_len"]), np.log1p(self.workload["batch"])]
            if self.known_decode:
                obs.append(np.log1p(self.decode_length))
        return np.asarray(obs, dtype=np.float32)

    def _read_segment_cache(self):
        cache = self.snapshot.scheduler.buffer.device_cache
        flags = []
        for op, groups in zip(self.weight_operators, self.weight_groups):
            resident = []
            for device, ids in zip(self.evaluator.compiler.devices, groups):
                present = [wid in cache[device].items for wid in ids]
                if any(present) and not all(present):
                    raise ValueError(f"Cannot encode partial segment cache: {op} on {device}")
                resident.append(int(all(present)))
            if resident[1] != resident[2]:
                raise ValueError(f"Cannot combine asymmetric PIM segment cache: {op}")
            flags.append((resident[0], resident[1]))
        return np.asarray(flags, dtype=np.float32).T.reshape(-1)

    def _start_segment(self):
        self.prefix = PrefixExecution(self.evaluator, self.snapshot, self.phase, self.workload["prefill_len"])
        if self.enhanced:
            self.segment_cache = self._read_segment_cache()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options is not None and "workload" in options:
            self.workload = options["workload"]
        elif self.workload_features:
            self.workload = self.workloads[int(self.np_random.integers(len(self.workloads)))]
        self.normalization_s = self.normalizations[workload_key(self.workload)]
        self.snapshot       = cold_snapshot(self.evaluator, self.workload["prefill_len"], self.workload["batch"], self.decode_length)
        self.segment        = self.operator_index = self.decode_completed = self.steps = 0
        self.choices        = {}
        self.reward_horizon = 0.0
        self._start_segment()
        return self._observation(), {"normalization_s": self.normalization_s,
                                     "workload": workload_key(self.workload)}

    def step(self, action):
        if self.steps >= self.episode_steps:
            raise RuntimeError("Call reset after the episode terminates")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid placement action: {action}")
        if self.segment_actions:
            return self._step_plan(action)
        self.choices[OPERATORS[self.operator_index]] = int(action)
        horizon                                      = self.prefix.advance(self.placement.partial(self.choices)) #推进执行
        self.steps += 1
        self.operator_index += 1
        if self.operator_index                       == len(OPERATORS): #一层结束之后的处理
            assert self.prefix.done
            if self.segment > 0:
                plan                                 = self.placement.plan([self.choices[op] for op in OPERATORS])
                # 当前 token 已通过前缀执行完成；其余最多 3 个 token 使用同一层间共享决策。
                tokens = segment_tokens(self.segment, self.decode_length)
                for _ in range(tokens - 1):
                    self.evaluator.evaluate(self.snapshot, plan, "decode",
                                            request_shape(self.snapshot, "decode", self.workload["prefill_len"]), inplace=True)
                self.decode_completed += tokens
            horizon                                  = self.snapshot.now
            self.segment += 1
            self.operator_index                      = 0
            self.choices                             = {}
            if self.steps < self.episode_steps:
                self._start_segment()

        return self._finish_step(horizon)

    def _step_plan(self, actions):
        plan = self.placement.plan([int(a) for a in actions])
        phase = self.phase
        tokens = segment_tokens(self.segment, self.decode_length)
        for _ in range(tokens):
            self.evaluator.evaluate(self.snapshot, plan, phase,
                                    request_shape(self.snapshot, phase, self.workload["prefill_len"]), inplace=True)
        if phase == "decode":
            self.decode_completed += tokens
        self.steps += 1
        self.segment += 1
        if self.steps < self.episode_steps:
            self._start_segment()
        return self._finish_step(self.snapshot.now)

    def _finish_step(self, horizon):
        # 包括 host 和链路预留的总时间上界；并行工作未延长上界时 reward 可以为 0。
        reward = -(horizon - self.reward_horizon) / self.normalization_s
        self.reward_horizon = horizon
        terminated = self.steps == self.episode_steps
        info = {"simulated_time_s": horizon, "decode_completed": self.decode_completed,
                "workload": workload_key(self.workload), "prefill_len": self.workload["prefill_len"],
                "batch": self.workload["batch"]}
        if self.known_decode:
            info["decode_len"] = self.decode_length
        if terminated:
            info.update(latency_s=self.snapshot.now,
                        speedup=self.normalization_s / self.snapshot.now)
        return self._observation(), float(reward), terminated, False, info
