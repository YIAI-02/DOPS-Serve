"""固定 64+64 请求的 Gymnasium 环境；策略只选择算子放置。"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .execution import (
    DECODE_LENGTH, EPISODE_STEPS, OPERATORS, REFRESH_TOKENS, REQUEST_ID, SEGMENTS,
    PrefixExecution, SharedPlacement, cold_snapshot, full_replay, make_evaluator, request_shape,
)


class PlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evaluator=None, normalization_s=None, variant="A"):
        super().__init__()
        self.evaluator = evaluator if evaluator is not None else make_evaluator()
        self.placement = SharedPlacement(self.evaluator)
        self.normalization_s = (full_replay(self.evaluator, [[0] * 9] * SEGMENTS).now
                                if normalization_s is None else float(normalization_s))
        self.variant = variant
        self.segment_actions = variant in ("C", "D")
        self.enhanced = variant in ("B", "D")
        self.episode_steps = SEGMENTS if self.segment_actions else EPISODE_STEPS
        self.action_space = spaces.MultiBinary(9) if self.segment_actions else spaces.Discrete(2)
        # 3 个设备绝对 timeline / C，2 维阶段 one-hot，decode 进度，9 维当前算子。
        dimensions = {"A": 15, "B": 23, "C": 6, "D": 15}[variant]
        low = np.zeros(dimensions, dtype=np.float32)
        high = np.ones(dimensions, dtype=np.float32)
        high[:1 if self.enhanced else 3] = np.inf
        if variant == "B":
            low[15:] = -1
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

    def _observation(self):
        if self.enhanced:
            obs = [np.log1p(self.snapshot.kv_written_tokens[REQUEST_ID]), *self.segment_cache]
            if self.variant == "B":
                obs.extend(self.choices.get(op, -1) for op in OPERATORS[:8])
            return np.asarray(obs, dtype=np.float32)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[:3] = [self.snapshot.scheduler.avail[d] / self.normalization_s
                   for d in self.evaluator.compiler.devices]
        obs[3 + int(self.segment > 0)] = 1.0
        obs[5] = self.decode_completed / DECODE_LENGTH
        if not self.segment_actions and self.steps < self.episode_steps:
            obs[6 + self.operator_index] = 1.0
        return obs

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
        self.prefix = PrefixExecution(self.evaluator, self.snapshot, self.phase)
        if self.enhanced:
            self.segment_cache = self._read_segment_cache()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.snapshot       = cold_snapshot(self.evaluator)
        self.segment        = self.operator_index = self.decode_completed = self.steps = 0
        self.choices        = {}
        self.reward_horizon = 0.0
        self._start_segment()
        return self._observation(), {"normalization_s": self.normalization_s}

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
                # 当前 token 已通过前缀执行完成；其余 3 个 token 使用同一层间共享决策。
                for _ in range(REFRESH_TOKENS - 1):
                    self.evaluator.evaluate(self.snapshot, plan, "decode",
                                            request_shape(self.snapshot, "decode"), inplace=True)
                self.decode_completed += REFRESH_TOKENS
            horizon                                  = self.snapshot.now
            self.segment += 1
            self.operator_index                      = 0
            self.choices                             = {}
            if self.segment < SEGMENTS:
                self._start_segment()

        return self._finish_step(horizon)

    def _step_plan(self, actions):
        plan = self.placement.plan([int(a) for a in actions])
        phase = self.phase
        for _ in range(1 if phase == "prefill" else REFRESH_TOKENS):
            self.evaluator.evaluate(self.snapshot, plan, phase,
                                    request_shape(self.snapshot, phase), inplace=True)
        if phase == "decode":
            self.decode_completed += REFRESH_TOKENS
        self.steps += 1
        self.segment += 1
        if self.segment < SEGMENTS:
            self._start_segment()
        return self._finish_step(self.snapshot.now)

    def _finish_step(self, horizon):
        # 包括 host 和链路预留的总时间上界；并行工作未延长上界时 reward 可以为 0。
        reward = -(horizon - self.reward_horizon) / self.normalization_s
        self.reward_horizon = horizon
        terminated = self.steps == self.episode_steps
        info = {"simulated_time_s": horizon, "decode_completed": self.decode_completed}
        if terminated:
            info.update(latency_s=self.snapshot.now,
                        speedup=self.normalization_s / self.snapshot.now)
        return self._observation(), float(reward), terminated, False, info
