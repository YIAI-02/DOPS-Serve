"""固定 64+64 请求的 Gymnasium 环境；策略只选择算子放置。"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .execution import (
    DECODE_LENGTH, EPISODE_STEPS, OPERATORS, REFRESH_TOKENS, SEGMENTS,
    PrefixExecution, SharedPlacement, cold_snapshot, full_replay, make_evaluator, request_shape,
)


class PlacementEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, evaluator=None, normalization_s=None):
        super().__init__()
        self.evaluator = evaluator if evaluator is not None else make_evaluator()
        self.placement = SharedPlacement(self.evaluator)
        self.normalization_s = (full_replay(self.evaluator, [[0] * 9] * SEGMENTS).now
                                if normalization_s is None else float(normalization_s))
        self.action_space = spaces.Discrete(2)
        # 3 个设备绝对 timeline / C，2 维阶段 one-hot，decode 进度，9 维当前算子。
        # 不含历史动作；这是刻意压缩的观测，并非完整 Markov 状态。
        self.observation_space = spaces.Box(
            low=np.zeros(15, dtype=np.float32),
            high=np.array([np.inf] * 3 + [1.0] * 12, dtype=np.float32), dtype=np.float32)

    @property
    def phase(self):
        return "prefill" if self.segment == 0 else "decode"

    def _observation(self):
        obs = np.zeros(15, dtype=np.float32)
        obs[:3] = [self.snapshot.scheduler.avail[d] / self.normalization_s
                   for d in self.evaluator.compiler.devices]
        obs[3 + int(self.segment > 0)] = 1.0
        obs[5] = self.decode_completed / DECODE_LENGTH
        if self.steps < EPISODE_STEPS:
            obs[6 + self.operator_index] = 1.0
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.snapshot = cold_snapshot(self.evaluator)
        self.segment = self.operator_index = self.decode_completed = self.steps = 0
        self.choices = {}
        self.reward_horizon = 0.0
        self.prefix = PrefixExecution(self.evaluator, self.snapshot, self.phase)
        return self._observation(), {"normalization_s": self.normalization_s}

    def step(self, action):
        if self.steps >= EPISODE_STEPS:
            raise RuntimeError("Call reset after the episode terminates")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid binary action: {action}")
        self.choices[OPERATORS[self.operator_index]] = int(action)
        horizon = self.prefix.advance(self.placement.partial(self.choices))
        self.steps += 1
        self.operator_index += 1
        if self.operator_index == len(OPERATORS):
            assert self.prefix.done
            if self.segment > 0:
                plan = self.placement.plan([self.choices[op] for op in OPERATORS])
                # 当前 token 已通过前缀执行完成；其余 3 个 token 使用同一层间共享决策。
                for _ in range(REFRESH_TOKENS - 1):
                    self.evaluator.evaluate(self.snapshot, plan, "decode",
                                            request_shape(self.snapshot, "decode"), inplace=True)
                self.decode_completed += REFRESH_TOKENS
            horizon = self.snapshot.now
            self.segment += 1
            self.operator_index = 0
            self.choices = {}
            if self.segment < SEGMENTS:
                self.prefix = PrefixExecution(self.evaluator, self.snapshot, self.phase)

        # 包括 host 和链路预留的总时间上界；并行工作未延长上界时 reward 可以为 0。
        reward = -(horizon - self.reward_horizon) / self.normalization_s
        self.reward_horizon = horizon
        terminated = self.steps == EPISODE_STEPS
        info = {"simulated_time_s": horizon, "decode_completed": self.decode_completed}
        if terminated:
            info.update(latency_s=self.snapshot.now,
                        speedup=self.normalization_s / self.snapshot.now)
        return self._observation(), float(reward), terminated, False, info
