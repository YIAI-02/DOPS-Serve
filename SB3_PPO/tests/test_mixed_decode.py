"""已知输出长度的批次执行、尾段和三元组归一化。"""
import json
from itertools import product

import numpy as np
import pytest

from dops_sb3.env import PlacementEnv
from dops_sb3.execution import ROOT, full_replay, make_evaluator, segment_count, workload_key


def test_formal_decode_grid():
    spec = json.loads((ROOT / "configs/mixed_decode_workloads.json").read_text())
    for split, lengths in (("train", [64, 128, 512, 1024]), ("heldout", [256, 796])):
        triples = [(w["prefill_len"], w["batch"], w["decode_len"]) for w in spec[split]]
        assert triples == list(product([64, 256, 1024], [1, 4, 16], lengths))
    assert set(map(workload_key, spec["train"])).isdisjoint(map(workload_key, spec["heldout"]))
    assert [segment_count(d) for d in [64, 128, 256, 512, 796, 1024]] == [17, 33, 65, 129, 200, 257]
    assert spec["training_timesteps"] == {"A": 1530000, "B": 1530000, "C": 170000, "D": 170000}
    assert spec["periodic_evaluations"] == 5


@pytest.fixture(scope="module", params=[(64, 4, 5), (1024, 16, 1024)])
def decode_case(request):
    p, b, d = request.param
    w = {"prefill_len": p, "batch": b, "decode_len": d}
    evaluator = make_evaluator()
    plans = np.random.default_rng(19).integers(0, 2, (segment_count(d), 9)).tolist()
    npu_s = full_replay(evaluator, [[0] * 9] * segment_count(d), p, b, d).now
    reference = full_replay(evaluator, plans, p, b, d)
    return w, evaluator, plans, npu_s, reference


@pytest.mark.parametrize("variant,dimension", [("A", 18), ("B", 26), ("C", 9), ("D", 18)])
def test_known_decode_matches_full_replay(decode_case, variant, dimension):
    w, evaluator, plans, npu_s, reference = decode_case
    env = PlacementEnv(evaluator, variant=variant, workloads=[w], normalizations={workload_key(w): npu_s})
    obs, _ = env.reset(seed=0)
    d, p, b = w["decode_len"], w["prefill_len"], w["batch"]
    assert obs.shape == (dimension,)
    np.testing.assert_allclose(obs[-3:], np.log1p([p, b, d]))
    assert set(env.snapshot.kv_reserved_tokens.values()) == {p + d}
    assert len(env.snapshot.kv_reserved_tokens) == b
    if (p, b, d) == (1024, 16, 1024):
        assert env.snapshot.scheduler.label.kv_bytes_by_pim == {"PIM0": 448 * 2**20, "PIM1": 448 * 2**20}
    total_reward = 0
    actions = plans if env.segment_actions else np.asarray(plans).ravel()
    assert len(actions) == env.episode_steps
    for index, action in enumerate(actions):
        obs, reward, done, truncated, info = env.step(np.asarray(action) if env.segment_actions else action)
        assert done == (index + 1 == len(actions)) and not truncated
        assert info["decode_len"] == d and env.observation_space.contains(obs)
        if not env.enhanced:
            assert obs[5] == pytest.approx(info["decode_completed"] / d)
        total_reward += reward
    assert info["decode_completed"] == d and set(env.snapshot.kv_written_tokens.values()) == {p + d}
    assert info["latency_s"] == reference.now
    assert env.snapshot.scheduler.buffer == reference.scheduler.buffer
    assert total_reward == pytest.approx(-reference.now / npu_s)


def test_reset_selects_length_and_uses_its_own_baseline():
    evaluator = make_evaluator()
    workloads = [{"prefill_len": 64, "batch": 1, "decode_len": d} for d in (4, 9)]
    env = PlacementEnv(evaluator, variant="C", workloads=workloads)
    assert len(set(env.normalizations.values())) == 2
    for w in workloads:
        obs, _ = env.reset(options={"workload": w})
        assert env.episode_steps == segment_count(w["decode_len"])
        rewards = []
        for _ in range(env.episode_steps):
            obs, reward, done, _, info = env.step(np.zeros(9, dtype=int))
            rewards.append(reward)
        assert done and info["workload"] == workload_key(w)
        assert sum(rewards) == pytest.approx(-1)
