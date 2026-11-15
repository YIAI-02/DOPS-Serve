"""验证混合工况的真实批执行、观测、采样与评估合同。"""
import numpy as np
import pytest

from dops_sb3.env import PlacementEnv
from dops_sb3.execution import SEGMENTS, full_replay, make_evaluator, workload_key
from dops_sb3.train import evaluate_workloads


WORKLOADS = [{"prefill_len": 64, "batch": 1}, {"prefill_len": 256, "batch": 4},
             {"prefill_len": 1024, "batch": 16}]


@pytest.fixture(scope="module")
def mixed_context():
    evaluator = make_evaluator()
    env = PlacementEnv(evaluator, variant="D", workloads=WORKLOADS)
    return evaluator, env.normalizations


@pytest.mark.parametrize("variant,dimension", [("A", 17), ("B", 25), ("C", 8), ("D", 17)])
@pytest.mark.parametrize("workload", WORKLOADS)
def test_batched_episode_matches_full_execution(mixed_context, variant, dimension, workload):
    evaluator, normalizations = mixed_context
    plans = np.random.default_rng(17).integers(0, 2, (SEGMENTS, 9)).tolist()
    oracle = full_replay(evaluator, plans, workload["prefill_len"], workload["batch"])
    env = PlacementEnv(evaluator, variant=variant, workloads=WORKLOADS, normalizations=normalizations)
    obs, _ = env.reset(seed=0, options={"workload": workload})
    assert obs.shape == (dimension,)
    np.testing.assert_allclose(obs[-2:], np.log1p([workload["prefill_len"], workload["batch"]]))
    actions = np.asarray(plans) if env.segment_actions else np.asarray(plans).reshape(-1)
    reward_sum = 0
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        reward_sum += reward
    assert done and not truncated
    assert info["latency_s"] == oracle.now
    assert reward_sum == pytest.approx(-oracle.now / normalizations[workload_key(workload)], abs=1e-12)
    assert len(env.snapshot.kv_written_tokens) == workload["batch"]
    assert set(env.snapshot.kv_written_tokens.values()) == {workload["prefill_len"] + 64}
    assert env.snapshot.scheduler.buffer == oracle.scheduler.buffer
    assert env.snapshot.scheduler.avail == oracle.scheduler.avail
    assert env.snapshot.scheduler.comm.timeline_end == oracle.scheduler.comm.timeline_end


def test_sampling_matches_across_variants_and_fixed_evaluation_does_not_consume_rng(mixed_context):
    evaluator, normalizations = mixed_context
    a = PlacementEnv(evaluator, variant="A", workloads=WORKLOADS, normalizations=normalizations)
    d = PlacementEnv(evaluator, variant="D", workloads=WORKLOADS, normalizations=normalizations)
    observed = set()
    for index in range(50):
        _, left = a.reset(seed=12 if index == 0 else None)
        _, right = d.reset(seed=12 if index == 0 else None)
        assert left["workload"] == right["workload"]
        observed.add(left["workload"])
        d.reset(options={"workload": WORKLOADS[0]})
    assert observed == {workload_key(w) for w in WORKLOADS}


def test_evaluation_covers_every_workload_and_keeps_raw_times(mixed_context):
    evaluator, normalizations = mixed_context
    env = PlacementEnv(evaluator, variant="D", workloads=WORKLOADS, normalizations=normalizations)

    class AllNpu:
        def predict(self, observation, deterministic):
            assert deterministic
            return np.zeros(9, dtype=int), None

    result = evaluate_workloads(AllNpu(), env, WORKLOADS)
    assert set(result["workloads"]) == {workload_key(w) for w in WORKLOADS}
    assert result["normalized_latency"] == pytest.approx(1.0)
    for key, values in result["workloads"].items():
        assert values["latency_s"] == normalizations[key]
    assert len({v["latency_s"] for v in result["workloads"].values()}) == 3
