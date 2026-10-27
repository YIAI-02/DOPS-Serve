"""验证新决策接口与原完整调度器的实际执行一致。"""
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from dops_sb3.env import PlacementEnv
from dops_sb3.execution import (
    EPISODE_STEPS, OPERATORS, REQUEST_ID, SEGMENTS, PrefixExecution,
    SharedPlacement, cold_snapshot, full_replay, make_evaluator, request_shape,
)


@pytest.fixture(scope="module")
def evaluator():
    return make_evaluator()


@pytest.fixture(scope="module")
def normalization(evaluator):
    return full_replay(evaluator, [[0] * 9] * SEGMENTS).now


def test_mapping(evaluator):
    placement = SharedPlacement(evaluator)
    mapping = placement.plan([1, 0, 1, 1, 0, 1, 0, 1, 1])["device_by_node"]
    npu = placement.npu
    for layer in range(28):
        for shard in range(2):
            assert mapping[f"L{layer}_Q_s{shard}"] == f"PIM{shard}"
            assert mapping[f"L{layer}_K_s{shard}"] == npu
            assert mapping[f"L{layer}_Softmax_s{shard}"] == f"PIM{shard}"
            assert mapping[f"L{layer}_SwiGLU_s{shard}"] == npu
            assert mapping[f"L{layer}_K_write_s{shard}"] == f"PIM{shard}"
            assert mapping[f"L{layer}_V_write_s{shard}"] == f"PIM{shard}"
        assert mapping[f"L{layer}_Add1"] == "PIM0"
        assert mapping[f"L{layer}_LN2"] == "PIM0"
        assert mapping[f"L{layer}_Add2"] == "PIM0"
        assert mapping[f"L{layer}_LN"] == (npu if layer == 0 else "PIM0")


def test_prefix_has_same_tasks_and_communication(evaluator):
    placement = SharedPlacement(evaluator)
    actions = [1, 0, 1, 1, 0, 1, 0, 1, 1]
    snapshot = cold_snapshot(evaluator)
    oracle = evaluator.evaluate(snapshot, placement.plan(actions), "prefill",
                                request_shape(snapshot, "prefill"))
    prefix = PrefixExecution(evaluator, snapshot, "prefill")
    choices = {}
    horizons = []
    for i, (op, action) in enumerate(zip(OPERATORS, actions)):
        choices[op] = action
        horizons.append(prefix.advance(placement.partial(choices)))
        if i < 8:
            assert not prefix.done
            assert all(t.node_id.startswith("L0_") for t in prefix.schedule)
    assert horizons == sorted(horizons)
    assert prefix.schedule == oracle.schedule
    assert snapshot.scheduler.stats.comm_events == oracle.comm_events
    assert snapshot.scheduler.buffer == oracle.snapshot.scheduler.buffer
    assert snapshot.now == oracle.snapshot.now


@pytest.mark.parametrize("kind", ["npu", "pim", "changing"])
def test_episode_matches_full_replay(evaluator, normalization, kind):
    if kind == "changing":
        plans = np.random.default_rng(17).integers(0, 2, (SEGMENTS, 9)).tolist()
    else:
        plans = [[int(kind == "pim")] * 9 for _ in range(SEGMENTS)]
    oracle = full_replay(evaluator, plans)
    env = PlacementEnv(evaluator, normalization)
    first_obs, _ = env.reset(seed=0)
    total = 0.0
    for step, action in enumerate(np.asarray(plans).reshape(-1)):
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        assert reward <= 0.0
        assert env.observation_space.contains(obs)
        assert not truncated
        assert terminated == (step == EPISODE_STEPS - 1)
        if (step + 1) % 9 == 0:
            assert info["decode_completed"] == max(0, (step + 1) // 9 - 1) * 4
    assert info["latency_s"] == oracle.now
    assert total == pytest.approx(-oracle.now / normalization, abs=1e-12)
    assert env.snapshot.kv_written_tokens == {REQUEST_ID: 128}
    assert env.snapshot.scheduler.avail == oracle.scheduler.avail
    assert env.snapshot.scheduler.comm.timeline_end == oracle.scheduler.comm.timeline_end
    assert env.snapshot.scheduler.buffer == oracle.scheduler.buffer
    reset_obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(first_obs, reset_obs)
    assert env.snapshot.now == 0.0
    assert env.snapshot.kv_written_tokens == {REQUEST_ID: 0}
    assert not env.snapshot.scheduler.weight_cached
    assert all(t == 0.0 for t in env.snapshot.scheduler.avail.values())
    assert not env.snapshot.scheduler.comm.timeline_end


def test_live_observation_and_sb3_contract(evaluator, normalization):
    env = PlacementEnv(evaluator, normalization)
    obs, _ = env.reset()
    assert obs.shape == (15,)
    obs, reward, _, _, _ = env.step(1)
    np.testing.assert_allclose(obs[:3], [env.snapshot.scheduler.avail[d] / normalization
                                       for d in evaluator.compiler.devices])
    assert reward < 0
    check_env(env, warn=True)
