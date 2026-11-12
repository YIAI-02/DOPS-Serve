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
@pytest.mark.parametrize("variant", list("ABCD"))
def test_episode_matches_full_replay(evaluator, normalization, kind, variant):
    if kind == "changing":
        plans = np.random.default_rng(17).integers(0, 2, (SEGMENTS, 9)).tolist()
    else:
        plans = [[int(kind == "pim")] * 9 for _ in range(SEGMENTS)]
    oracle = full_replay(evaluator, plans)
    env = PlacementEnv(evaluator, normalization, variant)
    first_obs, _ = env.reset(seed=0)
    total = 0.0
    actions = np.asarray(plans) if env.segment_actions else np.asarray(plans).reshape(-1)
    for step, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        assert reward <= 0.0
        assert env.observation_space.contains(obs)
        assert not truncated
        assert terminated == (step == env.episode_steps - 1)
        decisions_per_segment = 1 if env.segment_actions else 9
        if (step + 1) % decisions_per_segment == 0:
            assert info["decode_completed"] == max(0, (step + 1) // decisions_per_segment - 1) * 4
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


@pytest.mark.parametrize("variant,dimensions", [("B", 23), ("C", 6), ("D", 15)])
def test_variant_contract(evaluator, normalization, variant, dimensions):
    env = PlacementEnv(evaluator, normalization, variant)
    obs, _ = env.reset()
    assert obs.shape == (dimensions,)
    check_env(env, warn=True)


def advance_plan(env, plan):
    actions = [np.asarray(plan)] if env.segment_actions else plan
    for action in actions:
        obs, reward, done, _, info = env.step(action)
    return obs, reward, done, info


def test_enhanced_prefix_and_frozen_cache(evaluator, normalization):
    env = PlacementEnv(evaluator, normalization, "B")
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs[1:15], np.zeros(14))
    np.testing.assert_array_equal(obs[15:], [-1] * 8)
    for i, action in enumerate([1, 0, 1, 1, 1, 1, 1, 1]):
        obs, *_ = env.step(action)
        np.testing.assert_array_equal(obs[1:15], np.zeros(14))
        np.testing.assert_array_equal(obs[15:16+i], [1, 0, 1, 1, 1, 1, 1, 1][:i+1])
        np.testing.assert_array_equal(obs[16+i:], [-1] * (7-i))
    obs, *_ = env.step(1)
    assert obs[0] == np.float32(np.log1p(64))
    np.testing.assert_array_equal(obs[1:8], [0, 1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(obs[8:15], [1, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(obs[15:], [-1] * 8)


def test_enhanced_observation_resolves_history_alias(evaluator, normalization):
    histories = [[1, 0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1, 1, 1]]
    old_observations, enhanced_observations, rewards = [], [], []
    for plan in histories:
        old = PlacementEnv(evaluator, normalization, "A")
        enhanced = PlacementEnv(evaluator, normalization, "B")
        for env, observations in ((old, old_observations), (enhanced, enhanced_observations)):
            env.reset()
            advance_plan(env, plan)
            obs, *_ = env.step(1)
            observations.append(obs)
            _, reward, *_ = env.step(1)
            if env is old:
                rewards.append(reward)
    np.testing.assert_array_equal(*old_observations)
    assert not np.array_equal(*enhanced_observations)
    assert abs(rewards[0] - rewards[1]) > 1e-6


def test_cache_is_history_not_previous_plan(evaluator, normalization):
    observations = []
    for prefill in (0, 1):
        env = PlacementEnv(evaluator, normalization, "D")
        env.reset()
        advance_plan(env, [prefill] * 9)
        obs, *_ = advance_plan(env, [1] * 9)
        observations.append(obs)
    assert observations[0][0] == observations[1][0] == np.float32(np.log1p(68))
    np.testing.assert_array_equal(observations[0][1:8], np.ones(7))
    np.testing.assert_array_equal(observations[1][1:8], np.zeros(7))
    np.testing.assert_array_equal(observations[0][8:], observations[1][8:])
