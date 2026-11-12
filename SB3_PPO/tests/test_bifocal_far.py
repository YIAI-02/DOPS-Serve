"""验证 far 评分的运行状态隔离、未来长度边界和原 near/一致性机制。"""
import copy

import pytest

from dops_sb3.bifocal_search import (
    BifocalScheduler, ContinuousBifocal, FarFocusBifocal,
    scheduler_bifocal, scheduler_common, config,
)
from dops_sb3.execution import cold_snapshot, make_evaluator, request_shape


@pytest.fixture
def context(monkeypatch):
    for module in (config, scheduler_common, scheduler_bifocal):
        monkeypatch.setattr(module, "SCHED_WEIGHT_BIAS_ETA", 0.0)
        monkeypatch.setattr(module, "SCHED_JOINT_LK_CONSIST_LAMBDA", 0.25)
    evaluator = make_evaluator()
    base = evaluator.initial.scheduler
    scheduler = FarFocusBifocal(base.cluster, base.cost, base.label, batch=1, seq_len=64,
                               buffer=base.buffer, rand_seed=0)
    scheduler.reset_state()
    scheduler.set_storage_format_map(dict(base.storage_fmt_map))
    evaluator.initial.scheduler = scheduler
    snapshot = cold_snapshot(evaluator)
    scheduler = snapshot.scheduler
    scheduler.cost.configure_requests("prefill", request_shape(snapshot, "prefill"))
    return evaluator.compiler.dag, scheduler


def test_near_and_greedy_selection_are_inherited():
    assert FarFocusBifocal.schedule is BifocalScheduler.schedule
    for name in ("_earliest_finish_on_device", "_lookahead_window_estimate",
                 "_simulate_chain_finish", "_build_lookahead_chain",
                 "_update_plan_hints", "_update_weight_hint_after_commit"):
        assert getattr(FarFocusBifocal, name) is getattr(ContinuousBifocal, name)


def test_future_profile_does_not_change_execution_state(context):
    graph, scheduler = context
    original_cost = scheduler.cost
    shapes = original_cost.request_shapes
    flops_cache = original_cost._flops_cache
    buffer = copy.deepcopy(scheduler.buffer)
    avail = dict(scheduler.avail)
    links = dict(scheduler.comm.timeline_end)
    scheduler.prepare_far_focus(graph, "prefill", mode="both", reuse_window=64)
    assert scheduler.cost is original_cost
    assert scheduler.cost.request_shapes is shapes
    assert scheduler.cost._flops_cache is flops_cache
    assert scheduler.seq_len == 64
    assert scheduler.buffer == buffer
    assert scheduler.avail == avail
    assert scheduler.comm.timeline_end == links


def test_k_future_communication_uses_one_token_output(context):
    graph, scheduler = context
    scheduler.prepare_far_focus(graph, "prefill", mode="both", reuse_window=64)
    npu = scheduler.cluster.devices["Ascend_910B_NPU0"]
    pim = scheduler.cluster.devices["PIM0"]
    # Qwen 的单 shard K 输出是 1 个 KV head × 128 × FP16 = 256 bytes。
    expected = scheduler._estimate_transfer_time(
        npu, pim, 256, src_fmt=scheduler.cost.device_preferred_fmt(npu))
    costs = scheduler.future_costs["L0_K_s0"]
    assert costs[npu.name]["required_output_s"] == pytest.approx(expected, abs=1e-15)
    assert costs[pim.name]["required_output_s"] == 0
    assert costs[npu.name]["service_s"] < 1e-6


def test_window_counts_current_once_and_ignores_actual_stop(context):
    graph, scheduler = context
    npu = scheduler.cluster.devices["Ascend_910B_NPU0"]
    scheduler.prepare_far_focus(graph, "prefill", mode="both", reuse_window=1)
    assert scheduler._future_bias("L0_K_s0", npu, "decode") == 0
    one = scheduler._future_bias("L0_K_s0", npu, "prefill")
    assert one > 0
    scheduler.prepare_far_focus(graph, "prefill", mode="both", reuse_window=64)
    for stop in (1, 10000):
        scheduler.set_decode_context(cur_token_idx=0, total_decode_tokens=stop)
        assert scheduler._future_bias("L0_K_s0", npu, "prefill") == pytest.approx(64 * one)
        assert scheduler._future_bias("L0_K_s0", npu, "decode") == pytest.approx(63 * one)


def test_device_consistency_penalty_is_retained(context):
    graph, scheduler = context
    scheduler.prepare_far_focus(graph, "prefill", mode="both", reuse_window=64)
    idx = scheduler._get_graph_index(graph)
    node = graph.nodes["L0_K_s0"]
    npu = scheduler.cluster.devices["Ascend_910B_NPU0"]
    without = scheduler._weight_reuse_bias(graph, idx, node.id, npu, "prefill")
    scheduler._weight_plan_hint[node.weight_id] = "pim"
    with_hint = scheduler._weight_reuse_bias(graph, idx, node.id, npu, "prefill")
    expected = 0.25 * scheduler._estimate_weight_reload_time(
        node, npu, label=scheduler.label, batch=1, seq_len=64, phase="prefill")
    assert with_hint - without == pytest.approx(expected, abs=1e-15)
