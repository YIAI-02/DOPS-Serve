"""Replay recorded DOPS schedules through the native DOPS scheduler."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from .summary import iter_summary_array, read_summary_metadata


def load_fixed_plans(summary_path: Path) -> dict:
    def plan(rows):
        return {
            "order": tuple(row["node_id"] for row in rows),
            "device_by_node": {
                row["node_id"]: row["device"] for row in rows if row["device"] != "COMM"
            },
        }

    prefill = plan(list(iter_summary_array(summary_path, "prefill_schedule")))
    decode = []
    step_times = []
    for step in iter_summary_array(summary_path, "decode_steps"):
        if step["t"] != len(decode):
            raise ValueError(f"{summary_path}: non-contiguous decode steps")
        if step["schedule"] is not None:
            current = plan(step["schedule"])
        elif step["estimated"] and decode:
            current = decode[-1]
        else:
            raise ValueError(f"{summary_path}: unrecorded fresh schedule at token {step['t']}")
        decode.append(current)
        step_times.append(step["step_time"])
    return {"prefill": prefill, "decode": decode, "source_step_times": step_times}


def make_scheduler(backend_root: Path, case: dict, parameters: dict, log_dir: Path, *, simulation_log_file=None):
    sys.path.insert(0, str(backend_root / "src"))
    import config as scheduler_config

    for key, value in parameters.items():
        setattr(scheduler_config, key, value)

    from mainlib.cli import _apply_runtime_config_overrides, _load_cfg_from_json
    from mainlib.graph_utils import _fallback_npu_to_cpu_if_needed
    from mainlib.kv_policy import _apply_kv_place_constraints, _build_cost_model_for_run, _make_label_given_kv_place
    from mainlib.shared import GlobalMemoryManager, build_graph, demo_cluster, reset_simulation_logger
    from mainlib.simulator import _make_scheduler
    from mainlib.storage import _build_uniform_weight_storage_map

    cfg = _load_cfg_from_json(case["run_config_path"])
    _apply_runtime_config_overrides(cfg)
    cluster = demo_cluster(cfg)
    cfg["topology"] = cluster.topology
    graph, shape = build_graph(cfg)
    _fallback_npu_to_cpu_if_needed(graph, cluster)
    metadata = read_summary_metadata(case["summary_path"])
    label, feasible = _make_label_given_kv_place(
        cfg=cfg, cluster=cluster, graph=graph, shape=shape,
        kv_place=metadata["plan_label"]["kv_place"],
    )
    if not feasible:
        raise ValueError(f"{case['case_id']}: recorded KV placement is infeasible")
    graph = _apply_kv_place_constraints(graph, label.kv_place)
    log_dir.mkdir(parents=True, exist_ok=True)
    reset_simulation_logger()
    cost = _build_cost_model_for_run(cfg, cluster, shape, simulation_log_file=simulation_log_file or log_dir / "simulation.txt")
    cost.logger.start_simulation()
    sched = _make_scheduler(
        "Bifocal", cluster, cost, label, cfg["batch"], cfg["prefill_len"],
        GlobalMemoryManager(), rand_seed=cfg["scheduler_seed"],
    )
    sched.reset_state()
    sched.set_storage_format_map(_build_uniform_weight_storage_map(graph, cfg["weight_format"]))
    return sched, graph, shape, cfg


def replay_phase(sched, graph, phase: str, plan: dict) -> float:
    schedule = sched.schedule_with_plan(graph, phase=phase, plan=plan)
    if sched.export_fixed_plan(schedule) != plan:
        raise ValueError(f"DOPS changed the recorded {phase} plan during replay")
    end = float(sched.makespan(schedule))
    sched.stats.reset()
    return end


def replay_decode(sched, graph, cfg: dict, plans: list[dict], prefill_end: float) -> list[float]:
    """Use source token t's plan; hold its last plan beyond the source horizon."""
    global_end = prefill_end
    times = []
    try:
        for t in range(cfg["decode_len"]):
            sched.set_seq_len(cfg["prefill_len"] + t)
            sched.set_decode_context(
                cur_token_idx=t, total_decode_tokens=cfg.get("decode_horizon_len", cfg["decode_len"]), cfg=cfg,
            )
            end = replay_phase(sched, graph, "decode", plans[min(t, len(plans) - 1)])
            times.append(max(0.0, end - global_end))
            global_end = end
    finally:
        sched.clear_decode_context()
    return times


def compare_reference(case: dict, prefill_s: float, step_times: list[float], plans: dict) -> dict:
    expected = {
        "prefill": case["source_prefill_time_s"],
        "decode": case["source_decode_time_s"],
        "total": case["source_total_time_s"],
    }
    actual = {"prefill": prefill_s, "decode": sum(step_times), "total": prefill_s + sum(step_times)}
    matched = {
        phase: math.isclose(actual[phase], expected[phase], rel_tol=1e-9, abs_tol=1e-12)
        for phase in actual
    }
    source_steps = plans["source_step_times"]
    steps_match = len(step_times) == len(source_steps) == case["decode"] and all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(step_times, source_steps)
    )
    return {
        "case_id": case["case_id"], "model": case["model"],
        "passed": all(matched.values()) and steps_match,
        "expected_s": expected, "replayed_s": actual,
        "relative_error": {phase: (actual[phase] - expected[phase]) / expected[phase] for phase in actual},
        "phase_matches": matched, "all_decode_steps_match": steps_match,
        "decode_tokens": len(step_times),
        "max_decode_step_error_s": max(abs(a - b) for a, b in zip(step_times, source_steps)),
        "device_and_order_preserved": True,
    }
