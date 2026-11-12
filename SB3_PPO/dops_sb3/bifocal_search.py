"""在 PPO 的连续计时内核上搜索 Bifocal 评分参数，不读取真实剩余输出长度。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import sys
import time

from .execution import (ROOT, DECODE_LENGTH, REFRESH_TOKENS, REQUEST_ID,
                        PrefixExecution, cold_snapshot, full_replay,
                        make_evaluator, request_shape)

sys.path.insert(0, str(ROOT / "backend/src"))
import config
from scheduler import BifocalScheduler
from scheduler import scheduler_bifocal, scheduler_common
from .timing.placement_cost import ContinuousScheduler
from .timing.placement_evaluator import RequestShape


class ContinuousBifocal(BifocalScheduler, ContinuousScheduler):
    """保留原 Bifocal 决策，只接入与 PPO 相同的 collective 计时。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_calls = Counter()

    def _lookahead_window_estimate(self, *args, **kwargs):
        self.score_calls["lookahead"] += 1
        return super()._lookahead_window_estimate(*args, **kwargs)

    def _weight_reuse_bias(self, *args, **kwargs):
        value = super()._weight_reuse_bias(*args, **kwargs)
        self.score_calls["reuse_nonzero"] += int(value != 0)
        return value

    def _decode_phase_amortization_bias(self, *args, **kwargs):
        value = super()._decode_phase_amortization_bias(*args, **kwargs)
        self.score_calls["amort_nonzero"] += int(value != 0)
        return value

    def _simulate_chain_finish(self, g, chain, devs, first_finish, phase, *, apply_consistency_penalty):
        if apply_consistency_penalty and scheduler_bifocal.SCHED_JOINT_LK_CONSIST_LAMBDA > 0:
            self.score_calls["consistency_conflicts"] += sum(
                bool(self._plan_hint.get(nid)) and self._plan_hint[nid] != dev.name
                for nid, dev in zip(chain[1:], devs[1:]))
        return super()._simulate_chain_finish(
            g, chain, devs, first_finish, phase,
            apply_consistency_penalty=apply_consistency_penalty)


class FarFocusBifocal(ContinuousBifocal):
    """只替换 far 评分；继承原 EFT、短链前瞻、ready 选择和提交过程。"""

    def prepare_far_focus(self, g, phase, *, mode, reuse_window):
        self.far_mode = mode
        self.far_window = reuse_window
        self.future_costs = {}
        # 用当前已知长度估计下一次 decode，不读取请求的真实停止位置。
        # 独立成本视图避免预估改变当前 forward 的 shape 或成本缓存。
        current_cost, current_seq = self.cost, self.seq_len
        future_shapes = tuple(RequestShape(r.request_id, 1, r.kv_len + r.query_len)
                              for r in current_cost.request_shapes)
        self.cost = copy.copy(current_cost)
        self.cost.configure_requests("decode", future_shapes)
        self.seq_len = max(r.kv_len + r.query_len for r in future_shapes)
        try:
            total_weights = self._get_total_model_weight_bytes(g)
            devices = [d for t in self._executor_device_types()
                       for d in self.cluster.devices_by_type(t)]
            for nid, node in g.nodes.items():
                if not self._node_weight_id(node) or self._node_weight_size(node) <= 0:
                    continue
                costs = {}
                for dev in devices:
                    if not self._node_allowed_on(node, dev):
                        continue
                    reusable = self._device_can_hold_all_weights(dev, total_weights)
                    service = self._weight_service_profile_no_contention(
                        node, dev, cached=reusable, label=self.label,
                        batch=self.batch, seq_len=self.seq_len, phase="decode")
                    # 只计入目的设备已确定的后继传输；不假设未知后继都在 PIM。
                    transfers = {}
                    for successor in g.successors(nid):
                        child = g.nodes[successor]
                        if child.name.upper() not in ("K_WRITE", "V_WRITE", "KV_WRITE"):
                            continue
                        target = self._preferred_kv_write_device(g, successor)
                        # 执行器按生产者实际输出传输；far 不沿用短链的 16 KiB 下界。
                        _, size = self.cost.estimate_activation_bytes(
                            node, self.batch, self.seq_len, "decode")
                        transfers[target.name] = max(transfers.get(target.name, 0), size)
                    communication = sum(self._estimate_transfer_time(
                        dev, self.cluster.devices[target], size,
                        src_fmt=self.cost.device_preferred_fmt(dev))
                        for target, size in transfers.items())
                    costs[dev.name] = {"service_s": service["total_s"],
                                       "required_output_s": communication,
                                       "total_s": service["total_s"] + communication,
                                       "weights_reusable": reusable}
                reference = min(c["total_s"] for c in costs.values())
                for cost in costs.values():
                    cost["excess_s"] = cost["total_s"] - reference
                self.future_costs[nid] = costs
        finally:
            self.cost, self.seq_len = current_cost, current_seq
        self.score_calls["future_profile_builds"] += 1
        self.score_calls["future_device_profiles"] += sum(map(len, self.future_costs.values()))

    def _future_bias(self, nid, dev, phase):
        if nid not in self.future_costs:
            return 0.0
        repeats = self.far_window if phase == "prefill" else max(0, self.far_window - 1)
        value = repeats * self.future_costs[nid][dev.name]["excess_s"]
        self.score_calls[f"future_{phase}_nonzero"] += int(value != 0)
        return value

    def _weight_reuse_bias(self, g, idx, nid, dev, phase):
        # 新模式的 eta 固定为 0，原方法仍完整保留设备一致性惩罚。
        consistency = super()._weight_reuse_bias(g, idx, nid, dev, phase)
        return consistency + (self._future_bias(nid, dev, phase) if phase == "prefill" else 0.0)

    def _decode_phase_amortization_bias(self, g, nid, dev, phase):
        if self.far_mode == "prefill":
            return super()._decode_phase_amortization_bias(g, nid, dev, phase)
        # 当前冷加载已由 EFT 计算；只增加后续驻留执行代价，不重复减免加载。
        return self._future_bias(nid, dev, phase) if phase == "decode" else 0.0


KNOBS = {
    "h": "SCHED_JOINT_LK_H",
    "gamma": "SCHED_JOINT_LK_GAMMA",
    "lambda": "SCHED_JOINT_LK_CONSIST_LAMBDA",
    "plan_hint_max": "SCHED_JOINT_LK_PLAN_HINT_MAX",
    "eta": "SCHED_WEIGHT_BIAS_ETA",
}
COARSE_AXES = {
    "h": [2, 3, 4], "gamma": [0.1, 0.2, 0.4, 0.6, 0.8],
    "lambda": [0.25, 1.0, 4.0], "plan_hint_max": [1, 3, 5],
    "eta": [1.0, 5.0, 10.0], "amort_alpha": [0.5, 1.0, 2.0, 4.0, 6.0],
    "reuse_window": [4, 16, 64, 256],
}
FINE_AXES = {
    "h": [2, 3, 4], "gamma": [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0],
    "lambda": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    "plan_hint_max": [1, 2, 3, 5, 8],
    "eta": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0],
    "amort_alpha": [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
    "reuse_window": [4, 8, 16, 32, 64, 128, 256],
}


def key(params):
    return tuple(sorted(params.items()))


def run_trial(params, seed, save_plans=False):
    begin = time.perf_counter()
    far_mode = params.get("far_mode", "legacy")
    if far_mode not in ("legacy", "prefill", "both"):
        raise ValueError(f"Unknown far_mode: {far_mode}")
    if "far_estimator" in params or "far_slack_fraction" in params:
        raise ValueError("探索性队列/余量版本已移出正式实现，源码保存在 runs/bifocal_far_20261112/exploration_source.py")
    # 后端通过 from-import 读取常量；同一 worker 的每次试验都更新实际使用的模块。
    for module in (config, scheduler_common, scheduler_bifocal):
        module.SCHED_JOINT_LK_ENABLE = True
        for name, field in KNOBS.items():
            value = 0.0 if name == "eta" and far_mode != "legacy" else params[name]
            setattr(module, field, value)
    evaluator = make_evaluator()
    base = evaluator.initial.scheduler
    scheduler_type = ContinuousBifocal if far_mode == "legacy" else FarFocusBifocal
    scheduler = scheduler_type(
        base.cluster, base.cost, base.label, batch=1, seq_len=64,
        buffer=base.buffer, rand_seed=seed)
    scheduler.reset_state()
    scheduler.set_storage_format_map(dict(base.storage_fmt_map))
    evaluator.initial.scheduler = scheduler
    snapshot = cold_snapshot(evaluator)
    scheduler = snapshot.scheduler
    graph = evaluator.compiler.dag
    phases, plans = [], []
    future_profiles = []
    plan = None
    for step in range(1 + DECODE_LENGTH):
        phase, token = ("prefill", -1) if step == 0 else ("decode", step - 1)
        shapes = request_shape(snapshot, phase)
        scheduler.cost.configure_requests(phase, shapes)
        scheduler.seq_len = shapes[0].query_len + shapes[0].kv_len
        scheduler.stats.reset()
        scheduler.peak_runtime_bytes = {}
        scheduler.peak_activation_bytes = {}
        for device in scheduler.avail:
            scheduler.avail[device] = max(scheduler.avail[device], snapshot.now)
        for link in scheduler.comm.timeline_end:
            scheduler.comm.timeline_end[link] = max(scheduler.comm.timeline_end[link], snapshot.now)
        if step:
            # 传入的是每次固定的复用窗口；末 token 也不缩短窗口，不泄露实际停止时间。
            scheduler.set_decode_context(
                cur_token_idx=token, total_decode_tokens=token + params["reuse_window"],
                cfg={"decode_amort_enable": params.get("amort_enable", True),
                     "decode_amort_alpha": params["amort_alpha"],
                     "decode_amort_rmin": 1, "decode_amort_reuse_prob": 1.0})
        if step == 0 or token % REFRESH_TOKENS == 0:
            if far_mode != "legacy":
                scheduler.prepare_far_focus(graph, phase, mode=far_mode,
                                            reuse_window=params["reuse_window"])
                if save_plans:
                    future_profiles.append({"step": step, "costs": copy.deepcopy(scheduler.future_costs)})
            schedule = scheduler.schedule(graph, phase)
            plan = scheduler.export_fixed_plan(schedule)
            if save_plans:
                plans.append({"step": step, "plan": plan})
        else:
            schedule = scheduler.schedule_with_plan(graph, phase, plan)
        end = max(snapshot.now, scheduler.makespan(schedule),
                  *scheduler.avail.values(), *scheduler.comm.timeline_end.values())
        assert all(t.start >= snapshot.now - 1e-14 for t in schedule)
        phases.append((end - snapshot.now) * 1000)
        snapshot.now = end
        snapshot.kv_written_tokens[REQUEST_ID] += shapes[0].query_len
        scheduler.reset_state(clear_caches=False)
    result = {"params": dict(params), "seed": seed, "total_ms": snapshot.now * 1000,
              "prefill_ms": phases[0], "decode_ms": sum(phases[1:]),
              "wall_s": time.perf_counter() - begin,
              "score_calls": dict(scheduler.score_calls)}
    if save_plans:
        result.update(phases_ms=phases, plans=plans)
        if far_mode != "legacy":
            result["future_profiles"] = future_profiles
    return result


def verify_replay(result):
    evaluator = make_evaluator()
    snapshot = cold_snapshot(evaluator)
    plans = {entry["step"]: entry["plan"] for entry in result["plans"]}
    for step in range(1 + DECODE_LENGTH):
        if step in plans:
            plan = plans[step]
        execution = PrefixExecution(evaluator, snapshot, "prefill" if step == 0 else "decode")
        # 列表调度允许自由拓扑序；仅替换导出的顺序，其余复用原完整执行内核。
        execution.order = tuple(plan["order"])
        execution.advance(plan["device_by_node"])
        assert execution.done
    assert math.isclose(snapshot.now * 1000, result["total_ms"], rel_tol=0, abs_tol=1e-9)
    return snapshot.now * 1000


def coarse_candidates(count):
    rng = random.Random(20261112)
    values = []
    for h, gamma, eta in itertools.product([2, 3, 4], [0.1, 0.4, 0.8], [1.0, 5.0, 10.0]):
        values.append({"h": h, "gamma": gamma, "lambda": 1.0, "plan_hint_max": 3,
                       "eta": eta, "amort_alpha": 1.0, "reuse_window": 64})
    used = {key(p) for p in values}
    while len(values) < count:
        params = {name: rng.choice(axis) for name, axis in COARSE_AXES.items()}
        if key(params) not in used:
            values.append(params)
            used.add(key(params))
    return values[:count]


def fine_candidates(ranked, count, used):
    candidates = []
    # 各轴分别探索邻近值；再补充优胜区域内的组合，避免单项调参漏掉交互。
    for row in ranked[:3]:
        params = row["params"]
        for name, axis in FINE_AXES.items():
            near = sorted(axis, key=lambda value: abs(value - params[name]))
            for value in [v for v in near if v != params[name]][:2]:
                candidate = {**params, name: value}
                if key(candidate) not in used:
                    candidates.append(candidate)
                    used.add(key(candidate))
    rng = random.Random(20261113)
    while len(candidates) < count:
        params = dict(rng.choice(ranked[:3])["params"])
        for name in rng.sample(list(FINE_AXES), 2):
            near = sorted(FINE_AXES[name], key=lambda value: abs(value - params[name]))[:3]
            params[name] = rng.choice(near)
        if key(params) not in used:
            candidates.append(params)
            used.add(key(params))
    return candidates[:count]


def run_batch(pool, jobs, stage, out, records):
    futures = {pool.submit(run_trial, params, seed): index for index, (params, seed) in enumerate(jobs)}
    path = out / "results.jsonl"
    with path.open("a") as stream:
        for done, future in enumerate(as_completed(futures), 1):
            row = {**future.result(), "stage": stage, "index": futures[future]}
            records.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(f"{stage} {done}/{len(jobs)}: {row['total_ms']:.6f} ms; "
                  f"best={min(r['total_ms'] for r in records):.6f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--params", type=Path)
    parser.add_argument("--candidates", type=Path, help="使用已保存的参数列表继续搜索边界。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--coarse", type=int, default=96)
    parser.add_argument("--refine", type=int, default=48)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        parser.error("请使用 PYTHONHASHSEED=0，以固定 ready 集合的遍历顺序。")
    args.out.mkdir(parents=True, exist_ok=False)
    if args.check:
        evaluator = make_evaluator()
        reference = json.loads((ROOT / "runs/obs_action_ablation/A/seed0/baselines.json").read_text())
        for name, choices in (("all_npu", [[0] * 9] * 17), ("all_pim", [[1] * 9] * 17)):
            actual = full_replay(evaluator, choices).now
            assert math.isclose(actual, reference[name]["latency_s"], rel_tol=0, abs_tol=1e-12)
        control = {"h": 3, "gamma": 0, "lambda": 0, "plan_hint_max": 3,
                   "eta": .1, "amort_alpha": 1, "reuse_window": 4, "amort_enable": False}
        old = run_trial(control, 0, save_plans=True)
        assert math.isclose(old["total_ms"], 138.96964094653808, rel_tol=0, abs_tol=1e-9)
        params = {"h": 2, "gamma": .2, "lambda": 1, "plan_hint_max": 3,
                  "eta": 5, "amort_alpha": 2, "reuse_window": 32}
        active = run_trial(params, 0, save_plans=True)
        assert all(active["score_calls"].get(k, 0) > 0 for k in
                   ("lookahead", "reuse_nonzero", "amort_nonzero", "consistency_conflicts"))
        verify_replay(old)
        verify_replay(active)
        (args.out / "checks.json").write_text(json.dumps({"control": old, "active": active}, indent=2))
        print(json.dumps({"status": "passed", "active_ms": active["total_ms"],
                          "score_calls": active["score_calls"]}), flush=True)
        return
    if args.params:
        params = json.loads(args.params.read_text())
        result = run_trial(params, args.seed, save_plans=True)
        result["independent_replay_ms"] = verify_replay(result)
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k not in ("plans", "phases_ms", "future_profiles")}))
        return
    start = time.perf_counter()
    records = []
    coarse = json.loads(args.candidates.read_text()) if args.candidates else coarse_candidates(args.coarse)
    (args.out / "search_space.json").write_text(json.dumps({
        "objective": "full 64+64 request simulated latency", "coarse_axes": COARSE_AXES,
        "fine_axes": FINE_AXES, "amort_enable": True, "amort_rmin": 1,
        "amort_reuse_prob": 1, "coarse_candidates": coarse, "workers": args.workers,
        "refine_count": args.refine, "validation_top": args.top,
        "validation_seeds": [0, 1, 2], "actual_remaining_output_visible": False}, indent=2))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        run_batch(pool, [(p, 0) for p in coarse], "coarse", args.out, records)
        ranked = sorted(records, key=lambda r: (r["total_ms"], key(r["params"])))
        fine = fine_candidates(ranked, args.refine, {key(r["params"]) for r in records})
        (args.out / "fine_candidates.json").write_text(json.dumps(fine, indent=2))
        run_batch(pool, [(p, 0) for p in fine], "fine", args.out, records)
        ranked = sorted(records, key=lambda r: (r["total_ms"], key(r["params"])))
        top = [r["params"] for r in ranked[:args.top]]
        run_batch(pool, [(p, seed) for p in top for seed in (1, 2)], "validation", args.out, records)
    validated = []
    for params in top:
        rows = [r for r in records if key(r["params"]) == key(params)]
        times = [r["total_ms"] for r in rows]
        validated.append({"params": params, "mean_ms": statistics.mean(times),
                          "std_ms": statistics.stdev(times), "max_ms": max(times),
                          "seeds": [{"seed": r["seed"], "total_ms": r["total_ms"]} for r in rows]})
    validated.sort(key=lambda r: (r["mean_ms"], r["max_ms"], key(r["params"])))
    best = validated[0]
    (args.out / "best_params.json").write_text(json.dumps(best["params"], indent=2) + "\n")
    final = run_trial(best["params"], 0, save_plans=True)
    final["independent_replay_ms"] = verify_replay(final)
    (args.out / "best_seed0_plan.json").write_text(json.dumps(final, indent=2) + "\n")
    summary = {"best": best, "validated_candidates": validated,
               "unique_parameter_sets": len(coarse) + len(fine), "search_runs": len(records),
               "wall_s": time.perf_counter() - start, "all_pim_ms": 63.51781877121468,
               "ppo_best_ms": 63.41597837637215, "best_plan_replay_ms": final["independent_replay_ms"]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "comparison.csv").open("w", newline="") as stream:
        parameter_names = sorted({k for row in records for k in row["params"]})
        fields = ["stage", "seed", *parameter_names, "prefill_ms", "decode_ms", "total_ms", "wall_s"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({"stage": row["stage"], "seed": row["seed"], **row["params"],
                             **{k: row[k] for k in ("prefill_ms", "decode_ms", "total_ms", "wall_s")}})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
