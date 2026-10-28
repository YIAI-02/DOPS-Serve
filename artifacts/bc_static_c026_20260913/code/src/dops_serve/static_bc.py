"""Static-task, whole-trajectory winner BC. Run simulation/training on compute nodes."""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import multiprocessing
import os
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MODELS = ["qwen_1.8b", "qwen_7b", "qwen_14b"]
METHODS = ["Bifocal", "Naive", "PD", "AF", "PD+Linear", "PD+Attn", "PD+FFN", "weights_on_pim", "NeuPIMs"]
VAL = {(1, 512), (8, 2048)}
TEST = {(4, 128), (16, 1024)}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refresh(t):
    return t < 2 or t % 2 == 0


def prepare(root, backend, c026_root, other_root):
    parameters = read(c026_root / "F000/result.json")["candidate"]["parameters"]
    cases = [] # 任务列表
    for i, (model, batch, prompt, decode) in enumerate(itertools.product(MODELS, [1, 4, 8, 16], [128, 512, 1024, 2048], [128, 256, 512, 1024])):
        split = "val" if (batch, prompt) in VAL else "test" if (batch, prompt) in TEST else "train"
        case = dict(case_id=i, model=model, batch=batch, prompt=prompt, decode=decode, split=split)
        source_config = c026_root / f"F{i:03d}" / "run_config.json"
        cfg = read(source_config)
        if (cfg["model_variant"], cfg["batch"], cfg["prefill_len"], cfg["decode_len"]) != (model.removeprefix("qwen_"), batch, prompt, decode):
            raise ValueError(f"C026 shape does not match case {i}")
        for key in ("algo", "baselines", "result_dir"):
            cfg.pop(key)
        cfg.update(decode_horizon_len=decode,
                   hardware_json=str(backend / "src/examples/hardware_1npu_2aim.json"),
                   pim_config_path=str(backend / "src/aim_simulator/PIM_AiM.json"),
                   ramulator_config_path=str(backend / "src/aim_simulator/example.yaml"),
                   simulation_log_file="/dev/null")
        config_path = root / "configs" / f"{i:03d}.json"
        write(config_path, cfg)
        cases.append({**case, "run_config_path": str(config_path), "source_run_config_path": str(source_config)})
    write(root / "cases.json", cases)
    return write(root / "protocol.json", dict(parameters=parameters, methods=METHODS, models=MODELS,
        c026_root=str(c026_root), other_root=str(other_root), decode_plan_refresh_stride=2,
        pim_weight_load_overlap_ratio=cfg["pim_weight_load_overlap_ratio"],
        weight_load_compute_overlap_ratio=cfg["weight_load_compute_overlap_ratio"],
        epochs=10, seed=7, learning_rate=1e-4, batch_plans=16, max_grad_norm=1.0,
        hidden=256, depth=4, state_dim=47, node_dim=35, test_locked=True,
        val_groups=sorted(VAL), test_groups=sorted(TEST), objective="single whole-task throughput winner",
        loss="mean learned-node conditional NLL, then mean plans per task, then mean tasks",
        decision_runtime="CPU, one torch thread; JIT warmed before timing; includes features/tensors/compile",
        counts={split: sum(c["split"] == split for c in cases) for split in ("train", "val", "test")}))


def import_teachers(root):
    """Reference existing native schedules; never read test schedules or their scores."""
    from .summary import read_summary_metadata
    protocol = read(root / "protocol.json")
    c026_root, other_root = Path(protocol["c026_root"]), Path(protocol["other_root"])
    imported, missing = [], []
    for case in read(root / "cases.json"):
        if case["split"] == "test":
            continue
        i = case["case_id"]
        cfg = read(case["run_config_path"])
        other_config_path = other_root / "teacher_configs" / f"{i:03d}.json"
        other_cfg = read(other_config_path)
        # These are the inspected defaults of the existing teacher backend's config.py.
        other_cfg.setdefault("pim_weight_load_overlap_ratio", 0.5)
        other_cfg.setdefault("weight_load_compute_overlap_ratio", 1.0)
        other_cfg.setdefault("weight_source", "host")
        other_cfg.setdefault("weight_format", "ND")
        for key in ("model_family", "model_variant", "dtype", "batch", "prefill_len", "decode_len",
                    "decode_sample_stride", "decode_plan_refresh_stride", "scheduler_seed", "npu_backend",
                    "pim_fast_mode", "tp_qkv", "tp_ffn", "weight_source", "weight_format",
                    "pim_weight_load_overlap_ratio", "weight_load_compute_overlap_ratio"):
            if other_cfg[key] != cfg[key]:
                raise ValueError(f"old teacher config mismatch: case {i}, {key}")
        folder = f"{case['model']}_fp16_b{case['batch']}_s2"
        for method in METHODS:
            if method == "Bifocal":
                base = c026_root / f"F{i:03d}" / "dops_output" / folder
                source_config = case["source_run_config_path"]
                recorded = read(c026_root / f"F{i:03d}" / "result.json")
                if recorded["candidate"]["parameters"] != protocol["parameters"]:
                    raise ValueError(f"C026 parameters changed at case {i}")
            else:
                base = Path(other_cfg["result_dir"]) / folder
                source_config = str(other_config_path)
            summary = base / f"algo_{method}" / f"best_summary_{case['prompt']}x{case['decode']}.json"
            if not summary.exists():
                missing.append(dict(case_id=i, method=method, summary_path=str(summary)))
                continue
            meta = read_summary_metadata(summary)
            if meta["policy"].removeprefix("algo:") != method:
                raise ValueError(f"unexpected imported teacher: {summary}")
            entry = dict(summary_path=str(summary), method=method, source_kind="reused",
                source_run_config_path=source_config, host_wall_s=None, decision_components=None,
                parameters=protocol["parameters"], kv_place=meta["plan_label"]["kv_place"],
                source_times=meta["best_times"], host=None)
            write(root / "teachers" / f"{i:03d}" / method / "generated.json", entry)
            imported.append(dict(case_id=i, method=method, summary_path=str(summary)))
    return write(root / "teacher_import.json", dict(imported_count=len(imported), missing_count=len(missing),
        imported=imported, missing=missing, test_schedules_read=False))


def generate_method(root, backend, case_id, method, timing_dir=None):
    from .summary import read_summary_metadata
    case = read(root / "cases.json")[case_id]
    if case["split"] == "test":
        raise ValueError("test tasks are locked")
    out = (timing_dir if timing_dir is not None else root / "teachers") / f"{case_id:03d}" / method
    cfg = read(case["run_config_path"])
    cfg.update(result_dir=str(out), algo=[method] if method in METHODS[:2] else [],
               baselines=[] if method in METHODS[:2] else [method])
    config_path = out / "run.json"
    write(config_path, cfg)
    sys.path.insert(0, str(backend / "src"))
    import config as native_config
    parameters = read(root / "protocol.json")["parameters"]
    for key, value in parameters.items():
        setattr(native_config, key, value)
    from stats_recorder import StatsRecorder
    def omit_event_csv(self, ops_csv_path, comms_csv_path):
        """This workflow consumes full schedule JSON, not duplicate event CSV exports."""
        return None
    StatsRecorder.dump_trace_csv = omit_event_csv
    # Time only PD device selection and plan export, excluding simulator cost execution.
    from scheduler.scheduler_naive import NaiveTopoScheduler
    from scheduler.scheduler_base import SchedulerBase
    timer = {"device_selection_s": 0.0, "plan_export_s": 0.0}
    pick = NaiveTopoScheduler._pick_best_allowed_device
    export = SchedulerBase.export_fixed_plan
    def timed_pick(self, *args, **kwargs):
        start = time.perf_counter()
        result = pick(self, *args, **kwargs)
        timer["device_selection_s"] += time.perf_counter() - start
        return result
    def timed_export(self, *args, **kwargs):
        start = time.perf_counter()
        result = export(self, *args, **kwargs)
        timer["plan_export_s"] += time.perf_counter() - start
        return result
    if method == "PD":
        NaiveTopoScheduler._pick_best_allowed_device = timed_pick
        SchedulerBase.export_fixed_plan = timed_export
    sys.argv = ["main.py", "evaluate", "--config", str(config_path), "--algo", ",".join(cfg["algo"]), "--baselines", ",".join(cfg["baselines"])]
    from mainlib.app import main
    start = time.perf_counter()
    main()
    summaries = list(out.rglob(f"best_summary_{case['prompt']}x{case['decode']}.json"))
    if len(summaries) != 1:
        raise ValueError(f"expected one source for {method}: {summaries}")
    metadata = read_summary_metadata(summaries[0])
    if metadata["policy"].removeprefix("algo:") != method:
        raise ValueError("unexpected teacher method")
    return write(out / "generated.json", dict(summary_path=str(summaries[0]), method=method,
        host_wall_s=time.perf_counter()-start, decision_components=timer, parameters=parameters,
        kv_place=metadata["plan_label"]["kv_place"], source_times=metadata["best_times"], host=platform.node(), raw_event_csv=False))


def scheduler_for(root, backend, case, summary_path, name):
    from .dops_fixed_replay import make_scheduler
    from .placement_compiler import StaticPlacementCompiler
    sched, graph, shape, cfg = make_scheduler(backend, {**case, "summary_path": str(summary_path)},
        read(root / "protocol.json")["parameters"], root / "runtime" / name, simulation_log_file="/dev/null")
    return sched, graph, shape, cfg, StaticPlacementCompiler(graph, sched)


def compact_plans(compiler, source):
    return {"prefill": compiler.normalize(source["prefill"]),
            "decode": {str(t): compiler.normalize(plan) for t, plan in enumerate(source["decode"]) if refresh(t)}}


def tensor_inputs(inputs):
    import torch
    return (torch.tensor([inputs["state"]], dtype=torch.float32),
            torch.tensor([inputs["nodes"]], dtype=torch.float32),
            torch.tensor([inputs["masks"]], dtype=torch.bool),
            torch.tensor(inputs["fixed_from"], dtype=torch.long))


def replay_task(root, backend, case, summary_path, *, plans=None, actor=None, capture=False, name="replay"):
    from .dops_fixed_replay import replay_phase
    from .placement_dataset import static_policy_inputs
    setup_start = time.perf_counter()
    sched, graph, shape, cfg, compiler = scheduler_for(root, backend, case, summary_path, name)
    # Graph/backend initialization is shared infrastructure, not a per-decision cost.
    setup_s = time.perf_counter() - setup_start
    now, decision_s, samples = 0.0, 0.0, []
    prefill, previous_decode = None, None
    if actor is not None:
        # Exclude first-use TorchScript compilation/kernel warmup from every case.
        warm = static_policy_inputs(sched, compiler, shape, case, "prefill", -1, now)
        warm_tensors = tensor_inputs(warm)
        actor.generate(*warm_tensors)
        actor.generate(*warm_tensors)
    for t in range(-1, case["decode"]):
        phase = "prefill" if t == -1 else "decode"
        sched.set_seq_len(case["prompt"] + max(0, t))
        if t >= 0:
            sched.set_decode_context(cur_token_idx=t, total_decode_tokens=case["decode"], cfg=cfg)
        if t == -1 or refresh(t):
            start = time.perf_counter()
            inputs = None
            if capture or actor is not None:
                inputs = static_policy_inputs(sched, compiler, shape, case, phase, t, now, prefill, previous_decode)
            if actor is None:
                source = plans["prefill"] if t == -1 else plans["decode"][str(t)]
                plan = compiler.normalize(source)
            else:
                actions = actor.generate(*tensor_inputs(inputs))[0].tolist()
                plan = compiler.compile(actions)
            decision_s += time.perf_counter() - start
            if capture:
                samples.append({"case_id": case["case_id"], "model": case["model"], "phase": phase,
                    "token_index": t, "now_s": now, **inputs, "actions": compiler.actions(plan)})
            if t == -1:
                prefill = plan
            else:
                previous_decode = plan
        now = replay_phase(sched, graph, phase, plan)
    sched.clear_decode_context()
    sched.cost.logger.close()
    return dict(simulated_s=now, decision_s=decision_s, setup_s=setup_s,
                output_tokens=case["batch"]*case["decode"], samples=samples)


def teacher_case(root, backend, case_id):
    from .dops_fixed_replay import load_fixed_plans
    case = read(root / "cases.json")[case_id]
    if case["split"] == "test":
        raise ValueError("test is locked")
    generated = []
    for method in METHODS:
        out = root / "teachers" / f"{case_id:03d}" / method
        if not (out / "generated.json").exists():
            subprocess.run([sys.executable, "-m", "dops_serve.static_bc", "generate-method", "--root", str(root),
                            "--backend", str(backend), "--case-id", str(case_id), "--method", method], check=True)
        generated.append(read(out / "generated.json"))
    if len({r["kv_place"] for r in generated}) != 1:
        raise ValueError("candidate initial KV placement differs")
    scores, normalized = [], {}
    for item in generated:
        method = item["method"]
        source = load_fixed_plans(Path(item["summary_path"]))
        sched, _, _, _, compiler = scheduler_for(root, backend, case, item["summary_path"], f"score_{case_id}")
        plans = compact_plans(compiler, source)
        for plan in [plans["prefill"], *plans["decode"].values()]:
            assert compiler.compile(compiler.actions(plan)) == plan
        sched.cost.logger.close()
        normalized[method] = plans
        write(root / "normalized" / f"{case_id:03d}" / f"{method}.json", plans)
        result = replay_task(root, backend, case, item["summary_path"], plans=plans, name=f"score_{case_id}") #第一次完整回放：给九种候选打分
        scores.append(dict(method=method, simulated_s=result["simulated_s"],
            throughput=result["output_tokens"]/result["simulated_s"],
            normalization_s=result["decision_s"], source_times=item["source_times"],
            native_decision_s=sum(item["decision_components"].values()) if item["decision_components"] is not None else None,
            source_host_wall_s=item["host_wall_s"]))
    winner = max(scores, key=lambda row: row["throughput"]) #这里找到了最大的
    source = generated[METHODS.index(winner["method"])]
    captured = replay_task(root, backend, case, source["summary_path"], plans=normalized[winner["method"]],
                           capture=True, name=f"anchors_{case_id}") #第二次完整回放：生成训练样本
    if not math.isclose(captured["simulated_s"], winner["simulated_s"], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("teacher repeated replay changed time")
    anchors = root / "anchors" / f"{case_id:03d}.jsonl.gz"
    anchors.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(anchors, "wt") as stream:
        for sample in captured["samples"]:
            stream.write(json.dumps(sample, separators=(",", ":")) + "\n")
    pd = scores[METHODS.index("PD")]
    return write(root / "results" / f"{case_id:03d}.json", {**case, "complete": True,
        "winner": winner["method"], "summary_path": source["summary_path"], "teacher_s": winner["simulated_s"],
        "pd_s": None if pd["native_decision_s"] is None else pd["simulated_s"]+pd["native_decision_s"]+pd["normalization_s"],
        "pd_simulated_s": pd["simulated_s"],
        "pd_decision_s": None if pd["native_decision_s"] is None else pd["native_decision_s"]+pd["normalization_s"],
        "output_tokens": captured["output_tokens"], "samples": len(captured["samples"]), "scores": scores,
        "repeat_time_match": True, "exact_three_device_roundtrip": True})


def collect(root):
    cases = [c for c in read(root / "cases.json") if c["split"] != "test"]
    results = [read(root / "results" / f"{c['case_id']:03d}.json") for c in cases]
    if not all(r["complete"] and r["repeat_time_match"] and r["exact_three_device_roundtrip"] for r in results):
        raise ValueError("teacher gate failed")
    rows_csv(root / "teacher_summary.csv", [{k: r[k] for k in ("case_id", "model", "batch", "prompt", "decode", "split", "winner", "teacher_s", "pd_s", "samples")} for r in results])
    return write(root / "dataset_ready.json", {"complete": True, "cases": len(results), "samples": sum(r["samples"] for r in results), "test_executed": False})


def load_samples(root, cases):
    import torch
    samples = []
    for case in cases:
        with gzip.open(root / "anchors" / f"{case['case_id']:03d}.jsonl.gz", "rt") as stream:
            task = [json.loads(line) for line in stream]
        for row in task:
            samples.append({**{k: row[k] for k in ("case_id", "model")}, "task_weight": 1.0/len(task),
                **{k: torch.tensor(row[k], dtype=dtype) for k, dtype in (
                    ("state", torch.float32), ("nodes", torch.float32), ("actions", torch.long),
                    ("masks", torch.bool), ("learned", torch.float32))}})
    return samples


def batches(samples, batch_size, rng=None):
    grouped = {}
    for row in samples:
        grouped.setdefault(row["model"], []).append(row)
    result = []
    for rows in grouped.values():
        rows = list(rows)
        if rng is not None:
            rng.shuffle(rows)
        result.extend(rows[i:i+batch_size] for i in range(0, len(rows), batch_size))
    if rng is not None:
        rng.shuffle(result)
    return result


def batch_tensors(rows, device):
    import torch
    return [torch.stack([r[key] for r in rows]).to(device) for key in ("state", "nodes", "actions", "masks", "learned")]


def loss_evaluate(actor, samples, device, batch_size):
    import torch
    totals, counts = {}, {}
    actor.eval()
    with torch.inference_mode():
        for rows in batches(samples, batch_size):
            values, _ = actor(*batch_tensors(rows, device))
            for row, value in zip(rows, values.cpu().tolist()):
                totals[row["model"]] = totals.get(row["model"], 0.0)+value*row["task_weight"]
                counts.setdefault(row["model"], set()).add(row["case_id"])
    return {m: totals[m]/len(counts[m]) for m in totals}


def evaluate_case(arguments):
    root, backend, case, checkpoint, route, epoch = arguments
    os.environ["PIM_LATENCY_CACHE_FILE"] = str(root / "runtime" / f"pim_{route}_{epoch}_{case['case_id']}.json")
    import torch
    from .placement_policy import StaticPlacementActor
    torch.set_num_threads(1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**state["dimensions"])
    actor.load_state_dict(state["actor"])
    actor.eval()
    scripted = torch.jit.script(actor)
    reference = read(root / "results" / f"{case['case_id']:03d}.json")
    # Measure PD on the same allocated CPU as BC, once per route; freeze for its ten epochs.
    pd_timing = root / "training" / route / "pd_timing"
    pd_measurement = pd_timing / f"{case['case_id']:03d}" / "timing.json"
    if epoch == 1:
        from .dops_fixed_replay import load_fixed_plans
        pd_timing.mkdir(parents=True, exist_ok=True)
        with (pd_timing / f"{case['case_id']:03d}.log").open("w") as stream:
            subprocess.run([sys.executable, "-m", "dops_serve.static_bc", "generate-method", "--root", str(root),
                "--backend", str(backend), "--case-id", str(case["case_id"]), "--method", "PD", "--timing-dir", str(pd_timing)],
                stdout=stream, stderr=subprocess.STDOUT, check=True)
        measured = read(pd_timing / f"{case['case_id']:03d}" / "PD/generated.json")
        original = read(root / "teachers" / f"{case['case_id']:03d}" / "PD/generated.json")
        source = load_fixed_plans(Path(measured["summary_path"]))
        previous = load_fixed_plans(Path(original["summary_path"]))
        if source["prefill"] != previous["prefill"] or source["decode"] != previous["decode"]:
            raise ValueError("PD timing replay changed the baseline plan")
        sched, _, _, _, compiler = scheduler_for(root, backend, case, reference["summary_path"], f"pd_timing_{route}_{case['case_id']}")
        started = time.perf_counter()
        compact_plans(compiler, source)
        normalization_s = time.perf_counter()-started
        sched.cost.logger.close()
        write(pd_measurement, dict(decision_s=sum(measured["decision_components"].values())+normalization_s,
            host=platform.node(), same_native_plan=True, normalization_s=normalization_s))
    pd_measured = read(pd_measurement)
    if pd_measured["host"] != platform.node():
        raise ValueError("PD and BC decision timing must use the same allocated host")
    with torch.inference_mode():
        result = replay_task(root, backend, case, reference["summary_path"], actor=scripted,
                             name=f"{route}_e{epoch}_{case['case_id']}")
    total = result["simulated_s"]+result["decision_s"]
    return {**{k: case[k] for k in ("case_id", "model", "batch", "prompt", "decode")},
        "route": route, "epoch": epoch, "winner": reference["winner"],
        "output_tokens": result["output_tokens"], "bc_s": total, "bc_simulated_s": result["simulated_s"],
        "decision_s": result["decision_s"], "teacher_s": reference["teacher_s"], "pd_s": reference["pd_simulated_s"]+pd_measured["decision_s"],
        "bc_throughput": result["output_tokens"]/total, "teacher_throughput": result["output_tokens"]/reference["teacher_s"],
        "teacher_gap_pct": 100*(reference["teacher_s"]/total-1), "bc_over_pd": (reference["pd_simulated_s"]+pd_measured["decision_s"])/total,
        "host": platform.node(), "cpu": platform.processor(), "torch_version": torch.__version__, "threads": 1}


def report(root, route, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = root / "training" / route
    rows_csv(base / "training_progress.csv", history)
    titles = {"train_nll": "Training conditional NLL", "val_nll": "Validation conditional NLL",
              "bc_throughput": "Output throughput (tokens/s)",
              "teacher_gap_pct": "Throughput gap to teacher (%)", "bc_over_pd": "BC / PD throughput", "decision_s": "Decision time (s)"}
    figures = base / "figures"
    figures.mkdir(exist_ok=True)
    for key, title in titles.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for model in sorted({r["model"] for r in history}):
            rows = [r for r in history if r["model"] == model]
            line = ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o", label=model)
            if key == "bc_throughput":
                ax.axhline(rows[0]["teacher_throughput"], color=line[0].get_color(), linestyle="--", label=f"{model} teacher")
        ax.set(xlabel="Epoch", ylabel=title)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(figures / f"epoch_{key}.{extension}", dpi=170)
        plt.close(fig)
    lines = [f"# {route}: BC epoch history", "", "|Epoch|Model|Train NLL|Val NLL|BC token/s|Teacher gap|BC/PD|Decision s|", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"|{r['epoch']}|{r['model']}|{r['train_nll']:.5f}|{r['val_nll']:.5f}|{r['bc_throughput']:.4f}|{r['teacher_gap_pct']:+.2f}%|{r['bc_over_pd']:.4f}|{r['decision_s']:.3f}|" for r in history]
    (base / "report.md").write_text("\n".join(lines)+"\n")


def check_native_case(arguments):
    root, backend, case = arguments
    os.environ["PIM_LATENCY_CACHE_FILE"] = str(root / "runtime" / f"native_check_{case['case_id']}.json")
    from .dops_fixed_replay import load_fixed_plans, replay_phase, replay_decode
    raw_reference_checks = []
    for method in METHODS:
        source = read(root / "teachers" / f"{case['case_id']:03d}" / method / "generated.json")
        plans = load_fixed_plans(Path(source["summary_path"]))
        sched, graph, _, cfg, _ = scheduler_for(root, backend, case, source["summary_path"], f"native_check_{case['case_id']}")
        sched.set_seq_len(case["prompt"])
        prefill_s = replay_phase(sched, graph, "prefill", plans["prefill"])
        decode_s = sum(replay_decode(sched, graph, cfg, plans["decode"], prefill_s))
        sched.cost.logger.close()
        expected = source["source_times"]
        if not math.isclose(prefill_s, expected["prefill"], rel_tol=1e-9, abs_tol=1e-12) or not math.isclose(decode_s, expected["decode"], rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"native fixed replay does not reproduce DOPS source: {case['model']} {method}")
        raw_reference_checks.append(dict(method=method, reference_s=expected["total"], replay_s=prefill_s+decode_s))
    return raw_reference_checks


def smoke(root, backend):
    import torch
    from .placement_policy import StaticPlacementActor
    torch.set_num_threads(1)
    torch.manual_seed(7)
    if not torch.cuda.is_available():
        raise RuntimeError("smoke must exercise GPU training")
    cases = [read(root / "cases.json")[i] for i in (0, 64, 128)]
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(check_native_case, [(root, backend, c) for c in cases]))
    native_checks = {c["case_id"]: r for c, r in zip(cases, results)}
    actor = StaticPlacementActor(47, 35).cuda()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    checks = []
    for case in cases:
        raw_reference_checks = native_checks[case["case_id"]]
        samples = load_samples(root, [case])
        rows = samples[:1]
        inputs = batch_tensors(rows, "cuda")
        first = actor(*inputs)[0].item()
        for _ in range(3):
            optimizer.zero_grad()
            loss = actor(*inputs)[0].mean()
            loss.backward()
            optimizer.step()
        last = actor(*inputs)[0].item()
        if not math.isfinite(last) or last >= first:
            raise ValueError("GPU update did not learn the fixed smoke teacher")
        checks.append(dict(model=case["model"], state_dim=inputs[0].shape[-1], node_dim=inputs[1].shape[-1],
            nodes=inputs[1].shape[1], first_nll=first, last_nll=last, samples=len(samples), native_checks=raw_reference_checks))
    checkpoint = root / "smoke.pt"
    torch.save(dict(actor={k: v.cpu() for k, v in actor.state_dict().items()}, dimensions=actor.dimensions), checkpoint)
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        replay = list(pool.map(evaluate_case, [(root, backend, cases[0], checkpoint, "smoke", 1)]))[0]
    return write(root / "smoke_passed.json", dict(complete=True, checks=checks, autonomous_replay=replay,
        gpu=torch.cuda.get_device_name(0), parameter_count=sum(p.numel() for p in actor.parameters()),
        smoke_checkpoint_excluded_from_training=True))


def train(root, backend, route, workers):
    import torch
    from .placement_policy import StaticPlacementActor
    if not read(root / "dataset_ready.json")["complete"]:
        raise ValueError("teacher dataset not ready")
    if not read(root / "smoke_passed.json")["complete"]:
        raise ValueError("GPU/native replay smoke did not pass")
    if not read(root / "inference_passed.json")["complete"]:
        raise ValueError("optimized inference check did not pass")
    protocol = read(root / "protocol.json")
    torch.set_num_threads(1) #初始化网络参数
    torch.manual_seed(protocol["seed"])
    rng = random.Random(protocol["seed"])
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("BC training requires the requested GPU allocation")
    models = MODELS if route == "shared" else [route]
    cases = [c for c in read(root / "cases.json") if c["model"] in models]
    train_cases = [c for c in cases if c["split"] == "train"]
    val_cases = [c for c in cases if c["split"] == "val"]
    train_samples, val_samples = load_samples(root, train_cases), load_samples(root, val_cases)
    actor = StaticPlacementActor(protocol["state_dim"], protocol["node_dim"], protocol["hidden"], protocol["depth"]).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol["learning_rate"])
    base = root / "training" / route
    base.mkdir(parents=True, exist_ok=True)
    write(base / "initialized.json", dict(fresh_initialization=True, route=route, seed=protocol["seed"],
        parameter_count=sum(p.numel() for p in actor.parameters()), dimensions=actor.dimensions,
        train_tasks=len(train_cases), val_tasks=len(val_cases), train_plans=len(train_samples),
        host=platform.node(), torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0)))
    history = []
    for epoch in range(1, protocol["epochs"]+1):
        start = time.perf_counter()
        actor.train()
        update_batches = batches(train_samples, protocol["batch_plans"], rng)
        for rows in update_batches:
            optimizer.zero_grad(set_to_none=True)
            losses, _ = actor(*batch_tensors(rows, device))
            weights = torch.tensor([r["task_weight"] for r in rows], device=device)
            loss = (losses*weights).mean()*len(train_samples)/len(train_cases)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), protocol["max_grad_norm"])
            optimizer.step()
        train_nll = loss_evaluate(actor, train_samples, device, protocol["batch_plans"])
        val_nll = loss_evaluate(actor, val_samples, device, protocol["batch_plans"])
        checkpoint = base / f"epoch_{epoch:03d}.pt"
        torch.save(dict(actor={k: v.cpu() for k, v in actor.state_dict().items()}, optimizer=optimizer.state_dict(),
            dimensions=actor.dimensions, epoch=epoch, seed=protocol["seed"], random_state=rng.getstate(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), checkpoint)
        write(base / "status.json", dict(epoch=epoch, stage="validation", train_nll=train_nll, val_nll=val_nll))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(evaluate_case, [(root, backend, c, checkpoint, route, epoch) for c in val_cases]))
        rows_csv(base / f"epoch_{epoch:03d}_cases.csv", results)
        for model in models:
            rows = [r for r in results if r["model"] == model]
            output = sum(r["output_tokens"] for r in rows)
            bc, teacher, pd = (sum(r[k] for r in rows) for k in ("bc_s", "teacher_s", "pd_s"))
            simulated = sum(r["bc_simulated_s"] for r in rows)
            history.append(dict(route=route, epoch=epoch, model=model, train_nll=train_nll[model], val_nll=val_nll[model],
                bc_throughput=output/bc, teacher_throughput=output/teacher, pd_throughput=output/pd,
                teacher_gap_pct=100*(teacher/bc-1), bc_over_pd=pd/bc,
                simulation_only_teacher_gap_pct=100*(teacher/simulated-1),
                decision_s=sum(r["decision_s"] for r in rows), epoch_wall_s=time.perf_counter()-start))
        report(root, route, history)
        write(base / "status.json", dict(epoch=epoch, stage="epoch_complete", rows=history[-len(models):]))
        print(json.dumps(history[-len(models):]), flush=True)
    return write(base / "complete.json", dict(complete=True, epochs=protocol["epochs"], awaiting_user_before_continuation=True))


def inference_check(root, backend):
    import torch
    from .placement_policy import StaticPlacementActor
    torch.set_num_threads(1)
    checkpoint = root / "smoke.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**state["dimensions"])
    actor.load_state_dict(state["actor"])
    actor.eval()
    actor = torch.jit.script(actor)
    checked = 0
    with torch.inference_mode():
        for i in (0, 64, 128):
            with gzip.open(root / "anchors" / f"{i:03d}.jsonl.gz", "rt") as stream:
                for line in stream:
                    inputs = json.loads(line)
                    tensor_args = tensor_inputs(inputs)
                    actions = actor.generate(*tensor_args)
                    learned = torch.tensor([inputs["learned"]], dtype=torch.bool)
                    _, logits = actor(tensor_args[0], tensor_args[1], actions, tensor_args[2], learned)
                    torch.testing.assert_close(actions[learned], logits.argmax(-1)[learned], rtol=0, atol=0)
                    checked += 1
    case = read(root / "cases.json")[0]
    result = evaluate_case((root, backend, case, checkpoint, "smoke_optimized", 1))
    return write(root / "inference_passed.json", dict(complete=True, real_states_checked=checked,
        native_gru_conditionals_match=True, autonomous_replay=result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "import-teachers", "generate-method", "teacher", "collect", "smoke", "inference-check", "train"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--c026-root", type=Path)
    parser.add_argument("--other-root", type=Path)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--route", choices=[*MODELS, "shared"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timing-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        result = prepare(args.root, args.backend, args.c026_root, args.other_root)
    elif args.stage == "import-teachers":
        result = import_teachers(args.root)
    elif args.stage == "generate-method":
        result = generate_method(args.root, args.backend, args.case_id, args.method, args.timing_dir)
    elif args.stage == "teacher":
        result = teacher_case(args.root, args.backend, args.case_id)
    elif args.stage == "collect":
        result = collect(args.root)
    elif args.stage == "smoke":
        result = smoke(args.root, args.backend)
    elif args.stage == "inference-check":
        result = inference_check(args.root, args.backend)
    else:
        result = train(args.root, args.backend, args.route, args.workers)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
