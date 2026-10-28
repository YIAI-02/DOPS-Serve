"""Fresh fast-teacher / BC / value / PPO experiment. Workloads run on compute."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

MODELS = ["qwen_1.8b", "qwen_7b", "qwen_14b"]
SEEDS = [7, 17, 27]
METHODS = ["Bifocal", "Naive", "PD", "AF", "PD+Linear", "PD+Attn", "PD+FFN", "weights_on_pim", "NeuPIMs"]
BRANCHES = ["shared", *[f"independent/{model}" for model in MODELS]]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return value


def selected_models(branch):
    return MODELS if branch == "shared" else [branch.split("/")[1]]


def shape_cases():
    return [dict(case_id=i, model=m, batch=b, prompt=p, decode=d)
            for i, (m, b, p, d) in enumerate(itertools.product(MODELS, [1, 4, 8, 16], [128, 512, 1024, 2048], [128, 256, 512, 1024]))]


def prepare(repo, root, backend):
    from .placement_evaluator import build_backend
    source = repo / "artifacts/fast_bc_restart_20260908"
    for filename in ("workloads.jsonl", "workload_windows.jsonl", "workload_manifest.json"):
        target = root / filename
        if not target.exists():
            target.symlink_to(source / filename)
    config = read(repo / "configs/op_policy_v1.json")
    config["status"] = "fresh fast teacher and fresh BC; fixed throughput objective"
    write(root / "backend_config.json", config)
    # Keep the established split after retiring the earlier experiment.
    manifest = read(source / "trace_manifest.json")
    write(root / "trace_manifest.json", manifest)
    write(root / "teacher_cases.json", {"cases": shape_cases(), "methods": METHODS, "evaluations": 1728, "pilot_case_ids": [0, 64, 128]})
    for model in MODELS:
        build_backend(backend, root / "preflight_backend", model, config)
    return write(root / "prepared.json", {"complete": True, "config": config, "trace_manifest": manifest})


def teacher(root, backend, case_id):
    case = read(root / "teacher_cases.json")["cases"][case_id]
    status = root / "teacher_status" / f"{case_id:03d}.json"
    if status.exists():
        return read(status)
    config = read(root / "backend_config.json")
    output = root / "teacher_runs" / config["hardware_key"] / f"{case['model']}_fp16_b{case['batch']}_s2"
    cfg = {"model_family": "qwen", "model_variant": case["model"].removeprefix("qwen_"), "dtype": "fp16",
           "batch": case["batch"], "prefill_len": case["prompt"], "decode_len": case["decode"],
           "decode_sample_stride": 2, "decode_plan_refresh_stride": 2, "scheduler_seed": 7,
           "npu_backend": "fast", "pim_fast_mode": True, "tp_qkv": 2, "tp_ffn": 2,
           "hardware_json": str(backend / "src/examples/hardware_1npu_2aim.json"),
           "pim_config_path": str(backend / "src/aim_simulator/PIM_AiM.json"),
           "ramulator_config_path": str(backend / "src/aim_simulator/example.yaml"),
           "algo": METHODS[:2], "baselines": METHODS[2:], "dump_graph": False, "result_dir": str(output)}
    path = root / "teacher_configs" / f"{case_id:03d}.json"
    write(path, cfg)
    started = time.perf_counter()
    subprocess.run([sys.executable, str(backend / "src/main.py"), "evaluate", "--config", str(path),
                    "--algo", ",".join(METHODS[:2]), "--baselines", ",".join(METHODS[2:])], cwd=backend / "src", check=True)
    return validate_teacher(root, case_id, host_wall_s=time.perf_counter() - started)


def validate_teacher(root, case_id, *, host_wall_s=None):
    from .summary import read_summary_metadata
    case = read(root / "teacher_cases.json")["cases"][case_id]
    config = read(root / "teacher_configs" / f"{case_id:03d}.json")
    output = Path(config["result_dir"])
    summaries = sorted(output.rglob(f"best_summary_{case['prompt']}x{case['decode']}.json"))
    raw_policies = [read_summary_metadata(p)["policy"] for p in summaries]
    policies = [name.removeprefix("algo:") for name in raw_policies]
    if len(policies) != 9 or set(policies) != set(METHODS):
        raise ValueError(f"teacher methods differ from nine approved methods: {policies}")
    return write(root / "teacher_status" / f"{case_id:03d}.json", {"complete": True, **case, "policies": policies,
                 "raw_policies": raw_policies, "summaries": [str(p) for p in summaries], "host_wall_s": host_wall_s,
                 "existing_output_validation": host_wall_s is None})


def import_teachers(root, backend):
    from .placement_dataset import build
    for case in shape_cases():
        if not read(root / "teacher_status" / f"{case['case_id']:03d}.json")["complete"]:
            raise ValueError("teacher generation incomplete")
    result = build(root / "teacher_runs", MODELS, read(root / "backend_config.json")["hardware_key"],
                   backend_root=backend, artifact_dir=root, config=read(root / "backend_config.json"), workers=16, strict=True)
    if result["complete_sources"] != 1728:
        raise ValueError("new teacher import must include all 1728 evaluations")
    return result


def pilot_gate(root, backend):
    from .placement_dataset import build, reevaluate
    config = read(root / "backend_config.json")
    destination = root / "pilot_import"
    manifest = build(root / "teacher_runs", MODELS, config["hardware_key"], backend_root=backend,
                     artifact_dir=destination, config=config, workers=3, strict=True)
    if manifest["complete_sources"] != 27:
        raise ValueError("pilot requires three models times all nine methods")
    scores = reevaluate(backend, destination, config, workers=3)
    return write(root / "pilot_gate.json", {"complete": True, "import": manifest, "reevaluation": scores})


def profile_runtime(root, backend):
    import torch
    from .placement_evaluation import load_traces, host_signature
    from .placement_evaluator import PlacementEvaluator
    from .placement_policy import PlacementActor, LearnedPlacement, policy_inputs
    from .placement_training import run_update
    from .replay import PlacementReplay
    config = read(root / "backend_config.json")
    trace_id = read(root / "trace_manifest.json")["train_trace_ids"][0]
    records = [{**row, "arrival_time": 0.0} for row in load_traces(root, "train", [trace_id])[trace_id]]
    settings = {"ppo_epochs": 4, "clip": .2, "value_coefficient": .5, "entropy_coefficient": .001, "max_grad_norm": 1.0}
    results = []
    for model in MODELS:
        torch.set_num_threads(1); torch.manual_seed(7)
        evaluator = PlacementEvaluator(backend, root / "runtime_probe/backend", model, config)
        replay = PlacementReplay(records, evaluator, max_batch=config["max_batch"], decode_window=config["decode_window"])
        replay._arrivals()
        inputs = policy_inputs(replay.observation(), evaluator, "prefill")
        actor = PlacementActor(len(inputs["state"]), len(inputs["nodes"][0][0]))
        selector = LearnedPlacement(actor, evaluator, deterministic=False)
        windows = []
        started = time.perf_counter()
        for _ in range(4):
            result = replay.run_window(selector, {"a_N": 1., "a_T": 1.})
            windows.append({**copy.deepcopy(selector.last), "reward": result["reward"], "terminal": result["terminal"]})
            if replay.done:
                break
        sampling_s = time.perf_counter() - started
        for threads in (1, 2, 4):
            trials = []
            for _ in range(2):
                candidate = copy.deepcopy(actor)
                optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-4)
                trials.append(run_update(candidate, optimizer, windows, 0., settings, {"sample_threads": 1, "update_threads": threads, "update_device": "cpu"})["update_wall_s"])
            results.append({"model": model, "sample_threads": 1, "update_threads": threads, "sample_s": sampling_s, "update_s": statistics.median(trials), "update_repeats_s": trials})
    selected = min((1, 2, 4), key=lambda threads: sum(r["sample_s"] + r["update_s"] for r in results if r["update_threads"] == threads))
    return write(root / "runtime_probe/profile.json", {"rows": results, "runtime": {"sample_threads": 1, "update_threads": selected, "update_device": "cpu", "validation_signature": host_signature(1)}, "fresh_random_probe_only": True})


def freeze(root):
    from .placement_evaluation import assemble_protocol, host_signature
    config = read(root / "backend_config.json")
    protocol = assemble_protocol(root, config)
    manifest = read(root / "trace_manifest.json")
    protocol.update(objective="throughput", test_locked=True, confirmation_locked=True,
                    runtime=read(root / "runtime_probe/profile.json")["runtime"],
                    **{k: manifest[k] for k in ("train_trace_ids", "validation_trace_ids", "confirmation_trace_ids", "test_trace_ids")})
    write(root / "training_protocol.json", protocol)
    from .placement_comparison import prepare_comparison
    prepare_comparison(root, root / "validation/static", protocol, methods=[{"method": "static", "seed": None, "checkpoint": None}])
    for branch in BRANCHES:
        models = selected_models(branch)
        branch_protocol = {**protocol, "training_models": models, "models": {m: protocol["models"][m] for m in models}}
        write(root / branch / "training_protocol.json", branch_protocol)
    return protocol


def prepare_validation(root, branch, stage, epoch):
    from .placement_comparison import prepare_comparison
    base = root / branch
    protocol = read(base / "training_protocol.json")
    methods = [{"method": stage, "seed": seed, "checkpoint": str(base / stage / f"epoch_{epoch:03d}" / f"seed_{seed}" / ("bc_last.pt" if stage == "bc" else "ppo.pt")), "epoch": epoch} for seed in SEEDS]
    return prepare_comparison(root, base / stage / f"validation_{epoch:03d}", protocol, methods=methods)


def throughput_scores(rows, references):
    """Median within each case, then paired geometric mean across cases."""
    reference = defaultdict(list)
    for row in references:
        reference[(row["model"], row["trace_id"], row["load_multiplier"])].append(row["output_token_throughput"])
    cases = defaultdict(list)
    for row in rows:
        cases[(row["model"], row["seed"], row["trace_id"], row["load_multiplier"])].append(row)
    groups = defaultdict(list)
    for (model, seed, trace, load), repeats in cases.items():
        denominator = statistics.median(reference[(model, trace, load)])
        values = [r["output_token_throughput"] / denominator for r in sorted(repeats, key=lambda r: r["repeat"])]
        groups[(model, seed)].append(values)
    return [{"model": model, "seed": seed, "ratio": math.exp(statistics.fmean(math.log(statistics.median(v)) for v in values)),
             "repeats": [math.exp(statistics.fmean(math.log(v[i]) for v in values)) for i in range(3)], "conditions": len(values)}
            for (model, seed), values in sorted(groups.items())]


def select_progress(history, stage):
    """Common epoch across seeds; no two-seed stable model regression."""
    best = None
    stale = 0
    decisions = []
    origin = {(r["model"], r["seed"]): r for r in history[0]["scores"]}
    for item in history:
        score = statistics.fmean(math.log(row["ratio"]) for row in item["scores"])
        old = None if best is None else history[best]
        regressions = []
        if old is not None:
            for row in item["scores"]:
                baseline = origin[(row["model"], row["seed"])]
                noise = max(math.log(max(r["repeats"]) / min(r["repeats"])) for r in (row, baseline))
                if math.log(baseline["ratio"] / row["ratio"]) > noise:
                    regressions.append(row)
        stable = any(sum(r["model"] == m for r in regressions) >= 2 for m in {r["model"] for r in item["scores"]})
        def repeat_range(entry):
            values = [statistics.fmean(math.log(r["repeats"][i]) for r in entry["scores"]) for i in range(3)]
            return max(values) - min(values)
        noise = 0 if old is None else max(repeat_range(old), repeat_range(item))
        improved = old is None or (score > statistics.fmean(math.log(r["ratio"]) for r in old["scores"]) + noise and not stable)
        if improved:
            best, stale = history.index(item), 0
        else:
            stale += 1
        passed = stage == "bc" and all(r["ratio"] >= .95 for r in item["scores"])
        stop = passed or stale >= 3 or item["epoch"] >= 10
        # BC is frozen immediately on a common passing epoch.
        if passed:
            best = history.index(item)
        decisions.append({"epoch": item["epoch"], "best_epoch": history[best]["epoch"], "passed": passed,
                          "no_improvement": stale, "stop": stop, "regressions": [{"model": r["model"], "seed": r["seed"]} for r in regressions],
                          "open_cap": 10 if item["epoch"] >= 5 and stale < 3 else 5})
        if stop:
            break
    if len(decisions) != len(history):
        raise ValueError("epochs submitted after stage had already stopped")
    return decisions[-1]


def summarize(root, branch, stage, epoch):
    from .placement_comparison import read_throughput_results
    base = root / branch
    _, reference = read_throughput_results(root / "validation/static")
    first = 1 if stage == "bc" else 0
    history, all_rows = [], []
    for number in range(first, epoch + 1):
        _, rows = read_throughput_results(base / stage / f"validation_{number:03d}")
        history.append({"epoch": number, "scores": throughput_scores(rows, reference)})
        all_rows.extend({**r, "epoch": number, "stage": stage} for r in rows)
    state = {**select_progress(history, stage), "history": history, "stage": stage, "complete": True}
    write(base / stage / f"decision_{epoch:03d}.json", state)
    write(base / stage / "state.json", state)
    export_progress(base, stage, history, all_rows)
    progress = []
    for item in history:
        for score in item["scores"]:
            folder = base / stage / f"epoch_{item['epoch']:03d}" / f"seed_{score['seed']}"
            row = {"stage": stage, "epoch": item["epoch"], "model": score["model"], "seed": score["seed"], "relative_static_pct": 100 * score["ratio"]}
            if stage == "bc":
                training = read(folder / "bc_summary.json")
                current = training["history"][-1]
                row.update(training_loss=current["training_loss"], candidate_regret=current["validation"]["candidate_regret"],
                           cumulative_anchor_updates=sum(r["optimizer_steps"] for r in training["history"]),
                           cumulative_training_hours=sum(r["wall_s"] for r in training["history"]) / 3600)
            else:
                training = read(folder / "ppo_summary.json") if item["epoch"] else {"completed_traces": [], "updates": 0, "training_wall_s": 0}
                row.update(cumulative_requests=sum(r["completed"] for r in training["completed_traces"] if r["model"] == score["model"]),
                           cumulative_windows=sum(r["windows"] for r in training["completed_traces"] if r["model"] == score["model"]),
                           policy_update_batches=training["updates"], cumulative_training_hours=training["training_wall_s"] / 3600)
            progress.append(row)
    import csv
    with (base / stage / "training_progress.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(progress[0])); writer.writeheader(); writer.writerows(progress)
    return state


def export_progress(base, stage, history, rows):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = base / stage / "figures"
    folder.mkdir(exist_ok=True)
    compact = [{k: r[k] for k in ("epoch", "stage", "model", "seed", "trace_id", "load_multiplier", "repeat", "output_token_throughput", "e2e_p99_s", "decision_total_s")} for r in rows]
    aggregate = [{"epoch": item["epoch"], **{k: r[k] for k in ("model", "seed", "ratio", "conditions")}, "repeat_min": min(r["repeats"]), "repeat_max": max(r["repeats"])} for item in history for r in item["scores"]]
    for name, data in (("epoch_raw.csv", compact), ("epoch_aggregated.csv", aggregate)):
        with (folder / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    for metric, label in (("output_token_throughput", "Output tokens/s"), ("e2e_p99_s", "E2E p99 (s)"), ("decision_total_s", "Decision time (s)")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model, seed in sorted({(r["model"], r["seed"]) for r in rows}):
            epochs = sorted({r["epoch"] for r in rows})
            values = []
            for epoch in epochs:
                cases = defaultdict(list)
                for row in rows:
                    if row["model"] == model and row["seed"] == seed and row["epoch"] == epoch:
                        cases[(row["trace_id"], row["load_multiplier"])].append(row[metric])
                values.append(math.exp(statistics.fmean(math.log(max(statistics.median(v), 1e-12)) for v in cases.values())))
            ax.plot(epochs, values, marker="o", label=f"{model}, seed {seed}")
        ax.set_xticks(epochs)
        ax.set(xlabel=f"{stage.upper()} epoch", ylabel=f"Geometric mean of case medians: {label}", title="Decoder DAG scheduling replay")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.2); fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(folder / f"epoch_{metric}.{extension}", dpi=180)
        plt.close(fig)


def fit_value_head(actor, datasets, *, epochs=10, learning_rate=1e-4):
    import torch
    def load(item):
        return torch.load(item["path"], weights_only=False, mmap=True) if "path" in item else item

    backbone = {k: v.clone() for k, v in actor.state_dict().items() if not k.startswith("value_head.")}
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith("value_head."))
    optimizer = torch.optim.Adam(actor.value_head.parameters(), lr=learning_rate)
    models = sorted({r["model"] for r in datasets})
    constants = {}
    for model in models:
        total, count = 0., 0
        for item in datasets:
            if item["model"] == model and item["split"] == "fit":
                labels = load(item)["returns"]
                total += float(labels.sum()); count += len(labels)
        constants[model] = total / count

    def check():
        result = {}
        with torch.no_grad():
            for model in models:
                errors, baseline = [], []
                for item in datasets:
                    if item["model"] == model and item["split"] == "check":
                        data = load(item)
                        for x, y in zip(data["features"].split(2048), data["returns"].split(2048)):
                            errors.append(actor.value_head(x).squeeze(-1) - y)
                            baseline.append(constants[model] - y)
                errors, baseline = torch.cat(errors), torch.cat(baseline)
                result[model] = {"mse": float(errors.square().mean()), "bias": float(errors.mean()), "constant_mse": float(baseline.square().mean()), "blocks": len(errors)}
        return result

    before = check()
    best_state, best_error, best_epoch, stale = copy.deepcopy(actor.value_head.state_dict()), math.inf, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        squared, blocks, updates = 0.0, 0, 0
        for index in torch.randperm(len(datasets)).tolist():
            item = datasets[index]
            if item["split"] != "fit":
                continue
            data = load(item)
            for indices in torch.randperm(len(data["returns"])).split(2048):
                prediction = actor.value_head(data["features"][indices]).squeeze(-1)
                loss = (prediction - data["returns"][indices]).square().mean()
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.value_head.parameters(), 1.0)
                optimizer.step()
                squared += float(loss.detach()) * len(indices); blocks += len(indices); updates += 1
        validation = check()
        error = statistics.fmean(r["mse"] for r in validation.values())
        history.append({"epoch": epoch, "train_mse": squared / blocks, "train_blocks": blocks, "updates": updates, "check": validation})
        if error < best_error:
            best_state, best_error, best_epoch, stale = copy.deepcopy(actor.value_head.state_dict()), error, epoch, 0
        else:
            stale += 1
        if stale >= 3:
            break
    actor.value_head.load_state_dict(best_state)
    for key, value in backbone.items():
        torch.testing.assert_close(value, actor.state_dict()[key], rtol=0, atol=0)
    for parameter in actor.parameters():
        parameter.requires_grad_(True)
    after = check()
    return {"passed": all(after[m]["mse"] < before[m]["mse"] for m in models), "actor_parameters_unchanged": True,
            "before": before, "after": after, "best_epoch": best_epoch, "history": history}


def warmup(root, backend, branch, seed):
    import torch
    from .placement_training import load_actor, _save, throughput_schedule
    from .placement_policy import LearnedPlacement
    from .placement_evaluation import load_traces, scaled_trace
    from .placement_evaluator import PlacementEvaluator
    from .replay import PlacementReplay
    base = root / branch
    gate = read(base / "bc/state.json")
    if not gate["passed"] or not gate["stop"]:
        raise ValueError("BC is not frozen at a passing common epoch")
    source = base / "bc" / f"epoch_{gate['best_epoch']:03d}" / f"seed_{seed}/bc_last.pt"
    actor, checkpoint = load_actor(source)
    torch.manual_seed(seed)
    actor.eval()
    protocol = read(base / "training_protocol.json")
    folder = base / "value" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    traces = load_traces(root, "train", protocol["train_trace_ids"])
    datasets, accounting = [], []
    probe_inputs = None
    sampling_started = time.perf_counter()
    for model, trace, load in throughput_schedule(protocol, 0, seed):
        evaluator = PlacementEvaluator(backend, folder / "backend", model, protocol["backend_config"])
        calibration = protocol["models"][model]
        replay = PlacementReplay(scaled_trace(traces[trace], calibration["reference_rps"] * load), evaluator,
                                 max_batch=protocol["backend_config"]["max_batch"], decode_window=protocol["backend_config"]["decode_window"])
        selector = LearnedPlacement(actor, evaluator, deterministic=False)
        features, rewards = [], []
        while not replay.done:
            result = replay.run_window(selector, calibration["reward"])
            with torch.no_grad():
                feature = actor.pair(selector.last["inputs"], selector.last["programs"], value_features=True)["value_features"].detach()
            features.append(feature); rewards.append(result["reward"])
            probe_inputs = selector.last["inputs"]
            if len(rewards) % 50 == 0:
                print(json.dumps({"value_sampling": model, "seed": seed, "trace": trace, "windows": len(rewards), "completed": len(replay.completed)}), flush=True)
        running, returns = 0.0, []
        for feature, reward in reversed(list(zip(features, rewards))):
            running += reward
            returns.append(torch.full((len(feature),), running))
        split = "fit" if protocol["train_trace_ids"].index(trace) in (0, 2, 4) else "check"
        data = {"model": model, "trace_id": trace, "load_multiplier": load, "split": split,
                "features": torch.cat(features), "returns": torch.cat(list(reversed(returns)))}
        data_path = folder / f"{model}_{trace}.pt"
        torch.save(data, data_path)
        datasets.append({"model": model, "split": split, "path": str(data_path)})
        accounting.append({"model": model, "trace_id": trace, "load": load, "split": split, "requests": len(replay.completed), "windows": len(features), "blocks": len(data["returns"])})
        del data, features, returns
    sampling_s = time.perf_counter() - sampling_started
    with torch.no_grad():
        policy_before = actor.pair(probe_inputs, deterministic=True)
    started = time.perf_counter()
    result = fit_value_head(actor, datasets)
    with torch.no_grad():
        policy_after = actor.pair(probe_inputs, deterministic=True)
    if policy_before["programs"] != policy_after["programs"]:
        raise ValueError("value pretraining changed actor outputs")
    torch.testing.assert_close(policy_before["log_probs"], policy_after["log_probs"], rtol=0, atol=0)
    result.update(seed=seed, source_checkpoint=str(source), sampling_s=sampling_s, training_s=time.perf_counter() - started,
                  actor_outputs_unchanged=True, traces=accounting)
    write(folder / "value_summary.json", result)
    if result["passed"]:
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(actor.parameters(), lr=protocol["ppo"]["learning_rate"])
        destination = base / "ppo/epoch_000" / f"seed_{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        _save(actor, destination / "ppo.pt", stage="PPO", seed=seed, epoch=0, updates=0, exclude_model=None,
              models=selected_models(branch), protocol=protocol, completed_traces=[], schedule=[],
              optimizer_state_dict=optimizer.state_dict(), torch_rng_state=torch.get_rng_state(), python_rng_state=random.Random(seed).getstate(),
              source_checkpoint=str(source), continuation="fresh BC plus value-only pretraining", training_wall_s=0.0)
    return result


def freeze_final(root):
    candidates = []
    for branch in BRANCHES:
        base = root / branch
        bc = read(base / "bc/state.json")
        if not bc["stop"]:
            raise ValueError("global confirmation/test lock: BC still active")
        for seed in SEEDS:
            candidates.append({"branch": branch, "method": "bc", "epoch": bc["best_epoch"], "seed": seed,
                               "checkpoint": str(base / "bc" / f"epoch_{bc['best_epoch']:03d}" / f"seed_{seed}/bc_last.pt")})
        if not bc["passed"]:
            continue
        values = [read(base / "value" / f"seed_{seed}/value_summary.json") for seed in SEEDS]
        if not all(value["passed"] for value in values):
            continue
        ppo = read(base / "ppo/state.json")
        if not ppo["stop"]:
            raise ValueError("global confirmation/test lock: PPO still active")
        for stage, epoch in (("ppo", 0), ("ppo", ppo["best_epoch"])):
            for seed in SEEDS:
                row = {"branch": branch, "method": stage, "epoch": epoch, "seed": seed,
                       "checkpoint": str(base / stage / f"epoch_{epoch:03d}" / f"seed_{seed}" / ("bc_last.pt" if stage == "bc" else "ppo.pt"))}
                if row not in candidates:
                    candidates.append(row)
    return write(root / "final_candidates.json", {"all_branches_frozen": True, "candidates": candidates})


def prepare_final(root, split):
    from .placement_comparison import prepare_comparison
    frozen = read(root / "final_candidates.json")
    if not frozen["all_branches_frozen"]:
        raise ValueError("confirmation and test remain locked")
    protocol = read(root / "training_protocol.json")
    traces = protocol["confirmation_trace_ids" if split == "confirmation" else "test_trace_ids"]
    dataset_split = "val" if split == "confirmation" else "test"
    folder = root / split
    manifest = prepare_comparison(root, folder, protocol, methods=[{"method": "static", "seed": None, "checkpoint": None}], split=dataset_split, trace_ids=traces)
    for method in frozen["candidates"]:
        for model, load, trace in itertools.product(selected_models(method["branch"]), protocol["test_load_multipliers"], traces):
            manifest["cases"].append({"case_id": len(manifest["cases"]), **method, "model": model, "load_multiplier": load, "trace_id": trace})
    return write(folder / "comparison_cases.json", manifest)


def final_report(root, split):
    import csv
    from .placement_comparison import read_throughput_results
    _, rows = read_throughput_results(root / split)
    references = [r for r in rows if r["method"] == "static"]
    summaries = []
    for branch, method, epoch in sorted({(r["branch"], r["method"], r["epoch"]) for r in rows if "branch" in r}):
        group = [r for r in rows if (r.get("branch"), r["method"], r.get("epoch")) == (branch, method, epoch)]
        summaries.extend({"branch": branch, "stage": method, "epoch": epoch, **r} for r in throughput_scores(group, references))
    write(root / split / "summary.json", {"complete": True, "split": split, "scores": summaries, "selection_feedback_allowed": False})
    if summaries:
        with (root / split / "summary.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    return {"complete": True, "split": split, "scores": summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--branch", default="shared", choices=BRANCHES)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--phase", choices=["bc", "ppo"], default="bc")
    parser.add_argument("--comparison-dir")
    parser.add_argument("--split", choices=["confirmation", "test"], default="confirmation")
    args = parser.parse_args()
    repo, root = Path(args.repo_root), Path(args.root)
    backend = root / "backend/DOPS"
    import torch
    torch.set_num_threads(1)
    if args.stage == "prepare":
        result = prepare(repo, root, backend)
    elif args.stage == "teacher":
        result = teacher(root, backend, args.case_id)
    elif args.stage == "import":
        result = import_teachers(root, backend)
    elif args.stage == "profile":
        result = profile_runtime(root, backend)
    elif args.stage == "pilot-gate":
        result = pilot_gate(root, backend)
    elif args.stage == "check-pilot":
        result = [validate_teacher(root, case) for case in (0, 64, 128)]
    elif args.stage == "reevaluate":
        from .placement_dataset import reevaluate
        result = reevaluate(backend, root, read(root / "backend_config.json"), workers=16)
    elif args.stage in ("calibrate", "calibrate-recover"):
        from .placement_evaluation import calibrate_model
        result = calibrate_model(backend, root, read(root / "backend_config.json"), MODELS[args.case_id], train_count=5, validation_count=5, throughput_only=True, resume_trials=args.stage == "calibrate-recover")
    elif args.stage == "freeze":
        result = freeze(root)
    elif args.stage == "bc":
        from .placement_training import train_bc
        base = root / args.branch / "bc"
        previous = base / f"epoch_{args.epoch - 1:03d}" / f"seed_{args.seed}/bc_last.pt" if args.epoch > 1 else None
        result = train_bc(root, base / f"epoch_{args.epoch:03d}" / f"seed_{args.seed}", seed=args.seed, epochs=1,
                          models=selected_models(args.branch), checkpoint_path=previous)
    elif args.stage == "warmup":
        result = warmup(root, backend, args.branch, args.seed)
    elif args.stage == "ppo":
        from .placement_training import train_ppo
        base = root / args.branch
        protocol = read(base / "training_protocol.json")
        result = train_ppo(backend, root, protocol["backend_config"], base / "ppo" / f"epoch_{args.epoch - 1:03d}" / f"seed_{args.seed}/ppo.pt",
                           base / "ppo" / f"epoch_{args.epoch:03d}" / f"seed_{args.seed}", protocol, seed=args.seed, trace_epochs=1)
    elif args.stage == "prepare-validation":
        result = prepare_validation(root, args.branch, args.phase, args.epoch)
    elif args.stage == "evaluate":
        from .placement_comparison import evaluate_case
        result = evaluate_case(backend, root, args.comparison_dir, args.case_id)
    elif args.stage == "summarize":
        result = summarize(root, args.branch, args.phase, args.epoch)
    elif args.stage == "freeze-final":
        result = freeze_final(root)
    elif args.stage == "prepare-final":
        result = prepare_final(root, args.split)
    elif args.stage == "final-report":
        result = final_report(root, args.split)
    else:
        raise ValueError(args.stage)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
