from __future__ import annotations

import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .replay import PlacementReplay, _quantile


def host_signature(threads):
    import torch
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    return {"cpu_model": cpu, "torch_version": torch.__version__, "sample_threads": threads}


def load_traces(artifact_dir, split, trace_ids):
    wanted = set(trace_ids)
    traces = defaultdict(list)
    with (Path(artifact_dir) / "workloads.jsonl").open() as handle:
        for row in map(json.loads, handle):
            if row["trace_id"] in wanted:
                if row["split"] != split:
                    raise ValueError("requested trace crosses the chronological split")
                traces[row["trace_id"]].append(row)
                if set(traces) == wanted and all(len(rows) == 1000 for rows in traces.values()):
                    break
    if set(traces) != wanted or any(len(rows) != 1000 for rows in traces.values()):
        raise ValueError("placement experiments require complete 1000-request traces")
    return dict(traces)


def scaled_trace(records, target_rps):
    duration = records[-1]["arrival_time"] - records[0]["arrival_time"]
    scale = (len(records) - 1) / duration / target_rps
    origin = records[0]["arrival_time"]
    return [{**row, "arrival_time": (row["arrival_time"] - origin) * scale} for row in records]


def choose_trace_ids(artifact_dir, split, count):
    windows = [row["trace_id"] for row in map(json.loads, (Path(artifact_dir) / "workload_windows.jsonl").open()) if row["split"] == split]
    if count > len(windows):
        raise ValueError("requested more traces than available in split")
    # Uniform positions in time, chosen before looking at performance.
    return [windows[i * len(windows) // count] for i in range(count)]


class StaticPlacement:
    def __init__(self, evaluator, programs):
        self.evaluator, self.programs = evaluator, programs
        self.plans = None

    def __call__(self, observation, current):
        if self.plans is None:
            self.plans = {phase: self.evaluator.compiler.compile(program) for phase, program in self.programs.items()}
        return self.plans


def replay_trace(records, evaluator, selector, config, slo_s):
    replay = PlacementReplay(records, evaluator, slo_s=slo_s, max_batch=config["max_batch"], decode_window=config["decode_window"])
    started = time.perf_counter()
    while not replay.done:
        replay.run_window(selector, {"a_N": 0, "a_T": 0, "lambda_q": 0, "lambda_l": 0})
    return replay, replay_metrics(replay, time.perf_counter() - started)


def replay_metrics(replay, host_wall_s):
    metrics = replay.request_metrics()
    duration = replay.snapshot.now - min(r.arrival_time for r in replay.requests)
    result = {
        "requests": len(metrics), "output_tokens": sum(len(r.token_times) for r in replay.requests),
        "duration_s": duration, "host_wall_s": host_wall_s,
        "e2e_p95_s": _quantile([r["e2e_s"] for r in metrics], 0.95),
        "e2e_p99_s": _quantile([r["e2e_s"] for r in metrics], 0.99),
        "ttft_p95_s": _quantile([r["ttft_s"] for r in metrics], 0.95),
        "tpot_p95_s": _quantile([r["tpot_s"] for r in metrics if r["tpot_s"] is not None], 0.95),
        "tbt_p95_s": _quantile([t for r in metrics for t in r["tbt_s"]], 0.95),
        "queue_wait_p95_s": _quantile([r["queue_wait_s"] for r in metrics], 0.95),
        "placement_switches": sum(w["placement_switches"] for w in replay.windows),
        "decision_s": [w["decision_s"] for w in replay.windows],
        "communication_bytes": sum(r["communication_bytes"] for r in replay.step_results),
        "communication_s_sum": sum(r["communication_s_sum"] for r in replay.step_results),
    }
    result["output_token_throughput"] = result["output_tokens"] / duration
    if replay.slo_s is not None:
        result["e2e_violations"] = sum(r["e2e_violated"] for r in metrics)
        result["e2e_violation_rate"] = result["e2e_violations"] / len(metrics)
    result["decision_total_s"] = sum(result["decision_s"])
    result["decision_p95_s"] = _quantile(result["decision_s"], 0.95)
    result["decision_fraction"] = result["decision_total_s"] / duration
    return result


def static_candidates(artifact_dir, model):
    directory = Path(artifact_dir)
    placements = {row["placement_id"]: row for row in map(json.loads, (directory / "placements.jsonl").open())}
    summary = json.loads((directory / "reevaluation_summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("calibration requires completed historical reevaluation")
    training_sources = {source_id for item in summary["files"] if item["split"] == "train" for source_id in item["source_ids"]}
    pairs = {}
    for source in map(json.loads, (directory / "history_sources.jsonl").open()):
        if source["model_id"] != model or source["source_id"] not in training_sources:
            continue
        prefill = next(p["placement_id"] for p in source["placements"] if p["phase"] == "prefill")
        for decode in {p["placement_id"] for p in source["placements"] if p["phase"] == "decode"}:
            pairs.setdefault((prefill, decode), []).append(source["source_id"])
    return [{
        "placement_ids": dict(zip(("prefill", "decode"), pair)),
        "programs": {phase: placements[p]["block_actions"] for phase, p in zip(("prefill", "decode"), pair)},
        "source_ids": sorted(set(pairs[pair])),
    } for pair in sorted(pairs)]


def calibrate_model(backend_root, artifact_dir, config, model, *, train_count=1, validation_count=1, candidate_budget=2, throughput_only=False, resume_trials=False):
    from .placement_evaluator import PlacementEvaluator, RequestShape

    directory = Path(artifact_dir)
    destination = directory / "calibration" / model
    destination.mkdir(parents=True, exist_ok=True)
    evaluator = PlacementEvaluator(backend_root, destination / "backend", model, config)
    train_ids = choose_trace_ids(directory, "train", train_count)
    val_ids = choose_trace_ids(directory, "val", validation_count)
    training = load_traces(directory, "train", train_ids)
    validation = {} if throughput_only else load_traces(directory, "val", val_ids)
    candidates = static_candidates(directory, model)
    # Evaluate every historical phase pair from the same cold training batch;
    # its decode probe inherits its own actual prefill state and caches.
    probe_records = next(iter(training.values()))[:config["max_batch"]]
    snapshot = evaluator.initial.clone()
    for row in probe_records:
        evaluator.reserve(snapshot, row["request_id"], row["prompt_len"], row["max_tokens"])
    prefill_shapes = [RequestShape(row["request_id"], row["prompt_len"], 0) for row in probe_records]
    # Output counts are offline train labels here, never policy observations.
    expected_iterations = statistics.fmean(row["output_len"] - 1 for row in probe_records)
    screening = json.loads((destination / "static_screening.json").read_text()) if resume_trials else []
    for index, candidate in (() if resume_trials else enumerate(candidates)):
        plans = {p: evaluator.compiler.compile(program) for p, program in candidate["programs"].items()}
        prefill = evaluator.evaluate(snapshot, plans["prefill"], "prefill", prefill_shapes)
        decode_shapes = [RequestShape(row["request_id"], 1, row["prompt_len"]) for row in probe_records]
        decode = evaluator.evaluate(prefill.snapshot, plans["decode"], "decode", decode_shapes)
        screening.append({"candidate": index, "placement_ids": candidate["placement_ids"], "screening_cost_s": prefill.duration_s + expected_iterations * decode.duration_s})
    shortlist = sorted(screening, key=lambda r: (r["screening_cost_s"], r["candidate"]))[:candidate_budget]
    if not resume_trials:
        (destination / "static_screening.json").write_text(json.dumps(screening, indent=2) + "\n")
    total_candidates = len(candidates)
    candidates = [candidates[row["candidate"]] for row in shortlist]
    trials = json.loads((destination / "static_trials.json").read_text()) if resume_trials else []
    if resume_trials:
        if len(trials) != len(candidates) or any(
            row["placement_ids"] != candidates[i]["placement_ids"] or [t["trace_id"] for t in row["traces"]] != train_ids
            for i, row in enumerate(trials)
        ):
            raise ValueError("calibration recovery requires all original candidates and full training traces")
        winner = max(range(len(trials)), key=lambda i: trials[i]["throughput"])
    selected_replays = {}
    recovery_host_wall_s = 0.0
    for index in ([winner] if resume_trials else range(len(candidates))):
        candidate = candidates[index]
        trial_metrics, replays = [], []
        for trace_id, rows in training.items():
            saturated = [{**row, "arrival_time": 0.0} for row in rows]
            replay, metrics = replay_trace(saturated, evaluator, StaticPlacement(evaluator, candidate["programs"]), config, None if throughput_only else math.inf)
            replays.append(replay)
            trial_metrics.append({"trace_id": trace_id, **metrics})
        throughput = sum(r["output_tokens"] for r in trial_metrics) / sum(r["duration_s"] for r in trial_metrics)
        selected_replays[index] = replays
        if resume_trials:
            recovery_host_wall_s = sum(row["host_wall_s"] for row in trial_metrics)
            print(json.dumps({"calibration_model": model, "recovered_candidate": index, "coefficient_replay_throughput": throughput}), flush=True)
        else:
            trials.append({"candidate": index, "placement_ids": candidate["placement_ids"], "throughput": throughput, "traces": trial_metrics})
            print(json.dumps({"calibration_model": model, "static_candidate": index, "throughput": throughput}), flush=True)
            (destination / "static_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    best = max(range(len(trials)), key=lambda i: trials[i]["throughput"])
    replays = selected_replays[best]
    # All requests are waiting initially. The queue stays nonempty until the
    # last admission; the later drain does not enter the reference-rate window.
    busy_times = [max(r.prefill_start for r in replay.requests) for replay in replays]
    busy_completions = [sum(r.completion_time <= cutoff for r in replay.requests) for replay, cutoff in zip(replays, busy_times)]
    reference_rps = sum(busy_completions) / sum(busy_times)
    val_metrics, latencies, val_replays = [], [], []
    for trace_id, rows in validation.items():
        replay, metrics = replay_trace(scaled_trace(rows, 0.5 * reference_rps), evaluator, StaticPlacement(evaluator, candidates[best]["programs"]), config, math.inf)
        latencies.extend(r["e2e_s"] for r in replay.request_metrics())
        val_replays.append(replay)
        val_metrics.append({"trace_id": trace_id, **metrics})
    slo_s = None if throughput_only else _quantile(latencies, 0.99)
    windows = [w for replay in replays for w in replay.windows]
    output_scale = statistics.median(w["output_tokens"] for w in windows)
    time_scale = statistics.median(w["duration_s"] for w in windows)
    reward = {"a_N": 1 / max(1, output_scale), "a_T": 1 / time_scale}
    if not throughput_only:
        waiting_scale = statistics.median(w["waiting_s"] for w in windows)
        reward.update(lambda_q=1 / max(slo_s, waiting_scale), lambda_l=1.0)
    result = {
        "model_id": model, "reference_rps": reference_rps,
        **({"slo_s": slo_s} if not throughput_only else {}),
        "reward_scale_medians": {"output_tokens": output_scale, "duration_s": time_scale},
        "static": candidates[best], "reward": reward,
        "train_trace_ids": train_ids, "validation_trace_ids": val_ids,
        "static_candidate_budget": candidate_budget,
        "static_candidates_screened": total_candidates,
        "static_selection": "all training-history phase pairs scored on one common cold train batch with continuous prefill/decode; shortlist then full saturated training-trace throughput",
        "rate_window": "initial cold load through last admission; exclude final drain",
        "validation_results": val_metrics,
        "pilot_host_wall_s": sum(t["host_wall_s"] for trial in trials for t in trial["traces"]) + sum(t["host_wall_s"] for t in val_metrics),
        **({"recovery_host_wall_s": recovery_host_wall_s, "static_selection_reused": True} if resume_trials else {}),
    }
    (destination / "model_protocol.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def assemble_protocol(artifact_dir, config):
    directory = Path(artifact_dir)
    models = {model: json.loads((directory / "calibration" / model / "model_protocol.json").read_text()) for model in config["models"]}
    train_ids = models[config["models"][0]]["train_trace_ids"]
    validation_ids = models[config["models"][0]]["validation_trace_ids"]
    if any(row["train_trace_ids"] != train_ids or row["validation_trace_ids"] != validation_ids for row in models.values()):
        raise ValueError("model variants must use the same source windows")
    protocol = {
        "status": "calibrated pilot; formal testing remains locked", "test_frozen": False,
        "backend_config": config, "models": models,
        "train_trace_ids": train_ids, "validation_trace_ids": validation_ids,
        "training_load_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5],
        "test_load_multipliers": [0.5, 0.75, 1.0, 1.25, 1.5],
        "ppo": {"learning_rate": 1e-4, "ppo_epochs": 4, "clip": 0.2, "value_coefficient": 0.5, "entropy_coefficient": 0.001, "max_grad_norm": 1.0, "rollout_windows": 4, "trace_epochs": 1, "gamma": 1.0, "gae_lambda": 1.0},
        "scope": "decoder DAG timing/scheduling replay",
    }
    (directory / "training_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol
