"""Paired validation of Static, fixed-order Bifocal, BC and PPO checkpoints."""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

from .placement_evaluation import StaticPlacement, load_traces, replay_metrics, scaled_trace
from .placement_evaluator import PlacementEvaluator
from .replay import PlacementReplay


def prepare_comparison(artifact_dir, output_dir, protocol, *, methods, split="val", trace_ids=None):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    traces = trace_ids if trace_ids is not None else protocol["validation_trace_ids"]
    cases = [{"case_id": i, **method, "model": model, "load_multiplier": load, "trace_id": trace}
             for i, (method, model, load, trace) in enumerate(
                 (method, model, load, trace) for method in methods for model in protocol["models"]
                 for load in protocol["test_load_multipliers"] for trace in traces)]
    result = {"split": split, "trace_ids": traces, "protocol": protocol, "cases": cases,
              "repeats": 3, "scope": protocol["scope"], "checkpoint_compatibility": "same backend, feature ABI and deterministic CPU selector; rewards may differ"}
    (destination / "comparison_cases.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate_case(backend_root, artifact_dir, output_dir, case_id):
    import torch
    from .placement_policy import LearnedPlacement
    from .placement_training import load_actor
    from .placement_evaluation import host_signature

    destination = Path(output_dir)
    manifest = json.loads((destination / "comparison_cases.json").read_text())
    case = manifest["cases"][case_id]
    protocol = manifest["protocol"]
    runtime = protocol.get("runtime", {})
    torch.set_num_threads(runtime.get("sample_threads", 1))
    signature = host_signature(runtime.get("sample_threads", 1))
    if signature != runtime["validation_signature"]:
        raise ValueError(f"validation host/runtime changed: {signature}")
    config, model = protocol["backend_config"], case["model"]
    calibration = protocol["models"][model]
    trace_ids = [case["trace_id"]]
    traces = load_traces(artifact_dir, manifest["split"], trace_ids)
    folder = destination / "cases" / f"{case_id:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    actor, checkpoint = None, None
    if case["checkpoint"]:
        actor, checkpoint = load_actor(case["checkpoint"])
        expected_stage = "BC" if case["method"] == "bc" else "PPO"
        if checkpoint["stage"] != expected_stage or checkpoint["seed"] != case["seed"] or checkpoint["exclude_model"] is not None:
            raise ValueError("comparison checkpoint differs from the declared method, seed or model split")
        if checkpoint.get("models") is not None and model not in checkpoint["models"]:
            raise ValueError("checkpoint has not been trained for this comparison model")
        if expected_stage == "PPO":
            compatible = checkpoint["protocol"]["backend_config"] == config
            if not compatible:
                raise ValueError("PPO checkpoint and comparison backend/protocol differ")
            if "epoch" in case and checkpoint["epoch"] != case["epoch"]:
                raise ValueError("comparison checkpoint has the wrong epoch")
        actor.eval()
    results, baseline_config = [], None
    for trace_id, rows in traces.items():
        for repeat in range(manifest.get("repeats", 1)):
            evaluator = PlacementEvaluator(backend_root, folder / "backend", model, config)
            records = scaled_trace(rows, calibration["reference_rps"] * case["load_multiplier"])
            replay = PlacementReplay(records, evaluator, slo_s=calibration.get("slo_s"), max_batch=config["max_batch"], decode_window=config["decode_window"])
            if case["method"] == "static":
                selector = StaticPlacement(evaluator, calibration["static"]["programs"])
            elif case["method"] == "bifocal":
                from .placement_baselines import FixedOrderBifocalPlacement
                from scheduler import scheduler_bifocal as bifocal
                baseline_config = {key: getattr(bifocal, key) for key in (
                    "SCHED_JOINT_LK_ENABLE", "SCHED_JOINT_LK_H", "SCHED_JOINT_LK_GAMMA",
                    "SCHED_JOINT_LK_CONSIST_LAMBDA", "SCHED_JOINT_LK_PLAN_HINT_MAX", "SCHED_WEIGHT_BIAS_ETA",
                )}
                selector = FixedOrderBifocalPlacement(evaluator, lambda: replay.snapshot, max_batch=config["max_batch"], seed=7)
            else:
                selector = LearnedPlacement(actor, evaluator, deterministic=True)
            started = time.perf_counter()
            while not replay.done:
                replay.run_window(selector, calibration["reward"])
                if len(replay.windows) % 50 == 0:
                    print(json.dumps({"case_id": case_id, "trace_id": trace_id, "repeat": repeat, "windows": len(replay.windows), "completed": len(replay.completed)}), flush=True)
            metrics = replay_metrics(replay, time.perf_counter() - started)
            result = {"trace_id": trace_id, **metrics, "cumulative_reward": sum(w["reward"] for w in replay.windows), "windows": len(replay.windows)}
            result["repeat"] = repeat
            results.append(result)
            prefix = f"{trace_id}_repeat_{repeat}"
            (folder / f"{prefix}_windows.jsonl").write_text("".join(json.dumps(w) + "\n" for w in replay.windows))
            (folder / f"{prefix}_requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replay.request_metrics()))
    result = {
        **case, "split": manifest["split"], "scope": manifest["scope"], "host": platform.node(), "signature": signature,
        "reference_rps": calibration["reference_rps"], "traces": results, "baseline_config": baseline_config,
        "checkpoint_metadata": {k: checkpoint[k] for k in ("stage", "seed", "epoch", "updates", "continuation") if checkpoint is not None and k in checkpoint},
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result



def read_throughput_results(directory):
    """Read only complete, three-repeat, hardware-matched comparison cases."""
    directory = Path(directory)
    manifest = json.loads((directory / "comparison_cases.json").read_text())
    rows = []
    for case in manifest["cases"]:
        path = directory / "cases" / f"{case['case_id']:03d}" / "result.json"
        result = json.loads(path.read_text())
        if any(result[key] != value for key, value in case.items()):
            raise ValueError(f"case {case['case_id']} does not match its manifest")
        if result["signature"] != manifest["protocol"]["runtime"]["validation_signature"]:
            raise ValueError("comparison hardware/runtime differs from the frozen protocol")
        if [(r["trace_id"], r["repeat"]) for r in result["traces"]] != [(case["trace_id"], i) for i in range(3)]:
            raise ValueError("comparison requires three distinct complete repeats")
        for trace in result["traces"]:
            if trace["requests"] != 1000:
                raise ValueError("comparison requires all 1000 requests")
            rows.append({**case, **trace})
    return manifest, rows
