from __future__ import annotations


def validate_placement_backend(backend_root, artifact_dir, config):
    """Run scope/capacity/continuity probes; these are not serving comparisons."""
    import json
    import os
    import platform
    import time
    from pathlib import Path

    from .placement_evaluator import PlacementEvaluator, RequestShape

    cases = []
    for model in config["models"]:
        evaluator = PlacementEvaluator(backend_root, artifact_dir, model, config)
        compiler = evaluator.compiler
        for workload, prompts in (
            ("equal_single", [128]),
            ("ragged_full_batch", [128, 2048] * 8),
            ("worst_prefill", [2048] * 16),
        ):
            for policy in ("npu", "pim", "mixed_shards"):
                snapshot = evaluator.initial.clone()
                for index, prompt in enumerate(prompts):
                    evaluator.reserve(snapshot, str(index), prompt, 1024)
                started = time.perf_counter()
                program = [list(a) for a in compiler.uniform(int(policy == "pim"))]
                if policy == "mixed_shards":
                    for block, actions in zip(compiler.blocks, program):
                        for index, node in enumerate(block.node_ids):
                            if node.endswith("_s1") and compiler.mask[node][1]:
                                actions[index] = 1
                plans = {phase: compiler.compile(program) for phase in ("prefill", "decode")}
                decision_s = time.perf_counter() - started
                snapshot.now += decision_s
                prefill = evaluator.evaluate(snapshot, plans["prefill"], "prefill", [RequestShape(str(i), p, 0) for i, p in enumerate(prompts)])
                prefill.snapshot.now += 0.01
                decode = evaluator.evaluate(prefill.snapshot, plans["decode"], "decode", [RequestShape(str(i), 1, p) for i, p in enumerate(prompts)])
                cases.append({
                    "model": model, "workload": workload, "placement": policy,
                    "kv_reserved_bytes_by_pim": decode.snapshot.scheduler.label.kv_bytes_by_pim,
                    "prefill_s": prefill.duration_s, "decode_s": decode.duration_s,
                    "complete_plan_pair_compile_s": decision_s,
                    "prefill_host_evaluator_s": prefill.host_evaluator_s,
                    "decode_host_evaluator_s": decode.host_evaluator_s,
                    "prefill_peak_runtime_bytes": prefill.peak_runtime_bytes,
                    "decode_peak_runtime_bytes": decode.peak_runtime_bytes,
                    "communication_bytes": sum(e["bytes"] for step in (prefill, decode) for e in step.comm_events),
                    "activation_store_events": sum(e["tag"] == "act_store" for step in (prefill, decode) for e in step.comm_events),
                })
                print(json.dumps({"validated_cases": len(cases), "model": model, "workload": workload, "placement": policy}), flush=True)
    result = {
        "passed": True, "cases": cases,
        "cpu": next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
        "host": platform.node(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "npu_backend": config["npu_backend"], "pim_fast_mode": config["pim_fast_mode"],
        "scope": "decoder DAG integration and capacity probes, not policy training or serving performance comparisons",
    }
    (Path(artifact_dir) / "backend_gate.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"passed": True, "cases": len(cases)}
