from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .inventory import _MODEL_DIR, _assert_outside_source, _inventory_row
from .placement_evaluator import HARDWARE_KEY, MODEL_KEYS, build_backend
from .summary import extract_plan, read_summary_metadata


def _read_history_source(args):
    root, path, parsed_path = args
    try:
        if parsed_path.is_file():
            payload = json.loads(parsed_path.read_text())
            if payload["source"]["summary_path"] != str(path):
                raise ValueError("cached source path differs from current inventory")
            return payload, None
        row = _inventory_row(root, path)
        metadata = read_summary_metadata(path)
        if metadata["config"].get("dtype", "fp16").lower() != "fp16":
            raise ValueError("summary dtype disagrees with FP16 source directory")
        plan = extract_plan(path, max_decode_steps=int(row["decode_len"]), ops_trace_path=row["ops_trace_path"])
        if not plan["placement_complete"] or plan["decode_steps_covered"] != int(row["decode_len"]):
            raise ValueError("source does not contain a complete decode trajectory")
        return {"source": row, "metadata": metadata, "plan": plan}, None
    except (ValueError, KeyError, TypeError, OSError) as error:
        return None, str(error)


def build(
    source: str | Path,
    model_keys: list[str],
    hardware_key: str,
    *,
    backend_root: str | Path,
    artifact_dir: str | Path,
    config: dict[str, Any],
    workers: int = 8,
    strict: bool = False,
) -> dict[str, Any]:
    """Inventory every in-scope source and project every complete trajectory.

    Historical times remain audit evidence. No training cost is emitted until
    continuous-state reevaluation has passed its separate integration gate.
    """
    if hardware_key != HARDWARE_KEY or set(model_keys) - set(MODEL_KEYS):
        raise ValueError("history manifest is restricted to Qwen / 1NPU2PIM")
    root, destination = Path(source).resolve(), Path(artifact_dir).resolve()
    _assert_outside_source(root, destination)
    scope = root / hardware_key
    if not scope.is_dir():
        raise NotADirectoryError(scope)
    destination.mkdir(parents=True, exist_ok=True)
    raw_dir = destination / "historical_plans"
    raw_dir.mkdir(exist_ok=True)
    compilers = {}
    graphs = {}
    for model in model_keys:
        scheduler, shape, compiler = build_backend(backend_root, destination, model, config)
        compilers[model] = compiler
        bytes_per_token_per_pim = 2 * (shape.n_kv_heads // 2) * shape.head_dim * shape.layer_num * 2
        graphs[model] = {
            "layers": shape.layer_num, "hidden_size": shape.dim,
            "q_heads": shape.n_heads, "kv_heads": shape.n_kv_heads, "head_dim": shape.head_dim,
            "nodes": len(compiler.order), "free_nodes": len(compiler.mask),
            "blocks": [asdict(b) for b in compiler.blocks],
            "pim_owner": compiler.pim_owner, "mask": compiler.mask,
            "fixed_devices": compiler.fixed_devices, "order": compiler.order,
            "kv_bytes_per_token_per_pim": bytes_per_token_per_pim,
            "worst_kv_reserved_bytes_per_pim": bytes_per_token_per_pim * config["max_batch"] * (config["prompt_max"] + config["max_tokens"]),
        }
    (destination / "graphs.json").write_text(json.dumps(graphs, indent=2) + "\n")
    paths = []
    for directory in sorted(scope.iterdir()):
        match = _MODEL_DIR.fullmatch(directory.name)
        if match and match["model"] in model_keys and match["dtype"].lower() == "fp16":
            paths.extend(sorted(directory.rglob("best_summary_*.json")))
    counts: Counter = Counter()
    by_model: Counter = Counter()
    by_policy: Counter = Counter()
    program_ids: dict[tuple, str] = {}
    outputs = (destination / "history_sources.jsonl", destination / "placements.jsonl", destination / "history_excluded.jsonl")
    jobs = ((root, path, raw_dir / f"source_{index:06d}.json") for index, path in enumerate(paths))
    with ProcessPoolExecutor(max_workers=workers) as pool, outputs[0].open("w") as sources, outputs[1].open("w") as placements, outputs[2].open("w") as excluded:
        for index, (path, (payload, error)) in enumerate(zip(paths, pool.map(_read_history_source, jobs))):
            counts["scanned_sources"] += 1
            source_id = f"source_{index:06d}"
            try:
                if error is not None:
                    raise ValueError(error)
                row, metadata, plan = payload["source"], payload["metadata"], payload["plan"]
                compiler = compilers[row["model_id"]]
                phase_mappings = [("prefill", -1, -1, plan["prefill"])] + [
                    ("decode", segment["start_t"], segment["end_t"], segment["placement"])
                    for segment in plan["decode_segments"]
                ]
                projected = []
                changed_pim = 0
                for phase, start, end, mapping_rows in phase_mappings:
                    mapping = dict(mapping_rows)
                    if len(mapping) != len(mapping_rows):
                        raise ValueError(f"{phase}: duplicate node IDs in historical mapping")
                    program = compiler.project(mapping)
                    fixed = compiler.compile(program)["device_by_node"]
                    changed_pim += sum(device.startswith("PIM") and fixed[node] != device for node, device in mapping.items() if node in compiler.pim_owner)
                    projected.append((phase, start, end, program))
            except (ValueError, KeyError, TypeError, OSError) as error:
                excluded.write(json.dumps({"source_id": source_id, "summary_path": str(path), "reason": str(error)}) + "\n")
                counts["excluded_sources"] += 1
                if strict:
                    raise ValueError(f"new teacher {path} cannot be imported: {error}") from error
                continue
            raw_path = raw_dir / f"{source_id}.json"
            raw_path.write_text(json.dumps({"source": row, "metadata": metadata, "plan": plan}) + "\n")
            source_programs = []
            for phase, start, end, program in projected:
                key = (row["model_id"], phase, program)
                if key not in program_ids:
                    program_id = f"placement_{len(program_ids):06d}"
                    program_ids[key] = program_id
                    placements.write(json.dumps({
                        "placement_id": program_id, "model_id": row["model_id"],
                        "phase": phase, "block_actions": program,
                        "fixed_plan": compiler.compile(program),
                    }) + "\n")
                source_programs.append({"phase": phase, "start_t": start, "end_t": end, "placement_id": program_ids[key]})
            sources.write(json.dumps({
                "source_id": source_id, **row, "raw_plan_path": str(raw_path),
                "raw_order_timing_source": str(path), "placements": source_programs,
                "reevaluation_status": "pending evaluator gate",
                "projected_pim_changes": changed_pim,
            }) + "\n")
            counts["complete_sources"] += 1
            counts["projected_pim_changes"] += changed_pim
            counts["decode_steps"] += plan["decode_steps_covered"]
            by_model[row["model_id"]] += 1
            by_policy[row["policy"]] += 1
            if (index + 1) % 25 == 0:
                print(json.dumps({"progress": index + 1, "total": len(paths), **counts}), flush=True)
    result = {
        **counts, "unique_projected_placements": len(program_ids),
        "by_model": dict(by_model), "by_policy": dict(by_policy),
        "hardware_key": hardware_key, "model_keys": model_keys,
        "plan_count_cap": None, "decode_step_cap": None,
        "training_costs_ready": False,
        "workers": workers,
    }
    (destination / "manifest_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if set(by_model) != set(model_keys):
        raise ValueError(f"no compatible complete histories for {sorted(set(model_keys) - set(by_model))}; see history_excluded.jsonl")
    return result


def _anchor_observation(snapshot, evaluator, batch, prompt, decode_step):
    scheduler = snapshot.scheduler
    requests = [{
        "request_id": f"anchor_{i}", "prompt_len": prompt, "max_tokens": 1024,
        "generated": decode_step + 1 if decode_step >= 0 else 0,
        "age_s": snapshot.now, "kv_len": snapshot.kv_written_tokens[f"anchor_{i}"],
    } for i in range(batch)]
    return {
        "now": snapshot.now,
        "waiting": requests if decode_step < 0 else [],
        "active": requests if decode_step >= 0 else [],
        "device_available_s": dict(scheduler.avail),
        "kv_reserved_bytes_by_pim": dict(scheduler.label.kv_bytes_by_pim),
        "kv_actual_bytes_by_pim": {p: sum(snapshot.kv_written_tokens.values()) * evaluator.kv_bytes_per_token_per_pim for p in ("PIM0", "PIM1")},
        "weight_cache_bytes": {d: cache.used for d, cache in scheduler.buffer.device_cache.items()},
        "weight_cache_capacity_bytes": {d: cache.capacity for d, cache in scheduler.buffer.device_cache.items()},
        "weight_cache_keys": {d: list(cache.items) for d, cache in scheduler.buffer.device_cache.items()},
    }


def _reevaluate_shape(job):
    from .placement_evaluator import PlacementEvaluator, RequestShape
    from .placement_policy import policy_inputs

    backend_root, artifact_dir, config, key, split, sources, placements = job
    model, batch, prompt = key
    destination = Path(artifact_dir) / "anchors"
    name = f"{model}_b{batch}_p{prompt}"
    evaluator = PlacementEvaluator(backend_root, destination / name, model, config)
    reference_program = evaluator.compiler.uniform(0)
    reference_plan = evaluator.compiler.compile(reference_program)
    snapshot = evaluator.initial.clone()
    for i in range(batch):
        evaluator.reserve(snapshot, f"anchor_{i}", prompt, config["max_tokens"])
    # Segment starts retain every projected historical action. A single fixed
    # NPU-preferred carrier constructs their shared, physically advanced state.
    candidates = {}
    for source in sources:
        for item in source["placements"]:
            step = -1 if item["phase"] == "prefill" else item["start_t"]
            candidates.setdefault(step, {}).setdefault(item["placement_id"], set()).add(source["source_id"])
    path = destination / f"{name}.jsonl"
    rows = 0
    covered = set()
    with path.open("w") as handle:
        for step in range(-1, max(candidates) + 1):
            phase = "prefill" if step < 0 else "decode"
            shapes = [RequestShape(f"anchor_{i}", prompt if step < 0 else 1, snapshot.kv_written_tokens[f"anchor_{i}"]) for i in range(batch)]
            if step in candidates:
                scores = []
                for placement_id, source_ids in sorted(candidates[step].items()):
                    result = evaluator.evaluate(snapshot, placements[placement_id]["fixed_plan"], phase, shapes)
                    scores.append({"placement_id": placement_id, "cost_s": result.duration_s, "host_evaluator_s": result.host_evaluator_s, "source_ids": sorted(source_ids)})
                    covered.add(placement_id)
                observation = _anchor_observation(snapshot, evaluator, batch, prompt, step)
                handle.write(json.dumps({
                    "anchor_id": f"{name}_{phase}_{step}", "model_id": model, "phase": phase,
                    "split": split, "batch": batch, "prompt_len": prompt, "decode_step": step,
                    "carrier": "NPU-preferred fixed plan; actual prefill then continuous decode",
                    "state_time_s": snapshot.now,
                    "inputs": policy_inputs(observation, evaluator, phase, reference_program if step >= 0 else None),
                    "candidates": scores,
                }) + "\n")
                handle.flush()
                rows += 1
            if step < max(candidates):
                snapshot = evaluator.evaluate(snapshot, reference_plan, phase, shapes).snapshot
    result = {"path": str(path), "anchors": rows, "placement_ids": sorted(covered), "source_ids": [s["source_id"] for s in sources], "split": split}
    print(json.dumps({"reevaluated": name, "anchors": rows, "placements": len(covered)}), flush=True)
    return result


def reevaluate(backend_root, artifact_dir, config, *, workers=8, shape_limit=0):
    """Score all projected programs at common continuous-state segment anchors."""
    directory = Path(artifact_dir)
    sources = [json.loads(line) for line in (directory / "history_sources.jsonl").open()]
    placements = {row["placement_id"]: row for row in map(json.loads, (directory / "placements.jsonl").open())}
    groups = {}
    for source in sources:
        key = (source["model_id"], source["batch"], source["prefill_len"])
        groups.setdefault(key, []).append(source)
    keys = sorted(groups)
    if shape_limit:
        keys = keys[:shape_limit]
    (directory / "anchors").mkdir(exist_ok=True)
    jobs = []
    for key in keys:
        model_keys = sorted(k for k in groups if k[0] == key[0])
        split = "val" if model_keys.index(key) % 5 == 4 else "train"
        candidate_ids = {p["placement_id"] for s in groups[key] for p in s["placements"]}
        jobs.append((backend_root, str(directory), config, key, split, groups[key], {p: placements[p] for p in candidate_ids}))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_reevaluate_shape, jobs))
    covered = {p for row in results for p in row["placement_ids"]}
    source_ids = {p for row in results for p in row["source_ids"]}
    complete = covered == set(placements) and source_ids == {s["source_id"] for s in sources}
    summary = {
        "complete": complete, "shape_groups": len(results), "anchors": sum(row["anchors"] for row in results),
        "placements_scored": len(covered), "sources_covered": len(source_ids),
        "missing_placements": sorted(set(placements) - covered), "files": results,
        "anchor_protocol": "each historical (model,batch,prompt,segment start); NPU-preferred carrier advanced through every intervening decode; candidates clone the same snapshot",
        "history_split": "hold out every fifth sorted (batch,prompt) per model; no condition shared between BC train and validation",
    }
    filename = "reevaluation_summary.json" if not shape_limit else "reevaluation_pilot.json"
    (directory / filename).write_text(json.dumps(summary, indent=2) + "\n")
    if not shape_limit:
        if not complete:
            raise ValueError("historical reevaluation did not cover all sources and placements")
        manifest = json.loads((directory / "manifest_summary.json").read_text())
        manifest["training_costs_ready"] = True
        (directory / "manifest_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def static_policy_inputs(scheduler, compiler, shape, case, phase, token_index, now,
                         prefill_plan=None, previous_decode_plan=None):
    """Features at a static decision boundary, with explicit physical-device history."""
    import math

    def scale(value):
        value = float(value)
        return math.copysign(math.log1p(abs(value)), value) / 10.0

    generated = max(0, token_index)
    context = 0 if phase == "prefill" else case["prompt"] + generated
    state = [scale(v) for v in (case["batch"], case["prompt"], case["decode"], generated, context)]
    state += [float(phase == "prefill"), float(phase == "decode")]
    state += [scale(v) for v in (shape.layer_num, shape.dim, shape.ffn_dim, shape.n_heads,
                                shape.n_kv_heads, shape.head_dim,
                                sum(n.weight_size for n in compiler.dag.nodes.values()) / 2**30)]
    host = scheduler.cost.get_host_device().name
    all_devices = (*compiler.devices, host)
    for device in compiler.devices:
        cache = scheduler.buffer.device_cache.get(device)
        pim = scheduler.buffer.pim_state.get(device)
        state += [scale(scheduler.avail[device] - now),
                  scale(cache.used / 2**30 if cache else 0),
                  scale(cache.capacity / 2**30 if cache else 0),
                  scale(pim.kv_reserved_bytes / 2**30 if pim else 0),
                  scale(pim.kv_used_bytes / 2**30 if pim else 0),
                  scale(pim.limit_bytes / 2**30 if pim else 0)]
    for source in all_devices:
        for destination in all_devices:
            if source != destination:
                state.append(scale(scheduler.comm.timeline_end.get((source, destination), 0.0) - now))
    state += [float(scheduler.label.kv_place == place) for place in ("pim", "host", "npu")]

    roles = ("LN", "Q", "K", "V", "QK", "SOFTMAX", "SV", "O", "ADD", "W1", "W3", "SWIGLU", "W2")
    tokens = case["batch"] * (case["prompt"] if phase == "prefill" else 1)
    pairs = case["batch"] * (case["prompt"] * (case["prompt"] + 1) / 2 if phase == "prefill" else context + 1)
    nodes = []
    for index, nid in enumerate(compiler.node_ids):
        node = compiler.dag.nodes[nid]
        role = node.name.upper().removeprefix("FFN_")
        attrs = node.attrs
        values = [float(role == name) for name in roles]
        values += [attrs["layer"] / shape.layer_num, index / len(compiler.node_ids),
                   float(attrs.get("head_shard", -1)), float(attrs.get("ffn_shard", -1)),
                   scale(node.weight_size / 2**20), scale(tokens), scale(pairs),
                   scale(attrs.get("q_heads", shape.n_heads)),
                   scale(attrs.get("kv_heads", shape.n_kv_heads)), scale(attrs.get("ffn_dim", shape.ffn_dim))]
        values += [float(scheduler.buffer.is_cached(device, node.weight_id)) for device in compiler.devices]
        values += [float(v) for v in compiler.masks[index]]
        for plan in (prefill_plan, previous_decode_plan):
            chosen = plan["device_by_node"][nid] if plan is not None else None
            values += [float(chosen == device) for device in compiler.devices]
        nodes.append(values)
    return {"state": state, "nodes": nodes, "masks": compiler.masks,
            "fixed_from": compiler.fixed_from, "learned": compiler.learned}


def serving_policy_inputs(scheduler, compiler, shape, phase, requests, waiting,
                          admitted_count, decode_feature, now,
                          prefill_plan=None, previous_decode_plan=None):
    """BC-compatible features from visible, actually admitted requests only."""
    import math
    from statistics import mean, pstdev

    def scale(value):
        return math.copysign(math.log1p(abs(value)), value) / 10.0

    prompts = [r["prompt_len"] for r in requests]
    kv = [r["kv_len"] for r in requests] if phase == "decode" else [0]
    progress = mean(r["generated"] - 1 for r in requests) if phase == "decode" else 0
    case = dict(batch=len(requests), prompt=mean(prompts), decode=0)
    inputs = static_policy_inputs(scheduler, compiler, shape, case, phase, progress, now,
                                  prefill_plan, previous_decode_plan)
    inputs["state"][2] = decode_feature  # Already scaled on BC train tasks.
    inputs["state"][4] = scale(mean(kv))
    ages = [r["age_s"] for r in waiting] or [0]
    inputs["state"] += [scale(v) for v in (
        7315, len(waiting), admitted_count, mean(ages), max(ages),
        max(prompts), pstdev(prompts), max(kv), pstdev(kv))]
    queries = sum(prompts) if phase == "prefill" else len(requests)
    pairs = (sum(p * (p + 1) / 2 for p in prompts) if phase == "prefill"
             else sum(k + 1 for k in kv))
    for node in inputs["nodes"]:
        node[18], node[19] = scale(queries), scale(pairs)
    return inputs
