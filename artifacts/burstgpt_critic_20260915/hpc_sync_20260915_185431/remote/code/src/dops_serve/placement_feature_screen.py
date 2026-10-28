from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .dops_fixed_replay import compare_reference, load_fixed_plans as _load_fixed_plans, make_scheduler, replay_decode, replay_phase

MODELS = ("qwen_7b", "qwen_1.8b", "qwen_14b")
FACTOR_VALUES = {
    "batch": (1, 4, 8, 16),
    "prompt": (128, 512, 1024, 2048),
    "decode": (128, 256, 512, 1024),
}
ACTOR_R2_LOW = 0.01


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def task_specs() -> list[dict]:
    tasks = []
    for model in MODELS:
        for factor in FACTOR_VALUES:
            background_names = tuple(name for name in FACTOR_VALUES if name != factor)
            for values in itertools.product(*(FACTOR_VALUES[name] for name in background_names)):
                tasks.append(
                    {
                        "task_id": len(tasks),
                        "model": model,
                        "factor": factor,
                        "background": dict(zip(background_names, values)),
                    }
                )
    return tasks


def factor_map_rows() -> list[dict]:
    rows = []

    def add(space: str, name: str, family: str, stage: str) -> None:
        index = sum(row["space"] == space for row in rows)
        rows.append(
            {
                "space": space,
                "index": index,
                "name": name,
                "factor_family": family,
                "experiment_stage": stage,
            }
        )

    for queue in ("waiting", "active"):
        add("state", f"{queue}.count", "request_count", "stage1:batch")
        for key, family in (
            ("prompt_len", "request_prompt_distribution"),
            ("kv_len", "request_kv_distribution"),
            ("age_s", "request_age_distribution"),
        ):
            add("state", f"{queue}.{key}.mean", family, "stage2")
            add("state", f"{queue}.{key}.max", family, "stage2")
            for edge in (256, 512, 1024, 2048) if key != "age_s" else (0.1, 1, 10, 100):
                add("state", f"{queue}.{key}.cdf_le_{edge}", family, "stage2")
        add("state", f"{queue}.max_tokens.mean", "requested_decode_length", "stage1:decode")
        add("state", f"{queue}.generated.mean", "decode_progress", "stage2")
    for device in ("NPU", "PIM0", "PIM1"):
        add("state", f"{device}.available_tail", "device_available_tails", "stage2")
        add("state", f"{device}.weight_cache_used_gib", "weight_cache_state", "stage2")
        add("state", f"{device}.weight_cache_capacity_gib", "weight_cache_capacity", "stage2")
    for device in ("PIM0", "PIM1"):
        add("state", f"{device}.kv_reserved_gib", "kv_reservation", "stage1:batch,prompt,decode")
        add("state", f"{device}.kv_actual_gib", "kv_progress", "stage1:batch,prompt")
    for name in (
        "recent_arrivals_1s",
        "recent_arrivals_10s",
        "recent_arrivals_60s",
        "queue_growth_per_s",
        "recent_service_tokens_per_s",
    ):
        add("state", name, "recent_load", "stage2")
    for name in ("layers", "hidden_dim", "q_heads", "kv_heads", "head_dim", "kv_bytes_per_token"):
        add("state", f"model.{name}", "model", "stage1:model")
    add("state", "phase.prefill", "phase", "stage1")
    add("state", "phase.decode", "phase", "stage1")

    for role in ("LN", "Q", "K", "V", "QK", "SOFTMAX", "SV", "O", "ADD", "W1", "W3", "SWIGLU", "W2"):
        add("node", f"op_type.{role}", "operator_type", "stage1:model")
    for name, family, stage in (
        ("normalized_layer", "model_topology", "stage1:model"),
        ("is_ffn", "operator_type", "stage1:model"),
        ("shard", "operator_shard", "stage1:model"),
        ("owner_is_pim1", "pim_owner", "stage1:model"),
        ("weight_mib", "operator_weight", "stage1:model"),
        ("tokens", "request_shape", "stage1:batch,prompt"),
        ("attention_pairs", "request_shape", "stage1:batch,prompt"),
        ("cached_on_npu", "weight_cache_state", "stage2"),
        ("cached_on_owner", "weight_cache_state", "stage2"),
        ("current_program", "placement_history", "stage2"),
    ):
        add("node", name, family, stage)
    return rows


def _discover_cases(source_root: Path) -> list[dict]:
    from .summary import read_summary_metadata

    rows = []
    for result_path in sorted(source_root.glob("F*/result.json")):
        result = json.loads(result_path.read_text())
        case = dict(result["case"])
        if result["candidate"]["candidate_id"] != "C026":
            raise ValueError(f"{result_path}: expected C026")
        filename = f"best_summary_{case['prompt']}x{case['decode']}.json"
        summaries = list(result_path.parent.glob(f"dops_output/*/algo_Bifocal/{filename}"))
        if len(summaries) != 1:
            raise ValueError(f"{result_path.parent}: expected one {filename}")
        metadata = read_summary_metadata(summaries[0])
        observed = metadata["config"]
        if (observed["batch"], observed["prefill_len"], observed["decode_len"]) != (
            case["batch"], case["prompt"], case["decode"]
        ):
            raise ValueError(f"{summaries[0]}: summary shape differs from result.json")
        rows.append(
            {
                **case,
                "summary_path": str(summaries[0]),
                "run_config_path": str(result_path.parent / "run_config.json"),
                "source_prefill_time_s": result["prefill_time_s"],
                "source_decode_time_s": result["decode_time_s"],
                "source_total_time_s": result["total_time_s"],
            }
        )
    expected = set(itertools.product(MODELS, *FACTOR_VALUES.values()))
    observed = {(row["model"], row["batch"], row["prompt"], row["decode"]) for row in rows}
    if observed != expected:
        raise ValueError(f"C026 grid differs from the expected 192 states: missing={len(expected - observed)} extra={len(observed - expected)}")
    return rows


def prepare(source_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "groups").mkdir(exist_ok=True)
    cases = _discover_cases(source_root)
    tasks = task_specs()
    _write_jsonl(output_dir / "controlled_states.jsonl", cases)
    _write_jsonl(output_dir / "tasks.jsonl", tasks)
    with (output_dir / "factor_map.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("space", "index", "name", "factor_family", "experiment_stage"))
        writer.writeheader()
        writer.writerows(factor_map_rows())
    first = json.loads(next(source_root.glob("F*/result.json")).read_text())
    manifest = {
        "source": str(source_root),
        "reference": "DOPS Bifocal C026",
        "c026_parameters": first["candidate"]["parameters"],
        "models": MODELS,
        "factor_values": FACTOR_VALUES,
        "states": len(cases),
        "tasks": len(tasks),
        "evaluations_per_task": {"prefill": 16, "decode_full_sequence": 16},
        "evaluator": "native DOPS schedule_with_plan",
        "decode_transfer": "source plan at the same token index; hold final source plan beyond source horizon",
        "decode_initial_state": "target reference prefill state, identical for every candidate",
        "actor_r2_low_threshold": ACTOR_R2_LOW,
        "actor_r2_response": "log fixed-placement cost",
        "actor_r2_reduced": "background + target factor + source-placement factor",
        "actor_r2_full": "reduced + target-factor-by-source-placement interaction",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _case_key(row: dict) -> tuple:
    return row["model"], row["batch"], row["prompt"], row["decode"]


def validate_reference(backend_root: Path, output_dir: Path, case_index: int) -> dict:
    case = _read_jsonl(output_dir / "controlled_states.jsonl")[case_index]
    parameters = json.loads((output_dir / "run_manifest.json").read_text())["c026_parameters"]
    plans = _load_fixed_plans(Path(case["summary_path"]))
    sched, graph, _, cfg = make_scheduler(backend_root, case, parameters, output_dir / "backend_logs" / case["case_id"])
    prefill_s = replay_phase(sched, graph, "prefill", plans["prefill"])
    step_times = replay_decode(sched, graph, cfg, plans["decode"], prefill_s)
    result = compare_reference(case, prefill_s, step_times, plans)
    destination = output_dir / "reference_checks"
    destination.mkdir(exist_ok=True)
    (destination / f"{case['case_id']}.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise ValueError(f"Original DOPS timing was not reproduced: {result}")
    return result


def check_references(output_dir: Path) -> dict:
    cases = _read_jsonl(output_dir / "controlled_states.jsonl")
    checks = [json.loads((output_dir / "reference_checks" / f"{case['case_id']}.json").read_text()) for case in cases]
    if not all(row["passed"] for row in checks):
        raise ValueError("Reference replay failed; correlation experiments must not run")
    result = {"passed": True, "cases": len(checks), "decode_tokens": sum(row["decode_tokens"] for row in checks), "checks": checks}
    (output_dir / "reference_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "checks"}


def _node_role(node_id: str) -> str:
    role = node_id.split("_", 1)[1].split("_s")[0].upper()
    role = role.removeprefix("FFN_")
    if role.startswith("LN"):
        return "LN"
    if role.startswith("ADD"):
        return "ADD"
    return role


def _placement_diff(reference_plans: list[dict], candidate_plans: list[dict], sched, graph, shape) -> dict:
    executors = [dev for dev in sched.cluster.devices.values() if dev.type in sched._executor_device_types()]
    free_nodes = [
        node_id for node_id, node in graph.nodes.items()
        if not sched._is_comm_node(node)
        and sched._preferred_kv_write_device(graph, node_id) is None
        and sum(sched._node_allowed_on(node, dev) for dev in executors) > 1
    ]
    changed_counts = dict.fromkeys(free_nodes, 0)
    token_changes = []
    for t, reference in enumerate(reference_plans):
        candidate = candidate_plans[min(t, len(candidate_plans) - 1)]
        changed = [node_id for node_id in free_nodes
                   if reference["device_by_node"][node_id] != candidate["device_by_node"][node_id]]
        for node_id in changed:
            changed_counts[node_id] += 1
        token_changes.append({"t": t, "changed": len(changed), "flip_rate": len(changed) / len(free_nodes)})
    groups: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for node_id in free_nodes:
        node = graph.nodes[node_id]
        layer = int(node.attrs["layer"])
        bucket = min(3, 4 * layer // shape.layer_num)
        shard = node.attrs.get("head_shard", node.attrs.get("ffn_shard", -1))
        for kind, value in (("op_type", _node_role(node_id)), ("layer_bucket", bucket), ("shard", shard)):
            groups[(kind, value)][1] += len(reference_plans)
            groups[(kind, value)][0] += changed_counts[node_id]
    by = defaultdict(dict)
    for (kind, value), (count, total) in groups.items():
        by[kind][str(value)] = {"changed": count, "total": total, "flip_rate": count / total}
    changed = sum(changed_counts.values())
    return {
        "changed_nodes": {node_id: count for node_id, count in changed_counts.items() if count},
        "changed": changed,
        "total_free_nodes": len(free_nodes),
        "steps": len(reference_plans),
        "flip_rate": changed / (len(free_nodes) * len(reference_plans)),
        "per_token": token_changes,
        **by,
    }


def run_task(backend_root: Path, output_dir: Path, task_id: int) -> dict:
    validation = json.loads((output_dir / "reference_validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("Reference validation must pass before cross evaluation")
    task = _read_jsonl(output_dir / "tasks.jsonl")[task_id]
    cases = {_case_key(row): row for row in _read_jsonl(output_dir / "controlled_states.jsonl")}
    parameters = json.loads((output_dir / "run_manifest.json").read_text())["c026_parameters"]
    group = []
    for value in FACTOR_VALUES[task["factor"]]:
        target_shape = {**task["background"], task["factor"]: value}
        row = cases[(task["model"], target_shape["batch"], target_shape["prompt"], target_shape["decode"])]
        group.append({**row, "factor_value": value, "plans": _load_fixed_plans(Path(row["summary_path"]))})

    costs = []
    diffs = []
    for target in group:
        results = {"prefill": {}, "decode": {}}
        for candidate in group:
            log_dir = output_dir / "backend_logs" / f"task_{task_id:03d}" / f"{target['case_id']}_{candidate['case_id']}"
            sched, graph, shape, cfg = make_scheduler(backend_root, target, parameters, log_dir / "prefill")
            prefill_s = replay_phase(sched, graph, "prefill", candidate["plans"]["prefill"])
            results["prefill"][candidate["factor_value"]] = prefill_s
            if candidate["case_id"] != target["case_id"]:
                sched, graph, shape, cfg = make_scheduler(backend_root, target, parameters, log_dir / "decode")
                prefill_s = replay_phase(sched, graph, "prefill", target["plans"]["prefill"])
            step_times = replay_decode(sched, graph, cfg, candidate["plans"]["decode"], prefill_s)
            results["decode"][candidate["factor_value"]] = sum(step_times)
            if candidate["case_id"] == target["case_id"]:
                check = compare_reference(target, prefill_s, step_times, target["plans"])
                if not check["passed"]:
                    raise ValueError(f"Reference timing drifted during cross evaluation: {check}")
        for phase in ("prefill", "decode"):
            reference_cost = results[phase][target["factor_value"]]
            reference_plans = [target["plans"][phase]] if phase == "prefill" else target["plans"][phase]
            for candidate in group:
                value = candidate["factor_value"]
                cost = results[phase][value]
                common = {
                    "task_id": task_id, "model": task["model"], "factor": task["factor"],
                    "background": task["background"], "phase": phase,
                    "target_case_id": target["case_id"], "target_value": target["factor_value"],
                    "candidate_case_id": candidate["case_id"], "candidate_value": value,
                    "decode_tokens": target["decode"] if phase == "decode" else 0,
                }
                candidate_plans = [candidate["plans"][phase]] if phase == "prefill" else candidate["plans"][phase]
                detail = _placement_diff(reference_plans, candidate_plans, sched, graph, shape)
                costs.append({
                    **common, "cost_s": cost, "target_reference_cost_s": reference_cost,
                    "regret": (cost - reference_cost) / reference_cost, "flip_rate": detail["flip_rate"],
                })
                diffs.append({**common, **detail})

    destination = output_dir / "groups" / f"{task_id:03d}.json"
    result = {"task": task, "cross_costs": costs, "placement_diffs": diffs}
    destination.write_text(json.dumps(result) + "\n")
    return {"task_id": task_id, "cross_costs": len(costs), "placement_diffs": len(diffs), "path": str(destination)}


def actor_partial_r2(rows: list[dict]) -> dict:
    observations = [
        {
            "background": json.dumps(row["background"], sort_keys=True),
            "target": str(row["target_value"]),
            "candidate": str(row["candidate_value"]),
            "y": math.log(row["cost_s"]),
        }
        for row in rows
    ]
    grand = statistics.mean(row["y"] for row in observations)

    def means(key):
        grouped = defaultdict(list)
        for row in observations:
            grouped[row[key]].append(row["y"])
        return {value: statistics.mean(values) for value, values in grouped.items()}

    background_mean = means("background")
    target_mean = means("target")
    candidate_mean = means("candidate")
    cell_values = defaultdict(list)
    for row in observations:
        cell_values[(row["target"], row["candidate"])].append(row["y"])
    cell_mean = {cell: statistics.mean(values) for cell, values in cell_values.items()}

    reduced_sse = 0.0
    full_sse = 0.0
    total_sse = sum((row["y"] - grand) ** 2 for row in observations)
    for row in observations:
        reduced = background_mean[row["background"]] + target_mean[row["target"]] + candidate_mean[row["candidate"]] - 2 * grand
        full = background_mean[row["background"]] + cell_mean[(row["target"], row["candidate"])] - grand
        reduced_sse += (row["y"] - reduced) ** 2
        full_sse += (row["y"] - full) ** 2
    negligible = reduced_sse <= max(1e-24, total_sse * 1e-12)
    value = 0.0 if negligible else (reduced_sse - full_sse) / reduced_sse
    return {"actor_partial_r2": max(0.0, min(1.0, value)), "reduced_sse": reduced_sse, "full_sse": full_sse}


def best_candidate_changes(rows: list[dict]) -> set[str]:
    backgrounds = defaultdict(lambda: defaultdict(list))
    for row in rows:
        backgrounds[json.dumps(row["background"], sort_keys=True)][row["target_value"]].append(row)
    changed = set()
    for background, targets in backgrounds.items():
        best_sets = []
        for candidates in targets.values():
            best = min(row["cost_s"] for row in candidates)
            best_sets.append(frozenset(row["candidate_value"] for row in candidates
                                      if math.isclose(row["cost_s"], best, rel_tol=1e-9, abs_tol=1e-12)))
        if len(set(best_sets)) > 1:
            changed.add(background)
    return changed


def summarize(output_dir: Path, models: tuple[str, ...]) -> dict:
    validation = json.loads((output_dir / "reference_validation.json").read_text())
    if not validation["passed"]:
        raise ValueError("Cannot summarize unvalidated reference replays")
    selected_tasks = [task for task in _read_jsonl(output_dir / "tasks.jsonl") if task["model"] in models]
    missing = [task["task_id"] for task in selected_tasks if not (output_dir / "groups" / f"{task['task_id']:03d}.json").is_file()]
    if missing:
        raise ValueError(f"missing group results: {missing[:16]}")
    payloads = [json.loads((output_dir / "groups" / f"{task['task_id']:03d}.json").read_text()) for task in selected_tasks]
    costs = [row for payload in payloads for row in payload["cross_costs"]]
    diffs = [row for payload in payloads for row in payload["placement_diffs"]]
    destination = output_dir if set(models) == set(MODELS) else output_dir / "checkpoints" / "_".join(models)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "cross_costs.jsonl", costs)
    _write_jsonl(destination / "placement_diffs.jsonl", diffs)

    summary_rows = []
    for model in models:
        for factor in FACTOR_VALUES:
            for phase in ("prefill", "decode"):
                subset = [row for row in costs if row["model"] == model and row["factor"] == factor and row["phase"] == phase]
                nonself = [row for row in subset if row["target_value"] != row["candidate_value"]]
                r2 = actor_partial_r2(subset)
                affected_backgrounds = {
                    json.dumps(row["background"], sort_keys=True)
                    for row in nonself
                    if row["flip_rate"] > 0.0 and row["regret"] >= 0.01
                }
                max_regret = max(row["regret"] for row in nonself)
                best_changes = best_candidate_changes(subset)
                if len(best_changes & affected_backgrounds) >= 2:
                    decision = "keep_evidence"
                elif max(abs(row["regret"]) for row in nonself) < 0.01 and r2["actor_partial_r2"] <= ACTOR_R2_LOW:
                    decision = "no_effect_observed_pending_holdout_and_nine_methods"
                else:
                    decision = "review"
                summary_rows.append(
                    {
                        "model": model,
                        "factor": factor,
                        "phase": phase,
                        "groups": len({json.dumps(row["background"], sort_keys=True) for row in subset}),
                        "evaluations": len(subset),
                        "nonself_comparisons": len(nonself),
                        "changed_pairs": sum(row["flip_rate"] > 0.0 for row in nonself),
                        "mean_flip_rate": statistics.mean(row["flip_rate"] for row in nonself),
                        "max_flip_rate": max(row["flip_rate"] for row in nonself),
                        "mean_regret": statistics.mean(row["regret"] for row in nonself),
                        "max_regret": max_regret,
                        "regret_ge_1pct_pairs": sum(row["regret"] >= 0.01 for row in nonself),
                        "regret_ge_1pct_backgrounds": len(affected_backgrounds),
                        "negative_regret_pairs": sum(row["regret"] < 0.0 for row in nonself),
                        "best_candidate_change_backgrounds": len(best_changes),
                        **r2,
                        "decision": decision,
                    }
                )
    fields = tuple(summary_rows[0])
    with (destination / "factor_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# C026 placement feature decisions",
        "",
        "Native DOPS replay; original device IDs and execution order preserved. No neural network was trained.",
        "Decode covers every target token; source plans are aligned by token index and the final plan is held beyond the source horizon.",
        "Actor partial R2 measures relative-cost interaction, not prediction accuracy or proof of a ranking reversal. No feature deletion is approved by this screen alone.",
        "",
        "| Model | Factor | Phase | Actor partial R2 | Max regret | Affected backgrounds | Decision |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['factor']} | {row['phase']} | {row['actor_partial_r2']:.6f} | "
            f"{row['max_regret']:.2%} | {row['regret_ge_1pct_backgrounds']} | {row['decision']} |"
        )
    (destination / "feature_decisions.md").write_text("\n".join(lines) + "\n")
    result = {
        "complete": True,
        "reference_replay_validated": True,
        "evaluator": "native DOPS schedule_with_plan",
        "decode_scope": "full_sequence",
        "models": models,
        "tasks": len(selected_tasks),
        "cross_costs": len(costs),
        "placement_diffs": len(diffs),
        "factor_summaries": len(summary_rows),
        "destination": str(destination),
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-root", required=True, type=Path)
    prepare_parser.add_argument("--output-dir", required=True, type=Path)
    validate_parser = commands.add_parser("validate-reference")
    validate_parser.add_argument("--backend-root", required=True, type=Path)
    validate_parser.add_argument("--output-dir", required=True, type=Path)
    validate_parser.add_argument("--case-index", required=True, type=int)
    check_parser = commands.add_parser("check-references")
    check_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--backend-root", required=True, type=Path)
    run_parser.add_argument("--output-dir", required=True, type=Path)
    run_parser.add_argument("--task-id", required=True, type=int)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--output-dir", required=True, type=Path)
    summary_parser.add_argument("--models", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source_root, args.output_dir)
    elif args.command == "validate-reference":
        result = validate_reference(args.backend_root, args.output_dir, args.case_index)
    elif args.command == "check-references":
        result = check_references(args.output_dir)
    elif args.command == "run":
        result = run_task(args.backend_root, args.output_dir, args.task_id)
    else:
        models = MODELS if args.models == "all" else tuple(args.models.split(","))
        result = summarize(args.output_dir, models)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
