from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .summary import canonical_plan_text, extract_plan


def _policy_name(value: str) -> str:
    return value.rsplit(":", 1)[-1].strip().casefold()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _load_inventory(path: Path, policies: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if _policy_name(row["policy"]) in policies]


def _spread_limit(rows: list[dict[str, str]], limit: int) -> list[dict[str, str]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["batch"]),
            int(row["prefill_len"]),
            int(row["decode_len"]),
            row["summary_path"],
        ),
    )
    if limit <= 0 or len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def build_plan_bank(
    inventory_path: str | Path,
    artifact_dir: str | Path,
    policies: list[str],
    max_plans_per_group: int = 32,
    max_summaries_per_group: int = 0,
    max_decode_steps: int = 1024,
) -> dict[str, int]:
    inventory = Path(inventory_path).resolve()
    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plan_dir = destination / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)

    accepted_policies = {_policy_name(policy) for policy in policies}
    rows = _load_inventory(inventory, accepted_policies)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_id"], row["hardware_key"])].append(row)

    source_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, str]] = []
    parsed_count = 0

    for model_id, hardware_key in sorted(groups):
        selected_rows = _spread_limit(
            groups[(model_id, hardware_key)], max_summaries_per_group
        )
        unique: dict[str, dict[str, Any]] = {}
        for row in selected_rows:
            try:
                plan = extract_plan(
                    row["summary_path"],
                    max_decode_steps=max_decode_steps,
                    ops_trace_path=row["ops_trace_path"],
                )
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                excluded_rows.append(
                    {"summary_path": row["summary_path"], "reason": str(error)}
                )
                continue
            parsed_count += 1
            exact_text = canonical_plan_text(plan)
            if exact_text not in unique:
                unique[exact_text] = {
                    "plan": plan,
                    "representative_source": row,
                    "sources": [],
                    "incomplete_sources": 0,
                }
            elif (
                plan["decode_steps_covered"],
                plan["decode_steps_observed"],
            ) > (
                unique[exact_text]["plan"]["decode_steps_covered"],
                unique[exact_text]["plan"]["decode_steps_observed"],
            ):
                unique[exact_text]["plan"] = plan
                unique[exact_text]["representative_source"] = row
            unique[exact_text]["sources"].append(row)
            if not plan["placement_complete"]:
                unique[exact_text]["incomplete_sources"] += 1

        ranked = sorted(
            unique.values(),
            key=lambda item: (
                -len(item["sources"]),
                sum(float(row["total_time_s"]) for row in item["sources"])
                / len(item["sources"]),
                item["sources"][0]["summary_path"],
            ),
        )
        retained = ranked[:max_plans_per_group]
        dropped = ranked[max_plans_per_group:]
        for item in dropped:
            for source in item["sources"]:
                excluded_rows.append(
                    {
                        "summary_path": source["summary_path"],
                        "reason": "unique placement omitted by max_plans_per_group",
                    }
                )

        group_name = f"{_safe_name(model_id)}__{_safe_name(hardware_key)}"
        for plan_index, item in enumerate(retained, start=1):
            plan_id = f"{group_name}__p{plan_index:02d}"
            plan_path = plan_dir / f"{plan_id}.json"
            source_shapes = sorted(
                {
                    (
                        int(row["batch"]),
                        int(row["prefill_len"]),
                        int(row["decode_len"]),
                    )
                    for row in item["sources"]
                }
            )
            plan_payload = {
                "plan_id": plan_id,
                "model_id": model_id,
                "hardware_key": hardware_key,
                "source_policy": item["representative_source"]["policy"],
                "source_summary": item["representative_source"]["summary_path"],
                "source_shapes": [list(shape) for shape in source_shapes],
                "placement_equivalence": "prefill mapping and decode placement transitions",
                "incomplete_source_count": item["incomplete_sources"],
                **item["plan"],
            }
            with plan_path.open("w", encoding="utf-8") as handle:
                json.dump(plan_payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")

            manifest_rows.append(
                {
                    "plan_id": plan_id,
                    "model_id": model_id,
                    "hardware_key": hardware_key,
                    "source_policy": item["representative_source"]["policy"],
                    "plan_path": str(plan_path),
                    "profile_count": len(item["sources"]),
                    "decode_steps_covered": item["plan"]["decode_steps_covered"],
                    "decode_steps_observed": item["plan"]["decode_steps_observed"],
                    "decode_steps_estimated": item["plan"]["decode_steps_estimated"],
                    "decode_steps_recovered": item["plan"]["decode_steps_recovered"],
                    "decode_steps_missing": item["plan"]["decode_steps_missing"],
                    "placement_complete": item["plan"]["placement_complete"],
                    "placement_equivalence": "prefill mapping and decode placement transitions",
                    "incomplete_source_count": item["incomplete_sources"],
                    "source_shapes": [list(shape) for shape in source_shapes],
                }
            )
            for source in item["sources"]:
                source_rows.append(
                    {
                        "summary_path": source["summary_path"],
                        "plan_id": plan_id,
                        "model_id": model_id,
                        "hardware_key": hardware_key,
                    }
                )

    with (destination / "plan_bank.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with (destination / "plan_sources.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["summary_path", "plan_id", "model_id", "hardware_key"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    with (destination / "plan_excluded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["summary_path", "reason"])
        writer.writeheader()
        writer.writerows(excluded_rows)

    return {
        "inventory_rows": len(rows),
        "parsed_summaries": parsed_count,
        "plans": len(manifest_rows),
        "mapped_profiles": len(source_rows),
        "excluded": len(excluded_rows),
    }
