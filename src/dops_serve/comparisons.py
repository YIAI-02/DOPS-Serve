from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any


COMPARISON_FIELDS = [
    "comparison_id",
    "comparison_path",
    "source_candidate_count",
    "candidate_count",
    "model_id",
    "hardware_key",
    "dtype",
    "batch",
    "prefill_len",
    "decode_len",
    "summary_path",
    "policy",
    "prefill_time_s",
    "decode_time_s",
    "total_time_s",
    "output_token_throughput",
    "rank",
    "is_best",
]


def _assert_outside_source(source_root: Path, output_path: Path) -> None:
    source = str(source_root.resolve())
    destination = str(output_path.resolve())
    if os.path.commonpath([source, destination]) == source:
        raise ValueError("output must be outside the read-only DOPS output root")


def _read_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            str(Path(row["summary_path"]).resolve()): row
            for row in csv.DictReader(handle)
        }


def _result_dir(output_root: Path, value: Any) -> Path:
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "output":
        candidate = Path(*candidate.parts[1:])
    return output_root / candidate


def _summary_path(output_root: Path, config: dict[str, Any], policy: str) -> Path:
    policy_name = policy.rsplit(":", 1)[-1]
    return (
        _result_dir(output_root, config["result_dir"])
        / f"algo_{policy_name}"
        / f"best_summary_{int(config['prefill_len'])}x{int(config['decode_len'])}.json"
    ).resolve()


def _times_match(result: dict[str, Any], inventory: dict[str, str]) -> bool:
    pairs = (
        (result["prefill_time_s"], inventory["prefill_time_s"]),
        (result["decode_time_s"], inventory["decode_time_s"]),
        (result["total_time_s"], inventory["total_time_s"]),
    )
    return all(
        math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
        for left, right in pairs
    )


def build_anchor_comparisons(
    output_root: str | Path,
    inventory_path: str | Path,
    output_path: str | Path,
    path_prefix: str | list[str] | None = None,
) -> dict[str, int]:
    source = Path(output_root).resolve()
    destination = Path(output_path).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    _assert_outside_source(source, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = _read_inventory(Path(inventory_path))

    comparison_paths = sorted(source.rglob("baseline_compare_*.json"))
    if path_prefix:
        prefixes = [path_prefix] if isinstance(path_prefix, str) else path_prefix
        comparison_paths = [
            path
            for path in comparison_paths
            if any(str(path.relative_to(source)).startswith(prefix) for prefix in prefixes)
        ]

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    multi_candidate_files = 0
    paired_groups = 0
    excluded_groups = 0
    excluded_results = 0
    for comparison_path in comparison_paths:
        payload = json.loads(comparison_path.read_text(encoding="utf-8"))
        config = payload["config"]
        results = payload["results"]
        comparison_id = str(comparison_path.relative_to(source))
        if len(results) < 2:
            exclusions.append(
                {
                    "comparison_path": str(comparison_path),
                    "policy": "",
                    "reason": "comparison has fewer than two candidates",
                }
            )
            excluded_groups += 1
            continue

        multi_candidate_files += 1
        candidates: list[tuple[dict[str, Any], dict[str, str], Path]] = []
        for result in results:
            policy = str(result["policy"])
            summary_path = _summary_path(source, config, policy)
            inventory_row = inventory.get(str(summary_path))
            if inventory_row is None:
                exclusions.append(
                    {
                        "comparison_path": str(comparison_path),
                        "policy": policy,
                        "reason": f"linked summary absent from inventory: {summary_path}",
                    }
                )
                excluded_results += 1
                continue
            if not _times_match(result, inventory_row):
                exclusions.append(
                    {
                        "comparison_path": str(comparison_path),
                        "policy": policy,
                        "reason": f"timing mismatch with inventory: {summary_path}",
                    }
                )
                excluded_results += 1
                continue
            candidates.append((result, inventory_row, summary_path))

        if len(candidates) < 2:
            excluded_groups += 1
            continue

        paired_groups += 1
        ordered = sorted(candidates, key=lambda item: float(item[0]["total_time_s"]))
        candidate_count = len(ordered)
        for rank, (result, inventory_row, summary_path) in enumerate(ordered, start=1):
            batch = int(config["batch"])
            decode_len = int(config["decode_len"])
            total_time = float(result["total_time_s"])
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "comparison_path": str(comparison_path),
                    "source_candidate_count": len(results),
                    "candidate_count": candidate_count,
                    "model_id": inventory_row["model_id"],
                    "hardware_key": inventory_row["hardware_key"],
                    "dtype": inventory_row["dtype"],
                    "batch": batch,
                    "prefill_len": int(config["prefill_len"]),
                    "decode_len": decode_len,
                    "summary_path": str(summary_path),
                    "policy": result["policy"],
                    "prefill_time_s": float(result["prefill_time_s"]),
                    "decode_time_s": float(result["decode_time_s"]),
                    "total_time_s": total_time,
                    "output_token_throughput": batch * decode_len / total_time,
                    "rank": rank,
                    "is_best": rank == 1,
                }
            )

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    excluded_path = destination.with_name(destination.stem + "_excluded.csv")
    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["comparison_path", "policy", "reason"]
        )
        writer.writeheader()
        writer.writerows(exclusions)

    return {
        "comparison_files": len(comparison_paths),
        "multi_candidate_files": multi_candidate_files,
        "paired_groups": paired_groups,
        "rows": len(rows),
        "excluded_groups": excluded_groups,
        "excluded_results": excluded_results,
    }
