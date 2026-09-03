from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import DATASET_FIELDS
from .replay import ProfileEstimator


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _regular_state(batch: int, prefill_len: int) -> dict[str, float]:
    return {
        "waiting_count": float(batch),
        "active_count": 0.0,
        "total_waiting_prompt_tokens": float(batch * prefill_len),
        "total_active_context_tokens": 0.0,
        "arrival_rate_1s": float(batch),
        "arrival_rate_10s": batch / 10.0,
        "arrival_rate_60s": batch / 60.0,
        "waiting_prompt_mean": float(prefill_len),
        "waiting_prompt_p95": float(prefill_len),
        "waiting_prompt_max": float(prefill_len),
        "active_context_mean": 0.0,
        "active_context_p95": 0.0,
        "active_context_max": 0.0,
        "waiting_age_s_mean": 0.0,
        "waiting_age_s_p95": 0.0,
        "waiting_age_s_max": 0.0,
        "max_tokens_mean": 0.0,
        "max_tokens_p95": 0.0,
        "max_tokens_max": 0.0,
        "generated_tokens_mean": 0.0,
        "generated_tokens_p95": 0.0,
        "generated_tokens_max": 0.0,
    }


def _group_splits(
    groups: dict[str, list[tuple[dict[str, str], str]]]
) -> dict[str, str]:
    by_target: dict[tuple[str, str], list[str]] = defaultdict(list)
    for comparison_id, candidates in groups.items():
        row = candidates[0][0]
        by_target[(row["model_id"], row["hardware_key"])].append(comparison_id)

    splits: dict[str, str] = {}
    for comparison_ids in by_target.values():
        ordered = sorted(comparison_ids)
        if len(ordered) < 3:
            splits.update({comparison_id: "train" for comparison_id in ordered})
            continue
        train_end = max(1, int(len(ordered) * 0.7))
        validation_end = min(
            max(train_end + 1, int(len(ordered) * 0.85)), len(ordered) - 1
        )
        for index, comparison_id in enumerate(ordered):
            if index < train_end:
                splits[comparison_id] = "train"
            elif index < validation_end:
                splits[comparison_id] = "val"
            else:
                splits[comparison_id] = "test"
    return splits


def build_anchor_dataset(
    comparisons_path: str | Path,
    plan_sources_path: str | Path,
    profiles_path: str | Path,
    signatures_path: str | Path,
    output_path: str | Path,
    decode_len: int = 128,
) -> dict[str, int]:
    comparisons = [
        row
        for row in _read_csv(Path(comparisons_path))
        if int(row["decode_len"]) == decode_len
    ]
    source_to_plan = {
        row["summary_path"]: row["plan_id"]
        for row in _read_csv(Path(plan_sources_path))
    }
    estimator = ProfileEstimator(profiles_path, signatures_path)
    by_comparison: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in comparisons:
        by_comparison[row["comparison_id"]].append(row)

    exclusions: list[dict[str, str]] = []
    accepted: dict[str, list[tuple[dict[str, str], str]]] = {}
    for comparison_id, rows in sorted(by_comparison.items()):
        candidates: list[tuple[dict[str, str], str]] = []
        for row in rows:
            plan_id = source_to_plan.get(row["summary_path"])
            if plan_id is None:
                exclusions.append(
                    {
                        "comparison_id": comparison_id,
                        "summary_path": row["summary_path"],
                        "reason": "summary is not mapped to a retained plan",
                    }
                )
                continue
            signature = estimator.signatures.get(plan_id)
            if signature is None or signature["placement_complete"].casefold() != "true":
                exclusions.append(
                    {
                        "comparison_id": comparison_id,
                        "summary_path": row["summary_path"],
                        "reason": "plan signature is absent or incomplete",
                    }
                )
                continue
            candidates.append((row, plan_id))

        plan_ids = [plan_id for _, plan_id in candidates]
        if len(candidates) < 2 or len(set(plan_ids)) != len(plan_ids):
            for row, _ in candidates:
                exclusions.append(
                    {
                        "comparison_id": comparison_id,
                        "summary_path": row["summary_path"],
                        "reason": "comparison lacks two distinct complete plans",
                    }
                )
            continue
        accepted[comparison_id] = candidates

    splits = _group_splits(accepted)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for comparison_id, candidates in sorted(accepted.items()):
            for comparison, plan_id in candidates:
                batch = int(comparison["batch"])
                prefill_len = int(comparison["prefill_len"])
                duration = float(comparison["total_time_s"])
                signature = estimator.signature(plan_id)
                writer.writerow(
                    {
                        "state_id": f"anchor:{comparison_id}",
                        "trace_id": comparison_id,
                        "family": "dops_anchor_regular",
                        "split": splits[comparison_id],
                        "model_id": comparison["model_id"],
                        "hardware_key": comparison["hardware_key"],
                        "decision_index": 0,
                        "time_s": 0.0,
                        "plan_id": plan_id,
                        "horizon": decode_len,
                        **_regular_state(batch, prefill_len),
                        **signature,
                        "plan_shape_distance": estimator.shape_distance(
                            plan_id, batch, prefill_len + decode_len // 2
                        ),
                        "label_token_throughput": comparison[
                            "output_token_throughput"
                        ],
                        "label_duration_s": duration,
                        "label_generated_tokens": batch * decode_len,
                        "label_completed_requests": batch,
                        "label_p95_request_latency_s": duration,
                        "label_p95_queue_wait_s": 0.0,
                    }
                )
                rows_written += 1

    excluded_path = destination.with_name(destination.stem + "_excluded.csv")
    with excluded_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["comparison_id", "summary_path", "reason"]
        )
        writer.writeheader()
        writer.writerows(exclusions)

    return {
        "source_rows": len(comparisons),
        "states": len(accepted),
        "rows": rows_written,
        "excluded": len(exclusions),
    }
