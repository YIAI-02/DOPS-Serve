from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROFILE_FIELDS = [
    "plan_id",
    "model_id",
    "hardware_key",
    "policy",
    "batch",
    "prefill_len",
    "decode_len",
    "context_mid",
    "prefill_time_s",
    "decode_time_s",
    "total_time_s",
    "prefill_s_per_input_token",
    "decode_s_per_iteration",
    "output_token_throughput",
    "summary_path",
    "ops_trace_path",
    "comms_trace_path",
    "ops_prefill_count",
    "ops_decode_count",
    "ops_prefill_span_s",
    "ops_decode_span_s",
    "comm_prefill_bytes",
    "comm_decode_bytes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _trace_metrics(ops_path: Path, comms_path: Path) -> dict[str, float | int]:
    op_count = defaultdict(int)
    op_min_start: dict[str, float] = {}
    op_max_end: dict[str, float] = {}
    with ops_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            phase = row["phase"]
            start = float(row["start"])
            end = float(row["end"])
            op_count[phase] += 1
            op_min_start[phase] = min(op_min_start.get(phase, start), start)
            op_max_end[phase] = max(op_max_end.get(phase, end), end)

    comm_bytes = defaultdict(int)
    with comms_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            comm_bytes[row["phase"]] += int(float(row["bytes"]))

    return {
        "ops_prefill_count": op_count["prefill"],
        "ops_decode_count": op_count["decode"],
        "ops_prefill_span_s": op_max_end.get("prefill", 0.0)
        - op_min_start.get("prefill", 0.0),
        "ops_decode_span_s": op_max_end.get("decode", 0.0)
        - op_min_start.get("decode", 0.0),
        "comm_prefill_bytes": comm_bytes["prefill"],
        "comm_decode_bytes": comm_bytes["decode"],
    }


def _quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distance(
    query: tuple[float, ...],
    point: tuple[float, ...],
    ranges: tuple[float, ...],
) -> float:
    terms = [
        ((math.log1p(query[index]) - math.log1p(point[index])) / scale) ** 2
        for index, scale in enumerate(ranges)
        if scale > 0
    ]
    return math.sqrt(sum(terms)) if terms else 0.0


def _idw_estimate(
    query: tuple[float, ...],
    references: list[tuple[tuple[float, ...], float]],
    neighbors: int = 4,
) -> float:
    log_ranges = []
    for dimension in range(len(query)):
        values = [math.log1p(point[0][dimension]) for point in references]
        log_ranges.append(max(values) - min(values))
    ranked = sorted(
        (
            _distance(query, point, tuple(log_ranges)),
            value,
        )
        for point, value in references
    )[:neighbors]
    if ranked[0][0] == 0:
        return ranked[0][1]
    weights = [1.0 / distance for distance, _ in ranked]
    return sum(weight * ranked[index][1] for index, weight in enumerate(weights)) / sum(
        weights
    )


def _validate_profiles(rows: list[dict[str, object]], destination: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["plan_id"])].append(row)

    validation_rows: list[dict[str, object]] = []
    for plan_id, plan_rows in sorted(grouped.items()):
        if len(plan_rows) < 2:
            continue
        for held_index, held in enumerate(plan_rows):
            references = [row for index, row in enumerate(plan_rows) if index != held_index]
            targets = (
                (
                    "prefill_time_s",
                    (float(held["batch"]), float(held["prefill_len"])),
                    [
                        (
                            (float(row["batch"]), float(row["prefill_len"])),
                            float(row["prefill_time_s"]),
                        )
                        for row in references
                    ],
                ),
                (
                    "decode_s_per_iteration",
                    (float(held["batch"]), float(held["context_mid"])),
                    [
                        (
                            (float(row["batch"]), float(row["context_mid"])),
                            float(row["decode_s_per_iteration"]),
                        )
                        for row in references
                    ],
                ),
            )
            for metric, query, reference_values in targets:
                actual = float(held[metric])
                predicted = _idw_estimate(query, reference_values)
                validation_rows.append(
                    {
                        "plan_id": plan_id,
                        "summary_path": held["summary_path"],
                        "metric": metric,
                        "actual": actual,
                        "predicted": predicted,
                        "absolute_percentage_error": abs(predicted - actual)
                        / max(abs(actual), 1e-12),
                        "training_points": len(references),
                    }
                )

    fields = [
        "plan_id",
        "summary_path",
        "metric",
        "actual",
        "predicted",
        "absolute_percentage_error",
        "training_points",
    ]
    with (destination / "profile_validation.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows)

    errors = [float(row["absolute_percentage_error"]) for row in validation_rows]
    summary = {
        "validation_rows": len(validation_rows),
        "median_absolute_percentage_error": statistics.median(errors) if errors else None,
        "p95_absolute_percentage_error": _quantile(errors, 0.95) if errors else None,
        "note": "Leave-one-profile-out IDW validation; thresholds are not assumed.",
    }
    with (destination / "profile_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_profiles(
    inventory_path: str | Path,
    plan_sources_path: str | Path,
    artifact_dir: str | Path,
    include_trace_metrics: bool = False,
) -> dict[str, int]:
    inventory_rows = _read_csv(Path(inventory_path))
    source_rows = _read_csv(Path(plan_sources_path))
    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "plan_bank.jsonl").open("r", encoding="utf-8") as handle:
        plan_metadata = {
            row["plan_id"]: row
            for row in (json.loads(line) for line in handle if line.strip())
        }

    inventory_by_summary = {row["summary_path"]: row for row in inventory_rows}
    profiles: list[dict[str, object]] = []
    for source in source_rows:
        row = inventory_by_summary[source["summary_path"]]
        batch = int(row["batch"])
        prefill_len = int(row["prefill_len"])
        decode_len = int(row["decode_len"])
        prefill_time = float(row["prefill_time_s"])
        decode_time = float(row["decode_time_s"])
        total_time = float(row["total_time_s"])
        profile: dict[str, object] = {
            "plan_id": source["plan_id"],
            "model_id": row["model_id"],
            "hardware_key": row["hardware_key"],
            "policy": row["policy"],
            "batch": batch,
            "prefill_len": prefill_len,
            "decode_len": decode_len,
            "context_mid": prefill_len + decode_len / 2.0,
            "prefill_time_s": prefill_time,
            "decode_time_s": decode_time,
            "total_time_s": total_time,
            "prefill_s_per_input_token": prefill_time / (batch * prefill_len),
            "decode_s_per_iteration": decode_time / decode_len,
            "output_token_throughput": batch * decode_len / total_time,
            "summary_path": row["summary_path"],
            "ops_trace_path": row["ops_trace_path"],
            "comms_trace_path": row["comms_trace_path"],
            "ops_prefill_count": "",
            "ops_decode_count": "",
            "ops_prefill_span_s": "",
            "ops_decode_span_s": "",
            "comm_prefill_bytes": "",
            "comm_decode_bytes": "",
        }
        if include_trace_metrics:
            profile.update(
                _trace_metrics(Path(row["ops_trace_path"]), Path(row["comms_trace_path"]))
            )
        profiles.append(profile)

    with (destination / "profiles.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS)
        writer.writeheader()
        writer.writerows(profiles)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for profile in profiles:
        grouped[str(profile["plan_id"])].append(profile)
    signatures = []
    for plan_id, plan_rows in sorted(grouped.items()):
        signatures.append(
            {
                "plan_id": plan_id,
                "model_id": plan_rows[0]["model_id"],
                "hardware_key": plan_rows[0]["hardware_key"],
                "placement_complete": plan_metadata[plan_id]["placement_complete"],
                "incomplete_source_count": plan_metadata[plan_id][
                    "incomplete_source_count"
                ],
                "profile_count": len(plan_rows),
                "mean_prefill_s_per_input_token": statistics.fmean(
                    float(row["prefill_s_per_input_token"]) for row in plan_rows
                ),
                "mean_decode_s_per_iteration": statistics.fmean(
                    float(row["decode_s_per_iteration"]) for row in plan_rows
                ),
                "mean_output_token_throughput": statistics.fmean(
                    float(row["output_token_throughput"]) for row in plan_rows
                ),
                "mean_total_time_s": statistics.fmean(
                    float(row["total_time_s"]) for row in plan_rows
                ),
            }
        )
    signature_fields = [
        "plan_id",
        "model_id",
        "hardware_key",
        "placement_complete",
        "incomplete_source_count",
        "profile_count",
        "mean_prefill_s_per_input_token",
        "mean_decode_s_per_iteration",
        "mean_output_token_throughput",
        "mean_total_time_s",
    ]
    with (destination / "plan_signatures.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=signature_fields)
        writer.writeheader()
        writer.writerows(signatures)

    _validate_profiles(profiles, destination)
    return {"profiles": len(profiles), "plans": len(signatures)}


def estimate_idw(
    query: tuple[float, ...],
    references: list[tuple[tuple[float, ...], float]],
    neighbors: int = 4,
) -> float:
    return _idw_estimate(query, references, neighbors=neighbors)
