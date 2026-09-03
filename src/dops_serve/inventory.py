from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from .summary import read_summary_metadata


_MODEL_DIR = re.compile(
    r"^(?P<model>.+)_(?P<dtype>fp16|fp32|bf16|int\d+)_b(?P<batch>\d+)_s(?P<shards>\d+)$",
    re.IGNORECASE,
)
_SUMMARY_NAME = re.compile(r"^best_summary_(?P<prefill>\d+)x(?P<decode>\d+)\.json$")

INVENTORY_FIELDS = [
    "summary_path",
    "relative_summary_path",
    "experiment",
    "hardware_key",
    "model_id",
    "dtype",
    "batch",
    "shards",
    "policy",
    "algorithm_dir",
    "prefill_len",
    "decode_len",
    "prefill_time_s",
    "decode_time_s",
    "total_time_s",
    "ops_trace_path",
    "comms_trace_path",
    "summary_size_bytes",
]


def _assert_outside_source(source_root: Path, artifact_dir: Path) -> None:
    source = str(source_root.resolve())
    destination = str(artifact_dir.resolve())
    if os.path.commonpath([source, destination]) == source:
        raise ValueError("artifact_dir must be outside the read-only DOPS output root")


def _resolve_trace(output_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    candidate = Path(str(value))
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0] == "output":
        candidate = Path(*parts[1:])
    return output_root / candidate


def _path_metadata(output_root: Path, summary_path: Path) -> dict[str, str]:
    relative = summary_path.relative_to(output_root)
    model_index = -1
    model_match: re.Match[str] | None = None
    for index, part in enumerate(relative.parts):
        match = _MODEL_DIR.match(part)
        if match:
            model_index = index
            model_match = match
            break
    if model_match is None:
        raise ValueError(f"Cannot identify model directory from {relative}")
    if model_index + 1 >= len(relative.parts):
        raise ValueError(f"Cannot identify algorithm directory from {relative}")
    return {
        "relative_summary_path": str(relative),
        "experiment": relative.parts[0],
        "hardware_key": "/".join(relative.parts[:model_index]),
        "model_id": model_match.group("model"),
        "dtype": model_match.group("dtype").lower(),
        "batch": model_match.group("batch"),
        "shards": model_match.group("shards"),
        "algorithm_dir": relative.parts[model_index + 1],
    }


def _inventory_row(output_root: Path, summary_path: Path) -> dict[str, Any]:
    path_data = _path_metadata(output_root, summary_path)
    header = read_summary_metadata(summary_path)
    config = header.get("config", {})
    times = header.get("best_times", {})
    plan_label = header.get("plan_label", {})
    name_match = _SUMMARY_NAME.match(summary_path.name)
    if name_match is None:
        raise ValueError(f"Unexpected summary filename: {summary_path.name}")

    prefill_len = int(config.get("prefill_len", name_match.group("prefill")))
    decode_len = int(config.get("decode_len", name_match.group("decode")))
    batch = int(config.get("batch", path_data["batch"]))
    ops_path = _resolve_trace(output_root, plan_label.get("trace_ops_csv"))
    comms_path = _resolve_trace(output_root, plan_label.get("trace_comms_csv"))
    missing = [
        label
        for label, linked_path in (("ops_trace", ops_path), ("comms_trace", comms_path))
        if linked_path is None or not linked_path.is_file()
    ]
    if missing:
        raise ValueError("missing linked files: " + ", ".join(missing))

    return {
        "summary_path": str(summary_path),
        **path_data,
        "batch": batch,
        "policy": str(header["policy"]),
        "prefill_len": prefill_len,
        "decode_len": decode_len,
        "prefill_time_s": float(times["prefill"]),
        "decode_time_s": float(times["decode"]),
        "total_time_s": float(times["total"]),
        "ops_trace_path": str(ops_path),
        "comms_trace_path": str(comms_path),
        "summary_size_bytes": summary_path.stat().st_size,
    }


def build_inventory(
    output_root: str | Path,
    artifact_dir: str | Path,
    max_summaries: int = 0,
    path_prefix: str | list[str] | None = None,
) -> tuple[int, int]:
    source = Path(output_root).resolve()
    destination = Path(artifact_dir).resolve()
    if not source.is_dir():
        raise NotADirectoryError(source)
    _assert_outside_source(source, destination)
    destination.mkdir(parents=True, exist_ok=True)

    summaries = sorted(source.rglob("best_summary_*.json"))
    if path_prefix:
        prefixes = [path_prefix] if isinstance(path_prefix, str) else path_prefix
        summaries = [
            path
            for path in summaries
            if any(
                str(path.relative_to(source)).startswith(prefix) for prefix in prefixes
            )
        ]
    if max_summaries > 0:
        summaries = summaries[:max_summaries]

    inventory_path = destination / "inventory.csv"
    excluded_path = destination / "excluded.csv"
    valid_count = 0
    excluded_count = 0
    with inventory_path.open("w", encoding="utf-8", newline="") as inventory_handle, (
        excluded_path.open("w", encoding="utf-8", newline="")
    ) as excluded_handle:
        writer = csv.DictWriter(inventory_handle, fieldnames=INVENTORY_FIELDS)
        excluded_writer = csv.DictWriter(
            excluded_handle, fieldnames=["summary_path", "reason"]
        )
        writer.writeheader()
        excluded_writer.writeheader()
        for summary_path in summaries:
            try:
                writer.writerow(_inventory_row(source, summary_path))
                valid_count += 1
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
                excluded_writer.writerow(
                    {"summary_path": str(summary_path), "reason": str(error)}
                )
                excluded_count += 1
    return valid_count, excluded_count
