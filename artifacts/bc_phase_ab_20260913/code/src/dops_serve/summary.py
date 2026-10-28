from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterator


_TOP_LEVEL_KEY = re.compile(r'^  "(?P<key>[^"]+)"\s*:')


def read_summary_metadata(path: str | Path) -> dict[str, Any]:
    """Read only the metadata prefix before the large schedule arrays."""
    summary_path = Path(path)
    lines: list[str] = []
    plan_label_seen = False
    metadata_complete = False
    with summary_path.open("r", encoding="utf-8", newline=None) as handle:
        for line in handle:
            match = _TOP_LEVEL_KEY.match(line)
            if match:
                key = match.group("key")
                if plan_label_seen and key != "plan_label":
                    metadata_complete = True
                    break
                if key == "plan_label":
                    plan_label_seen = True
            lines.append(line)

    if not plan_label_seen:
        raise ValueError(f"Top-level plan_label not found in {summary_path}")
    if not metadata_complete:
        return json.loads("".join(lines))

    prefix = "".join(lines).rstrip()
    if prefix.endswith(","):
        prefix = prefix[:-1]
    return json.loads(prefix + "\n}")


def iter_summary_array(path: str | Path, key: str) -> Iterator[Any]:
    """Stream items from one pretty-printed top-level JSON array."""
    summary_path = Path(path)
    decoder = json.JSONDecoder()
    with summary_path.open("r", encoding="utf-8", newline=None) as handle:
        initial = ""
        for line in handle:
            match = _TOP_LEVEL_KEY.match(line)
            if not match or match.group("key") != key:
                continue
            bracket = line.find("[")
            if bracket < 0:
                raise ValueError(f"Top-level key {key!r} is not an array in {summary_path}")
            initial = line[bracket + 1 :]
            break
        else:
            raise KeyError(f"Top-level array {key!r} not found in {summary_path}")

        buffer = initial
        position = 0
        eof = False
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                chunk = handle.read(1 << 20)
                if not chunk:
                    eof = True
                    break
                buffer = buffer[position:] + chunk
                position = 0

            if eof:
                raise ValueError(f"Unterminated top-level array {key!r} in {summary_path}")
            if buffer[position] == "]":
                return

            try:
                item, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                chunk = handle.read(1 << 20)
                if not chunk:
                    raise ValueError(
                        f"Invalid JSON while reading {key!r} in {summary_path}: {error}"
                    ) from error
                buffer = buffer[position:] + chunk
                position = 0
                continue

            yield item
            position = end
            if position >= 1 << 20:
                buffer = buffer[position:]
                position = 0


def _trace_decode_placements(
    path: str | Path, expected_nodes: set[str], required_steps: int
) -> list[list[list[str]]]:
    counts = {node_id: 0 for node_id in expected_nodes}
    placements: list[dict[str, str]] = [dict() for _ in range(required_steps)]
    completed_nodes = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["phase"] != "decode":
                continue
            node_id = str(row["node_id"])
            if node_id not in expected_nodes:
                raise ValueError(f"Trace contains unknown decode node {node_id!r} in {path}")
            occurrence = counts[node_id]
            if occurrence < required_steps:
                placements[occurrence][node_id] = str(row["device"])
            counts[node_id] += 1
            if counts[node_id] == required_steps:
                completed_nodes += 1
                if completed_nodes == len(expected_nodes):
                    break

    incomplete = [
        node_id for node_id, count in counts.items() if count < required_steps
    ]
    if incomplete:
        raise ValueError(
            f"Trace does not cover {required_steps} decode steps for {len(incomplete)} nodes in {path}"
        )
    return [
        [[node_id, mapping[node_id]] for node_id in sorted(mapping)]
        for mapping in placements
    ]


def extract_plan(
    path: str | Path,
    max_decode_steps: int = 128,
    ops_trace_path: str | Path | None = None,
) -> dict[str, Any]:
    """Extract ordered device assignments without copying timing records."""
    prefill = [
        [str(item["node_id"]), str(item["device"])]
        for item in iter_summary_array(path, "prefill_schedule")
    ]

    selected_steps: list[dict[str, Any]] = []
    decode_iterator = iter_summary_array(path, "decode_steps")
    try:
        for index, step in enumerate(decode_iterator):
            if index >= max_decode_steps:
                break
            schedule = step.get("schedule")
            selected_steps.append(
                {
                    "t": int(step["t"]),
                    "placement": None
                    if schedule is None
                    else sorted(
                        [str(item["node_id"]), str(item["device"])]
                        for item in schedule
                    ),
                }
            )
    finally:
        decode_iterator.close()

    observed_steps = sum(step["placement"] is not None for step in selected_steps)
    estimated_steps = len(selected_steps) - observed_steps
    if not prefill or not selected_steps or observed_steps == 0:
        raise ValueError(f"Incomplete placement in {path}")

    recovered: dict[int, list[list[str]]] = {}
    if estimated_steps and ops_trace_path is not None:
        first_schedule = next(
            step["placement"]
            for step in selected_steps
            if step["placement"] is not None
        )
        expected_nodes = {item[0] for item in first_schedule}
        required_steps = max(int(step["t"]) for step in selected_steps) + 1
        trace_placements = _trace_decode_placements(
            ops_trace_path, expected_nodes, required_steps
        )
        for step in selected_steps:
            t = int(step["t"])
            trace_placement = trace_placements[t]
            placement = step["placement"]
            if placement is None:
                recovered[t] = trace_placement
                continue
            if placement != trace_placement:
                raise ValueError(
                    f"Summary and ops trace placement disagree at decode step {t} in {path}"
                )

    decode_segments: list[dict[str, Any]] = []
    for step in selected_steps:
        placement = step["placement"]
        if placement is None:
            placement = recovered.get(int(step["t"]))
        if decode_segments and decode_segments[-1]["placement"] == placement:
            decode_segments[-1]["end_t"] = int(step["t"])
            decode_segments[-1]["count"] += 1
        else:
            decode_segments.append(
                {
                    "start_t": int(step["t"]),
                    "end_t": int(step["t"]),
                    "count": 1,
                    "placement": placement,
                }
            )

    recovered_steps = len(recovered)
    missing_steps = estimated_steps - recovered_steps
    return {
        "prefill": prefill,
        "decode_segments": decode_segments,
        "decode_steps_covered": sum(segment["count"] for segment in decode_segments),
        "decode_steps_observed": observed_steps,
        "decode_steps_estimated": estimated_steps,
        "decode_steps_recovered": recovered_steps,
        "decode_steps_missing": missing_steps,
        "placement_complete": missing_steps == 0,
    }
