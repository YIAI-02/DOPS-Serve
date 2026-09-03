from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator


_TOP_LEVEL_KEY = re.compile(r'^  "(?P<key>[^"]+)"\s*:')


def read_summary_metadata(path: str | Path) -> dict[str, Any]:
    """Read only the metadata prefix before the large schedule arrays."""
    summary_path = Path(path)
    lines: list[str] = []
    schedule_key_found = False
    with summary_path.open("r", encoding="utf-8", newline=None) as handle:
        for line in handle:
            match = _TOP_LEVEL_KEY.match(line)
            if match and match.group("key") in {"prefill_schedule", "decode_steps"}:
                schedule_key_found = True
                break
            lines.append(line)

    if not schedule_key_found:
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

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


def extract_plan(path: str | Path, max_decode_steps: int = 128) -> dict[str, Any]:
    """Extract ordered device assignments without copying timing records."""
    prefill = [
        [str(item["node_id"]), str(item["device"])]
        for item in iter_summary_array(path, "prefill_schedule")
    ]

    decode_segments: list[dict[str, Any]] = []
    observed_steps = 0
    estimated_steps = 0
    decode_iterator = iter_summary_array(path, "decode_steps")
    try:
        for index, step in enumerate(decode_iterator):
            if index >= max_decode_steps:
                break
            schedule = step.get("schedule")
            estimated = bool(step.get("estimated", False))
            if schedule is None:
                placement = None
                estimated_steps += 1
            else:
                placement = [
                    [str(item["node_id"]), str(item["device"])] for item in schedule
                ]
                observed_steps += 1
            if (
                decode_segments
                and decode_segments[-1]["placement"] == placement
                and decode_segments[-1]["estimated"] == estimated
            ):
                decode_segments[-1]["end_t"] = int(step["t"])
                decode_segments[-1]["count"] += 1
            else:
                decode_segments.append(
                    {
                        "start_t": int(step["t"]),
                        "end_t": int(step["t"]),
                        "count": 1,
                        "estimated": estimated,
                        "placement": placement,
                    }
                )
    finally:
        decode_iterator.close()

    if not prefill or not decode_segments:
        raise ValueError(f"Incomplete placement in {path}")
    return {
        "prefill": prefill,
        "decode_segments": decode_segments,
        "decode_steps_covered": sum(segment["count"] for segment in decode_segments),
        "decode_steps_observed": observed_steps,
        "decode_steps_estimated": estimated_steps,
        "placement_complete": estimated_steps == 0,
    }


def canonical_plan_text(plan: dict[str, Any]) -> str:
    """Deterministic exact-comparison text; it is not a content hash."""
    core = {
        "prefill": plan["prefill"],
        "decode_segments": [
            {
                "count": segment["count"],
                "estimated": segment["estimated"],
                "placement": segment["placement"],
            }
            for segment in plan["decode_segments"]
        ],
    }
    return json.dumps(core, ensure_ascii=False, separators=(",", ":"))
