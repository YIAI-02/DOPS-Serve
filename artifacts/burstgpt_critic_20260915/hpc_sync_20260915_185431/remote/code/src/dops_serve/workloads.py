from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def import_op_policy_burstgpt(
    input_path: str | Path,
    artifact_dir: str | Path,
    requests_per_trace: int = 1000,
) -> dict[str, Any]:
    """Chronological 70/15/15 source splits, shared by every model/load variant."""
    eligible = []
    excluded: Counter = Counter()
    with Path(input_path).open(encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if row["Model"] != "ChatGPT" or row["Log Type"] != "API log":
                excluded["not_chatgpt_api"] += 1
                continue
            prompt, output = int(row["Request tokens"]), int(row["Response tokens"])
            if not 128 <= prompt <= 2048:
                excluded["prompt_out_of_range"] += 1
                continue
            if not 1 <= output <= 1024:
                excluded["output_out_of_range"] += 1
                continue
            eligible.append((float(row["Timestamp"]), index, prompt, output))
    eligible.sort()
    boundaries = (0, int(len(eligible) * 0.70), int(len(eligible) * 0.85), len(eligible))
    destination = Path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_summary = {}
    with (destination / "workloads.jsonl").open("w") as records, (destination / "workload_windows.jsonl").open("w") as windows:
        for split, first, last in zip(("train", "val", "test"), boundaries, boundaries[1:]):
            complete = (last - first) // requests_per_trace
            split_summary[split] = {"eligible_requests": last - first, "traces": complete, "discarded_tail": (last - first) % requests_per_trace}
            if complete == 0:
                raise ValueError(f"{split} has no complete {requests_per_trace}-request trace")
            for window in range(complete):
                start = first + window * requests_per_trace
                selected = eligible[start:start + requests_per_trace]
                trace_id = f"burstgpt_{split}_{window:06d}"
                origin = selected[0][0]
                windows.write(json.dumps({
                    "trace_id": trace_id, "split": split,
                    "sorted_eligible_start": start, "sorted_eligible_end_exclusive": start + requests_per_trace,
                    "source_time_start": origin, "source_time_end": selected[-1][0],
                    "source_rows": [item[1] for item in selected],
                }) + "\n")
                for timestamp, index, prompt, output in selected:
                    records.write(json.dumps({
                        "trace_id": trace_id, "source_window_id": trace_id,
                        "split": split, "family": "burstgpt", "request_id": f"burstgpt_row_{index}",
                        "source_row": index, "source_timestamp": timestamp,
                        "arrival_time": timestamp - origin, "prompt_len": prompt,
                        "max_tokens": 1024, "output_len": output,
                    }) + "\n")
    result = {
        "source": str(Path(input_path).resolve()), "eligible_requests": len(eligible),
        "excluded": dict(excluded), "splits": split_summary,
        "requests_per_trace": requests_per_trace, "arrival_scaling": "unscaled; calibrate on train/validation",
        "variants": "model and load variants inherit source_window_id and split",
    }
    (destination / "workload_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def import_full_burstgpt(input_path, artifact_dir, requests_per_trace=1000):
    """All source rows, stable time order, chronological splits and retained tails."""
    rows = []
    with Path(input_path).open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            prompt, output = int(row["Request tokens"]), int(row["Response tokens"])
            if prompt <= 0 or not 1 <= output <= 7315:
                raise ValueError(f"source row {index}: invalid token lengths")
            rows.append((float(row["Timestamp"]), index, prompt, output))
    rows.sort(key=lambda r: (r[0], r[1]))
    root = Path(artifact_dir)
    boundaries = (0, int(len(rows) * .7), int(len(rows) * .85), len(rows))
    manifest = dict(source=str(input_path), requests=len(rows), max_tokens=7315,
                    requests_per_trace=requests_per_trace, splits={})
    for split, lo, hi in zip(("train", "val", "test"), boundaries, boundaries[1:]):
        folder = root / split
        folder.mkdir(parents=True, exist_ok=True)
        traces = []
        for trace_index, start in enumerate(range(lo, hi, requests_per_trace)):
            selected = rows[start:min(start + requests_per_trace, hi)]
            origin = selected[0][0]
            records = [dict(request_id=f"burstgpt_row_{i}", source_row=i, source_timestamp=t,
                            arrival_time=t-origin, prompt_len=p, output_len=d, max_tokens=7315)
                       for t, i, p, d in selected]
            (folder / f"{trace_index:06d}.json").write_text(json.dumps(records) + "\n")
            traces.append(dict(index=trace_index, requests=len(records), sorted_start=start,
                               sorted_end=start+len(records), source_start=origin,
                               source_end=selected[-1][0], output_tokens=sum(r[3] for r in selected)))
        selected = rows[lo:hi]
        duration = selected[-1][0] - selected[0][0]
        manifest["splits"][split] = dict(requests=hi-lo, traces=traces,
            source_duration_s=duration, output_tokens=sum(r[3] for r in selected),
            arrival_output_tokens_per_s=sum(r[3] for r in selected)/duration,
            max_prompt=max(r[2] for r in selected), max_output=max(r[3] for r in selected))
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_full_trace(data_dir, split, index, arrival_scale=1.0):
    rows = json.loads((Path(data_dir) / split / f"{index:06d}.json").read_text())
    return [{**r, "arrival_time": r["arrival_time"] * arrival_scale} for r in rows]
