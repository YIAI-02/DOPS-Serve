"""Full chronological BurstGPT traces and arrival-time scaling."""
from __future__ import annotations

import csv
import json
from pathlib import Path


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
