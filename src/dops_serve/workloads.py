from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


FAMILIES = ("poisson", "bursty", "periodic", "heavy_tailed")


def _bounded_int(value: float, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _next_arrival(
    family: str,
    request_index: int,
    current_time: float,
    base_rate: float,
    rng: random.Random,
) -> float:
    if family == "poisson":
        return current_time + rng.expovariate(base_rate)
    if family == "bursty":
        in_burst = request_index % 24 < 8
        rate = base_rate * (8.0 if in_burst else 0.35)
        return current_time + rng.expovariate(rate)
    if family == "periodic":
        burst_size = 8
        period = 12.0
        cycle = request_index // burst_size
        within = request_index % burst_size
        return cycle * period + within * 0.025 + rng.uniform(0.0, 0.01)
    if family == "heavy_tailed":
        return current_time + min(rng.paretovariate(1.6) / (4.0 * base_rate), 20.0)
    raise ValueError(f"Unknown workload family: {family}")


def _request_lengths(
    family: str,
    rng: random.Random,
    prompt_min: int,
    prompt_max: int,
    max_token_choices: list[int],
) -> tuple[int, int, int]:
    if family == "heavy_tailed":
        prompt_len = _bounded_int(
            96.0 * rng.paretovariate(1.35), prompt_min, prompt_max
        )
    else:
        prompt_len = _bounded_int(
            math.exp(rng.gauss(math.log(512), 0.8)), prompt_min, prompt_max
        )
    max_tokens = rng.choice(max_token_choices)
    output_len = _bounded_int(
        math.exp(rng.gauss(math.log(max_tokens / 3.0), 0.65)), 1, max_tokens
    )
    return prompt_len, max_tokens, output_len


def generate_synthetic_workloads(
    output_path: str | Path,
    model_ids: list[str],
    families: list[str] | None = None,
    traces_per_family: int = 4,
    requests_per_trace: int = 200,
    base_rate: float = 2.0,
    seed: int = 7,
    prompt_min: int = 128,
    prompt_max: int = 4096,
    max_token_choices: list[int] | None = None,
) -> dict[str, int]:
    selected_families = families or list(FAMILIES)
    unknown = sorted(set(selected_families) - set(FAMILIES))
    if unknown:
        raise ValueError(f"Unknown workload families: {', '.join(unknown)}")
    if prompt_min <= 0 or prompt_max < prompt_min:
        raise ValueError("prompt bounds must satisfy 0 < prompt_min <= prompt_max")
    token_choices = sorted(set(max_token_choices or [128, 256, 512, 1024]))
    if not token_choices or token_choices[0] <= 0:
        raise ValueError("max token choices must be positive")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for model_id in model_ids:
        for family in selected_families:
            for trace_index in range(traces_per_family):
                trace_id = f"{family}_{model_id}_{trace_index:03d}"
                current_time = 0.0
                for request_index in range(requests_per_trace):
                    current_time = _next_arrival(
                        family, request_index, current_time, base_rate, rng
                    )
                    prompt_len, max_tokens, output_len = _request_lengths(
                        family,
                        rng,
                        prompt_min,
                        prompt_max,
                        token_choices,
                    )
                    records.append(
                        {
                            "trace_id": trace_id,
                            "family": family,
                            "request_id": f"{trace_id}_r{request_index:06d}",
                            "model_id": model_id,
                            "arrival_time": current_time,
                            "prompt_len": prompt_len,
                            "max_tokens": max_tokens,
                            "output_len": output_len,
                        }
                    )

    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "traces": len(model_ids) * len(selected_families) * traces_per_family,
        "requests": len(records),
    }


def import_trace_csv(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str,
    trace_id: str,
    arrival_column: str,
    prompt_column: str,
    output_column: str,
    max_tokens_column: str | None = None,
    max_tokens_default: int | None = None,
    family: str = "real",
) -> dict[str, int]:
    if max_tokens_column == output_column:
        raise ValueError("actual output length cannot be reused as max_tokens")
    if max_tokens_column is None and max_tokens_default is None:
        raise ValueError("provide a real max_tokens column or an independent default")

    source = Path(input_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open("r", encoding="utf-8", newline="") as input_handle, (
        destination.open("w", encoding="utf-8")
    ) as output_handle:
        for index, row in enumerate(csv.DictReader(input_handle)):
            output_len = int(float(row[output_column]))
            max_tokens = (
                int(float(row[max_tokens_column]))
                if max_tokens_column is not None
                else int(max_tokens_default)
            )
            if output_len > max_tokens:
                raise ValueError(
                    f"row {index}: output_len {output_len} exceeds max_tokens {max_tokens}"
                )
            record = {
                "trace_id": trace_id,
                "family": family,
                "request_id": f"{trace_id}_r{index:06d}",
                "model_id": model_id,
                "arrival_time": float(row[arrival_column]),
                "prompt_len": int(float(row[prompt_column])),
                "max_tokens": max_tokens,
                "output_len": output_len,
            }
            output_handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return {"traces": 1, "requests": count}


def import_burstgpt_windows(
    input_path: str | Path,
    output_path: str | Path,
    model_id: str,
    traces: int = 16,
    requests_per_trace: int = 100,
    target_median_rps: float = 2.0,
    max_tokens_default: int = 1024,
    prompt_min: int = 128,
    prompt_max: int = 2048,
    output_max: int = 1024,
    source_model: str | None = None,
    log_type: str | None = None,
    selection: str = "mixed",
    split_seed: int = 7,
) -> dict[str, Any]:
    if traces <= 0 or requests_per_trace < 2:
        raise ValueError("traces must be positive and requests_per_trace must be >= 2")
    if target_median_rps <= 0:
        raise ValueError("target_median_rps must be positive")
    if max_tokens_default <= 0 or output_max <= 0:
        raise ValueError("token limits must be positive")
    if prompt_min <= 0 or prompt_max < prompt_min:
        raise ValueError("prompt bounds must satisfy 0 < prompt_min <= prompt_max")
    if selection not in {"representative", "peak", "mixed"}:
        raise ValueError(f"unknown BurstGPT window selection: {selection}")

    source = Path(input_path)
    eligible: list[tuple[float, int, int, int]] = []
    source_rows = 0
    with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        required_columns = {"Timestamp", "Request tokens", "Response tokens"}
        if source_model is not None:
            required_columns.add("Model")
        if log_type is not None:
            required_columns.add("Log Type")
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"BurstGPT CSV is missing columns: {', '.join(sorted(missing))}"
            )

        for source_index, row in enumerate(reader):
            source_rows += 1
            if source_model is not None and row["Model"] != source_model:
                continue
            if log_type is not None and row["Log Type"] != log_type:
                continue
            prompt_len = int(float(row["Request tokens"]))
            output_len = int(float(row["Response tokens"]))
            if not prompt_min <= prompt_len <= prompt_max:
                continue
            if not 1 <= output_len <= min(output_max, max_tokens_default):
                continue
            eligible.append(
                (float(row["Timestamp"]), prompt_len, output_len, source_index)
            )

    eligible.sort(key=lambda item: (item[0], item[3]))
    if len(eligible) < traces * requests_per_trace:
        raise ValueError(
            f"only {len(eligible)} eligible requests for "
            f"{traces} traces x {requests_per_trace} requests"
        )

    windows: list[tuple[int, str, list[tuple[float, int, int, int]]]] = []
    for trace_index in range(traces):
        block_start = trace_index * len(eligible) // traces
        block_end = (trace_index + 1) * len(eligible) // traces
        method = selection
        if selection == "mixed":
            method = "representative" if trace_index % 2 == 0 else "peak"
        if method == "representative":
            window_start = block_start + (
                block_end - block_start - requests_per_trace
            ) // 2
        else:
            window_start = min(
                range(block_start, block_end - requests_per_trace + 1),
                key=lambda index: (
                    eligible[index + requests_per_trace - 1][0]
                    - eligible[index][0],
                    index,
                ),
            )
        windows.append(
            (
                trace_index,
                method,
                eligible[window_start : window_start + requests_per_trace],
            )
        )

    raw_rates = [
        (requests_per_trace - 1) / (window[-1][0] - window[0][0])
        for _, _, window in windows
        if window[-1][0] > window[0][0]
    ]
    if not raw_rates:
        raise ValueError("selected BurstGPT windows have no positive time span")
    raw_median_rps = statistics.median(raw_rates)
    arrival_time_scale = raw_median_rps / target_median_rps

    split_by_stratum: dict[int, str] = {}
    rng = random.Random(split_seed)
    for method in sorted({method for _, method, _ in windows}):
        strata = [
            trace_index
            for trace_index, window_method, _ in windows
            if window_method == method
        ]
        rng.shuffle(strata)
        if len(strata) < 3:
            train_end = len(strata)
            validation_end = len(strata)
        else:
            train_end = max(1, int(len(strata) * 0.7))
            validation_end = max(train_end + 1, int(len(strata) * 0.85))
            validation_end = min(validation_end, len(strata) - 1)
        for index, trace_index in enumerate(strata):
            if index < train_end:
                split_by_stratum[trace_index] = "train"
            elif index < validation_end:
                split_by_stratum[trace_index] = "val"
            else:
                split_by_stratum[trace_index] = "test"

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_count = 0
    with destination.open("w", encoding="utf-8") as output_handle:
        for trace_index, method, window in windows:
            trace_id = f"burstgpt_real_s{trace_index:03d}_{method}"
            first_timestamp = window[0][0]
            for request_index, (timestamp, prompt_len, output_len, _) in enumerate(
                window
            ):
                record = {
                    "trace_id": trace_id,
                    "family": f"burstgpt_{method}",
                    "split": split_by_stratum[trace_index],
                    "request_id": f"{trace_id}_r{request_index:06d}",
                    "model_id": model_id,
                    "arrival_time": (timestamp - first_timestamp)
                    * arrival_time_scale,
                    "prompt_len": prompt_len,
                    "max_tokens": max_tokens_default,
                    "output_len": output_len,
                }
                output_handle.write(json.dumps(record, sort_keys=True) + "\n")
                request_count += 1

    return {
        "source_rows": source_rows,
        "eligible_requests": len(eligible),
        "traces": len(windows),
        "requests": request_count,
        "representative_traces": sum(
            method == "representative" for _, method, _ in windows
        ),
        "peak_traces": sum(method == "peak" for _, method, _ in windows),
        "train_traces": sum(split == "train" for split in split_by_stratum.values()),
        "validation_traces": sum(
            split == "val" for split in split_by_stratum.values()
        ),
        "test_traces": sum(split == "test" for split in split_by_stratum.values()),
        "split_seed": split_seed,
        "raw_median_rps": raw_median_rps,
        "target_median_rps": target_median_rps,
        "arrival_time_scale": arrival_time_scale,
    }
