from __future__ import annotations

import csv
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .dataset import DATASET_FIELDS
from .replay import ProfileEstimator
from .summary import read_decode_step_times


DEFAULT_DECODE_LENGTHS = [128, 256, 512, 1024]
DEFAULT_HORIZONS = [32, 64, 128]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_candidate_timing(
    task: tuple[dict[str, str], str]
) -> tuple[list[float] | None, str | None]:
    row, _ = task
    source_decode_len = int(row["decode_len"])
    try:
        step_times = read_decode_step_times(
            row["summary_path"], max_decode_steps=source_decode_len
        )
    except (KeyError, OSError, ValueError) as error:
        return None, f"decode step timing unavailable: {error}"
    if len(step_times) != source_decode_len or not math.isclose(
        sum(step_times),
        float(row["decode_time_s"]),
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        return None, "decode step timings do not match the source decode total"
    prefix_times = [0.0]
    for step_time in step_times:
        prefix_times.append(prefix_times[-1] + step_time)
    return prefix_times, None


def _regular_state(
    batch: int, prefill_len: int, generated_tokens: int
) -> dict[str, float]:
    at_arrival = generated_tokens == 0
    context_len = prefill_len + generated_tokens
    return {
        "waiting_count": float(batch if at_arrival else 0),
        "active_count": float(0 if at_arrival else batch),
        "total_waiting_prompt_tokens": float(batch * prefill_len if at_arrival else 0),
        "total_active_context_tokens": float(0 if at_arrival else batch * context_len),
        "arrival_rate_1s": float(batch if at_arrival else 0),
        "arrival_rate_10s": batch / 10.0 if at_arrival else 0.0,
        "arrival_rate_60s": batch / 60.0 if at_arrival else 0.0,
        "waiting_prompt_mean": float(prefill_len if at_arrival else 0),
        "waiting_prompt_p95": float(prefill_len if at_arrival else 0),
        "waiting_prompt_max": float(prefill_len if at_arrival else 0),
        "active_context_mean": float(0 if at_arrival else context_len),
        "active_context_p95": float(0 if at_arrival else context_len),
        "active_context_max": float(0 if at_arrival else context_len),
        "waiting_age_s_mean": 0.0,
        "waiting_age_s_p95": 0.0,
        "waiting_age_s_max": 0.0,
        "max_tokens_mean": 0.0,
        "max_tokens_p95": 0.0,
        "max_tokens_max": 0.0,
        "generated_tokens_mean": float(generated_tokens),
        "generated_tokens_p95": float(generated_tokens),
        "generated_tokens_max": float(generated_tokens),
    }


def _group_splits(
    groups: dict[str, list[tuple[dict[str, str], str, list[float]]]]
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
    decode_len: int | None = None,
    decode_lengths: list[int] | None = None,
    horizons: list[int] | None = None,
    window_stride: int = 32,
    workers: int = 1,
) -> dict[str, int]:
    if decode_len is not None and decode_lengths is not None:
        raise ValueError("use decode_len or decode_lengths, not both")
    source_lengths = set(
        decode_lengths
        or ([decode_len] if decode_len is not None else DEFAULT_DECODE_LENGTHS)
    )
    actions = sorted(set(horizons or DEFAULT_HORIZONS))
    comparisons = [
        row
        for row in _read_csv(Path(comparisons_path))
        if int(row["decode_len"]) in source_lengths
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
    timing_inputs: list[tuple[str, dict[str, str], str]] = []
    for comparison_id, rows in sorted(by_comparison.items()):
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
            timing_inputs.append((comparison_id, row, plan_id))

    tasks = [(row, plan_id) for _, row, plan_id in timing_inputs]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            timing_results = list(executor.map(_read_candidate_timing, tasks))
    else:
        timing_results = [_read_candidate_timing(task) for task in tasks]

    timed_candidates: dict[
        str, list[tuple[dict[str, str], str, list[float]]]
    ] = defaultdict(list)
    for (comparison_id, row, plan_id), (prefix_times, reason) in zip(
        timing_inputs, timing_results
    ):
        if prefix_times is None:
            exclusions.append(
                {
                    "comparison_id": comparison_id,
                    "summary_path": row["summary_path"],
                    "reason": str(reason),
                }
            )
            continue
        timed_candidates[comparison_id].append((row, plan_id, prefix_times))

    accepted: dict[str, list[tuple[dict[str, str], str, list[float]]]] = {}
    for comparison_id in sorted(by_comparison):
        candidates = timed_candidates[comparison_id]
        plan_ids = [plan_id for _, plan_id, _ in candidates]
        if len(candidates) < 2 or len(set(plan_ids)) != len(plan_ids):
            for row, _, _ in candidates:
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
    states_written = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        for comparison_id, candidates in sorted(accepted.items()):
            source_decode_len = int(candidates[0][0]["decode_len"])
            batch = int(candidates[0][0]["batch"])
            prefill_len = int(candidates[0][0]["prefill_len"])
            for start_t in range(0, source_decode_len, window_stride):
                state_id = f"anchor:{comparison_id}:t{start_t:04d}"
                state = _regular_state(batch, prefill_len, start_t)
                states_written += 1
                for comparison, plan_id, prefix_times in candidates:
                    signature = estimator.signature(plan_id)
                    prefill_time = float(comparison["prefill_time_s"])
                    for horizon in actions:
                        end_t = min(start_t + horizon, source_decode_len)
                        generated_tokens = batch * (end_t - start_t)
                        duration = prefix_times[end_t] - prefix_times[start_t]
                        if start_t == 0:
                            duration += prefill_time
                        completed = end_t == source_decode_len
                        writer.writerow(
                            {
                                "state_id": state_id,
                                "trace_id": comparison_id,
                                "family": "dops_anchor_window",
                                "split": splits[comparison_id],
                                "model_id": comparison["model_id"],
                                "hardware_key": comparison["hardware_key"],
                                "decision_index": start_t // window_stride,
                                "time_s": 0.0,
                                "plan_id": plan_id,
                                "horizon": horizon,
                                **state,
                                **signature,
                                "plan_shape_distance": estimator.shape_distance(
                                    plan_id,
                                    batch,
                                    prefill_len + start_t + horizon // 2,
                                ),
                                "label_token_throughput": generated_tokens / duration,
                                "label_duration_s": duration,
                                "label_generated_tokens": generated_tokens,
                                "label_completed_requests": batch if completed else 0,
                                "label_p95_request_latency_s": (
                                    prefill_time + prefix_times[end_t]
                                    if completed
                                    else 0.0
                                ),
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
        "comparisons": len(accepted),
        "states": states_written,
        "rows": rows_written,
        "excluded": len(exclusions),
    }
