from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .dataset import PLAN_FEATURES, STATE_FEATURES, _trace_splits
from .replay import (
    ProfileEstimator,
    ReplayState,
    UnsupportedProfile,
    _quantile,
    request_states,
)
from .training import _encode_features, _network


POLICIES = ("robust_static", "nearest_shape", "mlp_selector", "myopic_oracle")
ScoreRows = Callable[[list[dict[str, Any]]], list[float]]


class ServingReplayUnsupported(ValueError):
    pass


def load_checkpoint_scorer(checkpoint_path: str | Path) -> ScoreRows:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for closed-loop MLP evaluation"
        ) from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preprocessor = checkpoint["preprocessor"]
    model = _network(torch, checkpoint["input_dim"], checkpoint["hidden_dims"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    def score_rows(rows: list[dict[str, Any]]) -> list[float]:
        for row in rows:
            for name in preprocessor["categorical_features"]:
                if row[name] not in preprocessor["vocabularies"][name]:
                    raise ServingReplayUnsupported(
                        f"checkpoint has no {name} category {row[name]}"
                    )
        features = torch.tensor(
            _encode_features(rows, preprocessor), dtype=torch.float32
        )
        with torch.no_grad():
            standardized = model(features).squeeze(1).tolist()
        return [
            float(preprocessor["target_mean"])
            + float(value) * float(preprocessor["target_scale"])
            for value in standardized
        ]

    return score_rows


def _target_shape(state: ReplayState) -> tuple[int, int]:
    batch = max(1, min(state.max_batch, len(state.active) or len(state.waiting) or 1))
    context = int(
        max(
            [request.context_len for request in state.active]
            or [request.prompt_len for request in state.waiting]
            or [1]
        )
    )
    return batch, context


def _project_action(
    state: ReplayState,
    estimator: ProfileEstimator,
    plan_id: str,
    horizon: int,
) -> tuple[float, float]:
    probe = state.clone()
    probe.requests = probe.requests[: probe.next_arrival]
    probe.next_arrival = len(probe.requests)
    current = [request for request in probe.requests if request.completion_time is None]
    for request in current:
        request.output_len = request.max_tokens
    metrics = probe.apply_action(estimator, plan_id, horizon)
    projected_ages = [
        (request.completion_time if request.completion_time is not None else probe.now)
        - request.arrival_time
        for request in current
    ]
    return metrics["duration_s"], max(projected_ages, default=0.0)


def _online_candidates(
    state: ReplayState,
    estimator: ProfileEstimator,
    model_id: str,
    hardware_key: str,
    actions: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    state.prepare_epoch()
    features = state.online_features()
    target_batch, target_context = _target_shape(state)
    candidates: list[dict[str, Any]] = []
    for plan_id, horizon in actions:
        try:
            projected_duration, projected_max_age = _project_action(
                state, estimator, plan_id, horizon
            )
        except UnsupportedProfile:
            continue
        signature = estimator.signature(plan_id)
        feature_row = {
            "model_id": model_id,
            "hardware_key": hardware_key,
            "plan_id": plan_id,
            "horizon": horizon,
            **{name: features[name] for name in STATE_FEATURES},
            **{name: signature[name] for name in PLAN_FEATURES[:-1]},
            "plan_shape_distance": estimator.shape_distance(
                plan_id, target_batch, target_context
            ),
        }
        candidates.append(
            {
                "plan_id": plan_id,
                "horizon": horizon,
                "feature_row": feature_row,
                "projected_duration_s": projected_duration,
                "projected_max_age_s": projected_max_age,
            }
        )
    if not candidates:
        raise ServingReplayUnsupported("no online-valid plan/horizon action")
    return candidates


def _best_by_score(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["selection_score"]),
            candidate["plan_id"],
            int(candidate["horizon"]),
        ),
    )[0]


def _select_mlp(
    candidates: list[dict[str, Any]],
    score_rows: ScoreRows,
    previous_plan: str | None,
    latency_slo_s: float,
    plan_switch_margin: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    scores = score_rows([candidate["feature_row"] for candidate in candidates])
    for candidate, score in zip(candidates, scores):
        candidate["selection_score"] = score

    eligible = [
        candidate
        for candidate in candidates
        if latency_slo_s <= 0
        or float(candidate["projected_max_age_s"]) <= latency_slo_s
    ]
    filtered = len(candidates) - len(eligible)
    forced = 0
    if not eligible:
        forced = 1
        eligible = [
            min(
                candidates,
                key=lambda candidate: (
                    float(candidate["projected_max_age_s"]),
                    -float(candidate["selection_score"]),
                    candidate["plan_id"],
                    int(candidate["horizon"]),
                ),
            )
        ]

    chosen = _best_by_score(eligible)
    blocked_switch = 0
    if previous_plan is not None and chosen["plan_id"] != previous_plan:
        previous_candidates = [
            candidate for candidate in eligible if candidate["plan_id"] == previous_plan
        ]
        if previous_candidates:
            previous = _best_by_score(previous_candidates)
            required_gain = plan_switch_margin * max(
                abs(float(previous["selection_score"])), 1e-12
            )
            if (
                float(chosen["selection_score"])
                - float(previous["selection_score"])
                < required_gain
            ):
                chosen = previous
                blocked_switch = 1
    return chosen, {
        "guardrail_filtered_actions": filtered,
        "guardrail_forced_decisions": forced,
        "hysteresis_blocked_switches": blocked_switch,
    }


def _robust_static_plan(
    estimator: ProfileEstimator, model_id: str, hardware_key: str
) -> str:
    plan_ids = estimator.plan_ids(model_id, hardware_key)
    if not plan_ids:
        raise ServingReplayUnsupported(
            f"no complete plans for model={model_id}, hardware={hardware_key}"
        )
    return sorted(
        plan_ids,
        key=lambda plan_id: (
            -estimator.signature(plan_id)["plan_profile_count"],
            -estimator.signature(plan_id)["plan_mean_output_token_throughput"],
            plan_id,
        ),
    )[0]


def _run_policy(
    records: list[dict[str, Any]],
    estimator: ProfileEstimator,
    hardware_key: str,
    policy: str,
    horizons: list[int],
    fixed_horizon: int,
    max_batch: int,
    max_decisions: int,
    score_rows: ScoreRows,
    latency_slo_s: float,
    plan_switch_margin: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model_ids = {str(row["model_id"]) for row in records}
    if len(model_ids) != 1:
        raise ServingReplayUnsupported("trace contains multiple model IDs")
    model_id = next(iter(model_ids))
    plan_ids = estimator.plan_ids(model_id, hardware_key)
    if not plan_ids:
        raise ServingReplayUnsupported(
            f"no complete plans for model={model_id}, hardware={hardware_key}"
        )
    static_plan = _robust_static_plan(estimator, model_id, hardware_key)
    state = ReplayState(request_states(records), max_batch=max_batch)
    previous_plan: str | None = None
    previous_horizon: int | None = None
    plan_switches = 0
    horizon_switches = 0
    guardrail_filtered_actions = 0
    guardrail_forced_decisions = 0
    hysteresis_blocked_switches = 0
    decision_rows: list[dict[str, Any]] = []

    decision_index = 0
    while not state.done and decision_index < max_decisions:
        state.prepare_epoch()
        decision_start = state.now
        if policy == "robust_static":
            actions = [(static_plan, fixed_horizon)]
        elif policy == "nearest_shape":
            actions = [(plan_id, fixed_horizon) for plan_id in plan_ids]
        else:
            actions = [
                (plan_id, horizon) for plan_id in plan_ids for horizon in horizons
            ]
        candidates = _online_candidates(
            state, estimator, model_id, hardware_key, actions
        )

        diagnostics = {
            "guardrail_filtered_actions": 0,
            "guardrail_forced_decisions": 0,
            "hysteresis_blocked_switches": 0,
        }
        if policy == "robust_static":
            chosen = candidates[0]
            chosen["selection_score"] = estimator.signature(static_plan)[
                "plan_mean_output_token_throughput"
            ]
        elif policy == "nearest_shape":
            chosen = min(
                candidates,
                key=lambda candidate: (
                    float(candidate["feature_row"]["plan_shape_distance"]),
                    candidate["plan_id"],
                ),
            )
            chosen["selection_score"] = -float(
                chosen["feature_row"]["plan_shape_distance"]
            )
        elif policy == "mlp_selector":
            chosen, diagnostics = _select_mlp(
                candidates,
                score_rows,
                previous_plan,
                latency_slo_s,
                plan_switch_margin,
            )
        elif policy == "myopic_oracle":
            for candidate in candidates:
                probe = state.clone()
                immediate = probe.apply_action(
                    estimator, candidate["plan_id"], int(candidate["horizon"])
                )
                candidate["selection_score"] = immediate["token_throughput"]
            chosen = _best_by_score(candidates)
        else:
            raise ValueError(f"unknown policy: {policy}")

        selected_plan = str(chosen["plan_id"])
        selected_horizon = int(chosen["horizon"])
        if previous_plan is not None and selected_plan != previous_plan:
            plan_switches += 1
        if previous_horizon is not None and selected_horizon != previous_horizon:
            horizon_switches += 1
        metrics = state.apply_action(
            estimator, selected_plan, selected_horizon
        )
        guardrail_filtered_actions += diagnostics["guardrail_filtered_actions"]
        guardrail_forced_decisions += diagnostics["guardrail_forced_decisions"]
        hysteresis_blocked_switches += diagnostics["hysteresis_blocked_switches"]
        decision_rows.append(
            {
                "trace_id": records[0]["trace_id"],
                "family": records[0]["family"],
                "policy": policy,
                "decision_index": decision_index,
                "time_s": decision_start,
                "plan_id": selected_plan,
                "horizon": selected_horizon,
                "selection_score": chosen["selection_score"],
                "projected_max_age_s": chosen["projected_max_age_s"],
                "duration_s": metrics["duration_s"],
                "generated_tokens": metrics["generated_tokens"],
                "completed_requests": metrics["completed_requests"],
            }
        )
        previous_plan = selected_plan
        previous_horizon = selected_horizon
        decision_index += 1

    if not state.done:
        raise ServingReplayUnsupported(
            f"trace exceeded max_decisions={max_decisions}"
        )
    latencies = [
        float(request.completion_time) - request.arrival_time
        for request in state.requests
        if request.completion_time is not None
    ]
    queue_waits = [
        float(request.prefill_start) - request.arrival_time
        for request in state.requests
        if request.prefill_start is not None
    ]
    if len(latencies) != len(records) or len(queue_waits) != len(records):
        raise ServingReplayUnsupported("trace ended without completing every request")
    makespan = max(float(request.completion_time) for request in state.requests) - min(
        request.arrival_time for request in state.requests
    )
    busy_time = sum(float(row["duration_s"]) for row in decision_rows)
    output_tokens = sum(request.output_len for request in state.requests)
    result = {
        "trace_id": records[0]["trace_id"],
        "family": records[0]["family"],
        "policy": policy,
        "requests": len(records),
        "output_tokens": output_tokens,
        "makespan_s": makespan,
        "busy_time_s": busy_time,
        "output_token_throughput": output_tokens / makespan,
        "busy_output_token_throughput": output_tokens / busy_time,
        "mean_request_latency_s": statistics.fmean(latencies),
        "p95_request_latency_s": _quantile(latencies, 0.95),
        "p95_queue_wait_s": _quantile(queue_waits, 0.95),
        "request_slo_violations": sum(
            latency_slo_s > 0 and latency > latency_slo_s for latency in latencies
        ),
        "decisions": len(decision_rows),
        "plan_switches": plan_switches,
        "horizon_switches": horizon_switches,
        "guardrail_filtered_actions": guardrail_filtered_actions,
        "guardrail_forced_decisions": guardrail_forced_decisions,
        "hysteresis_blocked_switches": hysteresis_blocked_switches,
        "_latencies": latencies,
        "_queue_waits": queue_waits,
    }
    return result, decision_rows


def _aggregate(policy: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [latency for row in rows for latency in row["_latencies"]]
    queue_waits = [wait for row in rows for wait in row["_queue_waits"]]
    output_tokens = sum(int(row["output_tokens"]) for row in rows)
    makespan = sum(float(row["makespan_s"]) for row in rows)
    busy_time = sum(float(row["busy_time_s"]) for row in rows)
    return {
        "policy": policy,
        "paired_traces": len(rows),
        "requests": sum(int(row["requests"]) for row in rows),
        "output_tokens": output_tokens,
        "output_token_throughput": output_tokens / makespan,
        "busy_output_token_throughput": output_tokens / busy_time,
        "mean_request_latency_s": statistics.fmean(latencies),
        "p95_request_latency_s": _quantile(latencies, 0.95),
        "p95_queue_wait_s": _quantile(queue_waits, 0.95),
        "request_slo_violations": sum(
            int(row["request_slo_violations"]) for row in rows
        ),
        "decisions": sum(int(row["decisions"]) for row in rows),
        "plan_switches": sum(int(row["plan_switches"]) for row in rows),
        "horizon_switches": sum(int(row["horizon_switches"]) for row in rows),
        "guardrail_filtered_actions": sum(
            int(row["guardrail_filtered_actions"]) for row in rows
        ),
        "guardrail_forced_decisions": sum(
            int(row["guardrail_forced_decisions"]) for row in rows
        ),
        "hysteresis_blocked_switches": sum(
            int(row["hysteresis_blocked_switches"]) for row in rows
        ),
    }


def evaluate_serving(
    workload_path: str | Path,
    profiles_path: str | Path,
    signatures_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    hardware_key: str,
    split: str = "test",
    horizons: list[int] | None = None,
    fixed_horizon: int = 64,
    max_batch: int = 16,
    max_decisions: int = 10000,
    latency_slo_s: float = 0.0,
    plan_switch_margin: float = 0.0,
    allow_extrapolation: bool = False,
    score_rows: ScoreRows | None = None,
) -> dict[str, Any]:
    with Path(workload_path).open(encoding="utf-8") as handle:
        workload_rows = [json.loads(line) for line in handle if line.strip()]
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in workload_rows:
        traces[str(row["trace_id"])].append(row)
    trace_splits = _trace_splits(traces)
    selected = {
        trace_id: rows
        for trace_id, rows in traces.items()
        if split == "all" or trace_splits[trace_id] == split
    }
    if not selected:
        raise ValueError(f"workload has no traces in split={split}")

    estimator = ProfileEstimator(
        profiles_path,
        signatures_path,
        allow_extrapolation=allow_extrapolation,
    )
    scorer = score_rows or load_checkpoint_scorer(checkpoint_path)
    actions = sorted(set(horizons or [32, 64, 128]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_results: list[dict[str, Any]] = []
    decision_results: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []

    for trace_id, records in sorted(selected.items()):
        paired_rows: list[dict[str, Any]] = []
        paired_decisions: list[dict[str, Any]] = []
        failed = False
        for policy in POLICIES:
            try:
                result, decisions = _run_policy(
                    records,
                    estimator,
                    hardware_key,
                    policy,
                    actions,
                    fixed_horizon,
                    max_batch,
                    max_decisions,
                    scorer,
                    latency_slo_s,
                    plan_switch_margin,
                )
            except ServingReplayUnsupported as error:
                exclusions.append(
                    {
                        "trace_id": trace_id,
                        "family": str(records[0]["family"]),
                        "policy": policy,
                        "reason": str(error),
                    }
                )
                failed = True
                break
            paired_rows.append(result)
            paired_decisions.extend(decisions)
        if not failed:
            trace_results.extend(paired_rows)
            decision_results.extend(paired_decisions)

    paired_trace_ids = {row["trace_id"] for row in trace_results}
    if not paired_trace_ids:
        raise ValueError("no trace completed under all four policies")
    summary_rows = [
        _aggregate(
            policy, [row for row in trace_results if row["policy"] == policy]
        )
        for policy in POLICIES
    ]
    family_summary_rows = []
    for family in sorted({str(row["family"]) for row in trace_results}):
        for policy in POLICIES:
            family_row = _aggregate(
                policy,
                [
                    row
                    for row in trace_results
                    if row["policy"] == policy and row["family"] == family
                ],
            )
            family_summary_rows.append({"family": family, **family_row})
    oracle_throughput = next(
        row["output_token_throughput"]
        for row in summary_rows
        if row["policy"] == "myopic_oracle"
    )
    for row in summary_rows:
        row["throughput_vs_myopic_oracle"] = (
            row["output_token_throughput"] / oracle_throughput
        )

    summary_fields = list(summary_rows[0])
    with (destination / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (destination / "summary_by_family.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(family_summary_rows[0])
        )
        writer.writeheader()
        writer.writerows(family_summary_rows)
    trace_fields = [
        key for key in trace_results[0] if not key.startswith("_")
    ]
    with (destination / "traces.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in trace_fields} for row in trace_results
        )
    with (destination / "decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decision_results[0]))
        writer.writeheader()
        writer.writerows(decision_results)
    with (destination / "excluded.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["trace_id", "family", "policy", "reason"]
        )
        writer.writeheader()
        writer.writerows(exclusions)

    result = {
        "evaluation_scope": {
            "requested_split": split,
            "input_traces": len(selected),
            "paired_traces": len(paired_trace_ids),
            "excluded_traces": len(selected) - len(paired_trace_ids),
        },
        "config": {
            "hardware_key": hardware_key,
            "horizons": actions,
            "fixed_horizon": fixed_horizon,
            "max_batch": max_batch,
            "latency_slo_s": latency_slo_s,
            "plan_switch_margin": plan_switch_margin,
            "allow_extrapolation": allow_extrapolation,
            "oracle_scope": "one-step counterfactual throughput with simulator truth",
        },
        "policies": summary_rows,
        "policies_by_family": family_summary_rows,
    }
    with (destination / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = [
        "# DOPS-Serve Closed-Loop Evaluation",
        "",
        "All policies replay the same paired request traces but advance independent",
        "runtime states. Results use DOPS profile timing, not tensor execution.",
        "The myopic oracle uses simulator truth for one-step action selection only.",
        "",
        "| Policy | Traces | Requests | Throughput | vs. oracle | p95 E2E | p95 wait | SLO violations | Plan switches | Horizon switches | Forced |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            "| {policy} | {paired_traces} | {requests} | "
            "{output_token_throughput:.6f} | {throughput_vs_myopic_oracle:.4f} | "
            "{p95_request_latency_s:.6f} | {p95_queue_wait_s:.6f} | "
            "{request_slo_violations} | {plan_switches} | {horizon_switches} | "
            "{guardrail_forced_decisions} |".format(**row)
        )
    report.extend(
        (
            "",
            f"Paired traces: {len(paired_trace_ids)}; excluded traces: "
            f"{len(selected) - len(paired_trace_ids)}.",
            f"MLP latency SLO: {latency_slo_s:.6f} s; plan switch margin: "
            f"{plan_switch_margin:.4f}.",
        )
    )
    (destination / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    return result
