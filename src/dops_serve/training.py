from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .dataset import PLAN_FEATURES, STATE_FEATURES


NUMERIC_FEATURES = [*STATE_FEATURES, *PLAN_FEATURES, "horizon"]
CATEGORICAL_FEATURES = ["model_id", "hardware_key"]
TARGET = "label_token_throughput"


def _read_dataset(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fit_preprocessor(rows: list[dict[str, str]]) -> dict[str, Any]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in NUMERIC_FEATURES:
        values = [float(row[name]) for row in rows]
        means[name] = statistics.fmean(values)
        scale = statistics.pstdev(values)
        scales[name] = scale if scale > 0 else 1.0
    target_values = [float(row[TARGET]) for row in rows]
    target_mean = statistics.fmean(target_values)
    target_scale = statistics.pstdev(target_values)
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "means": means,
        "scales": scales,
        "vocabularies": {
            name: sorted({row[name] for row in rows}) for name in CATEGORICAL_FEATURES
        },
        "target_mean": target_mean,
        "target_scale": target_scale if target_scale > 0 else 1.0,
    }


def _encode_rows(
    rows: list[dict[str, str]], preprocessor: dict[str, Any]
) -> tuple[list[list[float]], list[float]]:
    encoded: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        vector = [
            (float(row[name]) - float(preprocessor["means"][name]))
            / float(preprocessor["scales"][name])
            for name in preprocessor["numeric_features"]
        ]
        for name in preprocessor["categorical_features"]:
            vector.extend(
                1.0 if row[name] == category else 0.0
                for category in preprocessor["vocabularies"][name]
            )
        encoded.append(vector)
        targets.append(
            (float(row[TARGET]) - float(preprocessor["target_mean"]))
            / float(preprocessor["target_scale"])
        )
    return encoded, targets


def _network(torch: Any, input_dim: int, hidden_dims: list[int]) -> Any:
    layers = []
    previous = input_dim
    for width in hidden_dims:
        layers.extend((torch.nn.Linear(previous, width), torch.nn.ReLU()))
        previous = width
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _ranking_pairs(rows: list[dict[str, str]]) -> list[tuple[int, int]]:
    states: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        states[row["state_id"]].append(index)
    pairs: list[tuple[int, int]] = []
    for indices in states.values():
        best = max(indices, key=lambda index: float(rows[index][TARGET]))
        worst = min(indices, key=lambda index: float(rows[index][TARGET]))
        if float(rows[best][TARGET]) > float(rows[worst][TARGET]):
            pairs.append((best, worst))
    return pairs


def _mean_squared_error(torch: Any, model: Any, features: Any, targets: Any) -> float:
    model.eval()
    with torch.no_grad():
        predictions = model(features).squeeze(1)
        return float(torch.mean((predictions - targets) ** 2).item())


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    epochs: int = 80,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    ranking_weight: float = 0.2,
    ranking_margin: float = 0.05,
    hidden_dims: list[int] | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the train command") from error

    torch.manual_seed(seed)
    random.seed(seed)
    rows = _read_dataset(Path(dataset_path))
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "val"] or train_rows
    test_rows = [row for row in rows if row["split"] == "test"] or validation_rows
    if not train_rows:
        raise ValueError("Dataset has no training rows")

    preprocessor = _fit_preprocessor(train_rows)
    train_x, train_y = _encode_rows(train_rows, preprocessor)
    validation_x, validation_y = _encode_rows(validation_rows, preprocessor)
    widths = hidden_dims or [128, 64]
    model = _network(torch, len(train_x[0]), widths)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    regression_loss = torch.nn.SmoothL1Loss()

    train_features = torch.tensor(train_x, dtype=torch.float32)
    train_targets = torch.tensor(train_y, dtype=torch.float32)
    validation_features = torch.tensor(validation_x, dtype=torch.float32)
    validation_targets = torch.tensor(validation_y, dtype=torch.float32)
    pairs = _ranking_pairs(train_rows)
    best_validation = math.inf
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_rows))
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(train_rows), batch_size):
            indices = permutation[start : start + batch_size]
            predictions = model(train_features[indices]).squeeze(1)
            loss = regression_loss(predictions, train_targets[indices])
            if pairs:
                sampled = random.sample(pairs, min(len(pairs), max(1, len(indices) // 4)))
                better_indices = torch.tensor(
                    [pair[0] for pair in sampled], dtype=torch.long
                )
                worse_indices = torch.tensor(
                    [pair[1] for pair in sampled], dtype=torch.long
                )
                better = model(train_features[better_indices]).squeeze(1)
                worse = model(train_features[worse_indices]).squeeze(1)
                rank_loss = torch.relu(ranking_margin - (better - worse)).mean()
                loss = loss + ranking_weight * rank_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1

        validation_mse = _mean_squared_error(
            torch, model, validation_features, validation_targets
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss / max(batches, 1),
                "validation_mse": validation_mse,
            }
        )
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    model.load_state_dict(best_state)

    destination = Path(output_dir)
    checkpoint_dir = destination / "checkpoints"
    result_dir = destination / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": len(train_x[0]),
            "hidden_dims": widths,
            "preprocessor": preprocessor,
        },
        checkpoint_path,
    )

    evaluation = evaluate_rows(
        torch,
        model,
        train_rows=train_rows,
        test_rows=test_rows,
        preprocessor=preprocessor,
        result_dir=result_dir,
    )
    metrics = {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "ranking_pairs": len(pairs),
        "epochs": epochs,
        "best_validation_mse_standardized": best_validation,
        "checkpoint": str(checkpoint_path),
        "evaluation": evaluation,
        "history": history,
    }
    with (destination / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def _prediction_values(
    torch: Any,
    model: Any,
    rows: list[dict[str, str]],
    preprocessor: dict[str, Any],
) -> list[float]:
    features, _ = _encode_rows(rows, preprocessor)
    tensor = torch.tensor(features, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        standardized = model(tensor).squeeze(1).tolist()
    return [
        float(preprocessor["target_mean"])
        + float(value) * float(preprocessor["target_scale"])
        for value in standardized
    ]


def _action_switches(rows: list[dict[str, str]]) -> int:
    switches = 0
    previous: tuple[str, str, str] | None = None
    for row in sorted(
        rows, key=lambda item: (item["trace_id"], int(item["decision_index"]))
    ):
        current = (row["trace_id"], row["plan_id"], row["horizon"])
        if (
            previous is not None
            and current[0] == previous[0]
            and current[1:] != previous[1:]
        ):
            switches += 1
        previous = current
    return switches


def _aggregate_policy(name: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    throughputs = [float(row[TARGET]) for row in rows]
    latencies = [float(row["label_p95_request_latency_s"]) for row in rows]
    waits = [float(row["label_p95_queue_wait_s"]) for row in rows]
    return {
        "policy": name,
        "states": len(rows),
        "mean_token_throughput": statistics.fmean(throughputs) if throughputs else 0.0,
        "mean_window_p95_request_latency_s": statistics.fmean(latencies)
        if latencies
        else 0.0,
        "mean_window_p95_queue_wait_s": statistics.fmean(waits) if waits else 0.0,
        "action_switches": _action_switches(rows),
    }


def evaluate_rows(
    torch: Any,
    model: Any,
    train_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    preprocessor: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    train_plan_scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in train_rows:
        if int(row["horizon"]) == 64:
            key = (row["model_id"], row["hardware_key"], row["plan_id"])
            train_plan_scores[key].append(float(row[TARGET]))
    if not train_plan_scores:
        for row in train_rows:
            key = (row["model_id"], row["hardware_key"], row["plan_id"])
            train_plan_scores[key].append(float(row[TARGET]))
    static_plans: dict[tuple[str, str], str] = {}
    groups = {(key[0], key[1]) for key in train_plan_scores}
    for group in groups:
        candidates = [key for key in train_plan_scores if key[:2] == group]
        winner = max(
            candidates,
            key=lambda key: statistics.fmean(train_plan_scores[key]),
        )
        static_plans[group] = winner[2]

    grouped_test: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in test_rows:
        grouped_test[row["state_id"]].append(row)
    static_rows = []
    nearest_rows = []
    paired_groups: list[list[dict[str, str]]] = []
    excluded_unpaired_states = 0
    for _, candidates in sorted(grouped_test.items()):
        fixed_horizon = [row for row in candidates if int(row["horizon"]) == 64]
        horizon_candidates = fixed_horizon or candidates
        group = (candidates[0]["model_id"], candidates[0]["hardware_key"])
        static_plan = static_plans.get(group)
        if static_plan is None:
            static_plan = max(
                horizon_candidates,
                key=lambda row: float(row["plan_mean_output_token_throughput"]),
            )["plan_id"]
        static_candidates = [
            row for row in horizon_candidates if row["plan_id"] == static_plan
        ]
        if not static_candidates:
            excluded_unpaired_states += 1
            continue
        paired_groups.append(candidates)
        static_rows.append(static_candidates[0])
        nearest_rows.append(
            min(horizon_candidates, key=lambda row: float(row["plan_shape_distance"]))
        )

    paired_test_rows = [row for candidates in paired_groups for row in candidates]
    if not paired_test_rows:
        raise ValueError("No test states support the robust static baseline")
    inference_start = time.perf_counter()
    predictions = _prediction_values(torch, model, paired_test_rows, preprocessor)
    inference_seconds = time.perf_counter() - inference_start
    prediction_by_identity = {
        (row["state_id"], row["plan_id"], row["horizon"]): predictions[index]
        for index, row in enumerate(paired_test_rows)
    }
    mlp_rows = [
        max(
            candidates,
            key=lambda row: prediction_by_identity[
                (row["state_id"], row["plan_id"], row["horizon"])
            ],
        )
        for candidates in paired_groups
    ]
    oracle_rows = [
        max(candidates, key=lambda row: float(row[TARGET]))
        for candidates in paired_groups
    ]

    policy_rows = [
        _aggregate_policy("robust_static", static_rows),
        _aggregate_policy("nearest_shape", nearest_rows),
        _aggregate_policy("mlp_selector", mlp_rows),
        _aggregate_policy("offline_oracle", oracle_rows),
    ]
    oracle_throughput = policy_rows[-1]["mean_token_throughput"]
    for row in policy_rows:
        row["throughput_vs_oracle"] = (
            row["mean_token_throughput"] / oracle_throughput
            if oracle_throughput > 0
            else 0.0
        )

    summary_fields = list(policy_rows[0])
    with (result_dir / "summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(policy_rows)

    with (result_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ["state_id", "trace_id", "plan_id", "horizon", "actual", "predicted"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(paired_test_rows):
            writer.writerow(
                {
                    "state_id": row["state_id"],
                    "trace_id": row["trace_id"],
                    "plan_id": row["plan_id"],
                    "horizon": row["horizon"],
                    "actual": row[TARGET],
                    "predicted": predictions[index],
                }
            )
    result = {
        "evaluation_scope": {
            "paired_states": len(paired_groups),
            "excluded_unpaired_states": excluded_unpaired_states,
            "candidate_rows": len(paired_test_rows),
        },
        "mlp_batched_inference_us_per_state": inference_seconds
        * 1_000_000.0
        / len(paired_groups),
        "static_plans": {
            f"{model_id}|{hardware_key}": plan_id
            for (model_id, hardware_key), plan_id in sorted(static_plans.items())
        },
        "policies": policy_rows,
    }
    report_lines = [
        "# DOPS-Serve Evaluation",
        "",
        "All policies are evaluated on the same request-state rows. Values are",
        "profile-driven replay results, not physical hardware measurements.",
        "",
        "| Policy | States | Mean token throughput | vs. oracle | Mean window p95 latency | Mean window p95 wait | Switches |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in policy_rows:
        report_lines.append(
            "| {policy} | {states} | {mean_token_throughput:.6f} | "
            "{throughput_vs_oracle:.4f} | {mean_window_p95_request_latency_s:.6f} | "
            "{mean_window_p95_queue_wait_s:.6f} | {action_switches} |".format(**row)
        )
    report_lines.extend(
        (
            "",
            f"Paired states: {len(paired_groups)}; excluded unpaired states: {excluded_unpaired_states}.",
            "Batched CPU scorer timing: "
            f"{result['mlp_batched_inference_us_per_state']:.3f} us/state; "
            "this is not production online latency.",
        )
    )
    report_lines.extend(("", "Static plans:", ""))
    for group, plan_id in result["static_plans"].items():
        report_lines.append(f"- `{group}`: `{plan_id}`")
    (result_dir / "report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return result


def evaluate_checkpoint(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for the evaluate command") from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _network(torch, checkpoint["input_dim"], checkpoint["hidden_dims"])
    model.load_state_dict(checkpoint["state_dict"])
    rows = _read_dataset(Path(dataset_path))
    train_rows = [row for row in rows if row["split"] == "train"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        test_rows = [row for row in rows if row["split"] == "val"] or train_rows
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result = evaluate_rows(
        torch,
        model,
        train_rows=train_rows,
        test_rows=test_rows,
        preprocessor=checkpoint["preprocessor"],
        result_dir=destination,
    )
    with (destination / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result
