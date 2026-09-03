from __future__ import annotations

import argparse
import json

from .anchor_dataset import build_anchor_dataset
from .comparisons import build_anchor_comparisons
from .inventory import build_inventory
from .dataset import build_dataset
from .plan_bank import build_plan_bank
from .profiles import build_profiles
from .workloads import generate_synthetic_workloads, import_trace_csv


def _inventory(args: argparse.Namespace) -> None:
    valid, excluded = build_inventory(
        output_root=args.output_root,
        artifact_dir=args.artifact_dir,
        max_summaries=args.max_summaries,
        path_prefix=args.path_prefix,
    )
    print(json.dumps({"valid": valid, "excluded": excluded}, sort_keys=True))


def _plan_bank(args: argparse.Namespace) -> None:
    result = build_plan_bank(
        inventory_path=args.inventory,
        artifact_dir=args.artifact_dir,
        policies=args.policy,
        max_plans_per_group=args.max_plans_per_group,
        max_summaries_per_group=args.max_summaries_per_group,
        max_decode_steps=args.max_decode_steps,
    )
    print(json.dumps(result, sort_keys=True))


def _anchor_comparisons(args: argparse.Namespace) -> None:
    result = build_anchor_comparisons(
        output_root=args.output_root,
        inventory_path=args.inventory,
        output_path=args.output,
        path_prefix=args.path_prefix,
    )
    print(json.dumps(result, sort_keys=True))


def _anchor_dataset(args: argparse.Namespace) -> None:
    result = build_anchor_dataset(
        comparisons_path=args.comparisons,
        plan_sources_path=args.plan_sources,
        profiles_path=args.profiles,
        signatures_path=args.signatures,
        output_path=args.output,
        decode_lengths=args.decode_len,
        horizons=args.horizon,
        window_stride=args.window_stride,
    )
    print(json.dumps(result, sort_keys=True))


def _profiles(args: argparse.Namespace) -> None:
    result = build_profiles(
        inventory_path=args.inventory,
        plan_sources_path=args.plan_sources,
        artifact_dir=args.artifact_dir,
        include_trace_metrics=args.include_trace_metrics,
    )
    print(json.dumps(result, sort_keys=True))


def _generate_workloads(args: argparse.Namespace) -> None:
    result = generate_synthetic_workloads(
        output_path=args.output,
        model_ids=args.model,
        families=args.family,
        traces_per_family=args.traces_per_family,
        requests_per_trace=args.requests_per_trace,
        base_rate=args.base_rate,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))


def _import_workload(args: argparse.Namespace) -> None:
    result = import_trace_csv(
        input_path=args.input,
        output_path=args.output,
        model_id=args.model,
        trace_id=args.trace_id,
        arrival_column=args.arrival_column,
        prompt_column=args.prompt_column,
        output_column=args.output_column,
        max_tokens_column=args.max_tokens_column,
        max_tokens_default=args.max_tokens_default,
        family=args.family,
    )
    print(json.dumps(result, sort_keys=True))


def _dataset(args: argparse.Namespace) -> None:
    result = build_dataset(
        workload_path=args.workload,
        profiles_path=args.profiles,
        signatures_path=args.signatures,
        output_path=args.output,
        hardware_key=args.hardware_key,
        horizons=args.horizon,
        behavior_horizon=args.behavior_horizon,
        max_batch=args.max_batch,
        max_decisions_per_trace=args.max_decisions_per_trace,
        allow_extrapolation=args.allow_extrapolation,
        allow_incomplete_plans=args.allow_incomplete_plans,
    )
    print(json.dumps(result, sort_keys=True))


def _train(args: argparse.Namespace) -> None:
    from .training import train_model

    result = train_model(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        hidden_dims=args.hidden,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "checkpoint": result["checkpoint"],
                "best_validation_mse_standardized": result[
                    "best_validation_mse_standardized"
                ],
                "evaluation": result["evaluation"],
            },
            sort_keys=True,
        )
    )


def _evaluate(args: argparse.Namespace) -> None:
    from .training import evaluate_checkpoint

    result = evaluate_checkpoint(args.dataset, args.checkpoint, args.output_dir)
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dops-serve")
    commands = parser.add_subparsers(dest="command", required=True)

    inventory = commands.add_parser("inventory", help="Index DOPS outputs read-only")
    inventory.add_argument("--output-root", required=True)
    inventory.add_argument("--artifact-dir", required=True)
    inventory.add_argument("--max-summaries", type=int, default=0)
    inventory.add_argument("--path-prefix", action="append")
    inventory.set_defaults(handler=_inventory)

    comparisons = commands.add_parser(
        "anchor-comparisons", help="Build paired labels from DOPS baseline comparisons"
    )
    comparisons.add_argument("--output-root", required=True)
    comparisons.add_argument("--inventory", required=True)
    comparisons.add_argument("--output", required=True)
    comparisons.add_argument("--path-prefix", action="append")
    comparisons.set_defaults(handler=_anchor_comparisons)

    anchor_dataset = commands.add_parser(
        "anchor-dataset", help="Map paired DOPS anchors to complete retained plans"
    )
    anchor_dataset.add_argument("--comparisons", required=True)
    anchor_dataset.add_argument("--plan-sources", required=True)
    anchor_dataset.add_argument("--profiles", required=True)
    anchor_dataset.add_argument("--signatures", required=True)
    anchor_dataset.add_argument("--output", required=True)
    anchor_dataset.add_argument("--decode-len", action="append", type=int)
    anchor_dataset.add_argument("--horizon", action="append", type=int)
    anchor_dataset.add_argument("--window-stride", type=int, default=32)
    anchor_dataset.set_defaults(handler=_anchor_dataset)

    plan_bank = commands.add_parser("plan-bank", help="Extract and deduplicate plans")
    plan_bank.add_argument("--inventory", required=True)
    plan_bank.add_argument("--artifact-dir", required=True)
    plan_bank.add_argument("--policy", action="append", default=[])
    plan_bank.add_argument("--max-plans-per-group", type=int, default=32)
    plan_bank.add_argument("--max-summaries-per-group", type=int, default=0)
    plan_bank.add_argument("--max-decode-steps", type=int, default=1024)
    plan_bank.set_defaults(handler=_plan_bank)

    profiles = commands.add_parser("profiles", help="Build and validate profiles")
    profiles.add_argument("--inventory", required=True)
    profiles.add_argument("--plan-sources", required=True)
    profiles.add_argument("--artifact-dir", required=True)
    profiles.add_argument("--include-trace-metrics", action="store_true")
    profiles.set_defaults(handler=_profiles)

    workloads = commands.add_parser("generate-workloads", help="Generate synthetic traces")
    workloads.add_argument("--output", required=True)
    workloads.add_argument("--model", action="append", required=True)
    workloads.add_argument("--family", action="append")
    workloads.add_argument("--traces-per-family", type=int, default=4)
    workloads.add_argument("--requests-per-trace", type=int, default=200)
    workloads.add_argument("--base-rate", type=float, default=2.0)
    workloads.add_argument("--seed", type=int, default=7)
    workloads.set_defaults(handler=_generate_workloads)

    importer = commands.add_parser("import-workload", help="Import one real trace CSV")
    importer.add_argument("--input", required=True)
    importer.add_argument("--output", required=True)
    importer.add_argument("--model", required=True)
    importer.add_argument("--trace-id", required=True)
    importer.add_argument("--family", default="real")
    importer.add_argument("--arrival-column", required=True)
    importer.add_argument("--prompt-column", required=True)
    importer.add_argument("--output-column", required=True)
    importer.add_argument("--max-tokens-column")
    importer.add_argument("--max-tokens-default", type=int)
    importer.set_defaults(handler=_import_workload)

    dataset = commands.add_parser("dataset", help="Build counterfactual serving rows")
    dataset.add_argument("--workload", required=True)
    dataset.add_argument("--profiles", required=True)
    dataset.add_argument("--signatures", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--hardware-key", required=True)
    dataset.add_argument("--horizon", action="append", type=int)
    dataset.add_argument("--behavior-horizon", type=int, default=64)
    dataset.add_argument("--max-batch", type=int, default=16)
    dataset.add_argument("--max-decisions-per-trace", type=int, default=1000)
    dataset.add_argument("--allow-extrapolation", action="store_true")
    dataset.add_argument("--allow-incomplete-plans", action="store_true")
    dataset.set_defaults(handler=_dataset)

    train = commands.add_parser("train", help="Train the MLP plan scorer")
    train.add_argument("--dataset", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--ranking-weight", type=float, default=0.2)
    train.add_argument("--ranking-margin", type=float, default=0.05)
    train.add_argument("--hidden", action="append", type=int)
    train.add_argument("--seed", type=int, default=7)
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved scorer")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan-bank" and not args.policy:
        args.policy = ["hefthint", "bifocal"]
    args.handler(args)


if __name__ == "__main__":
    main()
