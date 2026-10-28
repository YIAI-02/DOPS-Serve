from __future__ import annotations

import argparse
import json
from pathlib import Path

def _placement_history(args: argparse.Namespace) -> None:
    from .placement_dataset import build

    config = json.loads(Path(args.config).read_text())
    result = build(
        args.output_root, config["models"], config["hardware_key"],
        backend_root=args.backend_root, artifact_dir=args.artifact_dir, config=config,
    )
    print(json.dumps(result, sort_keys=True))


def _placement_workloads(args: argparse.Namespace) -> None:
    from .workloads import import_op_policy_burstgpt

    result = import_op_policy_burstgpt(args.input, args.artifact_dir)
    print(json.dumps(result, sort_keys=True))


def _placement_validate(args: argparse.Namespace) -> None:
    from .serving_evaluation import validate_placement_backend

    result = validate_placement_backend(args.backend_root, args.artifact_dir, json.loads(Path(args.config).read_text()))
    print(json.dumps(result, sort_keys=True))


def _placement_reevaluate(args: argparse.Namespace) -> None:
    from .placement_dataset import reevaluate

    result = reevaluate(args.backend_root, args.artifact_dir, json.loads(Path(args.config).read_text()), workers=args.workers, shape_limit=args.shape_limit)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, sort_keys=True))


def _placement_bc(args: argparse.Namespace) -> None:
    from .placement_training import train_bc

    result = train_bc(args.artifact_dir, args.output_dir, seed=args.seed, epochs=args.epochs, temperature=args.temperature, learning_rate=args.learning_rate, exclude_model=args.exclude_model)
    print(json.dumps(result, sort_keys=True))


def _placement_calibrate(args: argparse.Namespace) -> None:
    from .placement_evaluation import calibrate_model

    result = calibrate_model(args.backend_root, args.artifact_dir, json.loads(Path(args.config).read_text()), args.model, train_count=args.train_traces, validation_count=args.validation_traces, candidate_budget=args.candidate_budget)
    print(json.dumps({key: value for key, value in result.items() if key not in {"static", "validation_results"}}, sort_keys=True))


def _placement_protocol(args: argparse.Namespace) -> None:
    from .placement_evaluation import assemble_protocol

    result = assemble_protocol(args.artifact_dir, json.loads(Path(args.config).read_text()))
    print(json.dumps({"status": result["status"], "models": list(result["models"])}))


def _placement_ppo(args: argparse.Namespace) -> None:
    from .placement_training import train_ppo

    result = train_ppo(args.backend_root, args.artifact_dir, json.loads(Path(args.config).read_text()), args.checkpoint, args.output_dir, json.loads(Path(args.protocol).read_text()), seed=args.seed, exclude_model=args.exclude_model, trace_epochs=args.trace_epochs)
    print(json.dumps(result, sort_keys=True))



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dops-serve")
    commands = parser.add_subparsers(dest="command", required=True)

    placement = commands.add_parser("placement-history", help="Build the uncapped Qwen binary placement manifest")
    placement.add_argument("--output-root", required=True)
    placement.add_argument("--backend-root", required=True)
    placement.add_argument("--artifact-dir", required=True)
    placement.add_argument("--config", required=True)
    placement.set_defaults(handler=_placement_history)

    op_workloads = commands.add_parser("placement-workloads", help="Build chronological BurstGPT splits for operator placement")
    op_workloads.add_argument("--input", required=True)
    op_workloads.add_argument("--artifact-dir", required=True)
    op_workloads.set_defaults(handler=_placement_workloads)

    op_validate = commands.add_parser("placement-validate", help="Validate Qwen fixed-placement backend and capacity")
    op_validate.add_argument("--backend-root", required=True)
    op_validate.add_argument("--artifact-dir", required=True)
    op_validate.add_argument("--config", required=True)
    op_validate.set_defaults(handler=_placement_validate)

    op_scores = commands.add_parser("placement-reevaluate", help="Reevaluate historical programs at continuous-state anchors")
    op_scores.add_argument("--backend-root", required=True)
    op_scores.add_argument("--artifact-dir", required=True)
    op_scores.add_argument("--config", required=True)
    op_scores.add_argument("--workers", type=int, default=8)
    op_scores.add_argument("--shape-limit", type=int, default=0, help="Pilot only; zero evaluates all historical shapes")
    op_scores.set_defaults(handler=_placement_reevaluate)

    op_bc = commands.add_parser("placement-bc", help="Cost-weighted BC on all reevaluated historical placements")
    op_bc.add_argument("--artifact-dir", required=True)
    op_bc.add_argument("--output-dir", required=True)
    op_bc.add_argument("--seed", type=int, default=7)
    op_bc.add_argument("--epochs", type=int, default=3)
    op_bc.add_argument("--temperature", type=float, default=1.0)
    op_bc.add_argument("--learning-rate", type=float, default=3e-4)
    op_bc.add_argument("--exclude-model")
    op_bc.set_defaults(handler=_placement_bc)

    op_calibrate = commands.add_parser("placement-calibrate", help="Select Static and calibrate a model's rate and SLO on train/validation")
    op_calibrate.add_argument("--backend-root", required=True)
    op_calibrate.add_argument("--artifact-dir", required=True)
    op_calibrate.add_argument("--config", required=True)
    op_calibrate.add_argument("--model", required=True)
    op_calibrate.add_argument("--train-traces", type=int, default=1)
    op_calibrate.add_argument("--validation-traces", type=int, default=1)
    op_calibrate.add_argument("--candidate-budget", type=int, default=2)
    op_calibrate.set_defaults(handler=_placement_calibrate)

    op_protocol = commands.add_parser("placement-protocol", help="Assemble the three calibrated model protocols")
    op_protocol.add_argument("--artifact-dir", required=True)
    op_protocol.add_argument("--config", required=True)
    op_protocol.set_defaults(handler=_placement_protocol)

    op_ppo = commands.add_parser("placement-ppo", help="Train a shared operator policy with block PPO")
    for flag in ("backend-root", "artifact-dir", "config", "checkpoint", "output-dir", "protocol"):
        op_ppo.add_argument("--" + flag, required=True)
    op_ppo.add_argument("--seed", type=int, default=7)
    op_ppo.add_argument("--exclude-model")
    op_ppo.add_argument("--trace-epochs", type=int, help="Additional complete passes through all training model traces")
    op_ppo.set_defaults(handler=_placement_ppo)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
