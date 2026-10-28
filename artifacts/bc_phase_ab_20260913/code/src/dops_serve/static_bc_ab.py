"""Fresh or resumed BC with B's equal-third phase objective."""
import argparse
import csv
import gzip
import json
import math
import multiprocessing
import os
import platform
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from . import static_bc as bc
from .placement_policy import StaticPlacementActor

GROUPS = ("prefill", "startup_decode", "steady_decode")


def phase_group(sample):
    return "prefill" if sample["phase"] == "prefill" else "startup_decode" if sample["token_index"] <= 2 else "steady_decode"


def add_phase_weights(samples):
    counts = Counter((row["case_id"], phase_group(row)) for row in samples)
    for row in samples:
        row["group"] = phase_group(row)
        row["phase_weight"] = 1.0 / (3 * counts[row["case_id"], row["group"]])
    return samples


def initialize(protocol, device):
    torch.manual_seed(protocol["seed"])
    actor = StaticPlacementActor(**{key: protocol[key] for key in ("state_dim", "node_dim", "hidden", "depth")}).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=protocol["learning_rate"])
    return actor, optimizer, random.Random(protocol["seed"])


def restore(checkpoint, device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**state["dimensions"]).to(device)
    actor.load_state_dict(state["actor"])
    optimizer = torch.optim.Adam(actor.parameters(), lr=state["optimizer"]["param_groups"][0]["lr"])
    optimizer.load_state_dict(state["optimizer"])
    rng = random.Random()
    rng.setstate(state["random_state"])
    torch.set_rng_state(state["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return actor, optimizer, rng, state["epoch"]


def load_history(base):
    with (base / "training_progress.csv").open() as stream:
        return [{key: value if key in ("route", "model") else int(value) if key in ("epoch", "update_batches") else float(value)
                 for key, value in row.items()} for row in csv.DictReader(stream)]


def update_batch(actor, optimizer, rows, device, sample_count, task_count, max_grad_norm):
    optimizer.zero_grad(set_to_none=True)
    losses, _ = actor(*bc.batch_tensors(rows, device))
    weights = torch.tensor([row["phase_weight"] for row in rows], device=device)
    loss = (losses * weights).mean() * sample_count / task_count
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach())


def teacher_metrics(actor, samples, device, batch_size):
    actor.eval()
    tasks = Counter(model for model, case_id in {(row["model"], row["case_id"]) for row in samples})
    totals = {model: Counter() for model in tasks}
    with torch.inference_mode():
        for rows in bc.batches(samples, batch_size):
            tensors = bc.batch_tensors(rows, device)
            losses, logits = actor(*tensors)
            correct = ((logits.argmax(-1) == tensors[2]) * tensors[4]).sum(1) / tensors[4].sum(1)
            for row, loss, accuracy in zip(rows, losses.tolist(), correct.tolist()):
                model = row["model"]
                totals[model]["common_nll"] += loss * row["task_weight"] / tasks[model]
                group_weight = 3 * row["phase_weight"] / tasks[model]
                totals[model][row["group"] + "_nll"] += loss * group_weight
                totals[model][row["group"] + "_teacher_forced_match"] += accuracy * group_weight
    for metrics in totals.values():
        metrics["loss"] = sum(metrics[group + "_nll"] for group in GROUPS) / 3
    return {model: dict(metrics) for model, metrics in totals.items()}


def evaluate_case(arguments):
    root, backend, case, checkpoint, route, epoch = arguments
    torch.set_num_threads(1)
    os.environ["PIM_LATENCY_CACHE_FILE"] = str(root / "runtime" / f"pim_{route}_{epoch}_{case['case_id']}.json")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**state["dimensions"])
    actor.load_state_dict(state["actor"])
    actor = torch.jit.script(actor.eval())
    reference = bc.read(root / "results" / f"{case['case_id']:03d}.json")
    counts = {group: Counter() for group in GROUPS}
    prefill_s = 0.0
    with gzip.open(root / "anchors" / f"{case['case_id']:03d}.jsonl.gz", "rt") as stream:
        def observe(sample):
            nonlocal prefill_s
            teacher = json.loads(next(stream))
            assert teacher["token_index"] == sample["token_index"]
            group = phase_group(sample)
            correct = total = pim_to_npu = 0
            for expected, chosen, learned in zip(teacher["actions"], sample["actions"], teacher["learned"]):
                if learned:
                    total += 1
                    correct += expected == chosen
                    pim_to_npu += expected in (1, 2) and chosen == 0
            counts[group].update(correct=correct, total=total, plans=1, exact=int(correct == total), pim_to_npu=pim_to_npu)
            if sample["token_index"] == 0:
                prefill_s = sample["now_s"]
        with torch.inference_mode():
            result = bc.replay_task(root, backend, case, reference["summary_path"], actor=actor,
                on_plan=observe, name=f"{route}_e{epoch}_{case['case_id']}")
        assert next(stream, None) is None
    output, elapsed = result["output_tokens"], result["simulated_s"]
    row = dict(case_id=case["case_id"], model=case["model"], batch=case["batch"], prompt=case["prompt"],
        decode=case["decode"], route=route, epoch=epoch, winner=reference["winner"], output_tokens=output,
        bc_execution_s=elapsed, teacher_execution_s=reference["teacher_s"], pd_execution_s=reference["pd_simulated_s"],
        decision_s=result["decision_s"], prefill_s=prefill_s, decode_s=elapsed-prefill_s,
        bc_throughput=output/elapsed, teacher_throughput=output/reference["teacher_s"],
        pd_throughput=output/reference["pd_simulated_s"], bc_over_teacher=reference["teacher_s"]/elapsed,
        bc_over_pd=reference["pd_simulated_s"]/elapsed, host=platform.node())
    for group, count in counts.items():
        row.update({group + "_autonomous_match": count["correct"]/count["total"],
                    group + "_exact_plan_fraction": count["exact"]/count["plans"],
                    group + "_pim_to_npu_fraction": count["pim_to_npu"]/count["total"]})
    return row


def evaluate(root, backend, cases, checkpoint, route, epoch, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        return list(pool.map(evaluate_case, [(root, backend, case, checkpoint, route, epoch) for case in cases]))



def prepare(root, backend, source):
    assert bc.read(source / "dataset_ready.json")["complete"]
    protocol = bc.read(source / "protocol.json")
    cases = [case for case in bc.read(source / "cases.json") if case["split"] != "test"]
    for case in cases:
        cfg = bc.read(case["run_config_path"])
        cfg.update(hardware_json=str(backend / "src/examples/hardware_1npu_2aim.json"),
            pim_config_path=str(backend / "src/aim_simulator/PIM_AiM.json"),
            ramulator_config_path=str(backend / "src/aim_simulator/example.yaml"))
        case["run_config_path"] = str(root / "configs" / f"{case['case_id']:03d}.json")
        bc.write(case["run_config_path"], cfg)
        for folder, suffix in (("anchors", ".jsonl.gz"), ("results", ".json")):
            (root / folder).mkdir(exist_ok=True)
            shutil.copy2(source / folder / f"{case['case_id']:03d}{suffix}", root / folder)
    protocol.update(source_root=str(source), models=bc.MODELS, routes=[*bc.MODELS, "shared"],
        fresh_initialization=True, epochs=50,
        loss="mean learned-node conditional NLL; equal thirds for prefill, decode t=0/1/2, decode t>=4; mean plans within group, mean tasks; equal model task counts",
        primary_metric="sum output tokens / sum simulated execution seconds; all decision overhead excluded",
        counts={split: sum(case["split"] == split for case in cases) for split in ("train", "val")})
    bc.write(root / "cases.json", cases)
    return bc.write(root / "protocol.json", protocol)


def aggregate(route, model, epoch, results, train_metrics, val_metrics, elapsed, updates):
    output = sum(row["output_tokens"] for row in results)
    actual, teacher, pd = (sum(row[key] for row in results) for key in ("bc_execution_s", "teacher_execution_s", "pd_execution_s"))
    row = dict(route=route, model=model, epoch=epoch,
        train_loss=train_metrics["loss"], val_loss=val_metrics["loss"],
        train_common_nll=train_metrics["common_nll"], val_common_nll=val_metrics["common_nll"],
        output_tokens=output, bc_execution_s=actual, teacher_execution_s=teacher, pd_execution_s=pd,
        bc_throughput=output/actual, teacher_throughput=output/teacher, pd_throughput=output/pd,
        teacher_gap_pct=100*(teacher/actual-1), bc_over_teacher=teacher/actual, bc_over_pd=pd/actual,
        decision_s=sum(item["decision_s"] for item in results), epoch_wall_s=elapsed, update_batches=updates)
    for split, metrics in (("train", train_metrics), ("val", val_metrics)):
        row.update({split + "_" + key: value for key, value in metrics.items() if key not in ("loss", "common_nll")})
    for group in GROUPS:
        for metric in ("autonomous_match", "exact_plan_fraction", "pim_to_npu_fraction"):
            key = group + "_" + metric
            row[key] = sum(item[key] for item in results)/len(results)
    return row


def report(root, route, history, target_epoch=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = root / "training" / route
    bc.rows_csv(base / "training_progress.csv", history)
    lines = [f"# {route}: B loss, target {target_epoch} epochs", "",
        "Loss = (prefill NLL + startup decode NLL + steady decode NLL) / 3, evaluated after each epoch on the full fixed train/val datasets.",
        "Throughput = total output tokens / total simulated execution seconds, excluding all decision overhead. Decision seconds are recorded separately.",
        "Each model is reported separately; teacher and PD use the same validation tasks. Test stays locked.", ""]
    for model in sorted({row["model"] for row in history}):
        rows = [row for row in history if row["model"] == model]
        bc.rows_csv(base / f"training_progress_{model}.csv", rows)
        figures = base / "figures" / model
        figures.mkdir(parents=True, exist_ok=True)
        plots = {
            "loss": ("Phase-balanced conditional NLL", [("train_loss", "Train"), ("val_loss", "Validation")]),
            "throughput": ("Output tokens / simulated second", [("bc_throughput", "BC"), ("teacher_throughput", "Teacher"), ("pd_throughput", "PD")]),
            "over_pd": ("BC / PD throughput", [("bc_over_pd", "BC / PD")]),
            "common_nll": ("Original task-mean conditional NLL (diagnostic)", [("train_common_nll", "Train"), ("val_common_nll", "Validation")]),
            "phase_val_nll": ("Validation conditional NLL by phase", [("val_"+group+"_nll", group) for group in GROUPS]),
            "autonomous_match": ("Autonomous learned-node agreement", [(group+"_autonomous_match", group) for group in GROUPS]),
        }
        for name, (ylabel, curves) in plots.items():
            fig, ax = plt.subplots(figsize=(7, 4))
            for key, label in curves:
                ax.plot([row["epoch"] for row in rows], [row[key] for row in rows], marker=".", label=label)
            if name == "over_pd":
                ax.axhline(1, color="black", linestyle="--", label="PD")
            ax.set(xlabel="Epoch", ylabel=ylabel, title=f"{route} / {model}")
            ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
            for extension in ("png", "svg"):
                fig.savefig(figures / f"epoch_{name}.{extension}", dpi=170)
            plt.close(fig)
        best = max(rows, key=lambda row: row["bc_throughput"])
        lines.extend([f"## {model}", "", f"Best validation throughput so far: E{best['epoch']}, {best['bc_throughput']:.4f} tokens/s.", "",
            "|Epoch|Train loss|Val loss|BC tokens/s|Teacher tokens/s|PD tokens/s|BC/teacher|BC/PD|Decision s|",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        lines.extend(f"|{row['epoch']}|{row['train_loss']:.6f}|{row['val_loss']:.6f}|{row['bc_throughput']:.4f}|{row['teacher_throughput']:.4f}|{row['pd_throughput']:.4f}|{100*row['bc_over_teacher']:.2f}%|{100*row['bc_over_pd']:.2f}%|{row['decision_s']:.3f}|" for row in rows)
        lines.append("")
    (base / "report.md").write_text("\n".join(lines)+"\n")


def preflight(root, backend, workers):
    protocol, cases = bc.read(root / "protocol.json"), bc.read(root / "cases.json")
    assert protocol["fresh_initialization"] and protocol["epochs"] == 50 and protocol["test_locked"]
    assert Counter((case["model"], case["split"]) for case in cases) == Counter({(model, split): count for model in bc.MODELS for split, count in (("train", 48), ("val", 8))})
    # Exercise all three node counts, shared batching, GPU updates and autonomous replay.
    train_cases = [next(case for case in cases if case["model"] == model and case["split"] == "train" and case["decode"] == 128) for model in bc.MODELS]
    val_cases = [next(case for case in cases if case["model"] == model and case["split"] == "val" and case["decode"] == 128) for model in bc.MODELS]
    samples = add_phase_weights(bc.load_samples(root, train_cases))
    actor, optimizer, rng = initialize(protocol, "cuda")
    assert not optimizer.state
    initial = teacher_metrics(actor, samples, "cuda", protocol["batch_plans"])
    actor.train()
    for rows in bc.batches(samples, protocol["batch_plans"], rng):
        update_batch(actor, optimizer, rows, "cuda", len(samples), len(train_cases), protocol["max_grad_norm"])
    metrics = teacher_metrics(actor, samples, "cuda", protocol["batch_plans"])
    assert all(math.isfinite(values["loss"]) for values in metrics.values())
    checkpoint = root / "preflight/smoke.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(actor={key: value.cpu() for key, value in actor.state_dict().items()}, dimensions=actor.dimensions), checkpoint)
    results = evaluate(root, backend, val_cases, checkpoint, "preflight", 0, workers)
    for row in results:
        assert row["output_tokens"] == row["batch"] * row["decode"]
        assert math.isfinite(row["bc_execution_s"]) and row["bc_execution_s"] > 0
    bc.rows_csv(root / "preflight/evaluation_cases.csv", results)
    return bc.write(root / "preflight_passed.json", dict(complete=True, fresh_optimizer_empty=True,
        smoke_checkpoint_excluded_from_training=True, test_executed=False, initial_metrics=initial, after_update_metrics=metrics,
        models=bc.MODELS, train_tasks=144, val_tasks=24, smoke_plans=len(samples),
        parameter_count=sum(parameter.numel() for parameter in actor.parameters()), gpu=torch.cuda.get_device_name(0)))


def resume_preflight(root, backend, workers, routes):
    protocol, cases = bc.read(root / "protocol.json"), bc.read(root / "cases.json")
    destination = root / "continuation_to_e100"
    previous = bc.read(destination / "preflight_passed.json") if (destination / "preflight_passed.json").exists() else {"checks": []}
    checks = [row for row in previous["checks"] if row["route"] not in routes]
    for route in routes:
        models = protocol["models"] if route == "shared" else [route]
        base = root / "training" / route
        assert bc.read(base / "complete.json")["final_epoch"] == 50
        history = load_history(base)
        for model in models:
            assert [row["epoch"] for row in history if row["model"] == model] == list(range(1, 51))
        checkpoint = base / "epoch_050.pt"
        actor, optimizer, rng, epoch = restore(checkpoint, "cuda")
        assert epoch == 50
        steps = sorted({int(value["step"]) for value in optimizer.state.values()})
        assert steps == [sum(row["update_batches"] for row in history if row["model"] == models[0])]
        assert optimizer.param_groups[0]["lr"] == protocol["learning_rate"]
        val_cases = [case for case in cases if case["model"] in models and case["split"] == "val"]
        samples = add_phase_weights(bc.load_samples(root, val_cases))
        metrics = teacher_metrics(actor, samples, "cuda", protocol["batch_plans"])
        replay_cases = [next(case for case in val_cases if case["model"] == model) for model in models]
        replays = evaluate(root, backend, replay_cases, checkpoint, "resume_preflight", 50, workers)
        with (base / "epoch_050_cases.csv").open() as stream:
            references = {int(row["case_id"]): row for row in csv.DictReader(stream)}
        for model, replay in zip(models, replays):
            last = next(row for row in history if row["model"] == model and row["epoch"] == 50)
            assert math.isclose(metrics[model]["loss"], last["val_loss"], rel_tol=1e-5, abs_tol=1e-7)
            reference = references[replay["case_id"]]
            assert math.isclose(replay["bc_execution_s"], float(reference["bc_execution_s"]), rel_tol=1e-8)
            checks.append(dict(route=route, model=model, source_checkpoint=str(checkpoint), epoch=epoch, optimizer_steps=steps,
                learning_rate=optimizer.param_groups[0]["lr"], val_loss=metrics[model]["loss"],
                expected_val_loss=last["val_loss"], replay_case_id=replay["case_id"],
                replay_execution_s=replay["bc_execution_s"], expected_execution_s=float(reference["bc_execution_s"])))
    checked_routes = list(dict.fromkeys(row["route"] for row in checks))
    bc.write(destination / "protocol.json", dict(models=protocol["models"] if "shared" in checked_routes else checked_routes,
        routes=checked_routes, source_epoch=50, target_epoch=100,
        additional_epochs=50, loss=protocol["loss"], learning_rate=protocol["learning_rate"],
        batch_plans=protocol["batch_plans"], max_grad_norm=protocol["max_grad_norm"],
        same_teacher_and_validation_data=True, restore_network_optimizer_and_rng=True, test_locked=True,
        primary_metric=protocol["primary_metric"]))
    return bc.write(destination / "preflight_passed.json", dict(complete=True, source_epoch=50, target_epoch=100,
        checks=checks, gpu=torch.cuda.get_device_name(0), test_executed=False))


def train(root, backend, route, workers, resume_epoch=0, target_epoch=None):
    assert bc.read(root / "preflight_passed.json")["complete"]
    protocol = bc.read(root / "protocol.json")
    target_epoch = protocol["epochs"] if target_epoch is None else target_epoch
    models = protocol["models"] if route == "shared" else [route]
    cases = [case for case in bc.read(root / "cases.json") if case["model"] in models]
    train_cases = [case for case in cases if case["split"] == "train"]
    val_cases = [case for case in cases if case["split"] == "val"]
    base = root / "training" / route
    if resume_epoch:
        continuation = bc.read(root / "continuation_to_e100/preflight_passed.json")
        assert continuation["complete"] and continuation["source_epoch"] == resume_epoch and continuation["target_epoch"] == target_epoch
        assert route in {row["route"] for row in continuation["checks"]}
        assert bc.read(base / "complete.json")["final_epoch"] == resume_epoch
    else:
        base.mkdir(parents=True)
    torch.set_num_threads(1)
    bc.write(base / "status.json", dict(epoch=resume_epoch, stage="loading_data", target_epoch=target_epoch))
    samples = add_phase_weights(bc.load_samples(root, train_cases))
    val_samples = add_phase_weights(bc.load_samples(root, val_cases))
    if resume_epoch:
        checkpoint = base / f"epoch_{resume_epoch:03d}.pt"
        actor, optimizer, rng, start_epoch = restore(checkpoint, "cuda")
        assert start_epoch == resume_epoch
        history = load_history(base)
        assert history[-1]["epoch"] == resume_epoch
        steps = sorted({int(value["step"]) for value in optimizer.state.values()})
        assert steps == [sum(row["update_batches"] for row in history if row["model"] == models[0])]
        shutil.copy2(base / "training_progress.csv", base / f"training_progress_epoch_{resume_epoch:03d}.csv")
        shutil.copy2(base / "report.md", base / f"report_epoch_{resume_epoch:03d}.md")
        (base / "complete.json").rename(base / f"complete_epoch_{resume_epoch:03d}.json")
        bc.write(base / "continuation_initialized.json", dict(route=route, source_checkpoint=str(checkpoint),
            start_epoch=resume_epoch, target_epoch=target_epoch, additional_epochs=target_epoch-resume_epoch,
            network_restored=True, optimizer_restored=True, random_states_restored=True, optimizer_steps=steps,
            learning_rate=optimizer.param_groups[0]["lr"], seed=protocol["seed"], scheme=protocol["loss"],
            train_tasks=len(train_cases), val_tasks=len(val_cases), train_plans=len(samples), val_plans=len(val_samples),
            host=platform.node(), gpu=torch.cuda.get_device_name(0), workers=workers))
    else:
        actor, optimizer, rng = initialize(protocol, "cuda")
        assert not optimizer.state
        bc.write(base / "initialized.json", dict(route=route, models=models, fresh_initialization=True,
            source_checkpoint=None, start_epoch=0, epochs=target_epoch, seed=protocol["seed"],
            scheme=protocol["loss"], optimizer_initial_steps=0, learning_rate=optimizer.param_groups[0]["lr"],
            parameter_count=sum(parameter.numel() for parameter in actor.parameters()), dimensions=actor.dimensions,
            train_tasks=len(train_cases), val_tasks=len(val_cases), train_plans=len(samples), val_plans=len(val_samples),
            host=platform.node(), gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, workers=workers))
        history = []
    for epoch in range(resume_epoch+1, target_epoch+1):
        started = time.perf_counter(); actor.train()
        bc.write(base / "status.json", dict(epoch=epoch, stage="training"))
        update_batches = bc.batches(samples, protocol["batch_plans"], rng)
        for rows in update_batches:
            update_batch(actor, optimizer, rows, "cuda", len(samples), len(train_cases), protocol["max_grad_norm"])
        bc.write(base / "status.json", dict(epoch=epoch, stage="teacher_loss"))
        train_metrics = teacher_metrics(actor, samples, "cuda", protocol["batch_plans"])
        val_metrics = teacher_metrics(actor, val_samples, "cuda", protocol["batch_plans"])
        checkpoint = base / f"epoch_{epoch:03d}.pt"
        torch.save(dict(actor={key: value.cpu() for key, value in actor.state_dict().items()}, optimizer=optimizer.state_dict(),
            dimensions=actor.dimensions, epoch=epoch, seed=protocol["seed"], random_state=rng.getstate(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), route=route,
            fresh_initialization=True, resumed_from_epoch=resume_epoch, target_epoch=target_epoch, loss=protocol["loss"]), checkpoint)
        bc.write(base / "status.json", dict(epoch=epoch, stage="validation", train_metrics=train_metrics, val_metrics=val_metrics))
        results = evaluate(root, backend, val_cases, checkpoint, route, epoch, workers)
        bc.rows_csv(base / f"epoch_{epoch:03d}_cases.csv", results)
        epoch_rows = [aggregate(route, model, epoch, [row for row in results if row["model"] == model],
            train_metrics[model], val_metrics[model], time.perf_counter()-started, len(update_batches)) for model in models]
        history.extend(epoch_rows)
        report(root, route, history, target_epoch)
        bc.write(base / "status.json", dict(epoch=epoch, stage="epoch_complete", rows=epoch_rows))
        print(json.dumps(epoch_rows), flush=True)
    return bc.write(base / "complete.json", dict(complete=True, fresh_initialization=True, epochs=target_epoch,
        resumed_from_epoch=resume_epoch, final_epoch=target_epoch, test_executed=False, awaiting_user_before_continuation=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "preflight", "resume-preflight", "train"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--route", choices=(*bc.MODELS, "shared"))
    parser.add_argument("--routes", nargs="+", choices=(*bc.MODELS, "shared"), default=["qwen_1.8b", "qwen_7b"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume-epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "prepare":
        result = prepare(args.root, args.backend, args.source_root)
    elif args.stage == "preflight":
        result = preflight(args.root, args.backend, args.workers)
    elif args.stage == "resume-preflight":
        result = resume_preflight(args.root, args.backend, args.workers, args.routes)
    else:
        result = train(args.root, args.backend, args.route, args.workers, args.resume_epoch, args.epochs)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
