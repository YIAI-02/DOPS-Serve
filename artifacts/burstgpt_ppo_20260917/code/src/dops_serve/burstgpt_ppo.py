"""BC-initialized node PPO on complete continuous BurstGPT traces."""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from .placement_policy import StaticPlacementActor, StaticPlacementCritic
from .placement_training import BurstWorkers, cpu_weights, update_static_ppo, group_rollout_fragments
from .io import read, write
from .workloads import load_full_trace

EPOCHS = {"qwen_1.8b": 57, "qwen_7b": 77, "qwen_14b": 42}
MODELS = list(EPOCHS)


def make_evaluator(root, model):
    from .placement_evaluator import BurstPlacementEvaluator
    protocol = read(root / "protocol.json")
    return BurstPlacementEvaluator(root / "backend/DOPS", root / "configs" / f"{model}.json",
                                   protocol["parameters"], protocol["kv_headroom_bytes"])


def calibrate(root, model, indices, label):
    from .replay import BurstPlacementReplay
    protocol = read(root / "protocol.json")
    evaluator = make_evaluator(root, model)
    destination = root / "calibration" / label / model
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index in indices:
        path = destination / f"{index:06d}.json"
        if path.exists():
            result = read(path)
        else:
            # Saturation removes only arrivals; requests, order, lengths, FIFO,
            # max batch and KV admission remain exactly the serving contract.
            rows = load_full_trace(root / "data", "train", index, 0.0)
            env = BurstPlacementReplay(rows, evaluator, protocol["decode_features"][model])
            started = time.perf_counter()
            while not env.done:
                decision_start = time.perf_counter()
                plan = evaluator.pd_plan(env.snapshot, env.pending[0], env.shapes())
                env.decision_s += time.perf_counter()-decision_start
                env.step(plan)
                if env.plan_count % 100 == 0:
                    print(json.dumps(dict(stage="calibration", model=model, trace=index,
                        plans=env.plan_count, completed=len(env.completed), output_tokens=env.output_tokens,
                        wall_s=time.perf_counter()-started)), flush=True)
            result = dict(trace=index, wall_s=time.perf_counter()-started, **env.metrics())
            write(path, result)
        results.append(result)
        print(json.dumps(dict(stage="calibration_trace_complete", model=model, **result)), flush=True)
    duration = sum(r["simulated_s"] for r in results)
    tokens = sum(r["output_tokens"] for r in results)
    arrival_rate = read(root / "data/manifest.json")["splits"]["train"]["arrival_output_tokens_per_s"]
    write(root / "calibration" / label / f"{model}.json", dict(model=model, trace_indices=indices,
        source_split="train", reference_output_tokens_per_s=tokens/duration,
        reward_scale_s=duration/len(results), original_arrival_output_tokens_per_s=arrival_rate,
        rows=results, label=label))


def route_groups(route, protocol):
    return [(route, load) for load in protocol["loads"]]


def calibrated_records(root, split, index, groups, calibration):
    return [load_full_trace(root / "data", split, index,
                calibration[model]["original_arrival_output_tokens_per_s"] /
                (load*calibration[model]["reference_output_tokens_per_s"])) for model, load in groups]


def evaluate_with_workers(root, workers, weights, indices, split, calibration):
    if split == "test":
        raise ValueError("This PPO round keeps test locked")
    records = []
    load_groups = list(dict.fromkeys(workers.groups))
    cohort_size = len(workers.groups) // len(load_groups)
    for lo in range(0, len(indices), cohort_size):
        cohort = indices[lo:lo+cohort_size]
        payloads = [rows for index in cohort
                    for rows in calibrated_records(root, split, index, load_groups, calibration)]
        payloads += [Ellipsis] * (len(workers.groups)-len(payloads))
        workers.call("start", payloads)
        results = workers.call("evaluate", [Ellipsis if rows is Ellipsis else weights for rows in payloads])
        for i, result in enumerate(results):
            if result is not None:
                model, load = workers.groups[i]
                records.append(dict(model=model, load=load, trace=cohort[i//len(load_groups)],
                                    split=split, **result))
        print(json.dumps(dict(stage="evaluation_cohort_complete", split=split, traces=cohort)), flush=True)
    cells = []
    for model, load in load_groups:
        selected = [r for r in records if (r["model"], r["load"]) == (model, load)]
        duration = sum(r["simulated_s"] for r in selected)
        tokens = sum(r["output_tokens"] for r in selected)
        cells.append(dict(model=model, load=load, throughput=tokens/duration,
                          simulated_s=duration, output_tokens=tokens))
    return dict(indices=indices, split=split, cells=cells, traces=records)


def score(result, baseline):
    reference = {(r["model"], r["load"]): r["throughput"] for r in baseline["cells"]}
    ratios = [r["throughput"]/reference[r["model"], r["load"]] for r in result["cells"]]
    return math.exp(sum(math.log(v) for v in ratios)/len(ratios))


def atomic_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def strict_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_ppo_start(root, route, device):
    protocol = read(root / "protocol.json")
    initial = torch.load(root / "initial" / route / "e0.pt", map_location="cpu", weights_only=False)
    trained = torch.load(root / "initial" / route / "critic.pt", map_location="cpu", weights_only=False)
    if (initial["bc_epoch"], trained["bc_epoch"], trained["epoch"], trained["route"]) != (
            EPOCHS[route], EPOCHS[route], protocol["critic_epochs"][route], route):
        raise ValueError("BC/critic provenance differs from the agreed starts")
    if not trained["validation"]["passed"] or trained["dimensions"] != initial["dimensions"]:
        raise ValueError("Pretrained critic did not pass or its dimensions differ")
    actor = StaticPlacementActor(**initial["dimensions"])
    actor.load_state_dict(initial["actor"])
    critic = StaticPlacementCritic(actor)
    critic.load_state_dict(trained["critic"])
    return actor.to(device), critic.to(device), initial


def write_epoch_report(root, base, route, seed, epoch, result, baseline, history, diagnostics):
    from datetime import date
    reference = {r["load"]: r["throughput"] for r in baseline["cells"]}
    best = max(history, key=lambda row: row["score"])
    lines = [f"# PPO {route} / seed {seed} / E{epoch}", "",
        f"固定 {len(result['indices'])} 条 val；test 未使用。当前三负载几何平均相对 E0："
        f"{score(result, baseline):.6f}；最佳 E{best['epoch']}：{best['score']:.6f}。", "",
        "| 负载 | E0 token/s | 当前 token/s | 相对 E0 |", "| --- | ---: | ---: | ---: |"]
    for cell in result["cells"]:
        lines.append(f"| {cell['load']:g}× | {reference[cell['load']]:.6f} | "
                     f"{cell['throughput']:.6f} | {cell['throughput']/reference[cell['load']]:.6f} |")
    lines += ["", "吞吐来自完整排空的分析模拟；CPU 决策时间不计入目标。", "",
              "诊断为上次报告以来已完成 cohort 的更新统计：", "", "```json",
              json.dumps(diagnostics, indent=2), "```", "",
              f"结果：`{base / 'validation' / f'epoch_{epoch:03d}.json'}`",
              f"恢复：`{base / 'last.pt'}`；最佳：`{base / 'best.pt'}`。", ""]
    path = root.parents[1] / "md" / f"{date.today().isoformat()}_PPO_{route}_seed{seed}_E{epoch:03d}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return str(path)


def train(root, route, seed, rounds, pilot_traces, calibration_label, resume, device):
    protocol = read(root / "protocol.json")
    settings = protocol["ppo"]
    pilot = pilot_traces > 0
    base = root / ("pilot/training" if pilot else "training") / route / f"seed_{seed}"
    if not (root / "preflight_passed.json").exists():
        raise RuntimeError("preflight has not passed")
    if not pilot:
        if not (root / "formal_budget.json").exists():
            raise RuntimeError("Formal PPO requires user confirmation of the pre-submission report")
        if rounds != protocol["epochs"] or seed not in protocol["seeds"] or route not in protocol["models"]:
            raise ValueError("Formal run differs from the frozen model/seed/epoch contract")
    base.mkdir(parents=True, exist_ok=True)
    strict_fp32()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor, critic, initial = load_ppo_start(root, route, device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=settings["critic_lr"])
    load_groups = route_groups(route, protocol)
    groups = load_groups * protocol["cohort_traces"]
    calibration = {m: read(root / "calibration" / calibration_label / f"{m}.json") for m, _ in load_groups}
    order = list(protocol["train_indices"][:pilot_traces] if pilot else protocol["train_indices"])
    rng.shuffle(order)
    epoch, cursor, updates, completed_sources = 0, 0, 0, 0
    previous_wall_s = 0.
    history, worker_rng, pending_validation = [], None, None
    interval = dict(updates=0, optimizer_steps=0, kl_stops=0, sample_wall_s=0., update_wall_s=0.,
                    mean_kl=0., entropy=0., clip_fraction=0., value_loss=0., raw_advantage_std=0.)
    last = base / "last.pt"
    if resume:
        state = torch.load(last, map_location="cpu", weights_only=False)
        if (state["seed"], state["route"], state["protocol"], state["calibration"], state["pilot"]) != (seed, route, protocol, calibration, pilot):
            raise ValueError("resume must retain route, seed, protocol and calibration")
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        rng.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        epoch, cursor, order = state["epoch"], state["cursor"], state["order"]
        updates, completed_sources = state["updates"], state["completed_sources"]
        history, worker_rng = state["history"], state["worker_rng"]
        previous_wall_s = state["training_wall_s"]
        pending_validation, interval = state["pending_validation"], state["interval"]
    elif last.exists():
        raise FileExistsError("training already initialized; pass --resume")
    workers = BurstWorkers(root, groups, protocol["decode_features"][route], seed, worker_rng)
    curve = (base / f"updates_{time.time_ns()}.jsonl").open("w")
    started = time.perf_counter()

    def save(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_checkpoint(path, dict(actor=cpu_weights(actor), critic=cpu_weights(critic),
            dimensions=actor.dimensions, actor_optimizer=actor_optimizer.state_dict(),
            critic_optimizer=critic_optimizer.state_dict(), python_rng=rng.getstate(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            worker_rng=workers.call("rng", [None]*len(groups)), epoch=epoch, cursor=cursor, order=order,
            updates=updates, completed_sources=completed_sources, history=history, interval=interval,
            pending_validation=pending_validation, protocol=protocol, calibration=calibration,
            route=route, seed=seed, pilot=pilot,
            training_wall_s=previous_wall_s+time.perf_counter()-started,
            boundary="all 12 environments drained their fixed cohort and PPO update completed"))

    def validate_pending():
        nonlocal pending_validation
        result = evaluate_with_workers(root, workers, cpu_weights(actor),
                                       protocol["quick_val_indices"], "val", calibration)
        ranking = score(result, e0)
        history.append(dict(checkpoint=pending_validation, score=ranking, epoch=epoch))
        diagnostics = dict(interval)
        for key in ("mean_kl", "entropy", "clip_fraction", "value_loss", "raw_advantage_std"):
            diagnostics[key] /= max(1, interval["updates"])
        write(base / "validation" / f"epoch_{epoch:03d}.json",
              dict(epoch=epoch, score=ranking, diagnostics=diagnostics, **result))
        if ranking > max(row["score"] for row in history[:-1]):
            shutil.copy2(pending_validation, base / "best.pt")
        if not pilot:
            write_epoch_report(root, base, route, seed, epoch, result, e0, history, diagnostics)
        pending_validation = None
        for key in interval:
            interval[key] = 0
        save(last)

    try:
        if not resume:
            write(base / "status.json", dict(stage="initial_evaluation", route=route, seed=seed))
            e0 = evaluate_with_workers(root, workers, initial["actor"], protocol["quick_val_indices"], "val", calibration)
            write(base / "initial_evaluation.json", dict(e0=e0))
            write(base / "validation/epoch_000.json", dict(epoch=0, score=1., diagnostics={}, **e0))
            initial_path = base / "checkpoints/epoch_000.pt"
            history.append(dict(checkpoint=str(initial_path), score=1., epoch=0))
            save(initial_path)
            shutil.copy2(initial_path, base / "best.pt")
            save(last)
            if not pilot:
                write_epoch_report(root, base, route, seed, 0, e0, e0, history, {})
        else:
            e0 = read(base / "initial_evaluation.json")["e0"]
        if pending_validation is not None:
            validate_pending()
        while epoch < rounds:
            cohort = order[cursor:cursor+protocol["cohort_traces"]]
            payloads = [rows for index in cohort
                        for rows in calibrated_records(root, "train", index, load_groups, calibration)]
            workers.call("start", payloads)
            done, totals = [False]*len(groups), [None]*len(groups)
            cohort_started = time.perf_counter()
            while not all(done):
                sample_started = time.perf_counter()
                actor_state, critic_state = cpu_weights(actor), cpu_weights(critic)
                sampled = workers.call("rollout", [Ellipsis if finished else dict(actor=actor_state,
                    critic=critic_state, plans=settings["rollout_plans"],
                    reward_scale_s=calibration[model]["reward_scale_s"])
                    for (model, load), finished in zip(groups, done)])
                sample_wall = time.perf_counter()-sample_started
                fragments = [[] if result is None else result["rows"] for result in sampled]
                rollout_groups, fragment_diagnostics = group_rollout_fragments(fragments, groups, load_groups)
                for i, result in enumerate(sampled):
                    if result is not None:
                        done[i] = result["done"]
                        if result["done"]:
                            totals[i] = result["metrics"]
                update_started = time.perf_counter()
                log = update_static_ppo(actor, critic, actor_optimizer, critic_optimizer,
                                        rollout_groups, settings, rng)
                update_wall = time.perf_counter()-update_started
                updates += 1
                record = dict(stage="ppo_update", route=route, seed=seed, pilot=pilot, epoch=epoch+1,
                    traces=cohort, update=updates, plans_per_environment=list(map(len, fragments)),
                    plans_per_load=list(map(len, rollout_groups)), environment_done=done,
                    fragment_diagnostics=fragment_diagnostics,
                    reward=sum(r["reward"] for rows in fragments for r in rows),
                    sample_wall_s=sample_wall, update_wall_s=update_wall,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                    simulator_s=sum(r["simulator_s"] for r in sampled if r is not None), **log)
                interval["updates"] += 1
                interval["optimizer_steps"] += log["optimizer_steps"]
                interval["kl_stops"] += int(log["kl_stopped"])
                interval["sample_wall_s"] += sample_wall
                interval["update_wall_s"] += update_wall
                for key in ("mean_kl", "entropy", "clip_fraction", "value_loss"):
                    interval[key] += log[key]
                interval["raw_advantage_std"] += sum(d["raw_advantage_std"] for d in fragment_diagnostics)/len(fragment_diagnostics)
                curve.write(json.dumps(record)+"\n")
                curve.flush()
                write(base / "status.json", record)
                print(json.dumps(record), flush=True)
            cursor += len(cohort)
            completed_sources += len(cohort)
            write(base / "completed" / f"e{epoch+1:03d}_cohort_{cursor//protocol['cohort_traces']:02d}.json",
                dict(epoch=epoch+1, traces=cohort, groups=totals,
                     wall_s=time.perf_counter()-cohort_started, updates=updates))
            epoch_finished = cursor == len(order)
            if epoch_finished:
                epoch += 1
                cursor = 0
                if epoch < rounds:
                    rng.shuffle(order)
                if epoch % protocol["validation_every_epochs"] == 0 or epoch == rounds:
                    path = base / "checkpoints" / f"epoch_{epoch:03d}.pt"
                    pending_validation = str(path)
                    save(path)
            save(last)
            if pending_validation is not None:
                validate_pending()
        completion = dict(stage="complete", route=route, seed=seed, pilot=pilot, epochs=epoch,
            updates=updates, source_traces=completed_sources, group_tasks=completed_sources*len(load_groups),
            wall_s=previous_wall_s+time.perf_counter()-started, history=history, test_locked=True)
        write(base / "complete.json", completion)
        write(base / "status.json", completion)
    finally:
        curve.close()
        workers.close()


def finalize(root, route, seed):
    """只汇总固定验证集；E50 与包含 E0 的 validation-best 分别保留。"""
    base = root / "training" / route / f"seed_{seed}"
    complete = read(base / "complete.json")
    selected = max(complete["history"], key=lambda row: row["score"])
    write(base / "selection.json", dict(selected=selected, candidates=complete["history"],
        final_checkpoint=str(base / "checkpoints" / f"epoch_{complete['epochs']:03d}.pt"),
        criterion="fixed validation traces; equal-load geometric mean versus E0", test_locked=True))


def worker_check(root, device="cuda"):
    """真实 BC/critic 权重的 12 环境采样、广播和参数更新；不保存测试权重。"""
    from .burstgpt_preflight import request
    protocol = read(root / "protocol.json")
    strict_fp32()
    logs = []
    for route in protocol["models"]:
        torch.manual_seed(4)
        actor, critic, _ = load_ppo_start(root, route, device)
        before_actor, before_critic = cpu_weights(actor), cpu_weights(critic)
        before_actor = {k: v.clone() for k, v in before_actor.items()}
        before_critic = {k: v.clone() for k, v in before_critic.items()}
        aopt = torch.optim.Adam(actor.parameters(), lr=protocol["ppo"]["actor_lr"])
        copt = torch.optim.Adam(critic.parameters(), lr=protocol["ppo"]["critic_lr"])
        load_groups = route_groups(route, protocol)
        groups = load_groups * protocol["cohort_traces"]
        workers = BurstWorkers(root, groups, protocol["decode_features"][route], 4)
        rng = random.Random(4)
        try:
            workers.call("start", [[request(0, 128+i, 20), request(1, 256, 24)] for i in range(len(groups))])
            for iteration in range(2):
                sampled = workers.call("rollout", [dict(actor=cpu_weights(actor), critic=cpu_weights(critic),
                    plans=2, reward_scale_s=1.) for _ in groups])
                rows, _ = group_rollout_fragments([r["rows"] for r in sampled], groups, load_groups)
                log = update_static_ppo(actor, critic, aopt, copt, rows, protocol["ppo"], rng)
                assert abs(log["initial_kl"]) < 1e-5, "sampler/update policies disagree"
                assert all(math.isfinite(v) for v in log.values())
                logs.append(dict(route=route, iteration=iteration, plans_per_load=list(map(len, rows)), **log))
            assert any(not torch.equal(v, cpu_weights(actor)[k]) for k, v in before_actor.items())
            assert any(not torch.equal(v, cpu_weights(critic)[k]) for k, v in before_critic.items())
        finally:
            workers.close()
    write(root / f"worker_check_{device}_passed.json", dict(passed=True, environments=12, updates=logs,
        device=device, gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
        torch_version=torch.__version__, purpose="actual pretrained weights, IPC, bootstrap, broadcast and update; discarded"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "worker-check", "orchestration-check", "calibrate", "train", "finalize"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--route", choices=list(EPOCHS))
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--indices", default="0,983")
    parser.add_argument("--calibration-label", default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    strict_fp32()
    root = args.root.resolve()
    if args.stage == "calibrate":
        calibrate(root, args.model, [int(v) for v in args.indices.split(",")], args.calibration_label)
    elif args.stage == "preflight":
        from .burstgpt_preflight import preflight
        preflight(root)
    elif args.stage == "train":
        train(root, args.route, args.seed, args.rounds, 0,
              args.calibration_label, args.resume, args.device)
    elif args.stage == "worker-check":
        worker_check(root, args.device)
    elif args.stage == "orchestration-check":
        from .burstgpt_preflight import orchestration_check
        orchestration_check(root)
    else:
        finalize(root, args.route, args.seed)


if __name__ == "__main__":
    main()
