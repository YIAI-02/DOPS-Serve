"""BC-initialized node PPO on complete continuous BurstGPT traces."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

from .placement_policy import StaticPlacementActor, StaticPlacementCritic, migrate_bc_actor
from .placement_training import (BurstWorkers, cpu_weights, update_static_ppo,
                                 complete_critic_returns, critic_plan_mse)
from .io import read, write
from .workloads import import_full_burstgpt, load_full_trace

MODELS = ["qwen_1.8b", "qwen_7b", "qwen_14b"]
EPOCHS = {"qwen_1.8b": 57, "qwen_7b": 77, "qwen_14b": 42, "shared": 35}


def prepare(root, bc_root, source):
    if (root / "protocol.json").exists():
        raise FileExistsError("PPO protocol already prepared; use preflight/calibrate/train")
    original = read(root / "bc_inputs/protocol.json")
    cases = read(root / "bc_inputs/cases.json")
    source_checkpoints, decode_features = {}, {}
    for route, epoch in EPOCHS.items():
        source_path = bc_root / "training" / route / f"epoch_{epoch:03d}.pt"
        destination = root / "bc_inputs" / f"{route}_E{epoch}.pt"
        shutil.copy2(source_path, destination)
        checkpoint = torch.load(destination, map_location="cpu", weights_only=False)
        if checkpoint["epoch"] != epoch or checkpoint["route"] != route:
            raise ValueError(f"wrong BC checkpoint for {route} E{epoch}")
        if checkpoint["dimensions"] != dict(state_dim=47, node_dim=35, hidden=256, depth=4):
            raise ValueError("BC architecture differs from the agreed network")
        torch.manual_seed(7)
        actor = migrate_bc_actor(checkpoint)
        initial = root / "initial" / route
        initial.mkdir(parents=True, exist_ok=True)
        torch.save(dict(actor=cpu_weights(actor), dimensions=actor.dimensions,
                        bc_epoch=epoch, source_checkpoint=str(source_path)), initial / "e0.pt")
        selected_models = MODELS if route == "shared" else [route]
        per_model = []
        for model in selected_models:
            tasks = [c for c in cases if c["split"] == "train" and c["model"] == model]
            per_model.append(sum(math.log1p(c["decode"])/10 for c in tasks)/len(tasks))
        decode_features[route] = sum(per_model)/len(per_model)
        source_checkpoints[route] = dict(epoch=epoch, path=str(source_path), copy=str(destination))
    for model, index in zip(MODELS, (0, 64, 128)):
        cfg = read(root / "bc_inputs" / f"{index:03d}.json")
        if (cfg["tp_qkv"], cfg["tp_ffn"], cfg["npu_backend"], cfg["pim_fast_mode"],
            cfg["weight_load_compute_overlap_ratio"], cfg["pim_weight_load_overlap_ratio"]) != (2, 2, "fast", True, 1.0, .5):
            raise ValueError("BC hardware/runtime settings do not match the agreed C026 run")
        for key, suffix in (("hardware_json", "examples/hardware_1npu_2aim.json"),
                            ("pim_config_path", "aim_simulator/PIM_AiM.json"),
                            ("ramulator_config_path", "aim_simulator/example.yaml")):
            cfg[key] = str(root / "backend/DOPS/src" / suffix)
        write(root / "configs" / f"{model}.json", cfg)
    data = import_full_burstgpt(source, root / "data")
    counts = {split: (row["requests"], len(row["traces"])) for split, row in data["splits"].items()}
    if counts != {"train": (983005, 984), "val": (210644, 211), "test": (210645, 211)}:
        raise ValueError(f"BurstGPT source counts changed: {counts}")
    protocol = dict(parameters=original["parameters"], models=MODELS, loads=[.5, 1., 1.5],
        source_checkpoints=source_checkpoints, decode_features=decode_features,
        state_dim=56, node_dim=35, max_tokens=7315, max_batch=16,
        kv_headroom_bytes=2**30, kv_headroom_status="one GiB per PIM; gated by preflight extreme-length replay",
        ppo=dict(actor_lr=3e-5, critic_lr=1e-4, clip=.1, ppo_epochs=4, target_kl=.01,
                 entropy=.001, max_grad_norm=.5, rollout_plans=16, mini_plans=4, gamma=1., gae_lambda=1.),
        quick_val_indices=list(range(0, 211, 14)), seeds=[7, 17, 27],
        objective="sum output tokens / sum complete trace simulated seconds; decision CPU time reported separately",
        source_data=str(source), test_locked=True,
        pilot="seed 7; train-only calibration; complete short tail then shuffled complete traces; no formal continuation",
        formal_budget_status="pending pilot measurement and user resource decision")
    write(root / "protocol.json", protocol)
    print(json.dumps(dict(stage="prepared", counts=counts, checkpoints=source_checkpoints)), flush=True)


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
    return [(model, load) for model in (MODELS if route == "shared" else [route]) for load in protocol["loads"]]


def calibrated_records(root, split, index, groups, calibration):
    return [load_full_trace(root / "data", split, index,
                calibration[model]["original_arrival_output_tokens_per_s"] /
                (load*calibration[model]["reference_output_tokens_per_s"])) for model, load in groups]


def evaluate_with_workers(root, workers, weights, indices, split, calibration):
    records = []
    for index in indices:
        workers.call("start", calibrated_records(root, split, index, workers.groups, calibration))
        results = workers.call("evaluate", [weights for _ in workers.groups])
        for (model, load), result in zip(workers.groups, results):
            records.append(dict(model=model, load=load, trace=index, split=split, **result))
        print(json.dumps(dict(stage="evaluation_trace_complete", split=split, trace=index,
                             groups=len(results))), flush=True)
    cells = []
    for model, load in workers.groups:
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


def train(root, route, seed, rounds, pilot_traces, calibration_label, resume, device):
    protocol = read(root / "protocol.json")
    settings = protocol["ppo"]
    pilot = pilot_traces > 0
    base = root / ("pilot/training" if pilot else "training") / route / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)
    if not (root / "preflight_passed.json").exists():
        raise RuntimeError("compute-node preflight has not passed")
    if not pilot and not (root / "formal_budget.json").exists():
        raise RuntimeError("formal rounds/resources must be frozen after the pilot")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    initial = torch.load(root / "initial" / route / "e0.pt", map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**initial["dimensions"])
    actor.load_state_dict(initial["actor"])
    critic = StaticPlacementCritic(actor)
    actor.to(device)
    critic.to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=settings["actor_lr"])
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=settings["critic_lr"])
    groups = route_groups(route, protocol)
    calibration = {m: read(root / "calibration" / calibration_label / f"{m}.json") for m, _ in groups}
    order = list(range(984))
    rng.shuffle(order)
    if pilot:
        order = [983] + [i for i in order if i != 983][:pilot_traces-1]
    epoch, cursor, updates, completed_sources = 0, 0, 0, 0
    previous_wall_s = 0.
    history, worker_rng = [], None
    pending_validation = None
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
        pending_validation = state["pending_validation"]
    elif last.exists():
        raise FileExistsError("training already initialized; pass --resume")
    workers = BurstWorkers(root, groups, protocol["decode_features"][route], seed, worker_rng)
    attempt = time.time_ns()
    curve = (base / f"updates_{attempt}.jsonl").open("w")
    started = time.perf_counter()

    def save(path):
        atomic_checkpoint(path, dict(actor=cpu_weights(actor), critic=cpu_weights(critic),
            dimensions=actor.dimensions, actor_optimizer=actor_optimizer.state_dict(),
            critic_optimizer=critic_optimizer.state_dict(), python_rng=rng.getstate(),
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else [],
            worker_rng=workers.call("rng", [None]*len(groups)), epoch=epoch, cursor=cursor, order=order,
            updates=updates, completed_sources=completed_sources, history=history,
            pending_validation=pending_validation,
            protocol=protocol, calibration=calibration, route=route, seed=seed, pilot=pilot,
            training_wall_s=previous_wall_s+time.perf_counter()-started,
            boundary="all groups drained same source trace and PPO update completed"))

    def validate_pending():
        nonlocal e0, pending_validation
        indices = protocol["quick_val_indices"][:1] if pilot else protocol["quick_val_indices"]
        if pilot:
            e0 = evaluate_with_workers(root, workers, initial["actor"], indices, "val", calibration)
            write(base / "quick_val_e0.json", e0)
        result = evaluate_with_workers(root, workers, cpu_weights(actor), indices, "val", calibration)
        ranking = score(result, e0)
        write(base / "validation" / f"source_{completed_sources:06d}.json", dict(score=ranking, **result))
        history.append(dict(checkpoint=pending_validation, score=ranking, completed_sources=completed_sources))
        pending_validation = None
        save(last)

    try:
        if not resume:
            # The short complete train tail checks autonomous E0/PD behavior
            # before pilot updates. Formal selection always uses fixed val.
            baseline_indices = [983] if pilot else protocol["quick_val_indices"]
            baseline_split = "train" if pilot else "val"
            write(base / "status.json", dict(stage="initial_evaluation", route=route, seed=seed))
            e0 = evaluate_with_workers(root, workers, initial["actor"], baseline_indices, baseline_split, calibration)
            pd = evaluate_with_workers(root, workers, None, baseline_indices, baseline_split, calibration)
            write(base / "initial_evaluation.json", dict(e0=e0, pd=pd))
            save(last)
        else:
            e0 = read(base / "initial_evaluation.json")["e0"]
        if pending_validation is not None:
            validate_pending()
        while epoch < rounds:
            while cursor < len(order):
                index = order[cursor]
                workers.call("start", calibrated_records(root, "train", index, groups, calibration))
                done = [False]*len(groups)
                totals = [None]*len(groups)
                trace_started = time.perf_counter()
                while not all(done):
                    sample_started = time.perf_counter()
                    actor_state, critic_state = cpu_weights(actor), cpu_weights(critic)
                    sampled = workers.call("rollout", [Ellipsis if finished else dict(actor=actor_state,
                        critic=critic_state, plans=settings["rollout_plans"],
                        reward_scale_s=calibration[model]["reward_scale_s"])
                        for (model, load), finished in zip(groups, done)])
                    sample_wall = time.perf_counter()-sample_started
                    rollout_groups = []
                    for i, result in enumerate(sampled):
                        rollout_groups.append([] if result is None else result["rows"])
                        if result is not None:
                            done[i] = result["done"]
                            if result["done"]:
                                totals[i] = result["metrics"]
                    update_started = time.perf_counter()
                    log = update_static_ppo(actor, critic, actor_optimizer, critic_optimizer,
                                            rollout_groups, settings, rng)
                    updates += 1
                    record = dict(stage="ppo_update", route=route, seed=seed, pilot=pilot, round=epoch+1,
                        trace=index, update=updates, plans_per_group=list(map(len, rollout_groups)),
                        group_done=done, reward=sum(r["reward"] for rows in rollout_groups for r in rows),
                        sample_wall_s=sample_wall, update_wall_s=time.perf_counter()-update_started,
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                        simulator_s=sum(r["simulator_s"] for r in sampled if r is not None), **log)
                    curve.write(json.dumps(record)+"\n")
                    curve.flush()
                    write(base / "status.json", record)
                    print(json.dumps(record), flush=True)
                cursor += 1
                completed_sources += 1
                trace_record = dict(round=epoch+1, trace=index, groups=totals,
                                     wall_s=time.perf_counter()-trace_started, updates=updates)
                write(base / "completed" / f"r{epoch+1}_{index:06d}.json", trace_record)
                should_validate = (cursor == len(order) or (not pilot and cursor % 99 == 0))
                if should_validate:
                    path = base / "checkpoints" / f"source_{completed_sources:06d}.pt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending_validation = str(path)
                    save(path)
                save(last)
                if should_validate:
                    validate_pending()
            epoch += 1
            cursor = 0
            if epoch < rounds:
                rng.shuffle(order)
            save(last)
        completion = dict(stage="complete", route=route, seed=seed, pilot=pilot, rounds=rounds,
            updates=updates, source_traces=completed_sources, group_tasks=completed_sources*len(groups),
            wall_s=previous_wall_s+time.perf_counter()-started, history=history,
            formal_continuation_allowed=False if pilot else True)
        write(base / "complete.json", completion)
        write(base / "status.json", completion)
    finally:
        curve.close()
        workers.close()


def finalize(root, route, seed, calibration_label):
    protocol = read(root / "protocol.json")
    base = root / "training" / route / f"seed_{seed}"
    complete = read(base / "complete.json")
    candidates = sorted(complete["history"], key=lambda r: r["score"], reverse=True)[:3]
    initial = root / "initial" / route / "e0.pt"
    candidates = [dict(checkpoint=str(initial), score=1.)] + candidates
    groups = route_groups(route, protocol)
    calibration = {m: read(root / "calibration" / calibration_label / f"{m}.json") for m, _ in groups}
    workers = BurstWorkers(root, groups, protocol["decode_features"][route], seed)
    try:
        scores, e0 = [], None
        for i, row in enumerate(candidates):
            weights = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)["actor"]
            result = evaluate_with_workers(root, workers, weights, list(range(211)), "val", calibration)
            if i == 0:
                e0 = result
            value = score(result, e0)
            scores.append(dict(checkpoint=row["checkpoint"], score=value))
            write(base / "final_val" / f"candidate_{i}.json", dict(score=value, **result))
        selected = max(scores, key=lambda r: r["score"])
        write(base / "selection.json", dict(selected=selected, candidates=scores, criterion="full validation only"))
        for name, checkpoint in (("selected", selected["checkpoint"]), ("e0", str(initial)), ("pd", None)):
            weights = None if checkpoint is None else torch.load(checkpoint, map_location="cpu", weights_only=False)["actor"]
            result = evaluate_with_workers(root, workers, weights, list(range(211)), "test", calibration)
            write(base / "test" / f"{name}.json", result)
    finally:
        workers.close()


def worker_check(root):
    """Exercise all nine persistent groups and CUDA PPO before pilot jobs."""
    protocol = read(root / "protocol.json")
    torch.manual_seed(7)
    initial = torch.load(root / "initial/shared/e0.pt", map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**initial["dimensions"])
    actor.load_state_dict(initial["actor"])
    critic = StaticPlacementCritic(actor)
    actor.cuda()
    critic.cuda()
    aopt = torch.optim.Adam(actor.parameters(), lr=protocol["ppo"]["actor_lr"])
    copt = torch.optim.Adam(critic.parameters(), lr=protocol["ppo"]["critic_lr"])
    groups = route_groups("shared", protocol)
    workers = BurstWorkers(root, groups, protocol["decode_features"]["shared"], 7)
    rng = random.Random(7)
    logs = []
    try:
        workers.call("start", [load_full_trace(root / "data", "train", 983, 1/load) for _, load in groups])
        finished = [False]*len(groups)
        for iteration in range(2):
            sampled = workers.call("rollout", [Ellipsis if done else dict(actor=cpu_weights(actor),
                critic=cpu_weights(critic), plans=2, reward_scale_s=1.) for done in finished])
            rows = [[] if result is None else result["rows"] for result in sampled]
            log = update_static_ppo(actor, critic, aopt, copt, rows, protocol["ppo"], rng)
            assert abs(log["initial_kl"]) < 1e-5, "sampler/update policies disagree"
            assert all(math.isfinite(v) for v in log.values())
            logs.append(dict(iteration=iteration, plans_per_group=list(map(len, rows)), **log))
            finished = [old if result is None else result["done"] for old, result in zip(finished, sampled)]
            print(json.dumps(dict(stage="worker_check_update", **logs[-1])), flush=True)
        write(root / "worker_check_passed.json", dict(passed=True, groups=groups, updates=logs,
            gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
            purpose="IPC, actual-phase bootstrap, policy broadcast and CUDA update; discarded test training"))
    finally:
        workers.close()


def critic_tasks(protocol):
    tasks = []
    for split in ("train", "val"):
        for index in protocol["critic"][f"{split}_indices"]:
            for route in EPOCHS:
                for model, load in route_groups(route, protocol):
                    tasks.append(dict(task=len(tasks), route=route, model=model, load=load,
                                      split=split, trace=index))
    return tasks


def critic_rollout_path(root, task):
    return (root / "rollouts" / task["route"] / task["model"] / f"load_{task['load']:g}"
            / task["split"] / f"{task['trace']:06d}")


def collect_critic(root, task):
    '''
    用固定的 BC actor，把指定的一条 BurstGPT trace 完整跑完，保存每次调度决策，并计算用于训练 critic 的真实累计回报。
    task: (route:actor,model: 哪个模型)
    '''
    protocol = read(root / "protocol.json")
    destination = critic_rollout_path(root, task) #output path
    if (destination / "complete.json").exists():
        print(json.dumps(dict(stage="critic_trace_reused", **task)), flush=True)
        return
    if destination.exists():
        shutil.rmtree(destination)  # An interrupted trace has no valid MC labels.
    destination.mkdir(parents=True)
    initial = torch.load(root / "initial" / task["route"] / "e0.pt", map_location="cpu", weights_only=False)
    calibration = {task["model"]: read(root / "calibration/pilot" / f"{task['model']}.json")}
    groups = [(task["model"], task["load"])]
    records = calibrated_records(root, task["split"], task["trace"], groups, calibration)[0]
    worker_seed = protocol["critic"]["seed"] + 1009 * (task["task"] + 1)
    workers = BurstWorkers(root, groups, protocol["decode_features"][task["route"]], worker_seed - 1009)
    labels = {key: [] for key in ("duration", "terminal", "phase")}
    chunks, offset, started = [], 0, time.perf_counter()
    try:
        workers.call("start", [records])
        while True:
            result = workers.call("collect_critic", [dict(actor=initial["actor"] if offset == 0 else None,
                plans=protocol["critic"]["chunk_plans"])])[0]
            batch = result["batch"]
            name = f"chunk_{len(chunks):06d}.pt"
            torch.save(batch, destination / name)
            count = len(batch["duration"])
            chunks.append(dict(file=name, offset=offset, plans=count))
            offset += count
            for key in labels:
                # Release IPC shared-memory handles after each persisted chunk.
                labels[key].append(batch[key].clone())
            if len(chunks) % 25 == 0:
                print(json.dumps(dict(stage="critic_collect", **task, plans=offset,
                                      wall_s=time.perf_counter()-started)), flush=True)
            if result["done"]:
                metrics = result["metrics"]
                break
        after = workers.call("actor_state", [None])[0] #检查 actor 没被修改，关闭子进程
        if not all(torch.equal(value, after[key]) for key, value in initial["actor"].items()):
            raise AssertionError("frozen collection actor changed")
    finally:
        workers.close()
    labels = {key: torch.cat(values) for key, values in labels.items()}
    labels["returns"] = complete_critic_returns(labels["duration"], labels["terminal"],
                                                calibration[task["model"]]["reward_scale_s"]) #计算每个trace的真实时间
    torch.save(labels, destination / "returns.pt")
    if metrics["completed"] != len(records):
        raise AssertionError("critic collection did not drain all requests")
    write(destination / "complete.json", dict(**task, worker_seed=worker_seed, plans=offset,
        chunks=chunks, actor_unchanged=True, source_checkpoint=initial["source_checkpoint"],#chunk 是组成了多份plan的集合
        bc_epoch=initial["bc_epoch"], wall_s=time.perf_counter()-started, metrics=metrics,
        bytes=sum(path.stat().st_size for path in destination.glob("*.pt"))))
    print(json.dumps(dict(stage="critic_trace_complete", **task, plans=offset,
                          wall_s=time.perf_counter()-started, metrics=metrics)), flush=True)


def load_critic_traces(root, route, split, protocol):
    traces = []
    for task in critic_tasks(protocol):
        if task["route"] != route or task["split"] != split:
            continue
        path = critic_rollout_path(root, task)
        record = read(path / "complete.json")
        if not record["actor_unchanged"] or record["metrics"]["completed"] != record["metrics"]["requests"]:
            raise ValueError(f"incomplete or non-frozen critic data: {path}")
        traces.append(dict(**task, path=path, record=record,
                           labels=torch.load(path / "returns.pt", map_location="cpu", weights_only=True)))
    return traces


def critic_baselines(traces):
    means = {}
    for trace in traces:
        key = (trace["model"], trace["load"])
        bucket = means.setdefault(key, {name: [] for name in ("all", "prefill", "decode")})
        labels = trace["labels"]
        bucket["all"].append(float(labels["returns"].mean()))
        for phase, name in enumerate(("prefill", "decode")):
            bucket[name].append(float(labels["returns"][labels["phase"] == phase].mean()))
    return {key: {name: sum(values)/len(values) for name, values in bucket.items()}
            for key, bucket in means.items()}


def critic_batch(batch, indices, device):
    return {key: batch[key][indices].to(device) for key in ("state", "nodes", "actions", "learned")}


def evaluate_critic(critic, traces, baselines, mini_plans):
    device = next(critic.parameters()).device
    groups = {}
    critic.eval() #评估模式，不更新梯度
    with torch.no_grad():
        for trace in traces:
            moments = {name: torch.zeros(6, dtype=torch.float64) for name in ("all", "prefill", "decode")}
            counts = dict(all=0, prefill=0, decode=0)
            for chunk in trace["record"]["chunks"]:
                raw = torch.load(trace["path"] / chunk["file"], map_location="cpu", weights_only=True)
                for lo in range(0, chunk["plans"], mini_plans):
                    hi = min(lo + mini_plans, chunk["plans"])
                    batch = critic_batch(raw, slice(lo, hi), device) #minibatch
                    values = critic(batch["state"], batch["nodes"], batch["actions"]).double()
                    target = trace["labels"]["returns"][chunk["offset"]+lo:chunk["offset"]+hi].to(device)
                    mask = batch["learned"]
                    count = mask.sum(1)
                    error = target[:, None] - values
                    row = torch.stack((target, target.square(), (values*mask).sum(1)/count,
                        (values.square()*mask).sum(1)/count, (error*mask).sum(1)/count,
                        (error.square()*mask).sum(1)/count), dim=1).cpu()
                    moments["all"] += row.sum(0)
                    counts["all"] += len(row)
                    for phase, name in enumerate(("prefill", "decode")):
                        selected = raw["phase"][lo:hi] == phase
                        moments[name] += row[selected].sum(0)
                        counts[name] += int(selected.sum())
            key = (trace["model"], trace["load"])
            bucket = groups.setdefault(key, {name: [] for name in moments})
            for name in moments:
                bucket[name].append(moments[name] / counts[name])
    cells = []
    for (model, load), bucket in groups.items(): #验证集统计数据
        stats = {}
        for name, rows in bucket.items():
            g, g2, v, v2, err, mse = torch.stack(rows).mean(0).tolist()
            constant = baselines[model, load][name]
            mse_constant = g2 - 2*constant*g + constant*constant
            variance = g2-g*g
            ev = 1 - (mse-err*err)/variance if variance > 0 else None
            stats[name] = dict(mse=mse, mse_zero=g2, mse_constant=mse_constant, ev=ev,
                target_mean=g, target_std=math.sqrt(max(0., g2-g*g)), value_mean=v,
                value_std=math.sqrt(max(0., v2-v*v)), constant=constant,
                passed=mse < g2 and mse < mse_constant and ev is not None and ev > 0)
        cells.append(dict(model=model, load=load, **stats))
    score = sum(cell["all"]["mse"]/cell["all"]["mse_constant"] for cell in cells)/len(cells) #cell 中各mse的平均值
    return dict(cells=cells, score=score, passed=all(cell["all"]["passed"] for cell in cells))


def train_critic(root, route, device):
    '''
    读取 collect_critic() 已经生成的数据，训练一个网络去预测“从当前决策位置到 trace 结束的累计回报
    train_critic()
    ├── read()                         读取配置
    ├── StaticPlacementActor()         加载、冻结 BC 网络
    ├── StaticPlacementCritic(actor)    复制 BC 骨干，创建价值网络
    ├── Adam()                         创建优化器
    ├── load_critic_traces()            读取采样数据索引和标签
    │   ├── critic_tasks()
    │   └── critic_rollout_path()
    ├── critic_baselines()              计算训练均值基线
    │
    ├── 每个 epoch
    │   ├── 读取、打乱 chunk
    │   ├── 每个 minibatch
    │   │   ├── critic_batch()          选择数据，送入 GPU
    │   │   ├── critic_plan_mse()
    │   │   │   └── critic.forward()   预测每个节点动作前的价值
    │   │   ├── backward()             计算梯度
    │   │   └── optimizer.step()       更新 critic 参数
    │   ├── evaluate_critic()           独立验证
    │   ├── cpu_weights()              准备保存参数
    │   └── atomic_checkpoint()        保存 checkpoint
    │
    └── write()                        写最终完成结果
    '''
    # 1. 读取参数，创建输出目录
    protocol = read(root / "protocol.json")
    settings = protocol["critic"]
    seed = settings["seed"]
    base = root / "training" / route / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)
    if (base / "complete.json").exists():
        print(json.dumps(dict(stage="critic_training_reused", route=route)), flush=True)
        return

    # 2. 加载 BC actor，冻结
    torch.manual_seed(seed)
    rng = random.Random(seed)
    initial = torch.load(root / "initial" / route / "e0.pt", map_location="cpu", weights_only=False)
    actor = StaticPlacementActor(**initial["dimensions"])
    actor.load_state_dict(initial["actor"])
    actor.requires_grad_(False)

    # 3. 创建真正要训练的网络
    critic = StaticPlacementCritic(actor).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=settings["lr"])

    # 4. 训练数据和验证数据, basline 作为对照
    train_traces = load_critic_traces(root, route, "train", protocol)
    val_traces = load_critic_traces(root, route, "val", protocol)
    baselines = critic_baselines(train_traces)
    chunks = [(trace, chunk) for trace in train_traces for chunk in trace["record"]["chunks"]]
    batches_per_epoch = sum(math.ceil(chunk["plans"]/settings["mini_plans"]) for _, chunk in chunks)
    denominator = len(route_groups(route, protocol)) * len(settings["train_indices"])
    first_epoch, stale, best_score, best_rank = 0, 0, math.inf, (True, math.inf)

    # 5. 如果有last，恢复上次训练状态
    last = base / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["route"] != route or saved["settings"] != settings:
            raise ValueError("critic resume must retain route and training settings")
        critic.load_state_dict(saved["critic"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        first_epoch, stale = saved["epoch"], saved["stale"]
        best_score, best_rank = saved["best_score"], tuple(saved["best_rank"])
    started = time.perf_counter()

    # 6. 进入训练循环：先打乱 chunk，再打乱其中的 plans
    for epoch in range(first_epoch, settings["max_epochs"]):
        if stale >= settings["patience"]:
            break
        critic.train()
        ordered = list(chunks)
        rng.shuffle(ordered)
        online_mse, steps = 0., 0
        for trace, chunk in ordered:
            raw = torch.load(trace["path"] / chunk["file"], map_location="cpu", weights_only=True)
            targets = trace["labels"]["returns"][chunk["offset"]:chunk["offset"]+chunk["plans"]]
            order = torch.randperm(chunk["plans"])
            trace_weight = 1/(denominator * trace["record"]["plans"])
            for lo in range(0, chunk["plans"], settings["mini_plans"]):
                indices = order[lo:lo+settings["mini_plans"]]
                batch = critic_batch(raw, indices, device)
                returns = targets[indices].to(device=device, dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                losses = critic_plan_mse(critic, batch, returns)
                loss = losses.sum() * trace_weight * batches_per_epoch
                loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), settings["max_grad_norm"])
                optimizer.step() #更新参数
                online_mse += float(losses.detach().sum()) * trace_weight
                steps += 1
                if steps % 200 == 0:
                    print(json.dumps(dict(stage="critic_fit", route=route, epoch=epoch+1,
                        steps=steps, total_steps=batches_per_epoch, wall_s=time.perf_counter()-started)), flush=True)
        validation = evaluate_critic(critic, val_traces, baselines, settings["mini_plans"])
        rank = (not validation["passed"], validation["score"]) #选最佳checkpoint
        checkpoint = dict(critic=cpu_weights(critic), dimensions=initial["dimensions"], route=route,
            seed=seed, epoch=epoch+1, settings=settings, source_checkpoint=initial["source_checkpoint"],
            bc_epoch=initial["bc_epoch"], actor_checkpoint=str(root/"initial"/route/"e0.pt"), validation=validation)
        if rank < best_rank:
            best_rank = rank
            atomic_checkpoint(base / "best.pt", checkpoint)
        if validation["score"] < best_score:
            best_score, stale = validation["score"], 0
        else:
            stale += 1 #连续多少轮没有刷新最佳分数
        if not all(torch.equal(value, actor.state_dict()[key]) for key, value in initial["actor"].items()):
            raise AssertionError("actor changed during critic training")
        record = dict(stage="critic_epoch", route=route, epoch=epoch+1, online_train_mse=online_mse,
            steps=steps, validation=validation, wall_s=time.perf_counter()-started, actor_unchanged=True)
        write(base / f"epoch_{epoch+1:03d}.json", record)
        atomic_checkpoint(last, dict(**checkpoint, optimizer=optimizer.state_dict(), stale=stale,
            best_score=best_score, best_rank=best_rank, python_rng=rng.getstate(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device == "cuda" else []))
        print(json.dumps(record), flush=True)
    best = torch.load(base / "best.pt", map_location="cpu", weights_only=False)
    last_epoch = torch.load(last, map_location="cpu", weights_only=False)["epoch"]
    write(base / "complete.json", dict(route=route, status="passed" if best["validation"]["passed"] else "failed_validation",
        best_epoch=best["epoch"], last_epoch=last_epoch,
        validation=best["validation"], actor_unchanged=True, ppo_started=False))


def critic_check(root, device):
    '''
    hpc 运行试验
    '''
    import tempfile
    protocol = read(root / "protocol.json")
    protocol["critic"] = dict(protocol["critic"], train_indices=[983], val_indices=[983], max_epochs=2)
    results = []
    with tempfile.TemporaryDirectory(prefix="critic_check_", dir=root) as temporary:
        test_root = Path(temporary)
        for name in ("backend", "configs", "initial", "calibration"):
            (test_root/name).symlink_to(root/name, target_is_directory=True)
        for split in ("train", "val"):
            (test_root/"data"/split).mkdir(parents=True)
            shutil.copy2(root/"data/train/000983.json", test_root/"data"/split/"000983.json")
        write(test_root/"protocol.json", protocol)
        # Reuse the short training fixture for validation wiring only; no quality claim.
        for task in critic_tasks(protocol):
            if task["split"] == "train":
                collect_critic(test_root, task)
            else:
                source = critic_rollout_path(test_root, dict(task, split="train"))
                destination = critic_rollout_path(test_root, task)
                shutil.copytree(source, destination)
        for route in EPOCHS:
            train_critic(test_root, route, device) #检查训练链路
            base = test_root/"training"/route/"seed_7"
            completion = read(base/"complete.json")
            before = torch.load(base/"last.pt", map_location="cpu", weights_only=False)
            (base/"complete.json").unlink()
            train_critic(test_root, route, device)
            after = torch.load(base/"last.pt", map_location="cpu", weights_only=False)
            if before["epoch"] != after["epoch"] or not all(torch.equal(v, after["critic"][k])
                                                         for k, v in before["critic"].items()):
                raise AssertionError("finalization resume repeated critic updates")
            results.append(dict(route=route, completed_epochs=completion["last_epoch"],
                                frozen_actor=True, resume_without_extra_updates=True))
    write(root/"critic_check_passed.json", dict(routes=results, device=device, torch_version=torch.__version__,
        validation_reuses_training_fixture=True, purpose="collection, full MC, critic fitting and checkpoint wiring only"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "preflight", "worker-check", "orchestration-check", "calibrate", "train", "finalize", "collect-critic", "train-critic", "critic-check"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bc-root", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--route", choices=list(EPOCHS))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--pilot-traces", type=int, default=0)
    parser.add_argument("--indices", default="0,983")
    parser.add_argument("--calibration-label", default="pilot")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    if args.stage == "critic-check":
        critic_check(root, args.device)
    elif args.stage == "collect-critic":
        collect_critic(root, critic_tasks(read(root/"protocol.json"))[args.task])
    elif args.stage == "train-critic":
        train_critic(root, args.route, args.device)
    elif args.stage == "prepare":
        prepare(root, args.bc_root, args.source)
    elif args.stage == "calibrate":
        calibrate(root, args.model, [int(v) for v in args.indices.split(",")], args.calibration_label)
    elif args.stage == "preflight":
        from .burstgpt_preflight import preflight
        preflight(root)
    elif args.stage == "train":
        train(root, args.route, args.seed, args.rounds, args.pilot_traces,
              args.calibration_label, args.resume, args.device)
    elif args.stage == "worker-check":
        worker_check(root)
    elif args.stage == "orchestration-check":
        from .burstgpt_preflight import orchestration_check
        orchestration_check(root)
    else:
        finalize(root, args.route, args.seed, args.calibration_label)


if __name__ == "__main__":
    main()
