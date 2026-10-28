from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import torch

from .placement_policy import PlacementActor, LearnedPlacement, policy_inputs


def static_tensors(inputs, device="cpu"):
    return (torch.tensor([inputs["state"]], dtype=torch.float32, device=device),
            torch.tensor([inputs["nodes"]], dtype=torch.float32, device=device),
            torch.tensor([inputs["masks"]], dtype=torch.bool, device=device),
            torch.tensor(inputs["fixed_from"], dtype=torch.long, device=device),
            torch.tensor([inputs["learned"]], dtype=torch.bool, device=device))


def node_returns(rows, bootstrap):
    running = float(bootstrap)
    for row in reversed(rows):
        if row["terminal"]:
            running = 0.0
        running += row["reward"]
        row["returns"] = [running] * len(row["actions"])
    return rows


def critic_bootstrap(critic, inputs):
    state, nodes, masks, fixed, learned = static_tensors(inputs)
    # All actions before the first learnable node are prescribed. Future
    # placeholders cannot enter its value because the GRU is right shifted.
    actions = torch.zeros((1, nodes.size(1)), dtype=torch.long)
    first = int(learned[0].nonzero()[0])
    for index in range(first):
        if int(fixed[index]) >= 0:
            actions[:, index] = actions[:, int(fixed[index])]
    return float(critic(state, nodes, actions)[0, first])


def cpu_weights(module):
    return {key: value.detach().cpu() for key, value in module.state_dict().items()}


def sample_nodes(actor, critic, env):
    started = time.perf_counter()
    inputs = env.inputs()
    state, nodes, masks, fixed, learned = static_tensors(inputs)
    with torch.no_grad():
        actions = actor.sample(state, nodes, masks, fixed, False)
        _, logits = actor(state, nodes, actions, masks, learned)
        logdist = logits.log_softmax(-1).masked_fill(~masks, 0)
        logp = logdist.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        values = critic(state, nodes, actions)
    plan = env.evaluator.compiler.compile(actions[0].tolist())
    decision_s = time.perf_counter() - started
    env.decision_s += decision_s
    dt = env.step(plan)
    return dict(inputs=inputs, actions=actions[0].tolist(), old_logp=logp[0].tolist(),
                old_logdist=logdist[0].tolist(), values=values[0].tolist(),
                duration_s=dt, terminal=env.done, decision_s=decision_s)


def sample_critic_nodes(actor, env):
    started = time.perf_counter()
    phase = env.pending[0]
    state, nodes, masks, fixed, learned = static_tensors(env.inputs())
    with torch.no_grad():
        actions = actor.sample(state, nodes, masks, fixed, False)
    plan = env.evaluator.compiler.compile(actions[0].tolist())
    env.decision_s += time.perf_counter() - started
    duration = env.step(plan)
    return dict(state=state[0], nodes=nodes[0], actions=actions[0], learned=learned[0],
                phase=torch.tensor(0 if phase == "prefill" else 1),
                duration=torch.tensor(duration, dtype=torch.float64), terminal=torch.tensor(env.done))


def complete_critic_returns(durations, terminals, reward_scale):
    if not bool(terminals[-1]) or bool(terminals[:-1].any()):
        raise ValueError("MC labels require exactly one complete trace")
    return -durations.double().flip(0).cumsum(0).flip(0) / reward_scale


def critic_plan_mse(critic, batch, returns):
    values = critic(batch["state"], batch["nodes"], batch["actions"])
    mask = batch["learned"]
    return ((values - returns[:, None]).square() * mask).sum(1) / mask.sum(1)


def burst_worker(connection, root, model, decode_feature, seed, rng_state=None):
    """One persistent environment per model/load group; policies fixed per rollout."""
    import traceback
    from .placement_policy import StaticPlacementActor, StaticPlacementCritic
    from .placement_evaluator import BurstPlacementEvaluator
    from .replay import BurstPlacementReplay
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    evaluator = BurstPlacementEvaluator(root / "backend/DOPS", root / "configs" / f"{model}.json",
        protocol["parameters"], protocol["kv_headroom_bytes"])
    base = StaticPlacementActor(56, 35)
    actor = torch.jit.script(base.eval())
    critic = StaticPlacementCritic(base).eval()
    if rng_state is not None:
        torch.set_rng_state(rng_state)
    env = None
    try:
        while True:
            command, payload = connection.recv()
            if command == "close":
                return
            if command == "start":
                env = BurstPlacementReplay(payload, evaluator, decode_feature)
                connection.send({"ready": True})
            elif command == "rng":
                connection.send(torch.get_rng_state())
            elif command == "actor_state":
                connection.send(cpu_weights(actor))
            elif command == "collect_critic":
                if payload["actor"] is not None:
                    actor.load_state_dict(payload["actor"])
                    for parameter in actor.parameters():
                        parameter.requires_grad_(False)
                rows = []
                for _ in range(payload["plans"]):
                    rows.append(sample_critic_nodes(actor, env))
                    if env.done:
                        break
                batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
                connection.send(dict(batch=batch, done=env.done, metrics=env.metrics() if env.done else None))
            elif command == "rollout":
                actor.load_state_dict(payload["actor"])
                critic.load_state_dict(payload["critic"])
                rows = []
                simulator_start = env.simulator_s
                started = time.perf_counter()
                for _ in range(payload["plans"]):
                    row = sample_nodes(actor, critic, env)
                    row["reward"] = -row["duration_s"] / payload["reward_scale_s"]
                    rows.append(row)
                    if env.done:
                        break
                with torch.no_grad():
                    bootstrap = 0.0 if env.done else critic_bootstrap(critic, env.inputs())
                node_returns(rows, bootstrap)
                connection.send(dict(rows=rows, done=env.done, bootstrap=bootstrap,
                    sample_wall_s=time.perf_counter()-started, simulator_s=env.simulator_s-simulator_start,
                    metrics=env.metrics() if env.done else None))
            elif command == "evaluate":
                if payload is not None:
                    actor.load_state_dict(payload)
                while not env.done:
                    started = time.perf_counter()
                    if payload is None:
                        plan = evaluator.pd_plan(env.snapshot, env.pending[0], env.shapes())
                    else:
                        state, nodes, masks, fixed, _ = static_tensors(env.inputs())
                        with torch.no_grad():
                            actions = actor.generate(state, nodes, masks, fixed)
                        plan = evaluator.compiler.compile(actions[0].tolist())
                    env.decision_s += time.perf_counter() - started
                    env.step(plan)
                connection.send(env.metrics())
    except BaseException:
        connection.send({"worker_error": traceback.format_exc()})
        raise
    finally:
        connection.close()


class BurstWorkers:
    def __init__(self, root, groups, decode_feature, seed, rng_states=None):
        import multiprocessing
        context = multiprocessing.get_context("spawn")
        self.groups, self.connections, self.processes = groups, [], []
        for i, (model, load) in enumerate(groups):
            parent, child = context.Pipe()
            process = context.Process(target=burst_worker, args=(child, str(root), model,
                decode_feature, seed + 1009 * (i + 1), None if rng_states is None else rng_states[i]))
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)

    def call(self, command, payloads):
        for connection, payload in zip(self.connections, payloads):
            if payload is not Ellipsis:
                connection.send((command, payload))
        results = []
        for connection, payload in zip(self.connections, payloads):
            if payload is Ellipsis:
                results.append(None)
                continue
            result = connection.recv()
            if isinstance(result, dict) and "worker_error" in result:
                raise RuntimeError(result["worker_error"])
            results.append(result)
        return results

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                connection.send(("close", None))
            connection.close()
        for process in self.processes:
            process.join()


def update_static_ppo(actor, critic, actor_optimizer, critic_optimizer, groups, settings, rng):
    """Conditional node PPO; fixed group weights, independent actor/critic graphs."""
    device = next(actor.parameters()).device
    for rows in groups:
        if not rows:
            continue
        values = torch.tensor([v for r in rows for v, keep in
            zip([g-v for g, v in zip(r["returns"], r["values"])], r["inputs"]["learned"]) if keep])
        mean, std = float(values.mean()), float(values.std(unbiased=False).clamp_min(1e-8))
        for row in rows:
            row["advantages"] = [(g-v-mean)/std for g, v in zip(row["returns"], row["values"])]

    def tensors(rows):
        return (torch.tensor([r["inputs"]["state"] for r in rows], device=device),
            torch.tensor([r["inputs"]["nodes"] for r in rows], device=device),
            torch.tensor([r["actions"] for r in rows], dtype=torch.long, device=device),
            torch.tensor([r["inputs"]["masks"] for r in rows], dtype=torch.bool, device=device),
            torch.tensor([r["inputs"]["learned"] for r in rows], dtype=torch.bool, device=device))

    def mean_kl():
        total = 0.0
        with torch.no_grad():
            for rows in groups:
                if not rows:
                    continue
                kl_sum, count = 0.0, 0
                for lo in range(0, len(rows), settings["mini_plans"]):
                    batch = rows[lo:lo+settings["mini_plans"]]
                    state, nodes, actions, masks, learned = tensors(batch)
                    _, logits = actor(state, nodes, actions, masks, learned)
                    new = logits.log_softmax(-1).masked_fill(~masks, 0)
                    old = torch.tensor([r["old_logdist"] for r in batch], device=device)
                    kl = (old.exp().masked_fill(~masks, 0) * (old-new)).sum(-1)
                    kl_sum += float(kl[learned].sum())
                    count += int(learned.sum())
                total += kl_sum / count / len(groups)
        return total

    logs = []
    epochs_done = 0
    initial_kl = mean_kl()
    for epoch in range(settings["ppo_epochs"]):
        ordered = [list(rows) for rows in groups]
        for rows in ordered:
            rng.shuffle(rows)
        for lo in range(0, max(map(len, ordered)), settings["mini_plans"]):
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            record = dict(actor_loss=0.0, value_loss=0.0, entropy=0.0, clip_fraction=0.0)
            for rows in ordered:
                batch = rows[lo:lo+settings["mini_plans"]]
                if not batch:
                    continue
                state, nodes, actions, masks, learned = tensors(batch)
                _, logits = actor(state, nodes, actions, masks, learned)
                logdist = logits.log_softmax(-1).masked_fill(~masks, 0)
                new_logp = logdist.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                old_logp = torch.tensor([r["old_logp"] for r in batch], device=device)
                advantages = torch.tensor([r["advantages"] for r in batch], device=device)
                returns = torch.tensor([r["returns"] for r in batch], device=device)
                ratio = (new_logp-old_logp).exp()
                surrogate = torch.minimum(ratio*advantages,
                    ratio.clamp(1-settings["clip"], 1+settings["clip"])*advantages)
                entropy = -(logdist.exp().masked_fill(~masks, 0)*logdist).sum(-1)[learned].mean()
                actor_loss = -surrogate[learned].mean()
                ((actor_loss-settings["entropy"]*entropy)/len(groups)).backward()
                values = critic(state, nodes, actions)
                value_loss = (values-returns).square()[learned].mean()
                (value_loss/len(groups)).backward()
                for name, value in dict(actor_loss=actor_loss, value_loss=value_loss, entropy=entropy,
                    clip_fraction=((ratio-1).abs()>settings["clip"]).float()[learned].mean()).items():
                    record[name] += float(value.detach())/len(groups)
            record["actor_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), settings["max_grad_norm"]))
            record["critic_grad_norm"] = float(torch.nn.utils.clip_grad_norm_(critic.parameters(), settings["max_grad_norm"]))
            actor_optimizer.step()
            critic_optimizer.step()
            logs.append(record)
        epochs_done = epoch+1
        kl = mean_kl()
        if kl >= settings["target_kl"]:
            break
    return dict(initial_kl=initial_kl, mean_kl=kl, epochs=epochs_done,
                **{key: sum(row[key] for row in logs)/len(logs) for key in logs[0]})


def cost_weights(costs, temperature):
    costs = torch.as_tensor(costs, dtype=torch.float64)
    # Relative costs compare only candidates sharing the exact same anchor.
    return torch.softmax(-(costs / costs.min() - 1) / temperature, dim=0)


def block_returns(windows, bootstrap):
    """Gamma=1, lambda=1; only the last block receives each window reward."""
    returns = []
    running = float(bootstrap)
    for window in reversed(windows):
        if window["terminal"]:
            running = 0.0
        running += window["reward"]
        returns.append([running] * len(window["values"]))
    return list(reversed(returns))


def ppo_loss(new_log_probs, old_log_probs, advantages, values, returns, entropies, *, clip, value_coefficient, entropy_coefficient):
    ratios = (new_log_probs - old_log_probs).exp()
    surrogate = torch.minimum(ratios * advantages, ratios.clamp(1 - clip, 1 + clip) * advantages)
    actor_loss, value_loss, entropy = -surrogate.mean(), (values - returns).square().mean(), entropies.mean()
    loss = actor_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    return loss, {"actor_loss": float(actor_loss.detach()), "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                  "return_mean": float(returns.mean()), "return_abs_mean": float(returns.abs().mean()),
                  "value_mean": float(values.mean().detach()), "value_abs_mean": float(values.abs().mean().detach()),
                  "clip_fraction": float(((ratios - 1).abs() > clip).float().mean().detach()), "approx_kl": float(((ratios - 1) - ratios.log()).mean().detach())}


def load_actor(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor = PlacementActor(**checkpoint["dimensions"])
    actor.load_state_dict(checkpoint["state_dict"])
    return actor, checkpoint


def _save(actor, path, **metadata):
    torch.save({"state_dict": actor.state_dict(), "dimensions": actor.dimensions, **metadata}, path)


def _anchor_files(directory, exclude_model, models=None):
    summary = json.loads((directory / "reevaluation_summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("BC requires complete historical reevaluation")
    files = {split: [] for split in ("train", "val")}
    for item in summary["files"]:
        if models is not None and not any(Path(item["path"]).name.startswith(model + "_") for model in models):
            continue
        if exclude_model and Path(item["path"]).name.startswith(exclude_model + "_"):
            continue
        files[item["split"]].append(Path(item["path"]))
    if not all(files.values()):
        raise ValueError("BC requires distinct training and validation conditions")
    return files


def _bc_validation(actor, files, programs):
    regret, nll, anchors = 0.0, 0.0, 0
    with torch.no_grad():
        for path in files:
            for row in map(json.loads, path.open()):
                scores = [float(actor.sequence(row["inputs"], programs[c["placement_id"]])["log_probs"].sum()) for c in row["candidates"]]
                best = max(range(len(scores)), key=scores.__getitem__)
                minimum = min(c["cost_s"] for c in row["candidates"])
                regret += row["candidates"][best]["cost_s"] / minimum - 1
                nll -= sum(scores) / len(scores) / sum(map(len, row["inputs"]["nodes"]))
                anchors += 1
    return {"candidate_regret": regret / anchors, "unweighted_nll": nll / anchors, "anchors": anchors}


def train_bc(artifact_dir, output_dir, *, seed=7, epochs=3, temperature=1.0, learning_rate=3e-4, exclude_model=None, models=None, checkpoint_path=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    directory, destination = Path(artifact_dir), Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = _anchor_files(directory, exclude_model, models)
    programs = {row["placement_id"]: row["block_actions"] for row in map(json.loads, (directory / "placements.jsonl").open())}
    with files["train"][0].open() as handle:
        sample = json.loads(next(handle))["inputs"]
    actor = PlacementActor(len(sample["state"]), len(sample["nodes"][0][0]))
    source = None
    if checkpoint_path is not None:
        actor, source = load_actor(checkpoint_path)
        if (source["stage"], source["seed"], source["models"], source["temperature"], source["learning_rate"], source["exclude_model"]) != ("BC", seed, models, temperature, learning_rate, exclude_model):
            raise ValueError("BC resume must retain its model selection, seed and training settings")
    parameter_count = sum(p.numel() for p in actor.parameters())
    if parameter_count >= 1_000_000:
        raise ValueError("actor and value exceed the proposal parameter budget")
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    history, best = [], (float("inf"), float("inf"))
    first_epoch = 0
    if source is not None:
        optimizer.load_state_dict(source["optimizer_state_dict"])
        torch.set_rng_state(source["torch_rng_state"])
        rng.setstate(source["python_rng_state"])
        first_epoch = source["epoch"]
        history = list(source["history"])
    with (destination / "bc_curve.jsonl").open("w") as curve:
        for epoch in range(first_epoch, first_epoch + epochs):
            started = time.perf_counter()
            paths = list(files["train"])
            rng.shuffle(paths)
            total_loss, count = 0.0, 0
            for path in paths:
                rows = [json.loads(line) for line in path.open()]
                rng.shuffle(rows)
                for row in rows:
                    weights = cost_weights([c["cost_s"] for c in row["candidates"]], temperature)
                    node_count = sum(map(len, row["inputs"]["nodes"]))
                    optimizer.zero_grad()
                    loss_value = 0.0
                    for weight, candidate in zip(weights, row["candidates"]):
                        result = actor.sequence(row["inputs"], programs[candidate["placement_id"]])
                        loss = -weight.float() * result["log_probs"].sum() / node_count
                        loss.backward()
                        loss_value += float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss_value
                    count += 1
                    if count % 500 == 0:
                        print(json.dumps({"bc_epoch": epoch + 1, "anchors": count, "loss": total_loss / count}), flush=True)
            validation = _bc_validation(actor, files["val"], programs)
            record = {"epoch": epoch + 1, "training_loss": total_loss / count, "train_anchors": count, "optimizer_steps": count, "validation": validation, "wall_s": time.perf_counter() - started}
            history.append(record)
            curve.write(json.dumps(record) + "\n")
            curve.flush()
            rank = (validation["candidate_regret"], validation["unweighted_nll"])
            metadata = dict(stage="BC", seed=seed, epoch=epoch + 1, temperature=temperature,
                            learning_rate=learning_rate, exclude_model=exclude_model, models=models,
                            validation=validation, optimizer_state_dict=optimizer.state_dict(),
                            torch_rng_state=torch.get_rng_state(), python_rng_state=rng.getstate(), history=history)
            _save(actor, destination / "bc_last.pt", **metadata)
            if rank < best:
                best = rank
                _save(actor, destination / "bc.pt", **metadata)
            print(json.dumps(record), flush=True)
    result = {"stage": "BC", "seed": seed, "epochs": first_epoch + epochs, "temperature": temperature, "parameter_count": parameter_count, "exclude_model": exclude_model, "models": models, "checkpoint": str(destination / "bc.pt"), "last_checkpoint": str(destination / "bc_last.pt"), "history": history}
    (destination / "bc_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def update_ppo(actor, optimizer, windows, bootstrap, settings):
    device = next(actor.parameters()).device
    returns = [torch.tensor(row, dtype=torch.float32, device=device) for row in block_returns(windows, bootstrap)]
    advantages = [ret - torch.tensor(row["values"], device=device) for ret, row in zip(returns, windows)]
    combined = torch.cat(advantages)
    mean, std = combined.mean(), combined.std(unbiased=False).clamp_min(1e-8)
    advantages = [(value - mean) / std for value in advantages]
    logs, norms = [], []
    for _ in range(settings["ppo_epochs"]):
        optimizer.zero_grad()
        for row, ret, advantage in zip(windows, returns, advantages):
            result = actor.pair(row["inputs"], row["programs"])
            loss, log = ppo_loss(result["log_probs"], torch.tensor(row["log_probs"], device=device), advantage, result["values"], ret, result["entropies"], clip=settings["clip"], value_coefficient=settings["value_coefficient"], entropy_coefficient=settings["entropy_coefficient"])
            (loss / len(windows)).backward()
            logs.append({"loss": float(loss.detach()), **log})
        norms.append(float(torch.nn.utils.clip_grad_norm_(actor.parameters(), settings["max_grad_norm"])))
        optimizer.step()
    return {"gradient_norm": sum(norms) / len(norms), **{key: sum(row[key] for row in logs) / len(logs) for key in logs[0]}}


def throughput_schedule(protocol, epoch_index, seed):
    models = list(protocol.get("training_models", protocol["backend_config"]["models"]))
    rotation = epoch_index % len(models)
    models = models[rotation:] + models[:rotation]
    loads = protocol["training_load_multipliers"]
    offset = (7, 17, 27).index(seed)
    return [(model, trace, loads[(index + epoch_index + offset) % len(loads)])
            for index, trace in enumerate(protocol["train_trace_ids"]) for model in models]



def run_update(actor, optimizer, windows, bootstrap, settings, runtime):
    """Sampling stays on the benchmark CPU; optionally offload only PPO updates."""
    device = runtime.get("update_device", "cpu")
    started = time.perf_counter()
    actor.to(device)
    if device != "cpu":
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
    torch.set_num_threads(runtime.get("update_threads", 1))
    log = update_ppo(actor, optimizer, windows, bootstrap, settings)
    actor.cpu()
    torch.set_num_threads(runtime.get("sample_threads", 1))
    return {**log, "update_wall_s": time.perf_counter() - started}


def train_ppo(backend_root, artifact_dir, config, checkpoint_path, output_dir, protocol, *, seed=7, exclude_model=None, trace_epochs=None):
    from .placement_evaluator import PlacementEvaluator
    from .placement_evaluation import load_traces, scaled_trace
    from .replay import PlacementReplay

    runtime = protocol.get("runtime", {})
    torch.set_num_threads(runtime.get("sample_threads", 1))
    if "validation_signature" in runtime:
        from .placement_evaluation import host_signature
        if host_signature(runtime["sample_threads"]) != runtime["validation_signature"]:
            raise ValueError("training sampler CPU/runtime differs from fixed validation")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    actor, checkpoint = load_actor(checkpoint_path)
    selected_models = protocol.get("training_models", config["models"])
    if checkpoint.get("models") is not None and checkpoint["models"] != selected_models:
        raise ValueError("checkpoint positive model selection differs from PPO")
    if checkpoint["exclude_model"] != exclude_model:
        raise ValueError("BC checkpoint model split differs from PPO")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if Path(checkpoint_path).resolve() == (destination / "ppo.pt").resolve():
        raise ValueError("PPO continuation requires a separate output directory")
    settings = protocol["ppo"]
    if protocol["backend_config"] != config or settings["gamma"] != 1.0 or settings["gae_lambda"] != 1.0:
        raise ValueError("PPO must use the calibrated backend and undiscounted block-return protocol")
    optimizer = torch.optim.Adam(actor.parameters(), lr=settings["learning_rate"])
    continuing = checkpoint["stage"] == "PPO"
    restore_optimizer = continuing
    if continuing and checkpoint["seed"] != seed:
        raise ValueError("PPO continuation must retain its original seed")
    if continuing and checkpoint["protocol"] != protocol:
        raise ValueError("PPO continuation must retain the original calibrated protocol")
    if restore_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        rng.setstate(checkpoint["python_rng_state"])
        if runtime.get("update_device") == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    traces = load_traces(artifact_dir, "train", protocol["train_trace_ids"])
    models = [m for m in selected_models if m != exclude_model]
    evaluators = {m: PlacementEvaluator(backend_root, destination / "backend", m, config) for m in models}
    schedule = [(model, trace_id) for model in models for trace_id in traces]
    updates = checkpoint["updates"] if continuing else 0
    totals = list(checkpoint["completed_traces"]) if continuing else []
    previous_epochs = max((row["epoch"] for row in totals), default=0)
    throughput = protocol.get("objective") == "throughput"
    training_wall_s = checkpoint.get("training_wall_s", 0.0)
    training_started = time.perf_counter()
    if restore_optimizer:
        schedule = [tuple(pair) for pair in checkpoint["schedule"]]
    epochs = settings["trace_epochs"] if trace_epochs is None else trace_epochs
    continuation = "exact optimizer/RNG resume" if restore_optimizer else "BC initialization"
    with (destination / "ppo_curve.jsonl").open("w") as curve:
        for epoch in range(previous_epochs, previous_epochs + epochs):
            if throughput:
                ordered = throughput_schedule(protocol, epoch, seed)
            else:
                rng.shuffle(schedule)
                ordered = [(model, trace_id, rng.choice(protocol["training_load_multipliers"])) for model, trace_id in schedule]
            for model, trace_id, load in ordered:
                trace_started = time.perf_counter()
                model_protocol = protocol["models"][model]
                records = scaled_trace(traces[trace_id], model_protocol["reference_rps"] * load)
                replay = PlacementReplay(records, evaluators[model], slo_s=model_protocol.get("slo_s"), max_batch=config["max_batch"], decode_window=config["decode_window"])
                selector = LearnedPlacement(actor, evaluators[model], deterministic=False)
                while not replay.done:
                    sample_started = time.perf_counter()
                    first_step = len(replay.step_results)
                    windows = []
                    rewards = []
                    for _ in range(settings["rollout_windows"]):
                        result = replay.run_window(selector, model_protocol["reward"])
                        rewards.append(result)
                        windows.append({**copy.deepcopy(selector.last), "reward": result["reward"], "terminal": result["terminal"]})
                        if replay.done:
                            break
                    bootstrap = 0.0
                    if not replay.done:
                        current = evaluators[model].compiler.project(replay.plans["prefill"]["device_by_node"])
                        inputs = policy_inputs(replay.observation(), evaluators[model], "prefill", current)
                        with torch.no_grad():
                            bootstrap = float(actor.sequence(inputs, deterministic=True)["values"][0])
                    sample_wall_s = time.perf_counter() - sample_started
                    log = run_update(actor, optimizer, windows, bootstrap, settings, runtime)
                    updates += 1
                    record = {"epoch": epoch + 1, "update": updates, "model": model, "trace_id": trace_id, "load": load, "terminal": replay.done, "windows": len(windows), "reward": sum(w["reward"] for w in windows),
                              "token_reward": sum(w["token_reward"] for w in rewards), "time_reward": sum(w["time_reward"] for w in rewards),
                              "sample_wall_s": sample_wall_s, "simulator_wall_s": sum(r["host_evaluator_s"] for r in replay.step_results[first_step:]),
                              "decision_wall_s": sum(w["decision_s"] for w in rewards),
                              "cumulative_requests": sum(r["completed"] for r in totals) + len(replay.completed),
                              "cumulative_windows": sum(r["windows"] for r in totals) + len(replay.windows),
                              "cumulative_training_wall_s": training_wall_s + time.perf_counter() - training_started, **log}
                    curve.write(json.dumps(record) + "\n")
                    curve.flush()
                    print(json.dumps(record), flush=True)
                totals.append({"epoch": epoch + 1, "model": model, "trace_id": trace_id, "load_multiplier": load, "completed": len(replay.completed), "windows": len(replay.windows)})
                print(json.dumps({"trace_complete": totals[-1], "wall_s": time.perf_counter() - trace_started}), flush=True)
            _save(actor, destination / "ppo.pt", stage="PPO", seed=seed, epoch=epoch + 1, updates=updates, exclude_model=exclude_model, models=models,
                  protocol=protocol, completed_traces=totals, optimizer_state_dict=optimizer.state_dict(),
                  torch_rng_state=torch.get_rng_state(), python_rng_state=rng.getstate(), schedule=schedule,
                  source_checkpoint=str(checkpoint_path), continuation=continuation,
                  training_wall_s=training_wall_s + time.perf_counter() - training_started,
                  **({"cuda_rng_state": torch.cuda.get_rng_state_all()} if runtime.get("update_device") == "cuda" else {}))
    result = {"stage": "PPO", "seed": seed, "updates": updates, "additional_trace_epochs": epochs, "continuation": continuation,
              "source_checkpoint": str(checkpoint_path), "completed_traces": totals, "checkpoint": str(destination / "ppo.pt"),
              "epoch": previous_epochs + epochs, "training_wall_s": training_wall_s + time.perf_counter() - training_started}
    (destination / "ppo_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
