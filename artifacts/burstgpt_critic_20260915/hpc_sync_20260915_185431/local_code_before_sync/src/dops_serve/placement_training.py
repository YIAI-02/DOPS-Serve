from __future__ import annotations

import json
import time
from pathlib import Path

import torch



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
    values = critic(batch["state"], batch["nodes"], batch["actions"]) #调用forward
    mask = batch["learned"]
    return ((values - returns[:, None]).square() * mask).sum(1) / mask.sum(1)


def burst_worker(connection, root, model, decode_feature, seed, rng_state=None):
    """One persistent environment per model/load group; policies fixed per rollout.
    受到 BurstWorkers 管理的进程
    """
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
        while True: #在这里进入循环，等待收到connection 的 command
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
    '''
    管理采样子进程
    '''
    def __init__(self, root, groups, decode_feature, seed, rng_states=None):
        import multiprocessing
        '''
        创建独立python进程，spawn启动新的python interpreter，Pipe一条双向通信通道，两端可以发送和接收 Python 对象或 tensor
        ppo 需要三个负载档位时，可以调用3个子进程
        '''
        context = multiprocessing.get_context("spawn")
        self.groups, self.connections, self.processes = groups, [], []
        for i, (model, load) in enumerate(groups):
            parent, child = context.Pipe()
            process = context.Process(target=burst_worker, args=(child, str(root), model,
                decode_feature, seed + 1009 * (i + 1), None if rng_states is None else rng_states[i])) #子进程启动函数burst_worker
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
