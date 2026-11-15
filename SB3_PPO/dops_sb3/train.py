"""直接调用 SB3 PPO、Monitor、EvalCallback 与原生日志。"""
import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from .env import PlacementEnv
from .execution import (DECODE_LENGTH, EPISODE_STEPS, OPERATORS, ROOT, SEGMENTS,
                        full_replay, make_evaluator, segment_count, workload_key)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_policy(model, env, workload=None):
    obs, _ = env.reset(options={"workload": workload} if workload is not None else None)
    rewards, actions = [], []
    for _ in range(env.unwrapped.episode_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        actions.extend(np.asarray(action).reshape(-1).astype(int).tolist())
        rewards.append(reward)
    assert terminated
    return {"reward": float(sum(rewards)), "latency_s": info["latency_s"],
            "speedup": info["speedup"], "actions": actions}


def evaluate_workloads(model, env, workloads):
    results = {workload_key(w): {**w, **evaluate_policy(model, env, w)} for w in workloads}
    reward = float(np.mean([r["reward"] for r in results.values()]))
    return {"reward": reward, "normalized_latency": -reward, "workloads": results}


def baselines(evaluator, normalization_s, random_episodes, seed, prompt_length=64, batch=1, decode_length=DECODE_LENGTH):
    segments = segment_count(decode_length)
    fixed = {
        "all_npu": [[0] * 9] * segments,
        "all_pim": [[1] * 9] * segments,
        "prefill_npu_decode_pim": [[0] * 9] + [[1] * 9] * (segments - 1),
    }
    result = {}
    for name, plans in fixed.items():
        latency = full_replay(evaluator, plans, prompt_length, batch, decode_length).now
        if name == "all_npu" and normalization_s is None:
            normalization_s = latency
        result[name] = {"latency_s": latency, "reward": -latency / normalization_s,
                        "speedup": normalization_s / latency}
    if random_episodes:
        rng = np.random.default_rng(seed)
        latencies = [full_replay(evaluator, rng.integers(0, 2, (segments, 9)).tolist(),
                                prompt_length, batch, decode_length).now for _ in range(random_episodes)]
        result["random"] = {"episodes": random_episodes, "latencies_s": latencies,
                            "reward_mean": -float(np.mean(latencies)) / normalization_s,
                            "latency_mean_s": float(np.mean(latencies)),
                            "latency_std_s": float(np.std(latencies))}
    return result


class WorkloadEvalCallback(BaseCallback):
    """明确遍历固定工况集；评估不使用训练环境的随机数。"""

    def __init__(self, env, workloads, frequency, out):
        super().__init__()
        self.env, self.workloads, self.frequency, self.out = env, workloads, frequency, out
        self.history = []
        self.complete_episodes = 0
        self.best_reward = -np.inf
        (out / "eval").mkdir()
        (out / "best").mkdir()

    def _on_step(self):
        self.complete_episodes += sum("episode" in info for info in self.locals["infos"])
        if self.n_calls % self.frequency:
            return True
        result = evaluate_workloads(self.model, self.env, self.workloads)
        self.history.append({"timesteps": self.num_timesteps, "complete_episodes": self.complete_episodes, **result})
        write_json(self.out / "eval/workloads.json", self.history)
        np.savez(self.out / "eval/evaluations.npz",
                 timesteps=[r["timesteps"] for r in self.history],
                 results=[[v["reward"] for v in r["workloads"].values()] for r in self.history],
                 latencies_s=[[v["latency_s"] for v in r["workloads"].values()] for r in self.history],
                 workload_ids=[workload_key(w) for w in self.workloads])
        self.logger.record("eval/mean_reward", result["reward"])
        self.logger.record("eval/normalized_latency", result["normalized_latency"])
        for key, values in result["workloads"].items():
            self.logger.record(f"eval/{key}/latency_s", values["latency_s"])
        if result["reward"] > self.best_reward:
            self.best_reward = result["reward"]
            self.model.save(self.out / "best/best_model")
        self.logger.dump(self.num_timesteps)
        return True


class LatencyCallback(BaseCallback):
    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.logger.record_mean("rollout/latency_s", info["latency_s"])
                self.logger.record_mean("rollout/speedup", info["speedup"])
        return True


def train(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = json.loads((ROOT / "configs/ppo.json").read_text())
    episode_steps = SEGMENTS if args.variant in ("C", "D") else EPISODE_STEPS
    workload_config = (json.loads(args.workloads.read_text()) if args.workloads is not None else None)
    known_decode = workload_config is not None and "decode_len" in workload_config["train"][0]
    if workload_config is not None:
        write_json(out / "workloads.json", workload_config)
    if known_decode:
        total_steps = workload_config["training_timesteps"][args.variant]
        period = total_steps // workload_config["periodic_evaluations"]
    else:
        if workload_config:
            assert workload_config["decode_len"] == DECODE_LENGTH
        total_steps = (workload_config["training_episodes"] if workload_config else 10_000) * episode_steps
        period = (workload_config["evaluation_episodes"] if workload_config else 100) * episode_steps
    config.update(seed=args.seed, total_timesteps=total_steps, eval_freq=period, checkpoint_freq=period)
    if args.smoke:
        config.update(total_timesteps=512, n_steps=256, eval_freq=256, checkpoint_freq=256)
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    write_json(out / "ppo_config.json", config)
    write_json(out / "versions.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        **{name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy", "tensorboard")},
    })
    environment_config = json.loads((ROOT / "configs/environment.json").read_text())
    write_json(out / "environment_config.json", environment_config)
    experiment = {
        "variant": args.variant, "seed": args.seed,
        "episode_unit": "synchronous batch" if workload_config else "single request",
    }
    if known_decode:
        choices_per_segment = 1 if args.variant in ("C", "D") else len(OPERATORS)
        experiment.update(requested_timesteps=config["total_timesteps"], evaluation_interval_steps=config["eval_freq"],
                          episode_steps_by_workload={workload_key(w): segment_count(w["decode_len"]) * choices_per_segment
                                                     for w in workload_config["train"] + workload_config["heldout"]})
    else:
        experiment.update(episode_steps=episode_steps, requested_episodes=config["total_timesteps"] / episode_steps,
                          evaluation_episodes=config["eval_freq"] / episode_steps)
        if not workload_config:
            experiment.update(requested_requests=experiment["requested_episodes"],
                              evaluation_requests=experiment["evaluation_episodes"])
    write_json(out / "experiment.json", experiment)
    evaluator = make_evaluator()
    if workload_config:
        train_workloads = workload_config["train"]
        all_workloads = train_workloads + workload_config["heldout"]
        if args.fixed_baselines is not None:
            shared = json.loads(args.fixed_baselines.read_text())
            if shared["environment"] != environment_config or shared["workloads"] != all_workloads:
                raise ValueError("Fixed baselines do not match this environment and workload set")
            comparisons = shared["baselines"]
        else:
            comparisons = {workload_key(w): baselines(evaluator, None, 0, args.seed, w["prefill_len"], w["batch"],
                                                      w.get("decode_len", DECODE_LENGTH)) for w in all_workloads}
        normalization = {key: values["all_npu"]["latency_s"] for key, values in comparisons.items()}
        eval_base = PlacementEnv(evaluator, variant=args.variant, workloads=all_workloads, normalizations=normalization)
        env = PlacementEnv(evaluator, variant=args.variant, workloads=train_workloads,
                           normalizations=normalization)
        write_json(out / "normalization.json", {"C_by_workload_s": normalization,
                   "source": "cold all_npu full batch for each workload"})
    else:
        env = PlacementEnv(evaluator, variant=args.variant)
        normalization = env.normalization_s
        eval_base = PlacementEnv(evaluator, normalization, args.variant)
        comparisons = baselines(evaluator, normalization, 2 if args.smoke else 20, config["seed"])
        write_json(out / "normalization.json", {"C_s": normalization, "source": "cold all_npu full request"})
    info_keys = ("latency_s", "speedup", "workload", "prefill_len", "batch")
    if known_decode:
        info_keys += ("decode_len",)
    eval_env = Monitor(eval_base, str(out / "eval.monitor.csv"), info_keywords=info_keys)
    train_env = Monitor(env, str(out / "train.monitor.csv"), info_keywords=info_keys)
    write_json(out / "baselines.json", comparisons)
    total_timesteps = config.pop("total_timesteps")
    eval_freq = config.pop("eval_freq")
    checkpoint_freq = config.pop("checkpoint_freq")
    model = PPO("MlpPolicy", train_env, verbose=1, **config)
    model.set_logger(configure(str(out), ["stdout", "csv", "tensorboard"]))
    def assess(policy):
        return (evaluate_workloads(policy, eval_env, train_workloads) if workload_config else
                evaluate_policy(policy, eval_env))

    initial = assess(model)
    write_json(out / "initial_policy.json", initial)
    model.save(out / "initial_model")
    evaluation = (WorkloadEvalCallback(eval_env, train_workloads, eval_freq, out) if workload_config else
                  EvalCallback(eval_env, n_eval_episodes=1, eval_freq=eval_freq,
                               deterministic=True, best_model_save_path=str(out / "best"),
                               log_path=str(out / "eval")))
    checkpoints = CheckpointCallback(checkpoint_freq, str(out / "checkpoints"), name_prefix="ppo")
    begin = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=[LatencyCallback(), evaluation, checkpoints])
    # SB3 在下一轮采样后才 dump 上次更新的 loss；这里也落盘最后一次原生更新日志。
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.dump(step=model.num_timesteps)
    model.save(out / "last_model")
    training_wall_s = time.perf_counter() - begin
    final = assess(model)
    best_path = out / "best/best_model.zip"
    best = assess(PPO.load(best_path, device="cpu")) if best_path.exists() else None
    budget_path = out / "checkpoints" / f"ppo_{total_timesteps}_steps.zip"
    budget = assess(PPO.load(budget_path, device="cpu")) if budget_path.exists() else None
    summary = {
        "variant": args.variant, "seed": args.seed,
        "smoke": args.smoke, "requested_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps,
        "rollouts": model.num_timesteps // config["n_steps"],
        "training_epochs": model._n_updates,
        "optimizer_steps": model._n_updates * int(np.ceil(config["n_steps"] / config["batch_size"])),
        "policy_parameters": sum(p.numel() for p in model.policy.parameters()),
        "training_wall_s": training_wall_s,
        "env_steps_per_s": model.num_timesteps / training_wall_s,
        "normalization_s": normalization, "initial": initial, "final": final, "best": best,
        "budget": budget,
    }
    if not known_decode:
        summary.update(episode_steps=episode_steps, requested_episodes=total_timesteps / episode_steps)
    if workload_config:
        import csv
        with (out / "train.monitor.csv").open() as handle:
            rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
        summary["normalization_by_workload_s"] = summary.pop("normalization_s")
        summary["complete_episodes"] = len(rows)
        summary["unfinished_steps"] = model.num_timesteps - sum(int(row["l"]) for row in rows)
        summary["complete_requests"] = sum(int(row["batch"]) for row in rows)
        summary["workload_counts"] = {workload_key(w): sum(row["workload"] == workload_key(w) for row in rows)
                                      for w in train_workloads}
        summary["generalization"] = evaluate_workloads(model, eval_env, workload_config["heldout"])
    else:
        completed, unfinished = divmod(model.num_timesteps, episode_steps)
        summary.update(requested_requests=total_timesteps / episode_steps, complete_requests=completed,
                       complete_episodes=completed, unfinished_steps=unfinished)
    write_json(out / "summary.json", summary)
    from .plot import plot_run
    plot_run(out)
    train_env.close()
    eval_env.close()
    print(f"Saved run: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--variant", choices=("A", "B", "C", "D"), default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="512-step integration test; not a learning result")
    parser.add_argument("--total-timesteps", type=int, help="Override formal training budget")
    parser.add_argument("--workloads", type=Path, help="Mixed workload configuration; omitted keeps the original experiment")
    parser.add_argument("--fixed-baselines", type=Path, help="Reuse fixed-policy replays from the same environment and workloads")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
