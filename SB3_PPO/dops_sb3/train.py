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
from .execution import EPISODE_STEPS, ROOT, SEGMENTS, full_replay, make_evaluator


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def evaluate_policy(model, env):
    obs, _ = env.reset()
    rewards, actions = [], []
    for _ in range(EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(int(action))
        actions.append(int(action))
        rewards.append(reward)
    assert terminated
    return {"reward": float(sum(rewards)), "latency_s": info["latency_s"],
            "speedup": info["speedup"], "actions": actions}


def baselines(evaluator, normalization_s, random_episodes, seed):
    fixed = {
        "all_npu": [[0] * 9] * SEGMENTS,
        "all_pim": [[1] * 9] * SEGMENTS,
        "prefill_npu_decode_pim": [[0] * 9] + [[1] * 9] * (SEGMENTS - 1),
    }
    result = {}
    for name, plans in fixed.items():
        latency = full_replay(evaluator, plans).now
        result[name] = {"latency_s": latency, "reward": -latency / normalization_s,
                        "speedup": normalization_s / latency}
    rng = np.random.default_rng(seed)
    latencies = [full_replay(evaluator, rng.integers(0, 2, (SEGMENTS, 9)).tolist()).now
                 for _ in range(random_episodes)]
    result["random"] = {"episodes": random_episodes, "latencies_s": latencies,
                        "reward_mean": -float(np.mean(latencies)) / normalization_s,
                        "latency_mean_s": float(np.mean(latencies)),
                        "latency_std_s": float(np.std(latencies))}
    return result


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
    if args.smoke:
        config.update(total_timesteps=512, n_steps=256, eval_freq=256, checkpoint_freq=256)
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    write_json(out / "ppo_config.json", config)
    write_json(out / "versions.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        **{name: version(name) for name in ("stable-baselines3", "gymnasium", "torch", "numpy", "tensorboard")},
    })
    write_json(out / "environment_config.json", json.loads((ROOT / "configs/environment.json").read_text()))
    evaluator = make_evaluator()
    env = PlacementEnv(evaluator)
    normalization = env.normalization_s
    write_json(out / "normalization.json", {"C_s": normalization, "source": "cold all_npu full request"})
    eval_env = Monitor(PlacementEnv(evaluator, normalization), str(out / "eval.monitor.csv"),
                       info_keywords=("latency_s", "speedup"))
    train_env = Monitor(env, str(out / "train.monitor.csv"), info_keywords=("latency_s", "speedup"))
    comparisons = baselines(evaluator, normalization, 2 if args.smoke else 20, config["seed"])
    write_json(out / "baselines.json", comparisons)
    total_timesteps = config.pop("total_timesteps")
    eval_freq = config.pop("eval_freq")
    checkpoint_freq = config.pop("checkpoint_freq")
    model = PPO("MlpPolicy", train_env, verbose=1, **config)
    model.set_logger(configure(str(out), ["stdout", "csv", "tensorboard"]))
    initial = evaluate_policy(model, eval_env)
    write_json(out / "initial_policy.json", initial)
    model.save(out / "initial_model")
    evaluation = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=eval_freq,
                              deterministic=True, best_model_save_path=str(out / "best"),
                              log_path=str(out / "eval"))
    checkpoints = CheckpointCallback(checkpoint_freq, str(out / "checkpoints"), name_prefix="ppo")
    begin = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=[LatencyCallback(), evaluation, checkpoints])
    # SB3 在下一轮采样后才 dump 上次更新的 loss；这里也落盘最后一次原生更新日志。
    model.logger.record("time/total_timesteps", model.num_timesteps)
    model.logger.dump(step=model.num_timesteps)
    model.save(out / "last_model")
    training_wall_s = time.perf_counter() - begin
    final = evaluate_policy(model, eval_env)
    best_path = out / "best/best_model.zip"
    best = evaluate_policy(PPO.load(best_path, device="cpu"), eval_env) if best_path.exists() else None
    write_json(out / "summary.json", {
        "smoke": args.smoke, "requested_timesteps": total_timesteps,
        "actual_timesteps": model.num_timesteps, "episode_steps": EPISODE_STEPS,
        "training_wall_s": training_wall_s,
        "env_steps_per_s": model.num_timesteps / training_wall_s,
        "normalization_s": normalization, "initial": initial, "final": final, "best": best,
    })
    from .plot import plot_run
    plot_run(out)
    train_env.close()
    eval_env.close()
    print(f"Saved run: {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="New output directory")
    parser.add_argument("--smoke", action="store_true", help="512-step integration test; not a learning result")
    parser.add_argument("--total-timesteps", type=int, help="Override formal training budget")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
