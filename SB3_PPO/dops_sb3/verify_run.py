"""验收真实 SB3 输出：有限 loss、参数更新、模型重载和日志/图表落盘。"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .env import PlacementEnv
from .train import evaluate_policy, evaluate_workloads


def verify_run(out):
    summary = json.loads((out / "summary.json").read_text())
    with (out / "progress.csv").open() as f:
        rows = list(csv.DictReader(f))
    for key in ("train/loss", "train/policy_gradient_loss", "train/value_loss", "train/approx_kl"):
        values = [float(r[key]) for r in rows if r.get(key)]
        assert values and all(math.isfinite(v) for v in values), key
    events = EventAccumulator(str(out)).Reload()
    assert "train/value_loss" in events.Tags()["scalars"]
    assert "eval/mean_reward" in events.Tags()["scalars"]
    initial = PPO.load(out / "initial_model.zip", device="cpu")
    final = PPO.load(out / "last_model.zip", device="cpu")
    assert any(not torch.equal(before, after)
               for before, after in zip(initial.policy.parameters(), final.policy.parameters()))
    workload_path = out / "workloads.json"
    workloads = json.loads(workload_path.read_text()) if workload_path.exists() else None
    if workloads:
        env = PlacementEnv(variant=summary["variant"], workloads=workloads["train"] + workloads["heldout"],
                           normalizations=summary["normalization_by_workload_s"])
    else:
        env = PlacementEnv(normalization_s=summary["normalization_s"], variant=summary.get("variant", "A"))

    def assess(model):
        return (evaluate_workloads(model, env, workloads["train"]) if workloads else
                evaluate_policy(model, env))

    restored = assess(final)
    assert restored == summary["final"]
    assert (out / "figures/reward_latency.png").stat().st_size > 0
    assert (out / "figures/loss.png").stat().st_size > 0
    assert (out / "best/best_model.zip").is_file()
    assert list((out / "checkpoints").glob("*.zip"))
    for name, path in (("best", out / "best/best_model.zip"),
                       ("budget", out / "checkpoints" / f"ppo_{summary['requested_timesteps']}_steps.zip")):
        if summary.get(name) is not None:
            assert assess(PPO.load(path, device="cpu")) == summary[name]
    if workloads:
        assert evaluate_workloads(final, env, workloads["heldout"]) == summary["generalization"]
        assert sum(summary["workload_counts"].values()) == summary["complete_episodes"]
        if "decode_len" in workloads["train"][0]:
            experiment = json.loads((out / "experiment.json").read_text())
            with (out / "train.monitor.csv").open() as handle:
                monitor = list(csv.DictReader(line for line in handle if not line.startswith("#")))
            counts = Counter(r["workload"] for r in monitor)
            assert all(counts[key] == count for key, count in summary["workload_counts"].items())
            lengths = experiment["episode_steps_by_workload"]
            assert all(int(r["l"]) == lengths[r["workload"]] for r in monitor)
            assert sum(int(r["l"]) for r in monitor) + summary["unfinished_steps"] == final.num_timesteps
            history = json.loads((out / "eval/workloads.json").read_text())
            period = experiment["evaluation_interval_steps"]
            assert [r["timesteps"] for r in history] == list(range(period, final.num_timesteps + 1, period))
            assert all(r["workloads"].keys() == restored["workloads"].keys() for r in history)
    env.close()
    print(json.dumps({"verified": True, "actual_timesteps": final.num_timesteps,
                      "env_steps_per_s": summary["env_steps_per_s"],
                      "final_reward": restored["reward"],
                      "smoke": summary["smoke"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify_run(parser.parse_args().run)
