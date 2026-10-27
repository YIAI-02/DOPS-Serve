"""验收真实 SB3 输出：有限 loss、参数更新、模型重载和日志/图表落盘。"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .env import PlacementEnv
from .train import evaluate_policy


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
    env = PlacementEnv(normalization_s=summary["normalization_s"])
    restored = evaluate_policy(final, env)
    assert restored == summary["final"]
    assert (out / "figures/reward_latency.png").stat().st_size > 0
    assert (out / "figures/loss.png").stat().st_size > 0
    assert (out / "best/best_model.zip").is_file()
    assert list((out / "checkpoints").glob("*.zip"))
    print(json.dumps({"verified": True, "actual_timesteps": final.num_timesteps,
                      "env_steps_per_s": summary["env_steps_per_s"],
                      "final_reward": restored["reward"],
                      "smoke": summary["smoke"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify_run(parser.parse_args().run)
