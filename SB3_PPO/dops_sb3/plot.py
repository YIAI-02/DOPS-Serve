"""从 SB3 Monitor/CSV/EvalCallback 原始日志导出图，不重新计算 PPO loss。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path, skip_comments=False):
    with path.open() as f:
        return list(csv.DictReader(line for line in f if not skip_comments or not line.startswith("#")))


def plot_run(out):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    monitor = read_csv(out / "train.monitor.csv", skip_comments=True)
    steps = np.cumsum([int(row["l"]) for row in monitor])
    rewards = np.asarray([float(row["r"]) for row in monitor])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(steps, rewards, alpha=0.35, label="Train episode return")
    if len(rewards) >= 20:
        axes[0].plot(steps[19:], np.convolve(rewards, np.ones(20) / 20, mode="valid"), label="Train mean (20 episodes)")
    evaluation_path = out / "eval/evaluations.npz"
    if evaluation_path.exists():
        with np.load(evaluation_path) as ev:
            axes[0].plot(ev["timesteps"], ev["results"].mean(axis=1), "o-", label="Deterministic evaluation")
    baseline = json.loads((out / "baselines.json").read_text())
    for color, (name, values) in zip(("#777777", "#2ca02c", "#9467bd", "#8c564b"), baseline.items()):
        axes[0].axhline(values.get("reward", values.get("reward_mean")), color=color,
                        linestyle="--", linewidth=0.8, label=name)
    initial = json.loads((out / "initial_policy.json").read_text())
    axes[0].scatter([0], [initial["reward"]], marker="s", color="black", label="Initial policy")
    summary_path = out / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        axes[0].scatter([summary["actual_timesteps"]], [summary["final"]["reward"]],
                        marker="x", color="red", label="Final policy after last update")
    axes[0].set_ylabel("Episode return (higher is better)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(steps, [float(r["latency_s"]) * 1000 for r in monitor])
    axes[1].set_ylabel("Simulated request time (ms)")
    axes[1].set_xlabel("Environment steps")
    fig.tight_layout()
    fig.savefig(figures / "reward_latency.png", dpi=160)
    plt.close(fig)

    # PPO total loss 是最后一个 minibatch 的原生日志值；不能解释为逐 episode loss。
    rows = read_csv(out / "progress.csv")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, key in zip(axes, ("train/loss", "train/policy_gradient_loss", "train/value_loss")):
        points = [(float(r["train/n_updates"]), float(r[key]))
                  for r in rows if r.get(key) and r.get("train/n_updates")]
        if points:
            x, y = zip(*points)
            ax.plot(x, y, "o-", markersize=3)
        ax.set_ylabel(key)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("SB3 train/n_updates (training epochs)")
    fig.tight_layout()
    fig.savefig(figures / "loss.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    plot_run(parser.parse_args().run)
