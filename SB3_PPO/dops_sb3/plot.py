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
    experiment_path = out / "experiment.json"
    episode_steps = (json.loads(experiment_path.read_text())["episode_steps"]
                     if experiment_path.exists() else int(monitor[0]["l"]))
    steps = np.cumsum([int(row["l"]) for row in monitor])
    requests = steps / episode_steps
    rewards = np.asarray([float(row["r"]) for row in monitor])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(requests, rewards, alpha=0.35, label="Train episode return")
    if len(rewards) >= 20:
        axes[0].plot(requests[19:], np.convolve(rewards, np.ones(20) / 20, mode="valid"), label="Train mean (20 episodes)")
    evaluation_path = out / "eval/evaluations.npz"
    if evaluation_path.exists():
        with np.load(evaluation_path) as ev:
            axes[0].plot(ev["timesteps"] / episode_steps, ev["results"].mean(axis=1), "o-", label="Deterministic evaluation")
    baseline = json.loads((out / "baselines.json").read_text())
    for color, (name, values) in zip(("#777777", "#2ca02c", "#9467bd", "#8c564b"), baseline.items()):
        axes[0].axhline(values.get("reward", values.get("reward_mean")), color=color,
                        linestyle="--", linewidth=0.8, label=name)
    initial = json.loads((out / "initial_policy.json").read_text())
    axes[0].scatter([0], [initial["reward"]], marker="s", color="black", label="Initial policy")
    summary_path = out / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        axes[0].scatter([summary["actual_timesteps"] / episode_steps], [summary["final"]["reward"]],
                        marker="x", color="red", label="Final policy after last update")
    axes[0].set_ylabel("Episode return (higher is better)")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].plot(requests, [float(r["latency_s"]) * 1000 for r in monitor])
    axes[1].set_ylabel("Simulated request time (ms)")
    axes[1].set_xlabel("Training requests")
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


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_runs(root):
    """并列汇总已完成的运行；不足十二次时明确记录覆盖数量，不判定策略优劣。"""
    from .execution import OPERATORS, REFRESH_TOKENS

    root = Path(root)
    runs = [(p.parent, json.loads(p.read_text()))
            for p in sorted(root.glob("[ABCD]/seed[012]/summary.json"))]
    if not runs:
        raise ValueError(f"No completed runs under {root}")
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    results, placements, stats, eval_rows, loss_rows = [], [], [], [], []
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    loss_fig, loss_axes = plt.subplots(4, 3, figsize=(14, 12), sharex="col")
    loss_keys = ("train/loss", "train/policy_gradient_loss", "train/value_loss")
    for group_i, (variant, color) in enumerate(zip("ABCD", ("#4477AA", "#EE6677", "#228833", "#AA3377"))):
        group = [(out, summary) for out, summary in runs if summary["variant"] == variant]
        series = []
        loss_series = {key: [] for key in loss_keys}
        for out, summary in group:
            with np.load(out / "eval/evaluations.npz") as ev:
                x = ev["timesteps"] / summary["episode_steps"]
                y = ev["results"].mean(axis=1)
            series.append((x, y, summary["normalization_s"]))
            axes[0].plot(x, y, color=color, alpha=0.25, linewidth=0.8)
            axes[1].plot(x, -y * summary["normalization_s"] * 1000, color=color, alpha=0.25, linewidth=0.8)
            rows = read_csv(out / "progress.csv")
            for ax, key in zip(loss_axes[group_i], loss_keys):
                points = [(float(r["train/n_updates"]), float(r[key]))
                          for r in rows if r.get(key) and r.get("train/n_updates")]
                x_loss, y_loss = zip(*points)
                loss_series[key].append((np.asarray(x_loss), np.asarray(y_loss)))
                ax.plot(x_loss, y_loss, linewidth=0.7, label=f"seed {summary['seed']}")
                ax.set_title(f"{variant}: {key}", fontsize=9)
                ax.grid(alpha=0.2)
            for stage in ("initial", "budget", "best", "final"):
                policy = summary.get(stage)
                if policy is None:
                    continue
                results.append({
                    "variant": variant, "seed": summary["seed"], "stage": stage,
                    "reward": policy["reward"], "latency_ms": policy["latency_s"] * 1000,
                    "speedup": policy["speedup"], "requested_requests": summary["requested_requests"],
                    "complete_requests": summary["complete_requests"],
                    "unfinished_steps": summary["unfinished_steps"],
                    "actual_timesteps": summary["actual_timesteps"],
                    "optimizer_steps": summary["optimizer_steps"],
                    "policy_parameters": summary["policy_parameters"],
                })
                for segment, plan in enumerate(np.asarray(policy["actions"]).reshape(-1, 9)):
                    for op, action in zip(OPERATORS, plan):
                        placements.append({
                            "variant": variant, "seed": summary["seed"], "stage": stage,
                            "segment": segment, "phase": "prefill" if segment == 0 else "decode",
                            "decode_start": 0 if segment == 0 else (segment - 1) * REFRESH_TOKENS,
                            "operator": op, "placement": "PIM" if action else "NPU",
                        })
        for ax, key in zip(loss_axes[group_i], loss_keys):
            curves = loss_series[key]
            if not curves:
                continue
            epochs = curves[0][0]
            for x, _ in curves[1:]:
                epochs = np.intersect1d(epochs, x)
            values = np.stack([y[np.searchsorted(x, epochs)] for x, y in curves])
            mean = values.mean(axis=0)
            ax.plot(epochs, mean, color="black", linewidth=1.2, label="Mean")
            if len(curves) > 1:
                std = values.std(axis=0, ddof=1)
                ax.fill_between(epochs, mean - std, mean + std, color="black", alpha=0.1)
            for i, epoch in enumerate(epochs):
                loss_rows.append({"variant": variant, "metric": key, "training_epochs": epoch,
                                  "n": len(curves), "mean": mean[i],
                                  "std": values[:, i].std(ddof=1) if len(curves) > 1 else ""})
        if series:
            common = series[0][0]
            for x, _, _ in series[1:]:
                common = np.intersect1d(common, x)
            rewards = np.stack([y[np.searchsorted(x, common)] for x, y, _ in series])
            latencies = np.stack([-y[np.searchsorted(x, common)] * c * 1000 for x, y, c in series])
            for ax, values in zip(axes, (rewards, latencies)):
                mean = values.mean(axis=0)
                ax.plot(common, mean, color=color, label=f"{variant}, n={len(group)}")
                if len(group) > 1:
                    std = values.std(axis=0, ddof=1)
                    ax.fill_between(common, mean - std, mean + std, color=color, alpha=0.12)
            for i, request in enumerate(common):
                eval_rows.append({
                    "variant": variant, "training_requests": request, "n": len(group),
                    "reward_mean": rewards[:, i].mean(),
                    "reward_std": rewards[:, i].std(ddof=1) if len(group) > 1 else "",
                    "latency_ms_mean": latencies[:, i].mean(),
                    "latency_ms_std": latencies[:, i].std(ddof=1) if len(group) > 1 else "",
                })
        for stage in ("initial", "budget", "best", "final"):
            selected = [r for r in results if r["variant"] == variant and r["stage"] == stage]
            if not selected:
                continue
            row = {"variant": variant, "stage": stage, "n": len(selected)}
            for metric in ("reward", "latency_ms", "speedup"):
                values = np.asarray([r[metric] for r in selected])
                row[f"{metric}_mean"] = values.mean()
                row[f"{metric}_std"] = values.std(ddof=1) if len(selected) > 1 else ""
            stats.append(row)
    axes[0].set_ylabel("Deterministic episode return")
    axes[1].set_ylabel("Simulated request time (ms)")
    axes[1].set_xlabel("Training requests")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    note = "SMOKE: integration checks only. " if any(s["smoke"] for _, s in runs) else ""
    fig.suptitle(note + "Evaluation mean ± sample SD; faint lines: individual seeds", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures / "comparison_reward_latency.png", dpi=160)
    plt.close(fig)
    for ax in loss_axes[-1]:
        ax.set_xlabel("Training epochs (SB3 n_updates)")
    for row in loss_axes:
        if row[0].lines:
            row[0].legend(fontsize=8)
    loss_fig.suptitle(note + "Raw SB3 loss; within-group mean ± sample SD", fontsize=11)
    loss_fig.tight_layout()
    loss_fig.savefig(figures / "comparison_loss.png", dpi=160)
    plt.close(loss_fig)
    write_csv(root / "comparison.csv", results)
    write_csv(root / "summary_statistics.csv", stats)
    write_csv(root / "evaluation_statistics.csv", eval_rows)
    write_csv(root / "loss_statistics.csv", loss_rows)
    write_csv(root / "placements.csv", placements)
    coverage = {"completed_runs": len(runs), "expected_runs": 12,
                "runs": [{"variant": s["variant"], "seed": s["seed"], "smoke": s["smoke"]} for _, s in runs],
                "std": "sample standard deviation (ddof=1); blank for n=1",
                "budget": "checkpoint at requested budget, before any remaining rollout update",
                "final": "policy after the last complete rollout update",
                "loss": "raw SB3 metrics; state and joint-action definitions differ between groups"}
    (root / "comparison.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    compare_runs(args.run) if args.compare else plot_run(args.run)
