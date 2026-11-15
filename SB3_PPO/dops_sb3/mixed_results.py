"""混合工况结果：逐工况真实耗时、等权归一化指标和三 seed 对照。"""
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np

from .execution import OPERATORS
from .plot import read_csv, write_csv


def training_axis(summary):
    if "episode_steps" in summary:
        return summary["episode_steps"], "Training batch episodes"
    return summary["requested_timesteps"] / 100, "Environment step budget (%)"


def workload_panels(count):
    columns = min(count, 4 if count > 9 else 3)
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(4.6 * columns, 3.1 * rows))
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    return fig, axes


def placement_rows(runs):
    # 完整变长计划的算子明细逐行写出，避免把十二个模型的几百万个字典一起放入内存。
    for _, summary in runs:
        for stage in ("initial", "budget", "best", "final", "generalization"):
            policy = summary.get(stage)
            if policy is None:
                continue
            for key, value in policy["workloads"].items():
                for segment, plan in enumerate(np.asarray(value["actions"]).reshape(-1, len(OPERATORS))):
                    for op, action in zip(OPERATORS, plan):
                        yield {"variant": summary["variant"], "seed": summary["seed"], "stage": stage,
                               "workload": key, "segment": segment, "operator": op,
                               "placement": "PIM" if action else "NPU"}


def plot_mixed_run(out):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    summary = json.loads((out / "summary.json").read_text())
    baselines = json.loads((out / "baselines.json").read_text())
    monitor = read_csv(out / "train.monitor.csv", skip_comments=True)
    history = json.loads((out / "eval/workloads.json").read_text())
    unit, xlabel = training_axis(summary)
    x = np.asarray([r["timesteps"] / unit for r in history])
    keys = list(summary["final"]["workloads"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    returns = np.asarray([float(r["r"]) for r in monitor])
    training_x = np.cumsum([int(r["l"]) for r in monitor]) / unit
    axes[0].plot(training_x, returns, alpha=0.15, label="Sampled training batches")
    axes[0].plot(x, [r["reward"] for r in history], "o-", label=f"Equal-weight evaluation, {len(keys)} workloads")
    axes[0].scatter([0], [summary["initial"]["reward"]], marker="s", label="Initial")
    axes[0].scatter([summary["actual_timesteps"] / unit],
                    [summary["final"]["reward"]], marker="x", label="Final after last update")
    for name in ("all_npu", "all_pim", "prefill_npu_decode_pim"):
        axes[0].axhline(np.mean([baselines[k][name]["reward"] for k in keys]),
                        linestyle="--", linewidth=0.8, label=name)
    if "decode_len" in summary["final"]["workloads"][keys[0]]:
        workloads = summary["final"]["workloads"]
        for length in sorted({v["decode_len"] for v in workloads.values()}):
            group = [k for k in keys if workloads[k]["decode_len"] == length]
            axes[1].plot(x, [np.mean([-r["workloads"][k]["reward"] for k in group]) for r in history],
                         label=f"decode {length}")
    else:
        for key in keys:
            axes[1].plot(x, [-r["workloads"][key]["reward"] for r in history], label=key)
    axes[0].set_ylabel("Mean return (higher is better)")
    axes[1].set_ylabel("Time / workload all-NPU time")
    axes[1].set_xlabel(xlabel)
    for ax in axes:
        ax.legend(fontsize=7, ncol=3)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "reward_latency.png", dpi=160)
    plt.close(fig)

    fig, axes = workload_panels(len(keys))
    for ax, key in zip(axes.flat, keys):
        ax.plot(x, [r["workloads"][key]["latency_s"] * 1000 for r in history], "o-", markersize=3)
        ax.scatter([summary["actual_timesteps"] / unit],
                   [summary["final"]["workloads"][key]["latency_s"] * 1000], color="red", marker="x")
        for name, values in baselines[key].items():
            ax.axhline(values["latency_s"] * 1000, linestyle="--", linewidth=0.8, label=name)
        ax.set_title(key)
        ax.set_ylabel("Simulated batch time (ms)")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(figures / "workload_latency.png", dpi=160)
    plt.close(fig)


def compare_mixed_runs(root, runs):
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    results, aggregate, evaluation, statistics, loss_rows = [], [], [], [], []
    fig, ax = plt.subplots(figsize=(10, 6))
    loss_fig, loss_axes = plt.subplots(4, 3, figsize=(14, 12))
    keys = list(runs[0][1]["final"]["workloads"])
    workload_fig, workload_axes = workload_panels(len(keys))
    heldout_count = len(runs[0][1]["generalization"]["workloads"])
    _, xlabel = training_axis(runs[0][1])
    for group_index, (variant, color) in enumerate(zip("ABCD", ("#4477AA", "#EE6677", "#228833", "#AA3377"))):
        group = [(out, s) for out, s in runs if s["variant"] == variant]
        curves = []
        for out, summary in group:
            baseline = json.loads((out / "baselines.json").read_text())
            history = json.loads((out / "eval/workloads.json").read_text())
            unit, _ = training_axis(summary)
            x = np.asarray([r["timesteps"] / unit for r in history])
            y = np.asarray([r["normalized_latency"] for r in history])
            curves.append((x, y, history))
            ax.plot(x, y, color=color, alpha=0.25, linewidth=0.8)
            for record in history:
                for key, value in record["workloads"].items():
                    evaluation.append({"variant": variant, "seed": summary["seed"],
                                       "episodes": (record["complete_episodes"] if "complete_episodes" in record
                                                    else record["timesteps"] / summary["episode_steps"]),
                                       "timesteps": record["timesteps"],
                                       "workload": key, "latency_ms": value["latency_s"] * 1000,
                                       "normalized_latency": -value["reward"]})
            for stage in ("initial", "budget", "best", "final", "generalization"):
                policy = summary.get(stage)
                if policy is None:
                    continue
                aggregate.append({"variant": variant, "seed": summary["seed"], "stage": stage,
                                  "normalized_latency": policy["normalized_latency"],
                                  "actual_timesteps": summary["actual_timesteps"],
                                  "complete_episodes": summary["complete_episodes"],
                                  "complete_requests": summary["complete_requests"],
                                  "optimizer_steps": summary["optimizer_steps"]})
                for key, value in policy["workloads"].items():
                    reference = min(v["latency_s"] for v in baseline[key].values())
                    results.append({"variant": variant, "seed": summary["seed"], "stage": stage,
                                    "workload": key, "prefill_len": value["prefill_len"], "batch": value["batch"],
                                    **({"decode_len": value["decode_len"]} if "decode_len" in value else {}),
                                    "latency_ms": value["latency_s"] * 1000,
                                    "normalized_latency": -value["reward"], "all_npu_speedup": value["speedup"],
                                    "best_fixed_baseline_ms": reference * 1000,
                                    "ratio_to_best_fixed": value["latency_s"] / reference})
            raw = read_csv(out / "progress.csv")
            for loss_ax, metric in zip(loss_axes[group_index],
                                       ("train/loss", "train/policy_gradient_loss", "train/value_loss")):
                points = [(float(r["train/n_updates"]), float(r[metric])) for r in raw
                          if r.get(metric) and r.get("train/n_updates")]
                lx, ly = zip(*points)
                loss_ax.plot(lx, ly, linewidth=0.8, label=f"seed {summary['seed']}")
                loss_ax.set_title(f"{variant}: {metric}", fontsize=9)
                loss_ax.set_xlabel("Training epochs")
                loss_ax.grid(alpha=0.2)
                loss_rows.extend({"variant": variant, "seed": summary["seed"], "metric": metric,
                                  "training_epochs": epoch, "value": value} for epoch, value in points)
        if curves:
            common = curves[0][0]
            for x, _, _ in curves[1:]:
                common = np.intersect1d(common, x)
            values = np.stack([y[np.searchsorted(x, common)] for x, y, _ in curves])
            mean = values.mean(axis=0)
            ax.plot(common, mean, color=color, label=f"{variant}, n={len(group)}")
            if len(group) > 1:
                sd = values.std(axis=0, ddof=1)
                ax.fill_between(common, mean - sd, mean + sd, color=color, alpha=0.12)
            for workload_ax, key in zip(workload_axes.flat, keys):
                times = np.stack([[h[i]["workloads"][key]["latency_s"] * 1000
                                   for i in np.searchsorted(x, common)] for x, _, h in curves])
                for individual in times:
                    workload_ax.plot(common, individual, color=color, alpha=0.2, linewidth=0.7)
                workload_ax.plot(common, times.mean(axis=0), color=color, label=variant)
                if len(group) > 1:
                    sd = times.std(axis=0, ddof=1)
                    workload_ax.fill_between(common, times.mean(axis=0) - sd,
                                              times.mean(axis=0) + sd, color=color, alpha=0.12)
                workload_ax.set_title(key)
                workload_ax.set_ylabel("Simulated batch time (ms)")
                workload_ax.set_xlabel(xlabel)
        for stage in ("initial", "budget", "best", "final", "generalization"):
            stage_rows = [r for r in results if r["variant"] == variant and r["stage"] == stage]
            for key in sorted({r["workload"] for r in stage_rows}):
                selected = [r for r in stage_rows if r["workload"] == key]
                row = {"variant": variant, "stage": stage, "workload": key, "n": len(selected)}
                for metric in ("latency_ms", "normalized_latency", "ratio_to_best_fixed"):
                    values = np.asarray([r[metric] for r in selected])
                    row[f"{metric}_mean"] = values.mean()
                    row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else ""
                statistics.append(row)
    ax.set_ylabel("Mean time / workload all-NPU time (lower is better)")
    ax.set_xlabel(xlabel)
    ax.legend()
    ax.grid(alpha=0.2)
    note = "SMOKE: integration only. " if any(s["smoke"] for _, s in runs) else ""
    fig.suptitle(note + f"Equal weight across {len(keys)} workloads; mean ± sample SD")
    fig.tight_layout()
    fig.savefig(figures / "comparison_reward_latency.png", dpi=160)
    plt.close(fig)
    workload_axes.flat[0].legend(fontsize=8)
    workload_fig.suptitle(note + "Individual workloads; mean ± sample SD across seeds")
    workload_fig.tight_layout()
    workload_fig.savefig(figures / "comparison_workloads.png", dpi=160)
    plt.close(workload_fig)
    for row in loss_axes:
        if row[0].lines:
            row[0].legend(fontsize=8)
    loss_fig.tight_layout()
    loss_fig.savefig(figures / "comparison_loss.png", dpi=160)
    plt.close(loss_fig)
    for name, rows in (("comparison", results), ("aggregate", aggregate), ("placements", placement_rows(runs)),
                       ("evaluation_statistics", evaluation), ("summary_statistics", statistics), ("loss_statistics", loss_rows)):
        write_csv(root / f"{name}.csv", rows)
    coverage = {"completed_runs": len(runs), "expected_runs": 12, "threshold": None,
                "metric": f"equal-weight mean T / all-NPU T over the {len(keys)} training workloads",
                "generalization": f"{heldout_count} held-out workloads, final models only; excluded from selection"}
    if len(runs) == 12 and not any(s["smoke"] for _, s in runs):
        selected_out, selected = min(runs, key=lambda pair: pair[1]["final"]["normalized_latency"])
        target = root / "selected_model"
        target.mkdir(exist_ok=True)
        shutil.copy2(selected_out / "last_model.zip", target / "model.zip")
        for name in ("workloads.json", "normalization.json", "environment_config.json", "experiment.json"):
            shutil.copy2(selected_out / name, target / name)
        coverage["selected"] = {"variant": selected["variant"], "seed": selected["seed"],
                                "source": str(selected_out.relative_to(root)), "stage": "final",
                                "normalized_latency": selected["final"]["normalized_latency"]}
        (target / "selection.json").write_text(json.dumps(coverage["selected"], indent=2) + "\n")
    (root / "comparison.json").write_text(json.dumps(coverage, indent=2) + "\n")
    lines = ["# 多工况 ABCD PPO 结果", "", f"已完成 {len(runs)}/12 次。" + note,
             "本轮不设置数值通过门槛。所有耗时均来自解析模拟器；未见工况不参与模型选择。", "",
             "| 组别 | seed | Initial | Best | Budget | Final | 未见工况 Final |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, summary in runs:
        scores = [f"{summary[stage]['normalized_latency']:.6f}" if summary.get(stage) else "—"
                  for stage in ("initial", "best", "budget", "final", "generalization")]
        lines.append(f"| {summary['variant']} | {summary['seed']} | " + " | ".join(scores) + " |")
    lines += ["", "表中指标为各工况 T / 全 NPU T 的等权均值，越低越好。",
              "逐工况耗时、三 seed 均值与样本标准差、完整评估曲线及放置分别见 comparison.csv、summary_statistics.csv、evaluation_statistics.csv 和 placements.csv。",
              "预算检查点位于指定步数；Final 为完成尾部 rollout 更新后的模型。各组实际优化次数保留在 aggregate.csv。",
              "", "![学习曲线](figures/comparison_reward_latency.png)",
              "![逐工况曲线](figures/comparison_workloads.png)", "![Loss](figures/comparison_loss.png)"]
    if "selected" in coverage:
        selected = coverage["selected"]
        lines += ["", f"交付候选：{selected['variant']} / seed {selected['seed']} 的 Final，见 selected_model/model.zip。",
                  f"选择只使用 {len(keys)} 种已见工况的综合指标，不代表通过预设稳定性或泛化门槛。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(coverage, indent=2))
