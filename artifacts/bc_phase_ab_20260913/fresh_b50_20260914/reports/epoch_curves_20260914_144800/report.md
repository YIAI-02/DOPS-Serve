# BC 累计 loss 与吞吐曲线：已完成验证的轮次

数据快照整理时间：2026-09-14 14:48:00（北京时间）。

完整验证范围：独立 1.8B、7B 为 E1–E100，独立 14B 为 E1–E56，共享网络三个模型均为 E1–E54。14B 和共享仍在运行。

**E0 没有正式完整验证记录**。初始化记录不等于验证成绩；预检使用小样本并包含更新，不能作为正式 E0。所有图横轴为0–100，实际曲线从E1开始，灰区表示尚无完整验证结果。

训练/验证 loss 均为每轮更新后在固定完整数据上计算的三阶段等权条件 NLL：prefill、启动 decode、后续 decode 各占1/3。吞吐为总输出 token / 总 fast 模拟执行秒数，BC、教师和 PD 均排除决策耗时；不是实测模型推理吞吐。没有平滑、插值补齐或混入旧 A/B 实验。

|路线|模型|最新完整epoch|训练loss|验证loss|吞吐 token/s|相对PD|达到教师|历史最佳|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|独立|Qwen 1.8B|100|0.006930|0.633254|2695.01|+14.85%|98.37%|E57: 2699.68|
|独立|Qwen 7B|100|0.001743|0.800740|613.03|+5.74%|95.47%|E77: 631.97|
|独立|Qwen 14B|56|0.007965|0.723171|354.36|+2.20%|94.26%|E42: 373.52|
|共享|Qwen 1.8B|54|0.006959|0.622350|2654.14|+13.11%|96.88%|E52: 2700.72|
|共享|Qwen 7B|54|0.001937|0.597850|581.63|+0.32%|90.58%|E24: 631.27|
|共享|Qwen 14B|54|0.002510|0.606725|323.10|-6.81%|85.95%|E35: 374.15|

不同模型的共享最佳轮次不能拼成同一个共享检查点成绩。最新轮次不一定是最佳轮次。

## Loss

上排独立网络，下排共享网络；蓝线为训练 loss，红色虚线为验证 loss。

![Loss overview](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/loss_overview.png)

[对数纵轴版本](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/loss_log_overview.png) · [可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/loss_overview.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/loss_overview.pdf)

## 吞吐

![Throughput overview](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/throughput_overview.png)

[可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/throughput_overview.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/throughput_overview.pdf)

## 与 PD 的直接比较

![Throughput vs PD](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/figures/throughput_vs_pd.png)

## 数据

- [全部逐轮数据（六组）](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/epoch_metrics.csv)
- [最新值与历史最佳汇总](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/latest_and_best.csv)
- [快照与口径说明](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/snapshot.json)
- [qwen_1.8b 原始累计 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/data/qwen_1.8b.csv)
- [qwen_7b 原始累计 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/data/qwen_7b.csv)
- [qwen_14b 原始累计 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/data/qwen_14b.csv)
- [shared 原始累计 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/epoch_curves_20260914_144800/data/shared.csv)
