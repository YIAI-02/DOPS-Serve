# BC E100 完成报告

整理时间：2026-09-14T19:53:11.420773+08:00。

四路作业全部正常完成，退出码均为 0:0；四路最终完成记录均为 E100。没有运行 PPO 或 test。用户取消的定时监控保持关闭。

|路线|续训作业|完成时间（北京时间）|E51–E100 耗时|
|---|---:|---|---|
|qwen_1.8b|14878513|2026-09-14 14:12:14|02:26:38|
|qwen_7b|14878514|2026-09-14 14:12:20|02:26:44|
|qwen_14b|14880554|2026-09-14 17:42:45|03:27:06|
|shared|14880555|2026-09-14 18:34:12|04:18:32|

## 统计口径

- 完整数据为 6 组 × 100 epoch = 600 行；独立三个模型各一组，共享网络按三个模型分开记录。每组每轮为固定 8 个验证任务。
- 正式曲线从 E1 开始。没有正式 E0 完整验证；不以初始化记录或小样本预检补作 E0。图中的 E50 虚线标记续训起点，没有平滑或插值。
- loss 是每轮更新后，在完整固定训练集/验证集计算的条件 NLL；prefill、启动 decode、后续 decode 各占 1/3。
- 吞吐 = 总输出 token / 总 fast 模拟执行秒数。BC、教师、PD 均不计决策耗时；decision_s 保留在 CSV。结果不是实际硬件推理吞吐。
- 已核对全部 600 组逐轮汇总与各自 8 个任务的输出量、执行时间、决策时间一致；并核对三阶段 loss 平均值等于主 loss。

## E100 与 E50

|路线|模型|E50 吞吐|E100 吞吐|变化|E100 相对 PD|E100 达到教师|
|---|---|---:|---:|---:|---:|---:|
|独立|Qwen 1.8B|2683.66|2695.01|+0.42%|+14.85%|98.37%|
|独立|Qwen 7B|593.25|613.03|+3.33%|+5.74%|95.47%|
|独立|Qwen 14B|350.63|319.87|-8.77%|-7.75%|85.09%|
|共享|Qwen 1.8B|2554.15|2654.80|+3.94%|+13.13%|96.90%|
|共享|Qwen 7B|629.85|630.49|+0.10%|+8.75%|98.19%|
|共享|Qwen 14B|369.61|361.91|-2.08%|+4.38%|96.27%|

## Loss 变化

|路线|模型|训练 loss E50 → E100|验证 loss E50 → E100|
|---|---|---:|---:|
|独立|Qwen 1.8B|0.019139 → 0.006930|0.468689 → 0.633254|
|独立|Qwen 7B|0.007694 → 0.001743|0.756913 → 0.800740|
|独立|Qwen 14B|0.012537 → 0.002861|0.551064 → 0.666913|
|共享|Qwen 1.8B|0.008929 → 0.005047|0.645093 → 0.759266|
|共享|Qwen 7B|0.002357 → 0.000639|0.560480 → 0.700314|
|共享|Qwen 14B|0.003284 → 0.000923|0.563267 → 0.719049|

六组训练 loss 均下降，六组验证 loss 均上升。这说明对训练教师动作的拟合继续增强，但没有同步改善未参与更新的验证集条件 NLL。它提示过拟合，以及模仿目标和自主吞吐可能不一致；仅凭这些汇总不能证明某个具体原因。

## 前后 50 轮的最佳成绩

|路线|模型|E1–E50 最佳|E51–E100 最佳|后半程最佳变化|全程最佳|
|---|---|---:|---:|---:|---:|
|独立|Qwen 1.8B|E41: 2693.55|E57: 2699.68|+0.23%|E57: 2699.68|
|独立|Qwen 7B|E48: 618.88|E77: 631.97|+2.11%|E77: 631.97|
|独立|Qwen 14B|E42: 373.52|E84: 372.29|-0.33%|E42: 373.52|
|共享|Qwen 1.8B|E36: 2697.18|E52: 2700.72|+0.13%|E52: 2700.72|
|共享|Qwen 7B|E24: 631.27|E88: 630.97|-0.05%|E24: 631.27|
|共享|Qwen 14B|E35: 374.15|E63: 372.81|-0.36%|E35: 374.15|

共享三模型的最佳 epoch 不同，不能把这些值拼成一个共享检查点的成绩；E100 三个共享结果来自同一检查点。

## 避免只看最后一轮

下面比较 E41–E50 与 E91–E100 的吞吐中位数。这只是同一训练过程的描述统计，不是独立随机种子实验，也不表示统计显著性。

|路线|模型|E41–E50 中位数|E91–E100 中位数|变化|高于 PD 的轮数（前 → 后）|
|---|---|---:|---:|---:|---:|
|独立|Qwen 1.8B|2557.17|2632.16|+2.93%|10/10 → 10/10|
|独立|Qwen 7B|542.75|608.78|+12.17%|2/10 → 8/10|
|独立|Qwen 14B|352.38|365.67|+3.77%|10/10 → 8/10|
|共享|Qwen 1.8B|2577.83|2657.10|+3.08%|10/10 → 10/10|
|共享|Qwen 7B|627.51|625.34|-0.35%|10/10 → 9/10|
|共享|Qwen 14B|366.97|365.03|-0.53%|9/10 → 8/10|

## 14B 最后一轮回落的位置

独立 14B 的 E99 → E100 吞吐由 372.09 降至 319.87 token/s；总模拟执行时间由 46.44 增至 54.02 秒。逐任务记录显示 prefill 时间全部不变，新增的约 7.58 秒都在 decode。主要增量来自 batch=8、prompt=2048 的四个验证任务，decode 长度为 128/256/512/1024；增量分别约 0.49/1.00/2.02/4.06 秒。

这把回落定位到自主 decode 安排的执行阶段，而不是决策计时变化。现有汇总尚不能区分具体哪个节点的分配、搬运或等待造成增加，不能据此声称已找到底层原因。详细逐任务对照保存在 independent14b_e99_e100_cases.csv。

## 对续训收益的判断

- 独立 1.8B：E100 比 E50 高 0.42%，后半程最佳仅提高 0.23%；收益较小。
- 独立 7B：E100 提高 3.33%，后半程最佳提高 2.11%；末十轮中位数也上升，本次续训有比较明确的收益迹象。
- 独立 14B：末十轮中位数比 E41–E50 高，但后半程最佳未超过 E42，E100 还明显回落；不能认为更晚的检查点更好，也不能用单个 E100 否定整个后半程。
- 共享 1.8B：末轮和末十轮中位数改善，但最佳只提高 0.13%。共享 7B、14B：后半程没有刷新最佳，末十轮中位数略降；额外 50 轮收益有限。
- 当前不建议只因为训练 loss 继续下降就原样追加轮次。先保留现有全部检查点，独立模型按固定验证吞吐考虑最佳检查点；共享必须选择同一个 epoch 联合评估三个模型。此次未切换检查点、追加训练或开启监控。

## 完整曲线

### Loss（上排独立、下排共享）

![Loss（上排独立、下排共享）](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/loss_overview.png)

[可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/loss_overview.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/loss_overview.pdf)

### 吞吐（蓝色 BC、青色教师、灰色 PD）

![吞吐（蓝色 BC、青色教师、灰色 PD）](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_overview.png)

[可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_overview.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_overview.pdf)

### 与 PD 比较

![与 PD 比较](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_vs_pd.png)

[可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_vs_pd.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_vs_pd.pdf)

### E40–E100 细节

![E40–E100 细节](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_late_epochs.png)

[可编辑 SVG](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_late_epochs.svg) · [PDF](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/throughput_late_epochs.pdf)

[loss 对数纵轴版本](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/figures/loss_log_overview.png)

[完整逐轮 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/epoch_metrics.csv) · [比较汇总 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/summary.csv) · [快照与作业证据](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/reports/final_e100_20260914/snapshot.json)
