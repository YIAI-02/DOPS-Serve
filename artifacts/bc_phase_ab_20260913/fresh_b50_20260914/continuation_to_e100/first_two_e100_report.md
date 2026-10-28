# 1.8B 与 7B：E50 续训至 E100 结果

2026-09-14 核对：作业 14878513、14878514 均 COMPLETED，ExitCode=0:0。两路 complete.json 为 E100，累计 CSV 均完整覆盖 E1–E100。

保持 E50 的网络、Adam、随机状态、B 三阶段等权 loss、教师和验证数据。主吞吐为固定验证集总输出 token / 总模拟执行秒数，BC、教师、PD 均排除决策耗时。这是 fast 模拟结果，不是真实硬件推理吞吐。

|模型|轮次|训练 loss|验证 loss|吞吐 token/s|比 PD 提高|达到教师吞吐|
|---|---:|---:|---:|---:|---:|---:|
|qwen_1.8b|50|0.019139|0.468689|2683.6583|+14.36%|97.95%|
|qwen_1.8b|100|0.006930|0.633254|2695.0135|+14.85%|98.37%|
|qwen_7b|50|0.007694|0.756913|593.2532|+2.32%|92.39%|
|qwen_7b|100|0.001743|0.800740|613.0338|+5.74%|95.47%|

|模型|E1–E50 最佳|E51–E100 最佳|最佳吞吐变化|E100 对比 E50|
|---|---:|---:|---:|---:|
|qwen_1.8b|E41: 2693.5509|E57: 2699.6824|+0.23%|+0.42%|
|qwen_7b|E48: 618.8814|E77: 631.9679|+2.11%|+3.33%|

本次续训在吞吐上有小幅收益：1.8B 最佳值提高约 0.23%，7B 约 2.11%。E100 不是各自最优检查点。两路训练 loss 下降，但 E100 验证 loss 都高于 E50；单一轮次或训练 loss 不能替代完整自主验证结果。

## qwen_1.8b

[逐轮 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_1.8b/training_progress.csv) · [逐轮报告](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_1.8b/report.md)

![qwen_1.8b loss](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_1.8b/figures/qwen_1.8b/epoch_loss.png)

![qwen_1.8b throughput](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_1.8b/figures/qwen_1.8b/epoch_throughput.png)

![qwen_1.8b over_pd](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_1.8b/figures/qwen_1.8b/epoch_over_pd.png)

## qwen_7b

[逐轮 CSV](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_7b/training_progress.csv) · [逐轮报告](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_7b/report.md)

![qwen_7b loss](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_7b/figures/qwen_7b/epoch_loss.png)

![qwen_7b throughput](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_7b/figures/qwen_7b/epoch_throughput.png)

![qwen_7b over_pd](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/fresh_b50_20260914/training/qwen_7b/figures/qwen_7b/epoch_over_pd.png)

