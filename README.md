# DOPS-Serve：PPO 与 Bifocal 算子放置实验

使用原生 Stable Baselines3 PPO 与 Bifocal 列表调度器。固定 Qwen 1.8B、28 层、FP16、TP2、batch=1、prefill 64 + 64 次 decode，模拟硬件为 1 NPU + 2 PIM，采用 fast/fast 成本模型。以下均为模拟完整请求耗时，越低越好。

## 实验结果（2026-11-12）

PPO 四组各完成 seeds 0/1/2、目标 10,000 请求；表中为均值 ± 样本标准差，单位 ms。Budget 为第 10,000 请求的检查点，Final 为最后完整 rollout 更新后的模型。

| 组别 | 观测 / 决策 | Budget | Final |
| --- | --- | ---: | ---: |
| A | 原始 / 逐算子 | 84.709 ± 20.021 | 85.237 ± 19.870 |
| B | 增强 / 逐算子 | 64.836 ± 0.747 | 64.953 ± 0.949 |
| C | 原始 / 整段 | 63.523 ± 0.000 | 63.519 ± 0.006 |
| D | 增强 / 整段 | 63.484 ± 0.059 | 63.484 ± 0.059 |

A/B 存在后期回退；C/D 的确定性结果接近全 PIM。全部 PPO 中最佳周期评估为 **63.415978 ms**，相对全 PIM 改善约 0.16%。确定性评估不代表随机采样同样稳定，详见 [PPO 结果分析](SB3_PPO/reports/2026-11-12_四组PPO实验结果分析.md)。

| 对照方法 | 耗时 / ms |
| --- | ---: |
| 全 NPU | 279.856 |
| Prefill NPU / Decode PIM | 149.054 |
| 全 PIM | 63.518 |
| 原 Bifocal 调参最优 | 64.616 |
| 修改 far focus 后的 Bifocal | 63.518 |

新 Bifocal 在三个 seed 下均为 **63.517612 ms**，由原贪心流程自行生成全 PIM 主算子放置；保留 EFT、短链前瞻和设备一致性机制，使用固定复用窗口估计未来成本，不读取真实剩余输出长度。与全 PIM 持平，尚未超过 PPO best。两者成本与连续计时对齐，但放置和排序自由度不同。详见 [Bifocal 结果](SB3_PPO/reports/2026-11-12_DOPS_list_scheduler/Bifocal_far修改与Qwen1.8B结果.md)。

## 代码与实验工件

| 内容 | 入口 |
| --- | --- |
| 实验合同、环境语义和运行命令 | [SB3_PPO/README.md](SB3_PPO/README.md) |
| PPO / Bifocal 入口 | [train.py](SB3_PPO/dops_sb3/train.py) / [bifocal_search.py](SB3_PPO/dops_sb3/bifocal_search.py) |
| 配置 | [configs](SB3_PPO/configs) |
| 四组完整训练输出、模型、曲线和检查点 | [obs_action_ablation](SB3_PPO/runs/obs_action_ablation) |
| Bifocal 生成计划、成本表和搜索记录 | [bifocal_far_20261112](SB3_PPO/runs/bifocal_far_20261112) |
| 详细分析与历史结果 | [reports](SB3_PPO/reports) |

结果来自单一固定 workload 的解析模拟，尚无跨 workload、真机或全局最优证明。历史 BC、Critic 预训练和自写 PPO 保存在 [archive/bc-ppo](https://github.com/YIAI-02/DOPS-Serve/tree/archive/bc-ppo)，历史单 seed 结果保留于 [ppo_seed0](SB3_PPO/runs/ppo_seed0)。
