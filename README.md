# DOPS-Serve：SB3 PPO 算子放置实验

main 保存当前基于 Stable Baselines3 的 PPO 实验：无 BC，固定 prompt 64、decode 64，使用 DOPS 的 NPU/PIM 解析模拟环境。

旧 BC、Critic 预训练和自写 PPO 的代码、结果与分析记录集中保存在 [archive/bc-ppo 分支](https://github.com/YIAI-02/DOPS-Serve/tree/archive/bc-ppo)。该分支也保留更早的 serving 实验代码；当前 main 的训练入口为 `SB3_PPO/`。

| 内容 | 入口 |
| --- | --- |
| 实验合同、环境语义、安装与训练命令 | [SB3_PPO/README.md](SB3_PPO/README.md) |
| 原生 PPO 训练入口 | [dops_sb3/train.py](SB3_PPO/dops_sb3/train.py) |
| 环境 / PPO 配置 | [environment.json](SB3_PPO/configs/environment.json) / [ppo.json](SB3_PPO/configs/ppo.json) |
| 正式训练结果报告 | [2026-10-28 正式训练结果](SB3_PPO/reports/2026-10-28_SB3_PPO_正式训练结果.md) |
| 正式训练完整输出 | [runs/ppo_seed0](SB3_PPO/runs/ppo_seed0) |
| 初次验收、作业启动、GPU 对比记录 | [reports](SB3_PPO/reports) |

## 已完成实验：seed 0

CPU 作业 16252628 于北京时间 2026-10-28 09:02:51 正常完成，运行 20 小时 55 分 41 秒，完成 1,531,904 个环境步和 10,012 个完整 episode。4 天为申请的运行上限。

| 策略 | Reward（越高越好） | 模拟请求耗时（ms，越低越好） |
| --- | ---: | ---: |
| 全 NPU | −1.000000 | 279.856 |
| 全 PIM | −0.226966 | 63.518 |
| 初始 PPO | −0.470974 | 131.805 |
| 最佳周期评估 PPO（737,280 步） | **−0.226603** | **63.416** |
| 最终 PPO | −0.370898 | 103.798 |

早期 Reward 明显改善，训练中多次回退，约 139 万步后出现持续退化。最佳策略仅比全 PIM 减少 0.16% 耗时，最终策略比最佳策略增加 63.68% 耗时；该实验不能判定为稳定收敛，也没有证明相对全 PIM 的实质性优势。

![Reward 与模拟请求耗时](SB3_PPO/runs/ppo_seed0/figures/reward_latency.png)

[Loss 曲线](SB3_PPO/runs/ppo_seed0/figures/loss.png) · [最佳模型](SB3_PPO/runs/ppo_seed0/best/best_model.zip) · [最终模型](SB3_PPO/runs/ppo_seed0/last_model.zip) · [74 个周期 checkpoint](SB3_PPO/runs/ppo_seed0/checkpoints)

完整结果保留 Monitor CSV、原生 SB3 progress.csv、TensorBoard 事件、评估 NPZ、运行配置与依赖版本、初始/最佳/最终模型和 Slurm 日志。最终模型在本地重载后的确定性评估与 HPC 输出完全一致。

结果来自单 seed、单一固定请求形状的解析模拟，不是真实 NPU/PIM 硬件测量。GPU 对比与已取消的重复测试另见报告，均不作为正式训练结果。
