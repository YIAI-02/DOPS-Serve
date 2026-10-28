# DOPS-Serve：历史 BC / Critic / PPO 实验归档

本分支 `archive/bc-ppo` 汇集 2026 年 9 月的 BC、Critic 预训练、自写 PPO、回放与性能排查尝试，保存当时的源代码、配置、测试、模型、日志、结果和分析记录。这里的失败与退化结果是后续研究的参照，不作为当前有效方案。

当前无 BC 的原生 Stable Baselines3 PPO 实验及完整结果位于 [main / SB3_PPO](https://github.com/YIAI-02/DOPS-Serve/tree/main/SB3_PPO)。

| 历史阶段 | 代码与工件 |
| --- | --- |
| 原始 serving / BurstGPT 回放 | 根目录 `src/`、`configs/`、`slurm/`、`tests/`；[原运行说明](README.serving.md) |
| 初期算子策略、BC 重启 | [qwen_1npu2pim_op_policy_v1](artifacts/qwen_1npu2pim_op_policy_v1)、[fast_bc_restart_20260908](artifacts/fast_bc_restart_20260908) |
| 静态 BC | [bc_static_c026_20260913](artifacts/bc_static_c026_20260913) |
| 分阶段均衡 BC | [bc_phase_ab_20260913](artifacts/bc_phase_ab_20260913) |
| Critic 预训练及 DOPS 后端 | [burstgpt_critic_20260915](artifacts/burstgpt_critic_20260915) |
| BC + Critic 起点后的自写 PPO | [burstgpt_ppo_20260917](artifacts/burstgpt_ppo_20260917) |
| 历史训练曲线与汇总 | [training_history_20260912](artifacts/training_history_20260912) |
| 性能、stride 和回放诊断 | [artifacts](artifacts) 下的其他对应实验目录 |
| 方案、运行记录与 E5/E10 退化分析 | [md](md) |
| 精简前的原始代码包 | [local-archives](local-archives) |

归档保留原目录层级和源文件字节，中文注释保持原样。目录中的 `code/` 是各实验实际使用的代码快照；根目录代码是更早的 serving 版本，两者不应混为一个可直接运行的环境。

本分支保存本机已有的历史实验工件，并不声称包含 HPC 上从未取回的文件。各报告和 `progress` 文件都是其记录日期的快照，不表示当前作业状态；本次归档未重新运行旧训练。部分历史 Markdown 使用旧 Mac 或 HPC 的绝对路径，查阅时可按本页相对路径找到相应实验目录。

Python 虚拟环境、安装包目录、Python/pytest 缓存、macOS 元数据和嵌套 Git 元数据不属于实验结果，未纳入归档。主仓库原有提交历史保留，可沿本分支追溯此前版本。
