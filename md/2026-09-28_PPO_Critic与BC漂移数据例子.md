# Critic 与 BC 策略漂移：具体数据例子

2026-09-28 整理。以下复核现有 JSON；没有新增训练或模拟。critic 与同状态回放例子来自 09-22 的 E0/E5 诊断；critic 原验收记录于 09-27 远端复核。不能把这些数值当成 E10 critic 的最新测量。

## 1. Critic：原验收总体通过，但 prefill 子项失败

对象：Qwen-1.8B，预训练 critic E2，旧验收 trace 5/158，1×负载。MSE 越小越好；常数基线来自训练数据，预测时不看输入状态。

| 统计范围 | Critic MSE | 常数基线 MSE | EV | passed |
|---|---:|---:|---:|---|
| 总体 | 0.168679 | 0.223762 | 0.7433 | true |
| prefill | 0.181466 | 0.164667 | 0.7603 | false |
| decode | 0.162384 | 0.243178 | 0.7253 | true |

prefill MSE 比常数基线高 10.20%。该子集回报均值为 -0.710320，预测均值为 -1.094956，平均偏差 V−G=-0.384636。EV 会去掉误差均值，因此可以同时出现 EV>0 与 MSE 不过关。

验收程序仅汇总各负载的 all.passed，所以最终总体仍通过。这证明验收范围存在缺口；共同偏移会被后续片段中心化消除，不能只凭该偏差认定 actor 更新方向错误。

原始记录：[E2 验收 JSON](/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_critic_20260915/hpc_sync_20260915_185431/remote/training/qwen_1.8b/seed_7/epoch_002.json)。

## 2. Critic：本轮训练片段的变化尺度

对象：BC E0 actor＋所选预训练 critic，train trace 646，1×负载，前 16 个连续 plan。按可学习节点统计；每个 plan 有 616 个节点，同一个 G 会重复到这些节点，不能将其视为 9856 个独立性能样本。

| 量 | 数值 |
|---|---:|
| 回报 G 的标准差 | 0.004586 |
| 预测 V 的标准差 | 0.358919 |
| 去均值后的 G−V 标准差 | 0.362116 |
| 残差波动 / 回报波动 | 78.95 倍 |

除以优势标准差后，G 部分的波动为 0.0127，V 部分为 0.9912。优势变化受 V 的变化强烈影响；这不代表相同比例的梯度必然错误。

G 是片段累计回报加共同 bootstrap；准确名称是“带 bootstrap 的回报估计”，不是独立测得的完整 MC 真值。这里的中心化波动不受共同尾部偏移影响。critic 预测的是条件期望，单次实际回报不是动作优劣的精确答案。本地随机采样没有逐位复现 HPC：此环境原日志优势标准差为 0.358862，本地为 0.362116。

09-29 补充数值：该 16-plan 片段累计模拟时间为 0.658434081 秒，奖励尺度 C=46.623281619 秒，所以片段奖励之和为 -0.014122431。critic 预测的片段之后剩余回报为 -0.730587363，故首个 plan 的 G=-0.744709794。前一项来自已执行的模拟，后一项来自模型预测。若同一条轨迹完整回放到结束，才能得到其完整 MC 实现回报；把 bootstrap 替换为实际尾部回报只会共同平移这 16 个 G，不改变所列标准差。上述 78.95 倍不能解释为整体预测误差或错误梯度比例。

原始记录：[首片段诊断 JSON](/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_ppo_20260917/diagnosis_actual_first_rollout_20260922.json)。

## 3. BC 漂移：同一状态下方案执行时间变化

对象：val177 的 BC 回放状态。请求、阶段、batch、KV/缓存初态相同，只替换 actor 生成的方案，再分别执行模拟。下表时间是这一步的模拟耗时，不是 CPU/GPU 运行诊断程序的墙钟时间，也不是完整 trace makespan。

| 状态 | BC E0 / ms | E5 seed 4 / ms | E5 seed 15 / ms | E5 seed 26 / ms |
|---|---:|---:|---:|---:|
| 第 11 个决策：prefill，batch13 | 111.852 | 914.421（8.18×） | 1715.335（15.34×） | 1552.702（13.88×） |
| 第 40 个决策：decode，batch16 | 2.838 | 15.725（5.54×） | 21.087（7.43×） | 15.426（5.44×） |

prefill batch13 的可学习节点放置如下：

| 策略 | NPU | PIM0 | PIM1 |
|---|---:|---:|---:|
| E0 | 616 | 0 | 0 |
| E5_seed15 | 298 | 142 | 176 |

这能确认放置方案已变化且在这些状态下显著变慢，但设备数量本身不能分解通信、计算和缓存开销。不能从这里推出所有 prefill 都该放 NPU，或所有 decode 都该放 PIM。

保留反例：prefill batch1 时，E0 为 87.757 ms，E5 seed4 为 86.328 ms，后者快约 1.63%。所以 BC 一致率不是最终优化目标，需要结合真实执行时间。

原始记录：[同状态执行 JSON](/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_ppo_20260917/diagnosis_same_state_step_20260922.json)。

## 4. BC 漂移：改变了多少节点

保存的 28 个 BC 状态中，11 个为 prefill，17 个为 decode；逐状态比较可学习节点的最终放置，再求一致率平均。

| E5 策略 | prefill 改变比例 | decode 改变比例 |
|---|---:|---:|
| seed 4 | 46.34% | 57.11% |
| seed 15 | 61.10% | 66.04% |
| seed 26 | 69.60% | 72.72% |

以 seed15 为例：E1–E5 已执行 317821 次 actor optimizer.step；日志相邻采样策略/更新后策略的平均节点 KL 为 0.00041692，KL 停止只有 1 次。同时 E5 对固定 E0 的放置一致率仅为 prefill 38.90%、decode 33.96%。这两组观测并不矛盾：比较参照不同。

不能将日志 KL 与固定 BC 状态上的 KL 简单相除来计算“漂移倍数”，两者参照和采样状态不同。也不能从节点改变比例反推出精确吞吐损失。

原始记录：[BC 一致率 JSON](/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_ppo_20260917/diagnosis_policy_drift_summary_20260922.json)。

## 解释边界

这些数据证明了 critic 验收不足的风险和 BC 行为明显漂移的事实；“critic 误导更新、缺少 BC 约束放大退化”仍是待单变量消融确认的因果解释。E10 完整 MC critic 误差和同状态策略对照尚未测量。
