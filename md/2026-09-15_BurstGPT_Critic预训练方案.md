# BurstGPT Critic 预训练方案

日期：2026-09-15。状态：Q1–Q6 及实施已获批准；旧结果清理与代码实现已完成。实际作业状态见 [执行记录](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/md/2026-09-15_BurstGPT_Critic预训练执行记录.md)。

## 1. 目标与固定合同

清除本版本旧 PPO 结果，四条路线一起完成 **冻结 BC actor → 完整 trace 采样 → critic 预训练 → 独立验证**。验证后暂停，不自动启动 PPO。

| 路线 | 冻结 actor 起点 | critic 覆盖组 |
|---|---|---|
| qwen_1.8b | BC E57 对应 E0 | 0.5×／1×／1.5× |
| qwen_7b | BC E77 对应 E0 | 0.5×／1×／1.5× |
| qwen_14b | BC E42 对应 E0 | 0.5×／1×／1.5× |
| shared | BC E35 对应 E0 | 3 模型 × 3 负载 |

- 沿用 state=56、node=35、现有节点前缀 GRU；critic 的 encoder／GRU 从对应 E0 独立复制，value head 的 weight 和 bias 置零。训练整个 critic，actor 全程冻结且不创建 actor optimizer。
- 保留连续到达、FIFO 排队、跨 phase 缓存状态、完整排空、C026、1 NPU＋2 PIM、TP=2、fast 后端、decode stride=2。
- 保留 max_tokens=7315、max_batch=16、每 PIM 1 GiB 非 KV 余量；不改变请求截断与 KV 预留规则。
- 保留奖励 `r_t = -Δt_t / C_m`。复用现有 PD 标定中的 `S_m`、`C_m` 和负载缩放，不重新标定。真实输出长度只用于模拟终止和回报，不进入网络输入。
- 原 BC 实验、原始 BurstGPT CSV、train/val/test 划分及其他实验保持不变；test 不参与本轮。

## 2. 目录与删除范围

| 用途 | 本地 | HPC |
|---|---|---|
| 旧实验根 | `/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_ppo_20260914` | `/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_ppo_20260914` |
| 新阶段根 | `/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_critic_20260915` | `/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_critic_20260915` |

实施顺序：核对旧作业已结束 → 将下列必要输入复制到新根并修正配置路径 → 确认新根输入可读取 → 删除本地与 HPC 的整个旧实验根。只在新根修改代码。

| 迁入新根 | 内容 |
|---|---|
| `code/`、`backend/DOPS/`、`configs/` | 沿用当前源码和后端；配置中的运行路径改为新根，不复制 Python 缓存 |
| `bc_inputs/`、`initial/<route>/e0.pt` | 四个 BC 起点、来源元数据及迁移后的 E0；不使用任何 PPO 训练权重 |
| `data/` | HPC 保留完整已导入 split；本地同步 manifest、六条选中 trace 及短测试用 train 983 |
| `calibration/pilot/` | 复用三模型已有 PD 标定；本地来源为旧根 `inspection/2026-09-15_process/calibration/` |
| `protocol.json` | 保留硬件、输入、奖励和 BC 来源合同，写入本轮 trace、seed 与 critic 参数；移除旧运行状态 |

**旧结果全部删除，不另留结果备份**：PPO checkpoint 及优化器状态、`pilot/`、旧 `training/`（若存在）、`inspection/`、`logs/`、`runtime/`、旧作业表和检查通过记录。标定与数据作为新阶段输入先迁移，不随旧根丢失。

同时删除 `md/` 中三份旧运行记录：

- `2026-09-14_BurstGPT_PPO启动记录.md`
- `2026-09-15_BurstGPT_PPO先导结果.md`
- `2026-09-15_PPO退化诊断与修正优先级.md`

两份历史设计文档 `2026-09-14_BurstGPT_BC后强化学习方案.md`、`2026-09-14_BurstGPT_BC后PPO方案_导师汇报版.md` 属于方案记录，保留；本轮执行以本文为准。原 BC 根 `artifacts/bc_phase_ab_20260913/` 和其他目录不在清理范围内。

## 3. 固定采样数据

只选请求数为 1000 的 trace。train、val 分别按 `(output_tokens, index)` 排序，等数量划为 4／2 个秩区间，每区间取下中位排名。区间边界为 `[floor(qN/K), floor((q+1)N/K))`；名单在训练前固定，不根据拟合结果挑选。

| split | trace | 请求数 | 输出 token | 原始到达跨度 |
|---|---:|---:|---:|---:|
| train | 315 | 1000 | 14,685 | 113 s |
| train | 758 | 1000 | 72,444 | 556 s |
| train | 858 | 1000 | 142,986 | 282 s |
| train | 403 | 1000 | 259,594 | 3,684 s |
| val | 5 | 1000 | 36,661 | 1,689 s |
| val | 158 | 1000 | 128,991 | 285 s |

- master seed=7，仅控制动作采样和训练 shuffle；选样本身确定性执行。每个 route/model/load/trace 只采样一次，并记录实际 worker seed。
- 四路线使用上述相同源 trace。每条路线均由自身冻结 E0 按现有随机策略采样；shared 的数据不能用三个独立 actor 的轨迹拼接替代。
- 每个完整 trace 从空环境开始，内部不断开、不重置；保持原始请求顺序，按既有标定缩放到达间隔，直到 1000 个请求全部完成。
- 合计 `(3×3 + 9)×(4+2) = 108` 次完整回放。train/val 按源 trace 隔离，不将同一 trace 的 plans 随机拆成训练集与验证集。

## 4. 回报标签与 critic 训练

**采集。** 沿用每 16 个 plan 分块传输，将原始 state、node、全部节点动作、learned mask、phase、持续时间和终止信息保存为紧凑张量。采集时不计算 critic value、old log-prob 或优势，也不缓存 critic hidden state。

整条 trace 排空后，以终止值 0 反向求和：

`G_t = Σ(u=t…T-1) r_u = -Σ(u=t…T-1) Δt_u / C_m`

每个 plan 只保存一个 G；该 plan 内各 learned 节点共用标签，预测仍依赖各自的动作前缀。16-plan 边界仅用于传输和存储，**不使用 critic bootstrap**。标签单独保存，避免再次写回整批 node 张量。

**优化默认值。** Adam，lr=`1e-4`，mini-batch=`4 plans`，梯度裁剪=`0.5`，最多 `20 epochs`。每 epoch 验证，分组等权平均归一化验证 MSE 连续 `3 epochs` 无改善则停止。采样只执行一次，训练各 epoch 复用已保存的数据。

损失使用 learned 节点上的 MSE：模型×负载等权，组内 trace 等权，trace 内 plan 等权，plan 内 learned 节点等权；固定／跟随节点仅参与前缀历史。共享模型的 mini-batch 按模型组织，以兼容不同节点数。prefill/decode 本轮不另加训练权重，但分别报告验证指标。

## 5. 最小代码改动与运行流程

修改新根 `code/src/dops_serve/` 下三个现有文件：

| 文件 | 修改 |
|---|---|
| `placement_policy.py` | `StaticPlacementCritic.value_head` 的 weight／bias 零初始化；网络结构和 actor 不改 |
| `placement_training.py` | 为现有 worker 增加冻结 actor 的原始轨迹采集路径；增加完整 MC 标签及 critic-only MSE 更新，不调用 `update_static_ppo` |
| `burstgpt_ppo.py` | 增加 `collect-critic`、`train-critic` 入口，训练中逐 epoch 验证；复用数据加载、分组、标定和 checkpoint 工具，本轮不进入原 PPO train 路径 |

执行顺序：

1. 迁移必要输入、清除旧结果；写入本轮 `protocol.json` 和固定任务清单。
2. 做最小检查：零初始化输出、critic 更新后 actor 参数逐项不变、跨 16-plan 边界的 MC 标签正确。再在 Slurm 计算节点用 train 983 的 5 个请求跑通四路线采集→拟合→保存→加载；测试数据不混入正式预训练数据。
3. 在 CPU 分区 `C064M0256G` 采集 108 个独立回放任务，初始并发上限 18，每任务 1 CPU；优先排训练 trace 315 的 18 组，记录实际耗时和数据量。完整完成的任务直接复用，未完整排空的任务不能进入训练。
4. 每条路线所需 train/val 数据齐备后，提交该路线 critic 训练；四路线可独立并行，每路线在 `GPU80G` 使用 1 GPU。每 epoch 保存训练恢复状态及验证结果。
5. 汇总四路线验收结果后停止，不启动 PPO，不自动增加 trace、seed 或 epoch 预算。

作业计划时限为 CPU 回放每任务 24 h、GPU 训练每作业 8 h，提交前核对分区允许时限。采样恢复以完整 route/model/load/trace 为边界，训练恢复以 epoch 为边界；不新增模拟器现场序列化。HPC 登录节点仅检查和提交，数据节点用于传输，所有回放及训练在计算节点运行；不使用该集群禁用的 `--mem` 参数。

新根工件：

| 路径 | 内容 |
|---|---|
| `rollouts/<route>/<model>/load_<load>/<split>/<trace>/` | 分块原始张量、完整 MC 标签、完成记录及回放指标 |
| `training/<route>/seed_7/` | `last.pt` 保存 critic、Adam、随机状态和 epoch 进度；`best.pt` 保存选中 critic、BC/数据来源及验证结果；另存训练曲线与最终状态 |
| `jobs.tsv`、`logs/` | 本轮作业及执行日志 |
| `md/2026-09-15_BurstGPT_Critic预训练结果.md`（项目根下） | 最终四路线报告；实际完成日期变化时使用完成日期命名 |

## 6. 验收与结果边界

先检查 108 次回放均完整完成、actor 与对应 E0 参数完全一致、训练只读取 train 样本。常数基线只由该 route/model/load 的训练回报计算，使用与验证相同的 trace/plan/node 权重。

每个 model×load 验证组必须同时满足：

1. `MSE(V,G) < MSE(0,G)`。
2. `MSE(V,G) < MSE(训练组回报均值,G)`。
3. `EV = 1 - Var(G-V)/Var(G) > 0`。

每个独立模型须 3 组全部达标，shared 须 9 组全部达标。每 epoch 对以上条件判定；优先从全部组达标的 checkpoint 中，按分组等权平均 `MSE / MSE(训练均值基线)` 选择 `best.pt`。若始终没有全组达标 checkpoint，则保存误差最小的诊断 checkpoint，明确标为未通过，不作为后续 PPO 起点。

报告包含每组三种 MSE、EV、V/G 均值与标准差、最佳 epoch、实际数据量和耗时；另列 prefill/decode 指标，其常数基线使用对应训练组／phase 的均值，不对这些小分组追加通过门槛。

当前 state 没有明确的 trace 剩余请求或未来到达信息，完整 MC 存在不可观测因素；零初始化和预训练是否足以得到有效价值预测需要本轮验证。两条 val 仅支持首轮判断。本轮 actor 冻结，结果不能作为 PPO 吞吐提升的证据。

总耗时和空间尚未实测。旧先导中单条完整 trace 的采样已达小时级，仅 7B 一条 trace 的三负载 node 输入就至少约 1.3 GiB；本轮策略、trace 和并行方式均有变化，不能据旧日志承诺完成时间。先用首条 trace 更新耗时和空间估计，不据此截断数据或放宽验收。
