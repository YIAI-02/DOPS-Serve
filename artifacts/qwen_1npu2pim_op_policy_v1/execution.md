# proposal.md 执行记录

更新：2026-09-05；业务代码同时落在本地 DOPS-Serve 与 HPC `/lustre/home/2501111916/workspace/DOPS-Serve`。

## 已完成

- 从 GitHub `micro26_pieak_final` 准备独立 DOPS 后端，具体版本见 `backend_revision.txt`。DOPS 依赖源码工作树干净。
- 建立三个 Qwen 的 block/node 对应、固定 PIM 归属、逐节点合法性 mask 和 Fixed Plan compiler。每层 26 个自由节点、2 个 block，KV write 与 COMM 固定。
- 实现逐请求 query/KV 成本适配、FC collective 链路计时、独立候选 snapshot、跨阶段缓存/时间线、KV 预留与实际写入账本。
- 实现严格 FIFO continuous batching、prefill 首 token、单 token 完成、补入与 drain、同步决策停顿，以及窗口 token/等待/超时积分。
- BurstGPT 筛选后 706125 个请求，按时间 70%/15%/15% 分段后构造 494/105/105 条完整 train/val/test trace，每条 1000 个请求。分段尾部共 2125 个请求不组成完整 trace，已在 manifest 记录。
- 全量历史作业 `14752068` 已完成，退出码 0，耗时 20:02：1524 份来源全部完整并通过结构投影，包含 731520 个 decode steps，保留 833 个按模型和 phase 去重的二元 placement。Qwen-1.8B/7B/14B 分别有 512/508/504 份来源；未设置 plan 或 decode-step 数量上限，无排除样本。

## 计算节点验证

- 作业 `14752093`：COMPLETED，退出码 0；41 项测试通过，27 组真实 Qwen DAG 容量与连续性探测通过。
- CPU：Intel Xeon Platinum 8358 @ 2.60 GHz；节点 `l06c44n3`；OMP/MKL/OpenBLAS 各 1 线程。
- 三种输入：单请求 128、16 请求混合 128/2048、16 请求全部 2048；各执行 NPU 优先、PIM 优先和混合分片三种合法 mapping 的 prefill/decode。
- 本地与 HPC 17 个本次实施文件逐字节一致，未使用文件摘要比较。

| 模型 | 探测中 PIM runtime 最高值 | 两阶段手工计划编译中位数 | decode evaluator 主机墙钟中位数 |
|---|---:|---:|---:|
| Qwen-1.8B | 3.14490 GiB | 0.473 ms | 174.794 ms |
| Qwen-7B | 10.07642 GiB | 0.526 ms | 177.359 ms |
| Qwen-14B | 15.15015 GiB | 0.619 ms | 296.608 ms |

PIM runtime 预算为每颗 15.2 GiB；上述探测未发生 activation store。Qwen-14B 的最坏 KV 预留为每 PIM 2,013,265,920 bytes，即 1.875 GiB，与原 proposal 一致。计算为 `2(K/V) × 2(KV heads/PIM) × 128(head_dim) × 40(layers) × 2(FP16 bytes) × 16(batch) × 3072(prompt + max_tokens)`；程序与实际探测均使用这一容量。

表中的主机 evaluator 时间是推进模拟所需的 CPU 时间，不加入模拟设备时间；手工计划编译时间尚未包含未来 actor 的完整生成耗时。正式结果需要实测 policy/compile/search 的完整同步开销。

## 产物与后续阶段

- 全量输出：`history_sources.jsonl`、`placements.jsonl`、`history_excluded.jsonl`、`historical_plans/`、`manifest_summary.json`。原始单进程作业 `14750423` 已取消，其工件被 8 进程作业复用。
- 历史 anchor 重计时、共享 block actor、BC/PPO 和 SLO/reward 校准代码已接入，第二阶段作业已提交，状态见下节；完整训练结果、统一基线和正式对比尚未产生。
- 当前显式使用 NPU fast / PIM fast、Host ND 权重源及关闭的 kernel launch 开关完成接入验证。正式训练与比较配置仍需按 proposal 校准并冻结。

所有结果均限定于 FP16 decoder DAG timing/scheduling replay，不代表真实 tensor 推理性能。

## 历史评分与学习试运行（2026-09-05 已启动）

- 历史评分规则：按原始 `(model,batch,prompt,decode segment start)` 建立 anchor；使用固定 NPU 优先方案实际推进 prefill 及中间每一步 decode，保留缓存、KV 和设备时间线。候选从同一状态副本执行。全部历史来源和二元安排保留，不使用原始日志耗时作为训练标签。
- 历史输入共有 48 个形状组合、18410 个 anchor、33365 次候选评分及 44624 次参考状态推进（包含 prefill）。试运行 `14754143` 已完成，耗时 6:25，单个形状产出 514 个 anchor、覆盖 35 种 placement 和 32 份来源。
- 共享 actor/value：128 维节点编码和 block GRU，逐节点应用 NPU/PIM mask；两阶段共享参数，各阶段重建自己的 block 前缀。观察只含当前已知请求与资源状态，包含分开的 PIM 预算和权重驻留信息。
- BC：同 anchor 内按相对重计时成本计算 softmax 权重；较差的合法方案仍保留。每个模型按排序后的 `(batch,prompt)` 每第五个条件留给验证，不把同一条件拆到训练和验证。验证候选 regret 与 NLL 用于选择 BC checkpoint。
- PPO：概率与裁剪按完整 block 计算；中间 block 奖励为零，窗口末统一给反馈；gamma=1、lambda=1，截断时 bootstrap，继续完成同一条 trace。补齐窗口末空闲推进，使 bootstrap 与下一次决策的观察边界一致。
- 最新验证 `14754368`：48 项测试通过，退出码 0。包含 BC 保存/加载、新生成方案实际执行与 PPO 更新、整 block 概率裁剪、未完成 trace 的 bootstrap、三个 Qwen 的随机生成方案在 batch16/prompt2048 下连续 prefill/decode。日志为 `learning_integration_tests.log`。

| 阶段 | Slurm 作业 | 提交时状态与依赖 |
|---|---|---|
| 全量历史重评分 | 14754168 | RUNNING，8 CPU；19:55 检查时完成 19/48 个形状 |
| BC，种子 7/17/27 | 14754530，array 0-2 | 等待全量重评分与接入测试成功 |
| 三个模型的 Static / R / SLO 校准 | 14754544，array 0-2 | 等待全量重评分与接入测试成功 |
| 汇总训练协议 | 14754572 | 等待三个校准任务全部成功 |
| PPO，种子 7/17/27 | 14754581，array 0-2 | 等待三个 BC 与训练协议生成成功 |

依赖采用 `afterok`，并启用无效依赖自动取消；前置步骤失败时后续训练不会启动。上述队列状态是提交时快照，最新状态应查 Slurm。

初次试训练参数：BC 3 epochs、temperature=1、学习率 3e-4；每个模型先用同一条完整 train trace 和一条完整 validation trace校准，均包含 1000 请求。Static 先在共同冷启动训练 batch 上对全部训练历史 phase pair 重新执行 prefill/decode，再取评分前两对跑完整训练 trace。R 统计饱和训练 trace 从开始到最后一次接纳之间的完成速率；SLO 为验证集在 0.5R 下的 p99 E2E。

PPO 每个 seed 回放三个模型的完整训练 trace，负载从 0.5/0.75/1/1.25/1.5R 均匀抽样；4 窗口一个 rollout、4 次 PPO 更新、学习率 1e-4、clip=0.2。奖励的 token、时间、排队项按训练窗口尺度归一化，超时项初值为 1。以上是需要用训练/验证结果检查的试运行设置，不是已选定的最终超参数。

产物：`anchors/*.jsonl`、`reevaluation_summary.json`、`learning/seed_*/bc.pt`、`bc_curve.jsonl`、`bc_summary.json`、`calibration/<model>/model_protocol.json`、`training_protocol.json`、`learning/seed_*/ppo.pt`、`ppo_curve.jsonl`、`ppo_summary.json`。未完成的阶段不会被标记为已产出结果。

正式 test trace 尚未进入作业。训练规模/超参数选择、统一 Nearest/Bifocal/Candidate Oracle、正式五档负载对比与留一模型评估仍需后续完成。

## 定时监控（2026-09-05 20:21，北京时间）

- 全量历史重评分 `14754168` 于 20:13:48 完成，退出码 0，耗时 39:08；48/48 个形状、18410 个 anchor、833/833 个 placement、1524/1524 份来源全部覆盖。
- 三个 BC 和三个校准任务于 20:13:48 开始运行。20:21:37 检查时，BC 种子 7/17/27 分别报告首轮 2500/2000/2500 个训练 anchor；协议汇总和 PPO 仍正常等待依赖。
- BC 每轮 15460 个训练 anchor、2950 个验证 anchor，共 3 轮。按开始后 7:49 的实测速率，并为验证及模型/形状耗时差异留余量，初步估计 BC 在 9 月 5 日 23:00 至 9 月 6 日 01:30 完成；尚无完整 epoch 耗时，置信度低。校准尚无完整 trace 耗时，PPO 未启动，暂不提供全部作业结束时间。
- 已创建 Codex 监控 `dops-serve`，每 15 分钟检查，仅在阶段完成、失败、明显延迟或需要用户操作时通知。状态保存在同目录 `monitor_status.json`。
- 按用户要求，只复用 `/private/tmp/dops-op-monitor-login.sock`，不自动重连；连接中断只提醒一次，等待用户明确要求恢复。监控持续期间保留连接，全部目标作业终止后关闭连接并暂停监控。定时检查需要本机开机且 Codex 运行。

## 监控恢复与阶段完成（2026-09-05 23:30，北京时间）

- 23:24 检查发现控制连接已断开；收到用户明确恢复请求并完成本次认证后，已恢复同一路径的 SSH 长连接。每 15 分钟监控继续有效，仍不自动重连。最新状态见 `monitor_status.json`。
- 三组 BC 均成功完成 3 个 epoch，退出码均为 0；种子 7/17/27 分别于 22:36:18、22:38:57、22:25:46 结束。三个 `bc_summary.json` 和 checkpoint 均已生成，网络实测参数量为 193795。
- 三个模型校准全部成功，退出码均为 0，最后一个 Qwen-14B 于 22:36:44 完成；三个 `model_protocol.json` 均存在。协议汇总 `14754572` 于 22:37:48 成功结束，`training_protocol.json` 已生成。
- 三组 PPO 均于 22:38:57 启动。23:30 检查时，种子 17/27 已各完成 Qwen-1.8B 的完整 1000 请求 trace，进入 Qwen-7B；种子 7 从 Qwen-14B 开始，仍在第一条 trace。三组训练日志持续推进，尚无最终 `ppo_summary.json`。
- 根据首条 Qwen-1.8B PPO trace 约 44–47 分钟，以及 Qwen-14B 当前推进速度，初步估计这批试训练于 9 月 6 日 01:00–03:00 完成。14B 尚无完整 PPO trace 耗时，估计置信度低；此区间不包含尚未提交的正式基线与 test 评估。

## 本轮作业完成（2026-09-06 08:49 核对，北京时间）

- 本轮 12 个作业（含接入验证及数组子任务）均为 `COMPLETED`、退出码 `0:0`；检查时账号 Slurm 队列为空。
- PPO 种子 7/17/27 分别于 01:29:49、01:31:03、01:16:44 完成，更新批次数为 166/165/163。每个种子完整完成三个 Qwen 模型各 1000 请求的训练 trace。
- 六个 BC/PPO checkpoint、六个训练汇总均存在；三个 PPO 曲线行数与更新数一致，各有 3 个终止记录，检查的训练数值没有 NaN/Inf。48 项接入测试通过；历史重评分 48 个 anchor 文件均存在，833 种方案和 1524 份来源全部覆盖。
- BC 最优验证轮次为 1/2/3，历史候选平均成本差距为 4.57%/1.85%/4.12%；这不是 PPO 或端到端服务性能提升的证据。
- 本轮为试训练，协议仍为 `test_frozen=false`；正式基线比较、五档负载测试和留一模型评估尚待完成。
- 监控 `dops-serve` 已暂停，检查连接已关闭。完整结果、产物路径和解释见同目录 [completion_summary.md](completion_summary.md)。
