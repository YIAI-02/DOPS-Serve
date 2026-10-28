# Critic 后 PPO 启动条件检查

2026-09-15，18:58–19:05 同步核对。范围：三个独立模型；只检查和同步，不提交 PPO、不改训练代码。以下依据为通过 data 节点拉取的远端实际源码、协议、checkpoint、指标和日志；本次未重建 login，未查询新的 Slurm 队列状态。

## 同步结果

- 远端副本：`artifacts/burstgpt_critic_20260915/hpc_sync_20260915_185431/remote/`，约 69 MiB，含代码、协议、配置、标定、E0、critic 训练 checkpoint/逐轮结果、日志和 108 份回放完成记录；未下载大型原始回放张量、后端、原始数据。
- 同步前本地代码保存在同目录 `local_code_before_sync/`。当前工作代码未被覆盖；远端文件未修改。
- 本地比远端少 19 个历史文件，多一个 `io.py`；11 个同名文件有文本差异。比较去除注释/docstring 的语法树后，actor、critic、PPO 主循环/更新、回报、采样、critic 采集/训练/验收等 11 个关键定义一致。差异清单为 `code_file_diff.json`，核心对比为 `core_comparison.json`。
- 三路 E0 与本地逐字节相同。三个 `best.pt` 均已同步，单文件 3,271,942 bytes；尚未进行真实 checkpoint 的 PyTorch 加载与 PPO 更新测试，本地已检查的 Python 环境没有 torch。
- data 持久连接已保留；login 尚未重建。共享网络快照已有 3 个 epoch 记录、无完成文件，本次三个独立模型不依赖它。

| 独立路线 | 最佳 critic epoch | 最终 epoch | 验收 | EV：0.5× / 1× / 1.5× |
| --- | --- | --- | --- | --- |
| 1.8B | E2 | E5 | passed | 0.7469 / 0.7433 / 0.7758 |
| 7B | E1 | E4 | passed | 0.1858 / 0.5677 / 0.6545 |
| 14B | E5 | E8 | passed | 0.0074 / 0.2553 / 0.8146 |

三路完成文件均确认 actor 未改变、PPO 未启动。108 份回放均完成 1000 个请求且 actor 未改变。14B 的低负载 EV 虽过线，但几乎没有解释回报波动；不据此修改既定验收门槛，也不将 passed 解读为稳定 PPO 的保证。

## 结论

**当前代码不能直接开始使用预训练 critic 的 PPO。** 即使三个 critic 最终均通过验收，下面的初始化、配置、目录和集成验证也未接通。

## 已确认的直接阻断

| 项目 | 当前证据 | 必须完成的衔接 |
| --- | --- | --- |
| PPO 参数 | `protocol.json` 为 `critic_pretraining`，缺少 `ppo`、`quick_val_indices`；`train()` 第一阶段直接访问 `protocol["ppo"]` | 在 PPO 实验根写入明确的 PPO 参数、采样范围和验证名单 |
| Critic 初始化 | `train()` 仅加载 BC E0，随后执行 `StaticPlacementCritic(actor)`；没有读取预训练 `best.pt` | BC E0 初始化 actor；对应路线已通过的 `best.pt` 初始化完整 critic；重新创建 PPO 优化器 |
| 输出路径 | 正式 PPO 与 critic 都写 `training/<route>/seed_7/` | 使用独立 PPO 实验根，保留 critic 原始工件 |
| 恢复格式 | critic 的 `last.pt` 没有 PPO 所需 actor、协议、两套优化器及 trace 游标 | 不将 critic checkpoint 当作 PPO `--resume`；先生成新的 PPO 初始状态 |
| 启动检查 | 同步后的远端新根只有 critic 检查记录；无 `preflight_passed.json`、`worker_check_passed.json`、`orchestration_check_passed.json`、`formal_budget.json` | 对真实预训练权重执行计算节点集成检查；正式预算另行确定 |
| Slurm 入口 | 当前两个脚本只提交回放与 `train-critic`；launcher 的其他分支也固定调用 `train-critic` | 添加最小 PPO 启动及提交入口，显式使用 PPO 根 |

源码位置：`artifacts/burstgpt_critic_20260915/code/src/dops_serve/burstgpt_ppo.py` 的 `train()`；`placement_training.py` 的 worker `rollout`、`update_static_ppo()`；`code/slurm/run_critic.sh`。

## 已保留的兼容合同

- actor 来源仍为 1.8B E57、7B E77、14B E42；critic 与 actor 结构兼容，但权重独立。
- state=56、node=35、三个设备动作、前缀因果性、learned mask 保留。
- 连续到达、FIFO、完整排空、1 NPU+2 PIM、max_tokens=7315、max_batch=16、每 PIM 1 GiB 余量沿用当前协议。
- 奖励仍为 `-Δt/C_m`，沿用同一 PD 标定；PPO 的终止 bootstrap=0，非终止使用下一真实状态第一个 learned 节点的价值。
- 原有 PPO 单元测试检查终止/截断回报、条件概率、actor/critic 梯度隔离与恢复。已完成的 GPU critic 检查仅覆盖采集、MC 拟合与 critic 恢复，未证明新权重与 PPO 主循环接通。

## 尚未验证的算法边界

Critic 预训练使用完整 MC 标签；保留的 PPO 每 16 个 plan 截断，用边界价值补足回报。这与既定 PPO 方法一致，但完整 MC 验收通过并不能证明截断后的优势可靠。

当前验收对 learned 节点整体平均，PPO bootstrap 只取下一 plan 的第一个 learned 节点；这一个位置的预测质量没有单独报告。PPO 又会按当前采样组将优势标准化到单位尺度，组内价值预测误差与有效时间信号的比例需要检查。

必须区分：当前 γ=1，每个组只采一条 trace 的一个连续分块，MC 与 bootstrap 的差 `V_end−G_end` 对该组所有节点相同。组内减均值、除标准差后，这个共同偏移会消失。因此，不能仅因原始 bootstrap 优势与 MC 优势符号不同就判定 actor 信号不可靠；应检查组内价值波动，以及 bootstrap 作为 critic 拟合目标时的偏差。终止块两者相同。

另一个实际差异是旧 `train()` 的先导固定以 train 983 开头。该尾段只有 5 请求，而本轮 critic 使用 1000 请求的完整 trace。短夹具适合接线检查；不建议将它作为预训练 critic 接入后的第一次正式策略更新。

进入小规模 PPO 前，最少需要用冻结 BC 的完整回放比较：

1. 三负载的边界 bootstrap 预测与对应完整 MC，单列第一个 learned 节点的误差。
2. 同一批数据记录 16-plan 组内回报、价值和原始优势的均值/标准差，核对 MC 与 bootstrap 优势标准化后的一致性，评估局部信号及 critic 目标偏差；不把未标准化的符号一致率作为硬门槛。
3. 实际 BC E0＋预训练 critic 的一次采样、更新、保存、恢复；检查初始新旧 log-prob 一致、数值有限、actor/critic 参数不共享且各自更新。

上述诊断没有预先约定统一数值阈值，本次不事后编造“通过线”。它们用于判断是否可以进入 seed=7 的小规模 PPO 先导；正式推进还需在相同完整验证 trace 上比较 E0/PD/PPO 吞吐，test 保持隔离。

## 最小后续顺序

以下为待用户讨论确认的修改范围，尚未执行：

1. 以已核对的本地精简代码为工作基础，保留注释；在独立 `artifacts/burstgpt_ppo_<实施日期>/` 根运行，保留全部现有 BC/critic 工件。
2. actor 分别加载 BC E57/E77/E42 的 E0；critic 分别加载本轮最佳 E2/E1/E5。只加载网络权重，新建 PPO 优化器，不使用 critic `last.pt` 作为 PPO 恢复文件。
3. 补齐 PPO 参数、输出路径、验证名单、真实权重检查与 Slurm 入口。初始沿用既定参数：actor lr=3e-5、critic lr=1e-4、clip=0.1、每批最多 4 遍、target KL=0.01、entropy=0.001、rollout=16 plans、mini-batch=4；不同时改奖励、网络或优化算法。
4. 先复用远端已存完整回放做冻结 actor 的诊断；随后用临时副本做一次采样/更新/保存/恢复检查，丢弃测试更新。仅在 Slurm 计算节点运行，不重做 BC 或重新标定。
5. 完成接线和诊断后先汇报，再讨论启动 seed=7 的三个独立 PPO 先导。建议先导使用已有 1000 请求 train `[315,758,858,403]`，val `[5,158]`，不使用 train 983 进行正式策略更新；完整正式训练与 test 暂不启动。
