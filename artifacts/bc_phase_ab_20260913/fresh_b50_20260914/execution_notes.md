# Fresh B50 执行记录

## 最新追加：14B 和共享网络到 E100

2026-09-14 用户追加要求 14B、共享网络也训练到 E100。两路当前实际完成 E50，因此按总目标新增 E51–E100 共 50 轮；不从最佳轮次改接，不重置模型、Adam 或随机状态，不改变数据、B loss、学习率及验证方式。

代码已扩展共享恢复：按每个 epoch 的一次实际更新统计 Adam 步数，避免共享日志的三模型记录重复累加。计算节点预检 `14880550` 已完成（0:0，2分36秒），10项测试通过。14B 从36300步、共享从108900步恢复；四组 E50 验证 loss 和抽查完整任务的自主执行耗时均与原记录完全一致。原 1.8B/7B 检查结果保留。

已提交且进入 RUNNING：14B=`14880554`（`bc100-qwen_14b`，16 CPU/1 GPU/8验证进程，l12gpu23）；共享=`14880555`（`bc100-shared`，32 CPU/1 GPU/24验证进程，l12gpu07）。证据见 `continuation_to_e100/preflight_passed.json`、`jobs.tsv` 和两路 `continuation_initialized.json`。旧两路作业表记录保留。

实际启动已核对：14B 使用 A800 80GB，已完成E51并进入E52；共享使用 A100 80GB，已完成E51参数更新并开始计算loss。两路 `continuation_initialized.json` 均确认网络、Adam、随机状态恢复，目标100，起始Adam步数分别36300和108900。

1.8B/7B 已正常完成 E100，作业分别为14878513/14878514，退出码均0:0；汇总见 `continuation_to_e100/first_two_e100_report.md`。四路总目标均为100，自动化 `dops` 已恢复每2小时检查，普通运行保持安静，结束后报告。

## 当前追加续训

2026-09-14 用户授权将独立 1.8B、7B 从 E50 继续到 E100。恢复网络、Adam、Python 批次随机状态和 torch CPU/CUDA 随机状态，沿用原 B loss、教师、验证数据和优化参数。每路新增 50 轮、36300 次更新；14B 和共享网络仍完成原定 50 轮。

续训预检 `14878508` 已在计算节点完成，ExitCode=0:0，耗时 1 分 34 秒。10 项测试通过；两路 E50 的验证 loss 和各一个完整任务的自主执行耗时均与原记录完全一致，Adam 步数均为 36300。证据见 `continuation_to_e100/preflight_passed.json`。

两路续训已提交：1.8B=`14878513`（`bc100-qwen_1.8b`），7B=`14878514`（`bc100-qwen_7b`），每路 16 CPU、1 GPU、8 验证进程。作业表见 `continuation_to_e100/jobs.tsv`，恢复证据见各模型的 `continuation_initialized.json`。原始四路作业表保留。

启动核对：两路均使用 A800 80GB，分别在 l12gpu23 和 l12gpu27；恢复记录确认 Adam 步数 36300、学习率 1e-4，网络和随机状态恢复成功。E51 已完成自主验证，两路均已进入 E52，累计 CSV 含 E1–E51。原 E1–E50 各列数值与保留副本逐项一致。自动化 `dops` 已启用，每 2 小时检查，完成后汇报。

续训结果追加到原模型目录，E1–E50 汇总另存 `training_progress_epoch_050.csv`、`report_epoch_050.md`、`complete_epoch_050.json`。每 2 小时复用连接只读检查，完成后提供累计 loss、吞吐和相对 PD 曲线；普通检查保持安静，失败或需要用户操作时通知。

2026-09-14 按用户要求，在父目录现有 `code/` 直接修改正式入口，保留原网络、教师和验证数据，采用 B 的三阶段各 1/3 loss，四路重新初始化训练 50 轮。

## 已完成

- 原 A/B 源码和 README 已归档于父目录 `history/ab_code_before_fresh50_20260914.tar.gz`；旧 A/B 检查点和结果保留。
- 修改 `static_bc_ab.py`、对应单元检查与两个 Slurm 脚本，正式入口不恢复旧检查点。
- 预检 14878054：9 项单元检查通过；随后发现预检计算初始 loss 后未切回训练模式，GPU GRU 反向更新报错。已补充 `actor.train()`，未更改目标或数据。
- 修正后预检 14878056：COMPLETED，ExitCode=0:0，耗时 50 秒。9 项检查通过，三模型 198 个小规模现场 GPU 更新及三项完整自主回放通过。预检权重不进入正式训练。
- 正式四路作业已提交并进入 RUNNING；各请求 1 GPU。独立路 16 CPU/8 验证进程，共享路 32 CPU/24 验证进程。
- 四路 `initialized.json` 均已核对：新网络、无来源检查点、Adam 初始步数为 0、起始 epoch 为 0、目标 50 轮。独立路各 11616/1936 个训练/验证现场，共享路 34848/5808 个。
- 1.8B、7B 的 E1 完整 CSV 与 PNG/SVG 已生成；核对三阶段 NLL 平均值等于主 loss，8 个任务的总输出量除以总执行时间等于汇总吞吐，结果见 `launch_verified.json`。

|路线|作业 ID|作业名|分配节点|
|---|---|---|---|
|1.8B 独立|14878057|bc50-qwen_1.8b|l12gpu06|
|7B 独立|14878058|bc50-qwen_7b|l12gpu06|
|14B 独立|14878059|bc50-qwen_14b|l12gpu23|
|共享|14878060|bc50-shared|l12gpu23|

## 结果口径

- 四路均为 E1…E50，固定 seed=7 仅供复现，不搜索种子。每模型训练 48 任务/11616 个现场，验证 8 任务/1936 个现场。
- `train_loss`、`val_loss`：epoch 更新完成后，在完整固定数据上重新计算的 B 目标；三个阶段的 NLL 各占 1/3。
- `train_common_nll`、`val_common_nll`：原计划等权目标，仅供辅助分析，不作为本轮主 loss。
- `bc_throughput`：完整自主验证总输出 token / 总模拟执行时间；不含决策时间。`decision_s` 单独记录。
- `training/<route>/training_progress.csv` 和 `report.md` 每轮更新；共享路逐模型三行，不合并吞吐。
- 各模型曲线在 `training/<route>/figures/<model>/`，含 `epoch_loss.png/svg`、`epoch_throughput.png/svg`。
- 每完成一轮自主验证才产生完整成绩；训练中的 loss 不代表吞吐成绩。
- test 锁定，无 PPO；50 轮完成后停止等待用户。

沿用既有每小时监控，自动化 ID 为 `dops`。仅复用现有登录/数据控制连接，不自动重连；全批终止后暂停。模型、优化器和随机状态是否从零初始化，以各路 `initialized.json` 为证。
