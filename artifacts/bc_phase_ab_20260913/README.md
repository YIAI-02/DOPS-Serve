# 三模型与共享 BC：从零训练 50 轮

## 当前追加：四路均续训至 E100

2026-09-14 用户先确认 1.8B、7B，随后追加 14B 和共享网络：四路均从各自 E50 恢复网络、Adam 和随机状态，继续 E51–E100；数据、loss、学习率及验证口径保持不变。追加请求同时提到“再 40 轮”，但当前实际为 E50，按明确的总目标 E100 执行，因此本次为再 50 轮。

- 先在计算节点运行 `code/slurm/run.sh resume-preflight --routes qwen_14b shared --workers 4`，核对 E50 的优化器步数、验证 loss 和自主执行结果；通过后执行 `code/slurm/submit_continuation.sh qwen_14b shared`。旧两路检查和作业记录保留。
- 续训记录和作业位于 `fresh_b50_20260914/continuation_to_e100/`。结果仍在各路原 `training/<route>/`，保留 E50 汇总副本并生成累计 E1–E100 CSV、loss、吞吐和 `epoch_over_pd.png/svg` 曲线。共享每轮三个模型分别一行；计算 Adam 步数时每轮只计一次。
- 每 2 小时检查一次，完成后汇报 E100 对比 E50、前后各 50 轮最佳成绩及相对 PD 的变化。禁止自动重连；异常或连接失效时通知。

2026-09-14 按用户要求，在本目录的 `code/` 直接改为 B 的三阶段等权 loss，四路均从零开始。

|项目|本轮设置|
|---|---|
|训练路线|qwen_1.8b、qwen_7b、qwen_14b 独立网络，另加一个三模型共享网络|
|初始化|每路新建网络、Adam 和随机状态；固定 seed=7 供复现；不加载旧检查点|
|loss|prefill、decode t=0/1/2、decode t>=4 各占 1/3；组内计划等权，计划内学习节点等权，任务等权|
|共享数据权重|三个模型各 48 个训练任务，因此模型等权；逐模型单独评价|
|数据|复用 `bc_static_c026_20260913` 的完整赢家教师；每模型训练 48 任务/11616 现场，验证 8 任务/1936 现场|
|网络与优化|原 4×256 MLP、GRU256；lr=1e-4，batch=16 份计划，梯度裁剪=1|
|轮次|从 E1 到 E50，完整遍历数据；独立网络每轮 726 次更新，共享网络每轮 2178 次更新|
|主吞吐|验证集总输出 token / 总模拟执行秒数；BC、教师和 PD 均排除决策耗时，BC 决策耗时另记|
|每轮 loss|本轮更新后，在完整固定训练集/验证集重新计算三阶段等权目标；原计划等权 NLL 只作辅助记录|

保留 C026 fast NPU/PIM、TP=2、原赢家轨迹、三设备动作、decode stride=2、原数据划分及后端。本轮不重新选教师，不引入新状态数据，不运行 PPO 或 test。50 轮完成后等待用户决定。

## 路径与工件

HPC 根目录：`/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_phase_ab_20260913/`。

- 当前源码：`code/src/dops_serve/static_bc_ab.py`；复用同目录 `static_bc.py` 和 `placement_policy.py`。
- 新结果：`fresh_b50_20260914/`，不覆盖旧 `training/A/`、`training/B/`、`initial/` 和诊断报告。
- 原 A/B 源码归档：`history/ab_code_before_fresh50_20260914.tar.gz`。
- 新配置与作业：`fresh_b50_20260914/{protocol.json,preflight_passed.json,jobs.tsv,logs/}`。
- 每路结果：`fresh_b50_20260914/training/<route>/training_progress.csv`、`report.md`、`epoch_NNN_cases.csv`、`epoch_NNN.pt`。
- 每个模型单独 CSV：`training_progress_<model>.csv`。
- 每个模型曲线：`figures/<model>/epoch_loss.png/svg`、`epoch_throughput.png/svg`；另存原 NLL、阶段 NLL、自主动作一致率曲线。
- `train_loss/val_loss` 是当前实际 B 目标；`train_common_nll/val_common_nll` 是原口径，不混用。

## 执行与验收

1. 数据节点上传代码，登录节点提交 `code/slurm/run.sh preflight`。
2. 计算节点执行单元检查、复制 168 个 train/val 教师任务、三模型 GPU 更新和完整自主回放。
3. 预检通过后，登录节点执行 `code/slurm/submit_training.sh`，并行提交四路 GPU 作业。
4. 各独立路 16 CPU/1 GPU/8 验证进程，共享路 32 CPU/1 GPU/24 验证进程；由调度器分配 GPU80G 节点。
5. 每轮生成完整 CSV 与曲线；验证没完成的 epoch 仅报告进度。按此前用户授权每小时检查累计结果，完成后暂停，不自动重连 SSH。
