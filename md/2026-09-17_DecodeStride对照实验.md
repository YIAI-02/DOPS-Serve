# Decode stride 对照实验

## 目标与固定合同

比较 stride=2/4/8/16 的 CPU 完整回放墙钟时间，以及模型执行完成的模拟 makespan。只有结果满足以下门槛，才按用户授权取消当前 PPO 并仅修改 stride 重启。

- 使用三模型原 BC E0 权重、argmax、三负载 0.5/1/1.5；复用当前标定、C026、长度/KV、FIFO 与逐 token 执行。
- 每个 batch 的局部 decode 计数在 0/1/2/4 密集决策，此后按 stride 刷新；新 prefill 和请求完成仍按当前逻辑重置计数。stride=2 与现行规则完全一致。
- 中间 token 继续模拟执行，仅复用 placement plan。记录完整 trace 的 `simulated_s`（包含空闲及排空）作为主要 makespan，同时记录逐请求完成时间，防止只看汇总而忽略内部变化。
- 冻结权重回放不能证明后续随机 PPO 的轨迹或收敛完全相同；本实验只验证当前策略的服务行为与 CPU 回放速度。

## 执行流程

1. 本地短检查：在真实请求上比较实验 stride=2 与正式代码的结果，检查 2/4/8/16 的实际刷新位置；不修改正式代码。
2. 第一阶段：完整 val trace 184（1,000 请求，9,538 输出 token），三个模型 × 三负载 × 四种 stride，共 36 次回放。
3. 任一模型/负载 makespan 改变的候选，不满足用户条件，停止扩大该候选。若全部候选失败，结束实验，保留当前 PPO。
4. 通过者继续完整 val trace 1（54,584 输出 token）与 191（302,552 输出 token），覆盖中、长输出；然后以反序再跑全部三条 trace，复核速度与 makespan。
5. 所有 trace 均完整排空，不以固定 plan 数或请求前缀代替完整结果。test 集保持锁定。

## 通过条件

- 每个模型/负载/trace/重复均完成全部 1,000 请求，输出 token 数一致。
- 每个单元的 makespan 与对应 stride=2 满足 `abs_tol=1e-9 秒、rel_tol=1e-9`；仅容忍浮点舍入，不接受百分比级退化。
- 在两轮完整测试中，候选的总回放墙钟耗时均至少减少 10%，避免将小幅计时波动当作明确提速。另报告各单元速度、决策数、actor 与模拟器耗时及逐请求完成时间差异。
- 多个候选通过时，选择两轮合计回放墙钟时间最短者。阶段一不能单独触发取消或重启。

## 资源、路径与边界

- 当前 PPO 数组 `14935360` 保持运行，不改源码、protocol、checkpoint。
- 测试使用 CPU 分区 C064M0256G，16 核、单 worker 单线程，最多 16 个并行回放；与现有 PPO 48 核合计 64 核，不申请 GPU。单次作业时限 24 小时，完整任务结果可复用。
- 实验根：`artifacts/burstgpt_stride_20260917/`；对应 HPC 路径在 `/lustre/home/2501111916/workspace/DOPS-Serve/` 下。
- `code/stride_replay.py` 是正式 replay 的独立副本，只增加 stride 参数并保留全部中文注释；其余网络、模拟器和输入代码引用现有正式实验。
- 测试代码先本地完成并检查，经 data 上传，逐文件读回比较一致后，由 login 提交到计算节点运行。
- 证据保存至 `results/`、`progress/`、`summary.json` 和 `logs/`；报告保存在本目录。实验结束后移除仅供测试的一次性助手，保留指标与日志。

## 条件通过后的重启

仅在全部门槛通过后：保留旧结果 → 在独立新实验根准备仅改 stride 的代码和 protocol → 最小检查及 data 同步逐文件核对 → 取消旧数组 `14935360` → 提交新数组。

仍从原 BC actor＋已验收 critic 起点重新开始，不接续 stride=2 的中间 optimizer/RNG；三模型、seeds 4/15/26、32 train/16 val、50 epoch、每五轮验证、学习率、rollout/minibatch、奖励、资源并行上限均不变。新 E0 必须按新 stride 计算。不借此合并 E0、改验证任务分发或增加其他优化。

## 当前进展与新增讨论

- 本地真实请求检查通过：prompt=352、output=40；实验 stride=2 与正式 replay 的 makespan、逐请求完成时间、输出量、plan 数、phase 步数完全一致。4/8/16 的实际刷新位置均通过检查。证据：`artifacts/burstgpt_stride_20260917/local_check/check_passed.json`。
- 该单请求检查只验证实现，不作为完整 trace 的效果或 HPC 提速证据。09:12 CST 已经 data 上传 3 个实验代码文件并逐文件读回一致，同时核对引用的 8 个正式源码文件及 protocol 一致。同步证据：`artifacts/burstgpt_stride_20260917/sync_verified.json`。09:15 CST 已通过 login 提交作业 `14938932`（16 CPU，C064M0256G，24 小时）；当前 PPO 数组未改动。
- 作业已在 09:36:17 CST 正常结束，36 次完整短 trace 回放完成。4/8/16 均在 7B、1.5× 负载下改变 makespan，故全部候选被筛除，未扩大测试，也未替换原 PPO。详见 `2026-09-17_DecodeStride对照结果.md`。自动化继续跟进正式 PPO 的每五轮汇报。
- 用户追加讨论缩减 trace 数：建议初步 50 epoch 使用固定 8 train＋4 val，保留完整到达/排队、三模型、三负载和三个种子。此项仍待用户选择，当前没有修改正式配置或取消原作业；若采纳，将替代上文 32/16 的保留约束，独立于 stride 是否通过。
- 当前 32 train 合计每负载 4,252,260 输出 token；16 val 合计每负载 1,434,933 输出 token。缩减子集应覆盖输出工作量与请求长度范围，实际速度不能只按 trace 数比例推算。
