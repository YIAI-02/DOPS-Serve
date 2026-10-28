# BC、PPO 与基线比较及一次 PPO 续训

最新完成核对（2026-09-06 12:59:21，北京时间）：全部 165 个比较条件与 3 个续训模型正常完成，最终汇总结束于 12:49:55。完整分析见 comparison_20260906/comparison_analysis.md；下文提交阶段状态保留作执行记录。

2026-09-06 09:35:47（北京时间）核对：51 项计算节点测试通过，165 条验证条件清单已生成。三个 PPO 续训任务和首批 24 个现有模型比较条件正在运行；续训后的 45 个条件及汇总已按依赖提交，尚无完整性能结果。

## 比较口径

- 模型：Qwen-1.8B、7B、14B；1 NPU、2 PIM；沿用 `training_protocol.json` 的后端、参考速率、SLO 和奖励参数。
- 数据：固定使用协议中的完整 validation trace，每条 1000 个请求；到达率为各模型参考速率的 0.5、0.75、1、1.25、1.5 倍。此次用于决定训练规模，不读取 test。
- 方法：Static、固定 order 的 Bifocal、三个种子的 BC、三个种子的现有 PPO，以及三个种子再完成一轮全部模型训练 trace 后的 PPO。
- 每个方法都走相同的 compiler、evaluator、FIFO 接纳、KV 管理和连续回放。模型决策使用确定性 argmax，完整特征、编译与搜索开销计入服务时间。
- Bifocal 复用当前后端的设备评分函数，并限制为统一拓扑顺序和每节点固定 PIM。其 prefill/decode 规划继承当前快照；不读取真实剩余输出长度。实际启发式参数写入每份结果。
- 现有 BC/PPO 与基线共 120 个条件，续训后另有 45 个条件，共 165 个完整条件。Static/Bifocal 为固定参考，不伪装成三个独立训练种子。

## 如何判断 PPO 是否改善

1. 在相同模型、种子、负载和 trace 下，配对比较 BC → 原 PPO，以及原 PPO → 续训 PPO。
2. 主指标是五档负载中 E2E 违约率不超过 1% 时的最高输出 token 吞吐；没有达标档位时明确记为未达标。
3. 同时检查 p99 E2E、违约率、排队时间和决策开销。吞吐增加而违约率恶化不等于可用性能改善。
4. 报告全部三个种子和各模型的表现；出现方向不一致时报告混合结果，不挑最好的一组代表整体。
5. 此轮只有一条固定验证 trace，最多支持该验证条件下的改善判断，不能直接证明稳定泛化或真实 tensor 推理收益。

## PPO 续训边界

旧 `ppo.pt` 没有 Adam 和随机数状态，第一次续训只能加载原权重，重新初始化 Adam 与 Torch 随机数状态；会根据已有轮次重建 Python 侧模型顺序/负载抽样位置。结果明确标记为 `weights only; fresh Adam and Torch RNG`，不称为无缝断点恢复。

新增 checkpoint 保存 optimizer、Torch/Python RNG 和训练顺序，之后的完整轮次边界可精确恢复。此次只额外跑一轮全部模型训练 trace；新文件放到独立目录，保留现有 `learning/seed_{7,17,27}/bc.pt` 和 `ppo.pt`。

## 代码及产物

- `src/dops_serve/placement_baselines.py`：固定 order 的 Bifocal 适配。
- `src/dops_serve/placement_comparison.py`：比较清单、单条件运行、配对汇总。
- `src/dops_serve/placement_evaluation.py`：复用完整回放指标，补充决策开销总量、p95 和时间占比。
- `src/dops_serve/placement_training.py`：从 PPO 权重继续、保存恢复状态、累计轮次和更新次数。
- `src/dops_serve/cli.py`、`slurm/op_policy_learning.sbatch`：比较与续训命令入口。
- `tests/test_placement_comparison.py`、`tests/test_placement_training.py`：固定动作空间与快照隔离、配对与达标口径、分段/连续训练一致性检查。

HPC 预定输出根目录：

```text
/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/qwen_1npu2pim_op_policy_v1/
  comparison_20260906/
    comparison_cases.json
    cases/NNN/result.json
    cases/NNN/<trace>_windows.jsonl
    cases/NNN/<trace>_requests.jsonl
    comparison_summary.json
    comparison_summary.md
  continuation_20260906/seed_{7,17,27}/
    ppo.pt
    ppo_curve.jsonl
    ppo_summary.json
```

## 验证与执行状态

- 本地 Python 语法、Shell 语法和 Git diff 空白检查通过。
- 不调用后端的配对汇总逻辑测试通过：同种子条件配对、违约率门槛、缺失条件不报整体完成。
- 计算节点集成测试 `14767579`：51 项通过，20.817 秒，作业退出码 0；其中分段 PPO 与连续训练的模型参数完全一致。日志：`comparison_integration_tests.log`。
- 以下作业均已提交；当前未观察到失败。HPC 登录节点只检查/提交，数据节点传输代码，计算节点运行测试与实验。


| 阶段 | 作业 ID | 状态（09:35:47） |
|---|---|---|
| 生成比较清单 | 14767580 | 完成，165 条，val split |
| 三种子 PPO 续训 | 14767581，array 0–2 | 3 个运行中，已进入第 2 轮 |
| 现有模型/基线比较 | 14767584，array 0–119 | 24 个运行、96 个等待并发空位 |
| 续训后比较 | 14767609，array 120–164 | 等待续训成功 |
| 最终配对汇总 | 14767610 | 等待两个比较数组结束 |

原始训练模型与协议未覆盖。首次续训从旧 PPO 权重加载，Adam/Torch RNG 重新初始化；新 checkpoint 将保存完整恢复状态。

定时监控 `dops` 每 15 分钟检查本任务，状态无重要变化时保持安静，只在失败、需要操作或全部完成时通知。不自动新建或重连 SSH。两条已认证控制连接保留给监控复用，任务全部结束后关闭并清理 socket。当前状态见 `comparison_monitor_status.json`，提交记录见 `comparison_jobs.json`。
