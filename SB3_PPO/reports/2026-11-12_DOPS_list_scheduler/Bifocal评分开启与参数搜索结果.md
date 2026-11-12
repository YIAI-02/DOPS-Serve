# Bifocal 评分开启与参数搜索结果

本报告针对原评分的参数搜索；后续修改 far focus 的实现与实验见同目录 `Bifocal_far修改与Qwen1.8B结果.md`。

日期：2026-11-12。已启用前瞻、权重复用、一致性惩罚和 decode 摊销，并完成 211 组不同参数、256 次搜索及多 seed 运行。当前找到的最佳参数在 seeds 0/1/2 下均为 **64.615654 ms**，比原 Bifocal 的 138.969641 ms 降低 **53.50%**；仍比全 PIM 慢 **1.73%**，比已有 PPO best 慢 **1.89%**。

这是当前固定 workload 和已搜索参数集合中的最佳结果，不是全局最优证明，也没有进行跨 workload 验证。以下时间均来自 fast 模拟，不是真机测量；目标不包含 Python 调度程序自身的运行耗时。

## 当前最佳配置

保存位置：`DOPS-Serve/SB3_PPO/configs/bifocal.json`。

| 参数 | 值 | 含义 |
| --- | ---: | --- |
| `h` | 3 | 前瞻链最多包含 3 个算子 |
| `gamma` | 0.1 | 前瞻完成时间在 EFT/前瞻混合评分中的权重；不同于 PPO 折扣 gamma |
| `lambda` | 0.25 | 违反前瞻放置提示时的传输代价惩罚系数 |
| `plan_hint_max` | 3 | 保留的前瞻节点提示数上限 |
| `eta` | 80 | prefill 权重复用收益的评分系数；不是 80 个 token |
| `amort_alpha` | 0.5 | decode 迁移摊销偏置的系数 |
| `reuse_window` | 64 | 每次评分使用的固定复用次数假设 |

入口强制打开前瞻，decode 摊销默认打开；摊销 `rmin=1`、`reuse_prob=1` 固定。评分形式仍沿用原实现：`0.9 × EFT + 0.1 × 前瞻窗口估计 + 权重复用偏置 + decode 摊销偏置`，一致性惩罚包含在前瞻估计中。

**固定窗口不读取真实剩余输出长度。** 每次传入 `total_decode_tokens = current_token + 64`，所以内部计算出的复用窗口始终为 64，最后一个 token 也不缩短。64 是本次离线调参选出的常量，不是运行时对终止位置的读取；它在其他输出长度上是否有效尚未验证。

当前最佳配置的实际计数：前瞻估计调用 93,789 次，一致性冲突候选 88,871 次，权重复用非零偏置 2,436 次，decode 摊销非零偏置 38,688 次。说明这些项实际进入了评分；计数不是各项独立性能收益的消融证明。沿用原 `reset_state()` 语义，未把一致性提示改造成跨 token 持久状态。

## 同一计时规则下的结果

| 策略 | Prefill / ms | Decode / ms | 完整请求 / ms |
| --- | ---: | ---: | ---: |
| 全 NPU | 86.622866 | 193.232694 | 279.855560 |
| Prefill NPU / Decode PIM | 86.622866 | 62.430770 | 149.053636 |
| 原 Bifocal 设置 | 43.673644 | 95.295997 | 138.969641 |
| **本次 Bifocal 最佳** | **43.897867** | **20.717787** | **64.615654** |
| 全 PIM | 44.629061 | 18.888757 | 63.517819 |
| 已有 PPO best：D/seed0 | 43.892663 | 19.523315 | 63.415978 |

Bifocal 最佳配置的 seed 0、1、2 均为 64.61565424835646 ms，样本标准差为 0。这是同一确定性 workload 下平局处理随机种子的结果，不代表不同请求之间没有波动。

相对全 PIM，当前 Bifocal 的 prefill 快约 0.731 ms，但 decode 慢约 1.829 ms，剩余差距主要体现在 decode。参数搜索已明显修复原设置的低效选择，尚未找到超过全 PIM 的组合；启用评分本身不保证贪心列表调度胜过所有固定策略。

## 搜索过程与范围

每个候选均完整执行 prefill 64 + 64 次 decode，先用 seed0 比较，再对每阶段前 5 名补 seeds 1/2，按三 seed 平均耗时选择。四阶段共有 256 条运行记录，其中包含 7 条跨阶段重复的参数/seed；去重后为 211 组参数、249 个参数/seed 组合，19 组参数完成三 seed 比较。单独的基线检查、最终计划回放和配置复现不计入 256 次搜索运行。

| 阶段 | 阶段内参数组数 | 运行记录数 | 阶段最佳三 seed 均值 / ms |
| --- | ---: | ---: | ---: |
| 96 组初筛 + 48 组局部细化 | 144 | 154 | 79.372218 |
| 扩展复用评分等参数边界 | 15 | 25 | 66.698062 |
| 高复用评分区域细化 | 40 | 50 | 65.085361 |
| 前瞻长度、权重与较弱摊销组合 | 17 | 27 | 64.615654 |

阶段间存在重复参数，组数不能直接相加当作不同参数总数。实际尝试的轴值如下，未穷举它们的笛卡尔积：

- H：2、3、4；gamma：0.025、0.1、0.2、0.4、0.6、0.8、0.9、1。
- lambda：0.01、0.05、0.1、0.25、0.5、1、4；提示数：1、2、3、5、8。
- eta：0.5、1、2、5、10、20、40、60、80、100、160。
- 摊销 alpha：0.1、0.25、0.5、1、1.5、2、4、6；固定复用窗口：4、16、32、64、128、256。

在阶段最优参数附近，把 eta 从 5 增至 80 带来明显改善；随后降低摊销强度，再联合调整 H/gamma，进一步缩小差距。不能把此处 eta=80 解释成通用最优值。

## 参数对齐与验证

沿用 PPO 的 Qwen 1.8B、28 层、FP16、batch=1、TP2、1 NPU + 2 PIM、fast/fast、冷缓存、host/ND 权重、真实 KV ledger 和连续通信计时；decode 在 token 0、4、8…60 刷新计划。Bifocal 仍允许逐节点/逐分片放置和自由拓扑顺序，PPO 使用九类共享二元动作；统一的是成本与执行规则，动作空间没有强制统一。

- 全 NPU、全 PIM 和原 Bifocal 设置复现既有数值，检查记录在 `runs/bifocal_search_check_20261112/checks.json`。
- 四阶段最佳计划均交给 `PrefixExecution` 独立回放，耗时与生成计划时一致。
- 使用保存后的 `configs/bifocal.json` 再次运行，得到 64.61565424835646 ms；独立回放也得到相同值。记录在 `runs/bifocal_selected_20261112/result.json`。
- 新入口更新 `config`、`scheduler_common` 和 `scheduler_bifocal` 的实际模块常量，避免只修改配置字典而未影响已导入常量。

本次新增可复用入口 `dops_sb3/bifocal_search.py`、最佳参数 JSON，并更新 README。未改动后端原生评分公式、PPO 训练实现或模型；本轮在本地完成，未同步或提交 HPC 作业。原生 DOPS CLI 不会自动加载新的参数 JSON，复现需使用下面的入口。

## 复现与工件

```bash
cd /Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/SB3_PPO
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search \
  --params configs/bifocal.json --seed 0 --out runs/new_bifocal_selected
```

输出目录须不存在；`PYTHONHASHSEED=0` 固定 ready 集合的遍历顺序。更改 `--seed` 可复现其他平局处理种子。

汇总工件在本报告同目录：

- `bifocal_search_summary/summary.json`：实际搜索轴值、三 seed 候选、最佳参数、评分计数及基线比较。
- `bifocal_search_summary/all_trials.csv`：256 条搜索运行记录。
- 四阶段日志：`bifocal_search.log`、`bifocal_search_boundary.log`、`bifocal_search_high_reuse.log`、`bifocal_search_joint.log`。

完整候选、逐次结果和节点计划位于 `DOPS-Serve/SB3_PPO/runs/` 下的 `bifocal_search_20261112`、`bifocal_search_boundary_20261112`、`bifocal_search_high_reuse_20261112`、`bifocal_search_joint_20261112`。
