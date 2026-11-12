# Bifocal far 修改与 Qwen1.8B 结果

日期：2026-11-12。已完成本地实现、调参和验证。修改后的 Bifocal 自行生成了全 PIM 主算子放置，三个 seed 的完整请求耗时均为 **63.517612 ms**。比修改前 Bifocal 调参最优的 64.615654 ms 降低 **1.70%**，与全 PIM 持平；比已有 PPO best 的 63.415978 ms 慢 **0.16%**。本轮未达到超过 PPO best 的目标。

所有计划均来自 Bifocal 原始贪心调度过程，没有生成后的迁移、节点放置修改或固定策略替换。此前手动迁移 K 得到的 63.396647 ms 仅保留为历史诊断，没有纳入新方法的性能结果。

## 修改范围与实际入口

实现位于 `DOPS-Serve/SB3_PPO/dops_sb3/bifocal_search.py`，新类为 `FarFocusBifocal`，通过 `far_mode=both` 启用。

| 部分 | 本次处理 |
| --- | --- |
| Near：EFT | 直接继承原实现，使用相同成本、通信、缓存和排队计时 |
| Near：短链前瞻 | 直接继承链构造及估计函数；H、gamma 可以调参 |
| Ready 选择和节点提交 | 直接继承原 `BifocalScheduler.schedule` |
| Far：prefill 权重复用 | 用后续 decode 的设备成本差替换原 `−eta × reload_cost` |
| Far：decode 摊销 | 用后续驻留执行的设备成本差替换原迁移抵扣项 |
| 设备一致性 | 保留原方法和 lambda；未增加跨 token 提示记忆 |

后端原 `scheduler_bifocal.py` 与下载的 DOPS 源码逐字节一致；本轮没有改动它。新评分以继承方式接入同一贪心调度器。原生 DOPS CLI 不会自动读取新实验配置，复现应使用本报告的入口。

## 最终保留的 far 评分

对有权重算子 v 和合法设备 d，预先估计下一次 decode 的服务时间，以及输出目的设备已确定时的必要传输代价。目前确定的输出目的地包括 K/V 写入所属 PIM。权重能够长期容纳时使用驻留服务成本，否则使用冷服务估计。

令 `C(v,d)` 为上述预估成本，`delta(v,d) = C(v,d) − min_device C(v,device)`。新的评分为：

```text
原 EFT / 短链混合评分 + 原一致性惩罚 + N_future × delta(v,d)

prefill: N_future = R
decode:  N_future = R − 1
```

当前一次执行的冷加载、通信和排队仍由 EFT 计算，不再使用旧 eta 收益或旧 decode 摊销重复抵扣。新模式将 eta 设为 0；`both` 模式不使用旧 `amort_alpha`。`prefill` 模式用于消融，只替换 prefill 项，decode 仍走原摊销。

成本表使用独立的单 token decode 视图，当前 forward 的 shape、成本缓存、设备时钟、链路和权重状态保持原样。下一 token 的 KV 长度由当前已知前缀计算。R 是固定预估窗口，末尾 token 也不缩短；真实输出停止位置不进入新评分。无权重算子和实际 KV 读取继续由原 EFT / 短链处理，本次没有把 far 扩展为完整未来 DAG 的精确求解。

每次刷新建立 1,064 条“有权重算子—合法设备”成本记录，17 次刷新合计 18,088 条。调度时仍逐步选择 ready 节点及设备，没有枚举完整放置组合。

## 最佳参数与生成的方案

配置：`DOPS-Serve/SB3_PPO/configs/bifocal_far.json`。这是本次搜索得到的并列最佳配置之一，选择沿用 lambda=0.25、提示数=3、R=64 的组合。

| 参数 | 值 |
| --- | ---: |
| far_mode | both |
| H | 3 |
| gamma | 0.025 |
| lambda | 0.25 |
| plan_hint_max | 3 |
| 固定复用窗口 R | 64 |
| eta / amort_alpha | 0 / 0，新 far 评分替换原收益与摊销 |

全部 17 个决策段的 504 个主算子节点均选择 PIM：PIM0、PIM1 各 252 个。prefill 的其余节点也位于 PIM；decode 仅第一层 LN 在 NPU，其余计算节点位于 PIM。节点顺序和具体分片选择由 Bifocal 决定，没有强制使用全 PIM 基线的计划。

seed 0、1、2 均为 63.51761173637886 ms，样本标准差为 0。这些 seed 对应同一确定性 workload 下的平局处理，不能据此推断不同请求间的波动或跨 workload 泛化。

## 同一成本与连续执行规则下的对比

| 方法 | Prefill / ms | Decode / ms | 完整请求 / ms |
| --- | ---: | ---: | ---: |
| 全 NPU | 86.622866 | 193.232694 | 279.855560 |
| Prefill NPU / Decode PIM | 86.622866 | 62.430770 | 149.053636 |
| HEFT | 43.679589 | 95.295997 | 138.975586 |
| 原 Bifocal 调参最优 | 43.897867 | 20.717787 | 64.615654 |
| **修改后 Bifocal，自行生成** | **44.628854** | **18.888757** | **63.517612** |
| 全 PIM 固定基线 | 44.629061 | 18.888757 | 63.517819 |
| 已有 PPO best：D/seed0 | 43.892663 | 19.523315 | 63.415978 |

新 Bifocal 与全 PIM 只相差 0.000207 ms，实际视为持平。相对 PPO，新 Bifocal 的 decode 快约 0.635 ms，但 prefill 慢约 0.736 ms，最终慢约 0.102 ms。

固定条件：Qwen1.8B、28 层、FP16、TP2、batch=1、prefill64 + 64 次 decode、1 NPU + 2 PIM、fast/fast、host/ND 权重、冷缓存。沿用 PPO 的 RaggedCostModel、真实 KV ledger、连续通信计时和每 4 token 刷新计划的规则。Bifocal 仍允许逐节点放置及自由拓扑顺序，PPO 使用九类共享二元动作；成本与计时对齐，动作空间没有强制统一。

表内时间是模拟执行耗时，不是真机测量，也不包含 Python 调度生成的墙钟时间。最终 seed0 生成及完整执行约 23.90 秒，独立回放另行执行。

## 搜索范围及未保留的探索

最终保留的评分实现比较了 **75 组不同参数、94 次运行**；其中 9 组参数完成三 seed 比较。H 尝试 2、3、4、5，gamma 尝试 0.01、0.025、0.05、0.1、0.2、0.4、0.8，R 尝试 2、4、8、16、32、40、48、56、64、128，lambda 尝试 0.01、0.1、0.25、1、4，提示数尝试 1、3、8。未穷举这些轴的笛卡尔积。

开发期间另比较了未来 decode 队列估计、依赖汇合余量及其折扣系数。这些探索没有超过最终保留的简单成本估计，相关运行代码已从正式入口移除，源码快照留在 `runs/bifocal_far_20261112/exploration_source.py`。包含这些探索在内，共有 **122 组不同参数/实现组合、166 条运行记录**，其中 21 组完成三 seed 比较。不能把 122 组全部称为最终实现的参数搜索。

各阶段记录如下；阶段间存在重复参数：

| 阶段目录 | 参数组数 | 运行数 | 最佳三 seed 均值 / ms |
| --- | ---: | ---: | ---: |
| first_search | 28 | 34 | 63.517612 |
| boundary_search | 32 | 38 | 63.517612 |
| timeline_search，探索版 | 6 | 10 | 63.517612 |
| slack_search_fixed，探索版 | 4 | 8 | 64.006454 |
| discount_search，探索版 | 8 | 12 | 63.517612 |
| long_search，探索版 | 18 | 24 | 63.519725 |
| joint_search，探索版 | 12 | 18 | 63.519725 |
| depth_search | 4 | 6 | 81.093874 |
| consistency_search | 12 | 16 | 63.517612 |

以上计数不包括初始草稿消融、测试运行和最终配置复现。最终实现没有找到先前手工 K 迁移诊断中的 63.396647 ms 混合放置。单算子的长期服务成本差仍是近似，未能准确刻画完整请求中的并行和关键路径影响；后续若继续研究，应先用具体评分轨迹定位这一差距。

## 验证与来源核对

- **26 项测试通过。** 新增 5 项检查覆盖 near / 贪心函数继承、未来估计不污染执行状态、K 的单 token 输出传输字节数、固定窗口不读取真实停止位置、设备一致性惩罚保留。
- 在最终配置运行时截获 17 次 `export_fixed_plan` 的原始输出，逐份与最终保存计划比较，完全相同。**生成后放置修改次数为 0。**
- 将这些计划交给原 `PrefixExecution` 独立执行，耗时与 Bifocal 生成及执行时完全一致，均为 63.51761173637886 ms。
- 核对全部计划的拓扑顺序、合法设备、固定 KV 写入位置和全 PIM 主算子计数。
- 原后端 `scheduler_bifocal.py` 与下载仓库源文件逐字节一致，EFT、短链、ready 选择和提示更新均直接继承。已有 PPO 训练改动、模型和实验配置保留原状。

本轮在本地完成，没有同步或提交 HPC 作业。

## 复现与工件

```bash
cd /Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/SB3_PPO
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search \
  --params configs/bifocal_far.json --seed 0 --out runs/new_bifocal_far
.venv/bin/python -m pytest tests -q
```

输出目录必须不存在。seeds 1/2 改动 `--seed` 即可。原评分的参数仍在 `configs/bifocal.json`，可通过同一入口复现。

- 本报告同目录的 `bifocal_far_summary.json`：搜索计数、三 seed 数值、基线、评分调用次数及生成来源核对。
- 本报告同目录的 `bifocal_far_trials.csv`：166 条参数/实现探索运行记录。
- `runs/bifocal_far_20261112/final_generated/result.json`：最终配置复现、17 份原始计划、65 阶段时间、未来成本表、独立回放和来源核对。
- `runs/bifocal_far_20261112/final_tests.log`：最终 26 项测试结果。
- `runs/bifocal_far_20261112/`：各阶段候选、逐次结果、三 seed 汇总、日志及探索版本源码快照。
