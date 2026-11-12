# DOPS list scheduler 核对

后续已完成配置核对、固定策略对照及参数探针，见 [参数核对与固定策略对比](参数核对与固定策略对比.md)。下文保留第一次原生入口试跑记录；138.789466 ms 对应关闭 decode 摊销的设置。

## 当前 PPO 的计算模式

本地 `DOPS-Serve/SB3_PPO/configs/environment.json` 与 A/B/C/D × seeds 0/1/2 的 12 份正式运行配置均为：

```json
{"npu_backend": "fast", "pim_fast_mode": true}
```

`FixedPlacementEvaluator` 将两项直接传给 `RaggedCostModel`。NPU 使用 fast 分析模型，PIM 使用 fast 分析路径；此次没有调用 PIM 的 Ramulator trace 模式。fast 模式仍保留缓存、权重加载及通信的模拟代价。

## 仓库与调度入口

- 来源：<https://github.com/YIAI-02/DOPS>
- 默认分支：`micro26_pieak_final`。
- 本地路径：`/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS`。
- 已连接 Git 上游分支，可正常拉取更新；本地工作区干净。采用 depth=1 和 sparse checkout，下载 503 个文件，仅排除 `experiment/demo/examples/` 中 9 个约 333 MB 的预生成演示文件。源码、配置和 PIM cache 均已下载。
- `src/scheduler/scheduler_heft.py`：`HEFTScheduler.schedule()`，先计算 upward rank，再从 ready queue 选择任务，对可用设备估算最早完成时间并提交选择。
- `src/scheduler/scheduler_bifocal.py`：`BifocalScheduler.schedule()`，在 HEFT 基础上提供前瞻、权重复用和跨 token 摊销评分。
- `src/main.py evaluate --config ... --algo HEFT,Bifocal`：现成评估入口，可导出设备放置、执行顺序与时间线。

已逐字节比较，新下载仓库的 `scheduler_heft.py`、`scheduler_bifocal.py`、`config.py` 和 `hardware_1npu_2aim.json` 与当前 PPO 的 `backend` 中对应文件一致。因此可以直接复用已有后端里的调度器。

HEFT/Bifocal 都是启发式策略生成器；能产生可执行候选，不保证全局最优。

## 与当前 PPO 对照前必须对齐的合同

| 项目 | 当前 PPO | 原生 DOPS scheduler |
| --- | --- | --- |
| 放置自由度 | 28 层共享 9 类算子选择，PIM 分片归属固定 | 可逐节点、逐层、逐分片选择 |
| 执行顺序 | 固定拓扑顺序 | 根据 ready queue / 策略排序 |
| 小算子 | LN、Softmax 等有固定跟随规则 | 原生规则未必相同 |
| Decode 刷新 | 第 0、4、8… 个 token 更新，共 16 次 | stride=4 之外，原生代码还在第 1 个 token 额外更新 |
| 输出长度信息 | 增强观测不含未来输出长度 | Bifocal 默认跨 token 摊销会读取 total_decode_tokens |
| 执行计时 | 连续 KV ledger、forward 边界、额外的 AllReduce 链路占用处理 | 原生 CLI 的模拟路径 |

因此原生 `best_summary` 的毫秒值不能直接与 PPO 的约 63.4 ms 做同口径排名。若要回答同一动作空间里是否存在更好的放置，应把候选转成 17×9 计划，并交给当前 `full_replay` 计时；如果候选本来就逐层不同，必须明确转换方式或另设允许逐层放置的对照。

另外，仓库默认 `SCHED_JOINT_LK_GAMMA=0`，`use_lookahead` 要求 gamma>0，因此默认前瞻并未实际启用。`decode_amort_enable=false` 可以关闭对剩余 decode 长度的利用；本目录的试跑配置已显式关闭。

## 原生入口试跑配置

`native_probe_config.json` 固定 Qwen 1.8B、FP16、batch=1、64+64、TP2、fast NPU/PIM、ND 权重和当前 PPO 使用的硬件文件，算法为 HEFT/Bifocal，scheduler seed=0。

该试跑用于检查候选生成和结果导出，不是完成了公平的 PPO/list scheduler 对照。原生默认前瞻权重保持不变，不修改现有 PPO 源码或模型。

## 原生入口试跑结果

两种算法均正常退出，结果保存在 `native_output/qwen_1.8b_fp16_b1_s4/`。

| 算法 | Prefill (ms) | Decode (ms) | 总时间 (ms) |
| --- | ---: | ---: | ---: |
| HEFT | 43.666347 | 95.123119 | 138.789466 |
| Bifocal | 43.666347 | 95.123119 | 138.789466 |

总时间相同不代表策略完全相同：prefill 有 165 个节点的设备选择不同，首个 decode token 有 102 个不同。当前配置下 Bifocal 的前瞻未启用、跨 token 摊销显式关闭，不能把本结果解释为调好参数后的 Bifocal 最佳表现。

HEFT 的第 0 层 Q 在 prefill 时把 shard0 放到 PIM0、shard1 放到 NPU；首个 decode token 又把两个 Q 分片都放到 PIM1。这两种选择均不属于当前 PPO 的九个二元动作所能表达的放置。原生日志也确认在 decode token 0、1、4、8、…、60 重新调度。

这次试跑确认了列表调度入口、候选生成和完整 trace 导出可用；没有证明它优于当前 PPO。下一步若比较策略质量，应在当前 `FixedPlacementEvaluator` 中固定相同执行合同，再为九类共享放置生成候选并回放，保留实际 KV 状态、缓存及通信代价。直接拿两个原生 CLI 的总时间排名会混入执行模型差异。

复现命令：

```bash
cd /Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS
/Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/SB3_PPO/.venv/bin/python src/main.py evaluate --config /Users/jiaqiyang/Desktop/Daily_2026/02_Projects/DOPSRT/reports/2026-11-12_DOPS_list_scheduler/native_probe_config.json
```
