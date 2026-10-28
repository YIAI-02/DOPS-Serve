# BurstGPT：BC 后 PPO 方案

2026-09-14｜用户已批准实现并开始训练。BC 起点已指定，先执行 seed=7 先导，再依据实测冻结正式资源预算。

## 1. 目标与基线

- **目标**：保留连续到达、FIFO 排队和动态 batch，优化 `总输出 token／完整 trace 模拟耗时`；包含空闲和最终排空，决策耗时另报。
- **路线**：PPO；Qwen-1.8B、7B、14B 三个独立 actor＋一个共享 actor。
- **基线**：[当前 BC 入口](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/bc_phase_ab_20260913/code/src/dops_serve/static_bc_ab.py)。沿用 StaticPlacementActor 的 4 层宽256编码器、GRU256及三设备动作头。
- **配置**：C026，1 NPU＋2 PIM，FP16，TP QKV/FFN=2，fast/fast，host ND，沿用原 overlap=1.0/0.5。
- **指定 BC 检查点**：独立1.8B E57、7B E77、14B E42，共享 E35；来自 `bc_phase_ab_20260913/fresh_b50_20260914/training/<route>/epoch_<NNN>.pt`。

## 2. 数据与容量

源文件：`/lustre/home/2501111916/workspace/DOPS-Serve/data/BurstGPT_without_fails_1.csv`，共 **1,404,294 条请求**。[统计证据](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_dataset_stats_20260914/summary.json)

- 保留全部类型、重复行及真实长度，统一公开 `max_tokens=7315`；真实 output_len 仅用于环境终止请求。
- 按时间稳定排序，以行数 70%／15%／15% 切分，再切连续 1,000 请求的 trace，保留尾段。每条 trace 从相同空队列、空 KV 和初始缓存状态开始，保留内部到达间隔并完整排空。

| 区间 | 请求数 | trace 数／尾段请求数 |
|---|---:|---:|
| train | 983,005 | 984／5 |
| val | 210,644 | 211／644 |
| test | 210,645 | 211／645 |

- **负载**：0.5×／1×／1.5×。仅用训练区 PD 饱和回放标定吞吐 S_m；设原始平均输出到达率为 A，到达间隔统一乘 `A/(负载×S_m)`，固定应用到 val/test。
- **KV**：逐 PIM 按 `Σ(prompt_i+7315)` 预留，`max_batch=16`；容量不足时排队，完成后释放。最大输入29,665，对应最大单请求预留36,980 token。
- 两块 PIM 的 runtime 总预算约 **30.4 GiB**，与权重、激活共享；安全 KV 预算待先导核对非 KV 峰值。保留14B超过标注8192上下文的请求，其结果按长度外推的分析模拟解释。

## 3. BC 迁移与执行

1. **输入从实际准入请求生成**。原 state[0:5] 依次改为：phase 请求数、平均 prompt、BC 训练集原已缩放 decode 特征均值、平均 generated−1、平均真实 kv_len；prefill 的后两项为0。沿用原缩放与 BC 任务／模型等权。
2. **state 47→56，node 保持35**。新增9项：公开上限、waiting数、已准入请求数、平均／最大等待时间、prompt最大值／标准差、decode KV长度最大值／标准差。新增输入列权重置零，其余网络权重继承；空等待统计和prefill历史KV统计为0。
3. **精确成本**：node[18]=Σquery_len；node[19] 在prefill为ΣPᵢ(Pᵢ+1)/2，decode为Σ(kv_lenᵢ+1)。历史输入使用最近实际执行的P/D计划；保留原kv_used槽位的预留记账语义，实际已写KV读请求台账。
4. **执行**：入队→准入→构造phase输入→生成／复用计划→StaticPlacementCompiler＋schedule_with_plan→更新输出／KV→完成释放。保留三设备身份、拓扑和合法跟随规则。
5. 每次prefill生成新计划并产生首token，随后执行output_len−1次decode。active加入／退出或新prefill后重置局部计数，按 **0、1、2、4、6…** 刷新decode计划；计数与请求生成进度分开。
6. 迁移后冻结actor为 **E0**，先自主评价；PPO直接从E0起步，不增加BC适配训练或教师辅助损失。critic独立复制BC编码器／GRU，新增价值头，仅训练时使用。

## 4. PPO 合同与参数

- **动作**：每个可学习节点选择NPU/PIM0/PIM1。训练采样，评价argmax；固定／跟随节点进入GRU历史，但不计随机动作损失。actor和critic均不得读取真实未来输出长度。
- **奖励**：`r=−Δt/C_m`；C_m取训练区PD饱和回放的平均trace耗时，三负载共用并冻结。计划内节点奖励为0，执行至下一决策的时间记在最后一个可学习动作之后。
- **回报**：γ=λ=1；`G=后续奖励和+截断处V_old`，`A=G−V_old`。截断取下一真实phase的价值，真正终止时为0；收益／优势停止梯度。
- **采样**：每个环境最多16次完整计划刷新为一段，保留连续现场。同一源trace分别跑独立路线的3个负载组／共享路线的9个模型×负载组；满批最多48／144份计划。
- **更新**：采样批次内策略固定；组内可学习节点loss平均，组系数固定为1/3或1/9。先完成组停止采样、贡献为0，其余组继续；全部完成才换源trace。各组rollout数量另报，组系数相同不代表累计梯度量相同。
- 按完整计划组成mini-batch，起步每组4份；重放原动作前缀，逐节点计算 `exp(new_logp−old_logp)`。优势按组标准化，actor用clipped surrogate＋熵，critic单独最小化回报MSE。

| 参数 | 起步值 |
|---|---|
| Actor／critic学习率 | 3e-5／1e-4 |
| PPO clip／重复训练 | 0.1／最多4遍 |
| 平均KL停止阈值／熵系数 | 0.01／0.001 |
| 梯度范数上限／优化器 | 0.5／两份独立Adam |

KL比较采样策略与更新策略。先导可用train/val调整参数，正式实验前冻结。

## 5. 覆盖、选模与验收

- **覆盖**：每条训练trace在每档负载各跑一次。每独立模型每种子每轮2,952个任务，共享8,856个；四路线×三种子共 **53,136个trace任务／轮**，至少完整覆盖一轮。
- **种子**：先导7，正式7／17／27。原始数据集合和切分相同，改变trace顺序、动作采样及新增参数初始化；正式三种子均重新从同一E0起步，critic／优化器重新初始化。
- **预算**：先导测采样、模拟、更新耗时和内存，再由用户确定正式轮数、并行度及资源预算。
- **日常验证**：固定16条val trace，索引0、14、…、210；约每完成10%覆盖及轮末，在所有组完成当前源trace后保存、验证。
- **最终选择**：每路线／种子的日常验证前3个PPO检查点＋E0，用全部211条val重评，选定后再跑全部211条test；测试不回流选模／调参。
- **评分**：每个模型×负载按Σ输出token／Σtrace耗时计算吞吐；独立路线取3组、共享取9组相对E0吞吐比的等权几何均值。E0、PD、PPO使用同一执行口径。
- **目标**：三个种子的平均综合吞吐比>1。完整报告每种子、各模型／负载、均值及标准差，保留退化与E0获选结果；另报相对PD吞吐、延迟、决策开销、峰值内存、训练成本及PPO诊断。
- **恢复**：仅在全部组完成同一源trace且更新结束的空环境边界保存；保存actor、critic、优化器、随机状态、trace排列、轮次和游标，中断后从最近该边界重跑。

## 6. 实施顺序与边界

1. 已指定四路检查点；核对源码／配置，复制code、backend、configs到独立实验目录。
2. 修改数据导入、输入迁移、实际phase决策、节点PPO和恢复；复用现有ragged成本及精确计划执行。
3. 通过核心检查，完成训练区标定及E0／PD评价，再做seed=7先导。
4. 冻结正式参数和预算，执行四路线×三种子，按第5节完成验证和测试。

主要修改模块：workloads（全量导入）、placement_dataset（输入）、placement_policy（采样／critic）、placement_evaluator＋replay（计划／KV／时间）、placement_training（PPO／恢复）；新增一个burstgpt_ppo编排入口。模块沿用上述BC入口所在源码目录，移除旧二动作actor、长度硬限制和32次decode窗口的依赖。

**最小检查**：数据全覆盖；首token、加入／退出、KV释放正确；三设备同计划单phase耗时对齐；无未来信息泄漏；未更新时ratio=1；终止／截断回报和时间奖励正确；actor/critic梯度分离；最大长度内存峰值及恢复接续通过。

**工件**：本地实验根目录为 `/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/burstgpt_ppo_<YYYYMMDD>/`，HPC对应 `/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_ppo_<YYYYMMDD>/`。保留code/backend/configs/data、protocol.json、jobs.tsv、logs和training/<route>/seed_<seed>下的检查点及结果。

**保留边界**：原BC、历史结果、原始数据及既有工作区改动均保留，无删除；代码改动只进入 `artifacts/burstgpt_ppo_20260914/` 独立副本。计划文档统一放在 `/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/md/`，按 `YYYY-MM-DD_文件名.md` 命名。HPC login只检查／提交，data只传输／下载，compute才运行。

实施进度见 `2026-09-14_BurstGPT_PPO启动记录.md`。正式轮数与资源仍待先导测量；不得把先导收益当成正式三种子结果。
