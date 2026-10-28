# 当前 BC 的代码、训练流程与整体分配问题

核对日期：2026-09-12。已复用持久 SSH 连接，只读核对 HPC 的 PlacementActor、train_bc、BC 阶段入口、anchor 构造、编译器及重评分关键代码，与本文引用的本地实现一致。数据总量与完整性也读取了 HPC 已有汇总；分模型样本数来自本地同轮汇总。本文没有运行新训练、回放或消融实验。

用户已取消自动监控：现有 dops 为 PAUSED。只有明确要求开启监控才恢复；恢复 SSH 连接不恢复监控。已有 Slurm 作业没有被取消。

## 1. 先回答“是不是在学整体分配”

**数据中确实保存了完整安排；当前模型用逐节点概率来学习这份安排，没有保证自主生成的结果保留教师方案的整体组合。**

需要区分三个层次：

| 层次 | 当前实现 |
|---|---|
| 一份训练答案 | 某个 prefill 或 decode 阶段，全部可选算子的二元设备动作 |
| 网络如何表示答案的概率 | 跨 block 使用 GRU；同一 block 内各节点分别预测 NPU/PIM 概率 |
| 正式使用时如何生成 | 每个节点取概率较大的设备，然后编译成完整可执行计划 |

它不是从 3,038 份候选中直接选择一个 placement_id，也不是学习整条 trace 从开始到结束的调度轨迹。原始 DOPS 的 prefill 安排、多个 decode 分段安排被拆成分别学习的阶段样本。

这里的“完整”仍受当前动作空间约束：0=NPU，1=该节点预先绑定的 PIM。PIM0/PIM1 的任意选择、固定 KV 写入位置、通信节点及执行顺序不由网络自由学习。投影可能改变原 DOPS 的 PIM 放置；所以训练 cost 用投影后的计划重新评分，不能拿原始 DOPS 耗时当成这个动作空间的成绩。

入口：[PlacementCompiler](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_compiler.py:14)、[compile/project](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_compiler.py:78)。

## 2. 为什么 loss 变好，整份安排仍可能变差

报告中独立 14B、种子 27 的 E5→E6 是 **loss 数值下降**：0.033190→0.030788；吞吐/Static 从 91.13%→60.06%。若“loss 提高”指的是学习指标改善，下面解释的就是这一现象。若某段 loss 数值实际升高，则表示该段模仿误差变大，需要指明相应日志。

### 2.1 一个小例子：三份好安排，也能学出一份坏组合

以下是说明结构问题的假设例子，不是实际 DOPS 测量。假设同一 block 的三个节点都允许在两个设备运行：

| 教师方案 | 节点 A | 节点 B | 节点 C |
|---|---|---|---|
| 方案一 | PIM | NPU | NPU |
| 方案二 | NPU | PIM | NPU |
| 方案三 | NPU | NPU | PIM |

假设三份方案耗时相同，目标权重各 1/3。每个节点在教师中有 2/3 次位于 NPU。若模型把每个节点的概率拟合得很好，逐节点 argmax 就会得到 **NPU、NPU、NPU**。

这份安排没有出现在任何教师方案中，且可能因为全部集中到 NPU 而变慢。它的出现不要求教师中存在坏方案；多个好方案之间的配合关系，也可能被逐节点预测拆散。

当前同一 block 的概率是：

```text
P(本 block 全部动作 | 现场, 之前 block 的摘要)
  = 各节点 P(自己的动作 | 自己的特征, 同一份摘要) 的乘积
```

这属于对整份答案概率的一种有约束的表示方式。问题不是“对数概率求和天然不能学习整体”，而是当前表示方式缺少同一 block 内当前动作之间的依赖。证据：[sequence 的 logits 与 chosen](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_policy.py:98)。

### 2.2 跨 block 的摘要还丢失了节点与动作的对应关系

实际代码：

```python
summary = (nodes + self.action_embedding(chosen)).mean(0)
hidden = self.gru(summary, hidden)
```

数学上第一行等于：

```text
节点特征的平均值 + 设备动作 embedding 的平均值
```

在同一个现场、同一个 block、相同前序 hidden 下，只要两份安排选择 NPU/PIM 的节点数量相同，即使把 W1 和 W2 的设备交换，传给下一 block 的摘要也相同。动作部分表达的是数量比例，没有保留“具体哪个节点去了哪个设备”。

所以 GRU 的存在确实提供了跨 block 关联，但不能据此说它记住了前面完整的节点—设备配对。这是从代码可直接推出的结构限制；**尚未用结构修复消融测出它对 E5→E6 吞吐下降贡献了多少。**

证据：[summary 与 GRU 更新](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_policy.py:113)。

### 2.3 训练时有教师带路，正式使用时自己带路

训练调用：

```python
actor.sequence(row["inputs"], teacher_program)
```

每个 block 的 chosen 被设成教师动作；计算网络给这些答案的概率，再用教师动作更新 GRU。前面某个 block 即使预测错误，后面的 hidden 仍沿教师给的动作继续。

自主生成调用没有 teacher_program，使用当前网络自己的 argmax。前面选错，后面接收的就是自己选出来的摘要。这是另一种训练与使用的差别。

此外，所有 anchor 的输入特征一次性从保存的现场构造，遍历 block 时不会重新执行模拟器，也不会在网络内准确更新“前面安排后还剩多少缓存”。这种影响只能由网络学习估计。

证据：[训练调用](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_training.py:122)、[自主生成](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_policy.py:138)。

### 2.4 loss 并不直接惩罚自主安排造成的吞吐损失

离线训练成本只给候选权重；网络自主生成的新组合若没有在候选里出现，它的实际执行耗时不会进入本轮 BC 梯度。

此前诊断已经实际测得：同一已有缓存现场，E5→E6 的计算时间之和从 43.28 ms 降到 39.51 ms，但权重加载增加 159.93 ms，decode 从 31.16 ms 增到 171.33 ms。另一个长队现场中，全 NPU 安排导致持续反复加载权重。

这些结果证明了“安排变化→缓存/搬运恶化→执行变慢”。它们没有证明“全部是多教师加权造成”，也没有证明“改成只学赢家就会解决”。

原始证据：[diagnosis_report.md](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/fast_bc_restart_20260908/diagnostics/e5_e6_20260911/diagnosis_report.md:1)。

## 3. 当前训练数据如何产生

调用链：

```text
placement_restart.teacher
  → 新克隆并冻结的 DOPS/src/main.py evaluate（fast/fast，九方法）
  → placement_dataset.build
  → compiler.project / compile
  → placement_dataset.reevaluate / _reevaluate_shape
  → 保存 placements.jsonl 和 anchors/*.jsonl
  → placement_training.train_bc
  → 每轮独立做自主服务回放验证
```

### 步骤 A：重新运行 DOPS 教师

3 模型 × batch {1,4,8,16} × prompt {128,512,1024,2048} × decode {128,256,512,1024} = 192 个形状。

每个形状显式运行九方法：Bifocal、Naive、PD、AF、PD+Linear、PD+Attn、PD+FFN、weights_on_pim、NeuPIMs。共 1,728 个结果来源，排除 HEFT。prompt 是输入长度，decode 是输出阶段长度；不是把二者都当 prefill。

实际入口：[teacher](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_restart.py:88)。它调用 DOPS 的 evaluate，不是由 BC 网络生成教师。

### 步骤 B：投影并整理完整候选

读取每条结果的 prefill 和 decode_segments，将设备映射投影为 block_actions。按模型、阶段、动作组合去重，本轮得到 **3,038 种投影后的安排**。

- placements.jsonl：placement_id、model_id、phase、block_actions、fixed_plan。
- history_sources.jsonl：原始来源、形状、分段位置、候选 ID、投影修改计数。
- 原始 DOPS 日志耗时保留作检查，不直接进入 BC 权重。

代码：[build 中的投影与写出](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_dataset.py:98)。

### 步骤 C：构造现场，并在同一现场重评分

按模型、batch、prompt 分组，共 48 组；不同 decode 总长度的教师在同组内贡献相应分段起点。

用固定 NPU 优先安排推进 prefill 和中间 decode，保留真实模拟进展中的 KV、缓存和设备时间线；在历史分段起点保存一个 anchor。对该 anchor 的各候选，复制同一快照，运行对应阶段：

- prefill 标签：这次完整 prefill 的模拟阶段耗时；
- decode 标签：当前这一步 decode 的模拟阶段耗时；
- 不是整条 trace 的吞吐，也不是未来整个 32 步窗口的累计耗时。

调用了 DOPS 的图、代价和调度模块，并由 DOPS-Serve 补连续状态、请求长度和固定计划接口。执行路径是 PlacementEvaluator.evaluate → snapshot.clone → scheduler.schedule_with_plan。不是 BC 训练时另有一个神经网络预测 cost。

代码：[anchor 构造和评分](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_dataset.py:179)、[DOPS 导入](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_evaluator.py:33)、[执行同一快照副本](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_evaluator.py:182)。

### 步骤 D：划分 BC 训练与验证

每模型按排序后的 batch/prompt 组合，每第五组留给 BC 验证，同一形状不跨 BC train/val。本轮：

| 数据范围 | 每 epoch 训练 anchor / 参数更新次数 | BC 候选验证 anchor |
|---|---:|---:|
| 独立 1.8B，每种子 | 5,200 | 1,033 |
| 独立 7B，每种子 | 5,215 | 1,033 |
| 独立 14B，每种子 | 5,255 | 1,061 |
| 共享网络，每种子 | 15,670 | 3,127 |

总共 18,797 个 anchor。3,038 是去重安排数，**不是候选评分调用次数**，更不是每个 anchor 都评分 3,038 份方案。

BC 参数更新使用上述离线 anchor，**不直接使用那五条 BurstGPT 训练 trace 作为逐请求监督样本**。五条 train trace 用于 Static 选择、奖励校准及计划中的 value/PPO；五条开发 val trace 用于每轮自主回放验收。BC 的 anchor 验证集和服务 trace 验证集是两种不同检查。

数据汇总：[reevaluation_summary.json](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/artifacts/fast_bc_restart_20260908/reevaluation_summary.json:1)。划分代码：[reevaluate](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_dataset.py:231)。

## 4. 网络输入与完整结构

输入字典是：

```text
inputs["state"]  : 68 个现场数字
inputs["nodes"]  : 按 block 分组，每个节点 23 个数字
inputs["masks"]  : 每节点两个布尔值，表示 NPU/PIM 是否合法
```

68 维现场包括：

- waiting、active 两组队列的长度，prompt/KV/等待时间的均值、最大值、阈值比例，输出长度信息；
- NPU、两个 PIM 的可用时间、权重缓存使用量与容量；
- 两个 PIM 的 KV 预留量与实际量；
- 到达速度、队列增长、服务速度等五项；
- 模型层数、隐藏维度、注意力头数、KV 头数、head_dim、KV 字节尺度；
- prefill/decode 阶段标记。

每节点 23 维：13 维算子类别，加层位置、attention/FFN 类型、分片号、PIM 归属、权重大小、当前 token 工作量、attention 工作量、两个缓存命中标记、此前计划在该节点的设备动作。部分数量经过 log1p 缩放。

这里没有 token 文本、词向量，也没有真正运行 Qwen 来训练其语言能力。训练的是一个小型设备安排网络。

代码：[state_features / policy_inputs](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_policy.py:17)。

网络结构：

```text
68 维现场 → Linear(68,128) → Tanh → 初始 hidden

23 维节点 → Linear(23,128) → Tanh → 节点向量
节点向量 + 拼接同一份 128 维 hidden
    → Linear(256,128) → Tanh → Linear(128,2)
    → 屏蔽非法设备 → NPU/PIM 概率

本 block 的节点和动作摘要 → GRUCell(128,128)
    → 下一个 block 的 hidden

value 分支：
hidden、节点平均向量、preceding_phase 拼成 384 维
    → Linear(384,128) → Tanh → Linear(128,1)
```

这里“拼接”是连接两段向量，不是前述摘要的逐元素加法。总参数 193,795，包括 value head。每层模型有 attention 和 FFN 两个 block；例如 1.8B 是 28 层、56 个 block、728 个自由节点。

BC loss 不使用 value 输出，因此 value head 没有 BC 训练梯度；尽管 Adam 接收全部参数，当前 BC 不会靠 value loss 更新网络，不应把这轮问题归因于 actor/critic 梯度互相干扰。

还有一个跨阶段细节：pair 会把 prefill 的 summary 传给 decode 的 preceding_phase，但此值只进入 value_input，**不进入 decode 的动作 logits**。因此当前 actor 没有通过这条连接显式地根据刚选出的 prefill 安排调整 decode 安排。

结构代码：[PlacementActor](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_policy.py:80)。

## 5. 一次参数更新究竟做什么

关键代码：[cost_weights](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_training.py:14)、[train_bc](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_training.py:84)。

先拿一个 anchor，设其候选耗时为 c_i，最低耗时为 c_min。权重：

```text
w_i = exp(−(c_i/c_min − 1)/τ) / 所有候选上述值之和
τ = 1
```

只有同一 anchor 的候选之间比较。cost 先化成相对比例，避免不同模型的绝对秒数直接决定权重。

例如同一现场两个候选耗时 100 和 110（任意相同单位），权重约 52.5% 与 47.5%；耗时 100 和 200，权重约 73.1% 与 26.9%。这两个是公式示例，不代表已经统计了本轮所有候选的真实权重分布。当前没有“超过某个成本就剔除”的规则。

对每个候选的完整动作：

```text
L_i = − 全部节点的 log(网络给教师设备的概率) 之和 / 节点数
L_anchor = Σ_i w_i × L_i
```

概率越接近教师动作，负对数损失越小。权重作用于整份候选的损失；**没有把多个二元答案先平均成一份 0/1 标签**。

实际循环：

```python
for anchor in 打乱顺序后的训练现场:
    optimizer.zero_grad()
    for weight, candidate in 该现场的候选:
        result = actor.sequence(anchor.inputs, candidate.block_actions)
        loss = -weight * result.log_probs.sum() / node_count
        loss.backward()       # 累加这份候选的梯度
    clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()          # 所有候选处理完，只更新一次
```

- 一个 anchor 一次 Adam 更新，不是每个节点更新一次，也不是每候选更新一次。
- Adam 学习率 3e-4，梯度范数上限 1；首次随机初始化，种子 7/17/27。
- 一 epoch 完整遍历所属模型的训练 anchor，文件和文件内的 anchor 都打乱。
- 下一 epoch 从本轮上一 epoch 的 bc_last.pt 恢复参数、Adam 状态及随机状态。
- 新路线第一轮不会读取旧 BC/PPO 权重。
- BC 没有 PPO reward、策略裁剪项或 value loss，也没有直接对吞吐求导。
- 每次更新只读取离线数据，**不调用 DOPS 重跑候选**。

日志 training_loss 是本轮训练过程中各 anchor loss 的平均。遍历时参数不断变化，因此不是“最后一个检查点在整个训练集上重新统一算一遍”的固定模型误差，也不是单个计划的执行时间。

实际分轮入口：[placement_restart 的 bc 阶段](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_restart.py:583)。Slurm 启动：[op_policy_restart.sbatch](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/slurm/op_policy_restart.sbatch:30)。

## 6. 两种验证分别测什么

### 候选验证：看教师方案的概率排序

_bc_validation 把已有候选逐个用 teacher forcing 算完整 log probability，取概率最高的已有候选，读取其离线 cost：

```text
candidate_regret = 被选候选 cost / 当前候选最低 cost − 1
```

最后按 anchor 平均。比如选到 110 ms、最快是 100 ms，则该现场 regret=0.10。

**它不是先让网络自主生成新方案，再测新方案 cost。** 候选集合本来就相近时，该值还可能很小。因此不能据此说完整安排已经学会。

unweighted_nll 则对候选不按 cost 加权，计算平均节点负对数概率。它和 training_loss 的定义、样本集、参数评估时点都不同，不能把二者的数值直接对齐。

代码：[候选验证](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_training.py:70)。

### 服务验证：看自己安排到底能跑多快

每轮用 bc_last.pt，网络自主 argmax 生成 prefill/decode 计划，编译并跑完整 trace。执行中队列、缓存和 KV 会随自己的安排变化；正常决策耗时计入时间线。沿用窗口内固定计划的机制，不在每个 decode 重新选择方案。

每模型、每种子：5 条 val trace × 5 档负载 × 3 次复测。每条件先取中位数，再对相同条件的 BC/Static 吞吐比取几何平均。BC 门槛仍是每模型每种子至少 Static 的 95%。

正式阶段选模使用这个回放成绩。单次 train_bc 里的 bc.pt 选择逻辑是候选 regret/NLL，但本轮分轮协调流程显式拿 bc_last.pt 做服务验证，并据此选择共同最佳 epoch，不是用候选 regret 放行 PPO。

代码：[prepare_validation](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_restart.py:207)、[自主回放](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_comparison.py:88)、[决策时间进入时间线](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/replay.py:127)、[汇总配对吞吐比](/Users/yangjiaqi/Desktop/Daily_2026/02_Projects/DOPSRT/DOPS-Serve/src/dops_serve/placement_restart.py:215)。

## 7. 是否应该改成只学习好的完整安排

**建议把“每个 anchor 只学一个最佳完整候选”作为第一项有限对照；暂时不把它当成已证实的修复。**

理由：你希望输出一份性能好的确定安排，而当前目标也在模仿较慢候选，并且会混合不兼容的分配模式。只学一个完整赢家，能直接去掉同一现场的多答案冲突，目标更符合这个要求。

需要守住几个条件：

1. 赢家按同一现场、当前固定动作空间重评分后的 cost 选择，不能按原始 DOPS 日志、算法名称或跨现场的最低绝对耗时选择。
2. 每个 anchor 固定一个完整赢家；并列时用固定规则选整份计划，不能逐节点拼“各自最优”，也不要每 epoch 随机换答案。
3. 首个对照保持模型、网络、训练 anchor、初始化种子、优化器和验证条件相同；分别从相同初始化训练“当前加权”与“只学赢家”。改了目标后从旧加权检查点接着训，会混入训练历史，难以判断效果来源。
4. 两组各自选开发验证最佳轮次，比较自主生成的同现场成本和完整 trace 吞吐；不只比较 loss。两种目标的 loss 定义已经变了，数值不能直接判胜。
5. 优先用诊断充分的独立 14B、种子 27 做机制试验；单种子改善只证明该试验，需要其他种子复核才推广。

“保留所有好方案再混合”与“只保留一个完整赢家”不是同一个实验。即使三份候选全都优秀，前面的三节点例子仍会发生。降低 temperature 可以更偏向最快候选，但不能自动修复同一 block 内的动作依赖或摘要信息丢失。

也要看到只学赢家解决不了的两类问题：

- **表征问题**：跨 block 摘要丢节点—设备配对，同一 block 内独立决策。若做结构修复，应单列一个对照；优先让摘要保留节点和动作的关联，不能仅凭“有 GRU”假设它已经记住了整份安排。进一步在同一 block 内逐节点条件生成，或直接选择完整候选，属于不同设计，不应一次全部改掉。
- **数据/成本目标问题**：当前 anchor 没有 active+waiting 混合排队现场，历史参考计划造成的缓存状态也不等于学生自行执行后的状态；当前单步最快不保证未来多步吞吐最好。此前诊断对这类输入差距和反复加载已有实际证据。若赢家对照仍失败，补训练 trace 上实际经过的现场并检查多步成本，比继续盲目堆 epoch 更有针对性。

关于模仿学习中的“自己行动后遇到新现场”，这是已有研究明确讨论的问题：[Ross 等人的 DAgger 原论文](https://proceedings.mlr.press/v15/ross11a.html)。这里引用的是问题机制，不声称当前流程已经实现 DAgger。BC 的一般含义是用专家的观察—动作样本做监督学习：[imitation 官方说明](https://imitation.readthedocs.io/en/stable/algorithms/bc.html)；本项目实际训练使用的是上述自定义 PyTorch 实现，不是直接调用该库的 BC 类。

本次只完成代码核对、说明和取消监控。没有改变训练目标、网络、教师数据或已有检查点，没有启动新对照，也没有取消已提交作业。

