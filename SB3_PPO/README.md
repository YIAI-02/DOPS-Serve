# 固定 64+64 的 SB3 PPO 实验

只训练算子放置策略，使用 `stable_baselines3.PPO`，不含 BC、旧策略/critic 权重或自写 PPO。
这是 DOPS fast/fast 解析模拟环境，不是 NPU/PIM 真机测量，也不执行真实模型生成。

当前支持下述 A/B/C/D 观测与决策粒度对照。原始实验合同描述 A 组；历史 `runs/ppo_seed0` 保留不覆盖。

四组训练方案、实际预算、结果与曲线见[仓库首页](../README.md)。

## 四组观测与决策粒度实验

| 组别 | 观测 | 动作 | 每请求步数 |
| --- | --- | --- | ---: |
| A | 原始 15 维 | Discrete(2)，逐算子选择 | 153 |
| B | 增强 23 维 | Discrete(2)，逐算子选择 | 153 |
| C | 原始信息范围 6 维：设备时间、阶段、decode 进度 | MultiBinary(9)，整段放置 | 17 |
| D | 增强 15 维 | MultiBinary(9)，整段放置 | 17 |

B/D 的共同输入依次为：`log1p(当前已写 KV token 数)`（1）、段开始 NPU 权重驻留（7）、段开始 PIM 权重驻留（7）。权重类别顺序为 Q、K、V、O、FFN_W1、FFN_W3、FFN_W2；驻留标记取 0/1，允许两边同时驻留。增强观测不使用最终生成长度、完成百分比或绝对设备时钟。

B 再附加 Q、K、V、QK、SV、O、FFN_W1、FFN_W3 的三值前缀（8）：−1 未选、0 NPU、1 PIM。首个未选位置确定当前算子；八项全选时当前为 FFN_W2，因此无需算子 one-hot 或额外 mask。

驻留标记描述整类权重在段开始时是否全有/全无，并在段内冻结。两个 PIM 各保存自己的 shard；仅在完整段边界合并标记。当前固定大容量配置下使用这一压缩；若出现部分层驻留或 PIM 分片不对称，环境明确报错，不能将其当作未驻留继续训练。后端身份、格式、容量与执行算法均沿用原实现。

四组共同使用原生 PPO：n_steps=2048、batch_size=64、n_epochs=10、learning_rate=3e-4、gamma=1、gae_lambda=0.95，其他参数沿用 `configs/ppo.json`。Actor/Critic 保持各两层 64 单元，CPU 单环境。C/D 为原生条件独立 Bernoulli 输出，不使用 512 类输出头或自回归策略。

四组共 12 次正式训练已完成。Budget 的三 seed 耗时分别为 A 84.709±20.021、B 64.836±0.747、C 63.523±0.000、D 63.484±0.059 ms；完整结果和训练稳定性分析见 [实验报告](reports/2026-11-12_四组PPO实验结果分析.md)。

每组 seed 0、1、2，目标各 10,000 个请求。原生 PPO 完成整个 rollout 后退出，因此 A/B 实际为 1,531,904 步（10,012 完整请求＋68 步），C/D 为 172,032 步（10,119 完整请求＋9 步）。相同请求预算下两种粒度的优化次数不同，按实际值记录，不单独归因于 reward 结算。

每 100 个请求评估并存档：A/B 间隔 15,300 步，C/D 间隔 1,700 步。第 10,000 请求处的 `budget` 检查点独立回放；`final` 是最后一次完整 rollout 更新后的策略，二者不混用。

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m dops_sb3.train --smoke --variant D --seed 0 --out runs/new_d_smoke
.venv/bin/python -m dops_sb3.verify_run runs/new_d_smoke
.venv/bin/python -m dops_sb3.train --variant B --seed 1 --out runs/obs_action_ablation/B/seed1
.venv/bin/python -m dops_sb3.plot --compare runs/obs_action_ablation
```

HPC 复用 `slurm/run.sh`，例如 `slurm/run.sh train runs/obs_action_ablation/D/seed2 --variant D --seed 2`。训练后自动执行结果核对。先本地修改和验证，再通过 data 节点上传并逐文件比较；只能在计算节点运行测试和训练。

12 次训练全部成功后，可通过依赖它们的 Slurm 汇总作业执行 `slurm/run.sh compare runs/obs_action_ablation`，自动生成四组 CSV 和图表。

每次运行新增 `experiment.json` 记录组别、seed、episode 长度及请求预算；其余沿用现有配置、Monitor/progress、TensorBoard、模型和图表工件。`summary.json` 记录实际完整请求/尾部、rollout、训练 epoch、优化次数、参数量及 initial/budget/best/final。四种策略结果的 actions 均为按段展开的153位，重排为17×9即可比较。

对照入口汇总已完成目录：`comparison.csv`（每个 seed 的策略结果）、`summary_statistics.csv`（组内均值与样本标准差）、`evaluation_statistics.csv`（共同评估点统计）、`loss_statistics.csv`（共同训练 epoch 的原生 loss 统计）、`placements.csv`（完整放置）和 `comparison.json`（覆盖数量/口径）。不足三个 seed 时明确 n，单个 seed 的标准差留空。图表输出到根目录 `figures/`，保留各 seed 曲线及组内均值±标准差；Loss 展示原始日志，不判定性能成败或策略优劣。短测图明确标为 SMOKE。

## 实验合同

- 单请求、batch=1、FP16；沿用本地 `qwen_1.8b_shape.json`：28 层、hidden=1536、FFN=8960、12 个 Q heads、2 个 KV heads；TP_QKV=TP_FFN=2。
- 设备为 1 NPU + PIM0/PIM1，CPU0 为 host。只改变放置，拓扑顺序不变。
- prompt=64；prefill 后实际执行 64 次单 token decode。对应 65 次 Transformer 层 DAG 执行；KV 写入共 128 tokens。若把 prefill 产生的首 token 也算作输出，则总输出为 65 tokens。
- 1 个 prefill 决策段 + 16 个 decode 决策段；每段 9 步，episode 共 **153 步**。同类主算子跨 28 层共享选择，decode 每 4 tokens 更新一次。

| 动作位置 | 独立主算子 | 跟随关系 |
| --- | --- | --- |
| 1–3 | Q、K、V 分开 | KV 写入始终位于对应 shard 的 PIM |
| 4 | QK | 同 shard Softmax 跟随 QK |
| 5 | SV | 独立放置 |
| 6 | O | Add1 跟随 O collective 代表设备；LN2 跟随 Add1 |
| 7 | FFN_W1 / gate | 同 shard SwiGLU 跟随 W1 |
| 8 | FFN_W3 / up | 独立放置；必要时向 SwiGLU 所在设备传输 |
| 9 | FFN_W2 / down | Add2 跟随 W2 collective 代表设备；下一层 LN 跟随 Add2 |

每步动作 `Discrete(2)`：0=两个 TP shards 都在 NPU；1=shard0 在 PIM0、shard1 在 PIM1。
PIM collective 的代表设备取 PIM0，但输出在全部参与设备可用。第一层 LN 固定在输入所在 NPU。

每选一次动作，执行已确定的拓扑前缀，并停在下一个未决定的算子；不会提前执行后续层。
第九次选择完成当前 forward 的剩余层；decode 段还完整执行另外 3 个 tokens。
动作之间不添加全局 barrier。段内/段间保留权重缓存、KV、设备与链路时钟；episode reset 克隆真正未执行过的初始状态。

观测共 15 维：NPU/PIM0/PIM1 的累计 available timeline / C（3），prefill/decode one-hot（2），已完成 decode 数 /64（1），当前算子 one-hot（9）。不含 153 个历史动作；隐藏缓存/链路等状态仍保留在模拟器，因此这是压缩观测，不宣称完整 Markov 性。

奖励为 `r_t = -(M_{t+1} - M_t) / C`。M 包括所有已提交任务、设备和链路预留的最大结束时间；初始自动执行部分也计入第一步。
C 固定为冷启动全主算子 NPU 基线的完整请求耗时。`gamma=1`，所以总 reward 精确对应 `-总耗时/C`，全 NPU 基线为 -1。
并行工作未延长 M 时 reward 可以为 0；第九步承担剩余层和 tokens 的时间，因此可能较大。reward 越接近 0 越好。

## 代码来源与入口

- `backend/`：复制自原 `artifacts/burstgpt_critic_20260915/backend/DOPS` 的环境 Python 依赖和两个 JSON，保留中文注释；不携带旧数据、外部模拟器二进制或训练产物。
- `dops_sb3/timing/`：来自原 `artifacts/burstgpt_ppo_20260917/code/src/dops_serve/placement_{compiler,cost,evaluator}.py`。只保留环境计算与完整执行参照，去掉旧 PD 入口并更名。
- `dops_sb3/execution.py`：九类决策的层间展开、跟随规则、原后端内核的可暂停执行。
- `dops_sb3/env.py`：Gymnasium reset/step、状态、增量 reward。
- `dops_sb3/train.py`：原生 SB3 PPO、Monitor、EvalCallback、CheckpointCallback。
- `dops_sb3/bifocal_search.py`：原 Bifocal / 新 far focus 评分、参数搜索和独立计划回放。
- `configs/environment.json`、`configs/ppo.json`：固定工作负载与训练参数。
- `tests/test_environment.py`：完整/增量执行的时间、通信、缓存等价性与 Gymnasium/SB3 接口。

## 本地运行

在本目录中执行，Python 3.12：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m dops_sb3.train --smoke --out runs/new_smoke
.venv/bin/python -m dops_sb3.verify_run runs/new_smoke
.venv/bin/python -m dops_sb3.train --out runs/new_training
```

输出目录必须是新目录。`--smoke` 使用 512 steps / n_steps=256，仅验证训练链路；正式参数保持在 JSON 中。
正式预算 1,530,000 env steps；SB3 按完整 2048-step rollout 采样，实际约 1,531,904 steps，约 10,000 episodes。
Actor/Critic 各 2×64，CPU 单环境、seed=0；lr=3e-4、gamma=1、GAE λ=0.95、batch=64、epochs=10、clip=0.2、entropy coef=0、value coef=0.5、grad norm=0.5、advantage normalization 开启。
历史 seed0 运行每 20,480 steps 原生确定性评估一次；当前入口按每 100 个请求评估并保存 checkpoint。固定初始条件下每次评估 1 episode。
历史运行覆盖 seed0；当前四组实验各运行 seed 0、1、2，工作负载仍固定。

## HPC

独立目录：`/lustre/home/2501111916/workspace/DOPSRT_SB3_PPO`。
本地先完成修改；经 data 节点上传并逐文件回读比较内容后，才提交作业。
不上传 `.venv`、本地 `runs` 和缓存。HPC 使用现有 PyTorch 容器的 Python 3.12，在本目录 `.hpc-venv` 安装独立 CPU 版依赖，不依赖容器自带旧 PyTorch。
`slurm/run.sh` 同时将动态库路径指向新环境的 `torch/lib`，并移除外部 PYTHONPATH/LD_PRELOAD，避免容器旧 Torch 动态库混入。

data 节点（只下载）：

```bash
cd /lustre/home/2501111916/workspace/DOPSRT_SB3_PPO
bash slurm/download.sh
```

login 节点（只提交；实际安装与运行在 Slurm 计算节点）：

```bash
cd /lustre/home/2501111916/workspace/DOPSRT_SB3_PPO
mkdir -p runs
sbatch -A hpc2006191081 -q high -p C064M0256G -N 1 -n 1 -c 2 -t 00:30:00 \
  -J sb3-check -o runs/check_%j.out slurm/run.sh setup-check runs/new_hpc_smoke
# 验收通过后，按实测吞吐选定 walltime；这里预留 4 天运行上限。
sbatch -A hpc2006191081 -q high -p C064M0256G -N 1 -n 1 -c 2 -t 4-00:00:00 \
  -J sb3-ppo-s0 -o runs/train_%j.out slurm/run.sh train runs/new_training
```

## 观测与工件

- `train.monitor.csv`：每个 episode 的 reward、长度、真实模拟 latency/speedup；`eval.monitor.csv` 同样记录确定性评估。
- `progress.csv` 和 `events.out.tfevents.*`：SB3 原生 `train/loss`、`policy_gradient_loss`、`value_loss`、entropy、KL、clip fraction、explained variance，以及训练/评估 reward。
- `eval/evaluations.npz`、`best/best_model.zip`、`last_model.zip`、`checkpoints/`。
- `baselines.json`：全 NPU、全 PIM、prefill NPU/decode PIM、20 次随机策略（smoke 为 2 次）。
- `initial_policy.json`、`summary.json`：初始、最终、周期评估最佳策略及动作；包含实测吞吐和实际采样量。
- `figures/reward_latency.png`、`figures/loss.png`：从原生日志生成，不自行计算 loss。
- `versions.json`、`ppo_config.json`、`environment_config.json`、`normalization.json`：复现所需运行参数。

```bash
.venv/bin/tensorboard --logdir runs
.venv/bin/python -m dops_sb3.plot runs/ppo_seed0
```

训练过程中 TensorBoard 可实时查看；PNG 在完成训练后生成，也可用上面的 plot 命令从已有日志生成。
Reward/latency 横轴用训练请求数；Loss 横轴用原生 `train/n_updates`（训练 epoch 计数），避免 SB3 延迟 dump 导致两个 loss 点落在相同 env step 上。
PPO total loss 是原生最后 minibatch 的值，不是整轮平均值；value/policy loss 的口径沿用 SB3。
周期评估在采样期间触发，因此最后一次梯度更新后的 final 策略另行保存、评估和标注；best 指周期评估最优。
先看确定性评估 reward/latency 相对初始、随机和固定策略是否改善，再结合 loss、KL 等诊断；不以 loss 下降或短 smoke 结果宣称学会调度。

## 已归档的运行结果

当前仓库已包含下列结果；重新运行须使用新的输出目录，不能覆盖归档目录。

| 目录 | 用途与状态 |
| --- | --- |
| [runs/ppo_seed0](runs/ppo_seed0) | 正式 CPU 训练，1,531,904 steps，已完成；含 74 个周期 checkpoint、初始/最佳/最终模型、配置、全部指标和曲线 |
| [runs/obs_action_ablation](runs/obs_action_ablation) | A/B/C/D 各三个 seed 的完整训练输出、模型、检查点和汇总分析 |
| [runs/bifocal_far_20261112](runs/bifocal_far_20261112) | 新 far 评分的候选、生成计划、成本表、参数搜索及验证记录 |
| [runs/hpc_smoke](runs/hpc_smoke)、[runs/local_smoke](runs/local_smoke) | HPC / 本地 512-step 链路验收，不用于判断学习效果 |
| [runs/gpu_benchmark](runs/gpu_benchmark) | 同一 A800 节点 CPU/CUDA 各 4096-step 对比，已完成，未观察到可靠 CUDA 加速 |
| [runs/gpu_benchmark_repeat](runs/gpu_benchmark_repeat) | 用户要求停止的 A100 复测，仅有部分输出，不能作完整对比 |
| [reports](reports) | 首次验收、正式提交、GPU 测速、最终结果报告 |

正式训练最佳模拟请求耗时为 63.416 ms，最终为 103.798 ms；全 PIM 基线为 63.518 ms。训练中存在回退，约 139 万步后持续退化，不能判定为稳定收敛。详见 [正式训练结果](reports/2026-10-28_SB3_PPO_正式训练结果.md)。

```bash
.venv/bin/python -m dops_sb3.verify_run runs/ppo_seed0
.venv/bin/tensorboard --logdir runs/ppo_seed0
```

历史 BC / Critic / 自写 PPO 的源码与工件位于 [archive/bc-ppo](https://github.com/YIAI-02/DOPS-Serve/tree/archive/bc-ppo)。

## Bifocal 评分参数搜索

`dops_sb3.bifocal_search` 使用与 PPO 相同的硬件、fast 成本、KV ledger 和连续计时；Bifocal 保留逐节点放置及自由拓扑序。原评分配置为 `configs/bifocal.json`，修改后的 far focus 配置为 `configs/bifocal_far.json`。两者均使用固定复用窗口，不读取真实剩余输出长度。

| 配置 | 三 seed 完整请求耗时 / ms |
| --- | ---: |
| 原 Bifocal 调参最优 | 64.615654 |
| 新 far focus | 63.517612 |
| 全 PIM 固定基线 | 63.517819 |

新 far focus 直接继承原 EFT、短链前瞻、ready 选择和一致性机制，以未来 decode 的设备成本差替换旧权重复用收益与摊销，自行生成全 PIM 主算子放置。与全 PIM 持平，尚未超过 PPO best 的 63.415978 ms。实现与完整结果见 [Bifocal 报告](reports/2026-11-12_DOPS_list_scheduler/Bifocal_far修改与Qwen1.8B结果.md)；原评分的 211 组参数搜索见 [搜索记录](reports/2026-11-12_DOPS_list_scheduler/Bifocal评分开启与参数搜索结果.md)。

```bash
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search --check --out runs/new_bifocal_check
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search --out runs/new_bifocal_search --workers 4 --coarse 96 --refine 48 --top 5
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search --params configs/bifocal.json --seed 0 --out runs/new_bifocal_legacy
PYTHONHASHSEED=0 .venv/bin/python -m dops_sb3.bifocal_search --params configs/bifocal_far.json --seed 0 --out runs/new_bifocal_far
```

输出包含候选、逐次结果、多 seed 汇总和节点计划；单次复现与最佳计划都会独立回放核对。`far_mode` 支持 `legacy`、`prefill`、`both`。指定候选集可使用 `--candidates path/to/candidates.json --refine 0`。输出目录必须不存在；原生 DOPS CLI 不会自动加载本入口的配置。结果仅代表当前 workload 下已搜索集合中的最优值。
