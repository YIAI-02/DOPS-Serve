# 固定 64+64 的 SB3 PPO 实验

只训练算子放置策略，使用 `stable_baselines3.PPO`，不含 BC、旧策略/critic 权重或自写 PPO。
这是 DOPS fast/fast 解析模拟环境，不是 NPU/PIM 真机测量，也不执行真实模型生成。

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
- `configs/environment.json`、`configs/ppo.json`：固定工作负载与训练参数。
- `tests/test_environment.py`：完整/增量执行的时间、通信、缓存等价性与 Gymnasium/SB3 接口。

## 本地运行

在本目录中执行，Python 3.12：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m dops_sb3.train --smoke --out runs/local_smoke
.venv/bin/python -m dops_sb3.verify_run runs/local_smoke
.venv/bin/python -m dops_sb3.train --out runs/ppo_seed0
```

输出目录必须是新目录。`--smoke` 使用 512 steps / n_steps=256，仅验证训练链路；正式参数保持在 JSON 中。
正式预算 1,530,000 env steps；SB3 按完整 2048-step rollout 采样，实际约 1,531,904 steps，约 10,000 episodes。
Actor/Critic 各 2×64，CPU 单环境、seed=0；lr=3e-4、gamma=1、GAE λ=0.95、batch=64、epochs=10、clip=0.2、entropy coef=0、value coef=0.5、grad norm=0.5、advantage normalization 开启。
每 20,480 steps 原生确定性评估一次，固定初始条件下评估 1 episode；同频保存 checkpoint。
本次先做 seed0 基础实验；多 seed、不同长度和结构均留到后续。

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
  -J sb3-check -o runs/check_%j.out slurm/run.sh setup-check runs/hpc_smoke
# 验收通过后，按实测吞吐选定 walltime；这里预留 48 小时。
sbatch -A hpc2006191081 -q high -p C064M0256G -N 1 -n 1 -c 2 -t 2-00:00:00 \
  -J sb3-ppo-s0 -o runs/train_%j.out slurm/run.sh train runs/ppo_seed0
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
Loss 横轴用原生 `train/n_updates`（训练 epoch 计数），避免 SB3 延迟 dump 导致两个 loss 点落在相同 env step 上。
PPO total loss 是原生最后 minibatch 的值，不是整轮平均值；value/policy loss 的口径沿用 SB3。
周期评估在采样期间触发，因此最后一次梯度更新后的 final 策略另行保存、评估和标注；best 指周期评估最优。
先看确定性评估 reward/latency 相对初始、随机和固定策略是否改善，再结合 loss、KL 等诊断；不以 loss 下降或短 smoke 结果宣称学会调度。
