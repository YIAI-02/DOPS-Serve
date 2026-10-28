# Phase-balanced BC

入口：`python -m dops_serve.static_bc_ab`；Slurm：`slurm/run.sh`。

- `static_bc_ab.py`：阶段各占 1/3 的 BC 训练、验证和恢复。
- `static_bc.py`：复用 teacher 数据、固定计划回放和 batch 工具。
- `placement_policy.py`：47 维 state、35 维 node 的 BC actor。

实验参数以 `../fresh_b50_20260914/protocol.json` 为准。
原目录携带的旧 serving/PPO/特征筛选代码已归档；精简记录见仓库 `md/2026-09-16_BC_PPO代码精简.md`。
