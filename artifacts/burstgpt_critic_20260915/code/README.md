# BurstGPT 节点 PPO 与 critic 预训练

当前目录为本地精简后的实验代码。当前运行阶段是冻结 BC actor 的 critic 预训练。

[文件说明与阅读顺序](../../../md/2026-09-15_PPO代码阅读与本地清理.md)

入口：`python -m dops_serve.burstgpt_ppo --help`。保留的 PPO `train` 入口尚未接入预训练 critic；当前 `protocol.json` 只包含 critic 训练配置。

HPC 副本未同步本次清理。
