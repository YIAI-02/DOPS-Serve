# 固定验证集上的策略比较

完成 165/165 个条件。范围：decoder DAG timing/scheduling replay。

吞吐只在已扫描负载中选择，E2E 违约率须不超过 1%；无达标档位记为未达标。测试集未使用。

| 方法 | 种子 | 模型 | 五档完整 | 达标最高 token/s | 对应负载 |
|---|---:|---|---|---:|---:|
| bc | 7 | qwen_1.8b | True | 未达标 | — |
| bc | 7 | qwen_14b | True | 未达标 | — |
| bc | 7 | qwen_7b | True | 未达标 | — |
| bc | 17 | qwen_1.8b | True | 未达标 | — |
| bc | 17 | qwen_14b | True | 未达标 | — |
| bc | 17 | qwen_7b | True | 未达标 | — |
| bc | 27 | qwen_1.8b | True | 未达标 | — |
| bc | 27 | qwen_14b | True | 未达标 | — |
| bc | 27 | qwen_7b | True | 未达标 | — |
| bifocal | 固定 | qwen_1.8b | True | 未达标 | — |
| bifocal | 固定 | qwen_14b | True | 未达标 | — |
| bifocal | 固定 | qwen_7b | True | 未达标 | — |
| ppo | 7 | qwen_1.8b | True | 未达标 | — |
| ppo | 7 | qwen_14b | True | 未达标 | — |
| ppo | 7 | qwen_7b | True | 未达标 | — |
| ppo | 17 | qwen_1.8b | True | 未达标 | — |
| ppo | 17 | qwen_14b | True | 未达标 | — |
| ppo | 17 | qwen_7b | True | 未达标 | — |
| ppo | 27 | qwen_1.8b | True | 未达标 | — |
| ppo | 27 | qwen_14b | True | 未达标 | — |
| ppo | 27 | qwen_7b | True | 未达标 | — |
| ppo_continued | 7 | qwen_1.8b | True | 未达标 | — |
| ppo_continued | 7 | qwen_14b | True | 未达标 | — |
| ppo_continued | 7 | qwen_7b | True | 未达标 | — |
| ppo_continued | 17 | qwen_1.8b | True | 未达标 | — |
| ppo_continued | 17 | qwen_14b | True | 未达标 | — |
| ppo_continued | 17 | qwen_7b | True | 未达标 | — |
| ppo_continued | 27 | qwen_1.8b | True | 未达标 | — |
| ppo_continued | 27 | qwen_14b | True | 未达标 | — |
| ppo_continued | 27 | qwen_7b | True | 未达标 | — |
| static | 固定 | qwen_1.8b | True | 267.5256 | 0.5 |
| static | 固定 | qwen_14b | True | 10.0799 | 0.5 |
| static | 固定 | qwen_7b | True | 63.2073 | 0.5 |

## 同条件配对变化

正吞吐变化表示更快；负 p99/违约率变化表示改善。随机种子全部保留，不按验证结果挑种子。

| 比较 | 模型 | 配对数 | 吞吐变化均值 | p99 变化均值 | 违约率变化（百分点） |
|---|---|---:|---:|---:|---:|
| bc → ppo | qwen_1.8b | 15 | +8.35% | -18.26% | -5.60 |
| bc → ppo | qwen_7b | 15 | +38.58% | -43.01% | -22.06 |
| bc → ppo | qwen_14b | 15 | -5.33% | +34.65% | +1.94 |
| ppo → ppo_continued | qwen_1.8b | 15 | +2.41% | -4.18% | -4.57 |
| ppo → ppo_continued | qwen_7b | 15 | +2.77% | -9.54% | -7.90 |
| ppo → ppo_continued | qwen_14b | 15 | +0.04% | -0.10% | -0.00 |

单条验证 trace 的比较不足以证明泛化或稳定收益。续训是否改善应同时检查各模型、各种子的达标吞吐及配对延迟，不能仅看训练 reward。
