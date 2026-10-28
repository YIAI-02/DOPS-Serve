#!/bin/bash
set -euo pipefail
PPO_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_ppo_20260917
# 用户确认提交前报告后才创建 formal_budget.json；本脚本不代替确认。
test -f "$PPO_ROOT/formal_budget.json"
test -f "$PPO_ROOT/hpc_cpu_check_passed.json"
test -f "$PPO_ROOT/worker_check_cuda_passed.json"
test -f "$PPO_ROOT/exact_cache_check_passed.json"
mkdir -p "$PPO_ROOT/logs"
PPO_ROUTE="${3:-qwen_1.8b}"
for PPO_SEED in 4 15 26; do
    sbatch --parsable --account=hpc2006191081 --qos=high --partition="${1:-GPU40G}" \
        --job-name="ppo-${PPO_ROUTE}-s${PPO_SEED}" --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 \
        --time="${2:-5-00:00:00}" --output="$PPO_ROOT/logs/train_${PPO_ROUTE}_seed_${PPO_SEED}_%j.out" \
        --error="$PPO_ROOT/logs/train_${PPO_ROUTE}_seed_${PPO_SEED}_%j.err" \
        "$PPO_ROOT/code/slurm/run_ppo.sh" train "$PPO_ROUTE" "$PPO_SEED"
done
