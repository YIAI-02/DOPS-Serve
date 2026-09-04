#!/bin/bash
#SBATCH --job-name=dops-burstgpt-real-v2
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full_trace_windows_parallel_v7}
SOURCE_CSV=${SOURCE_CSV:-$REPO_ROOT/data/BurstGPT_without_fails_1.csv}
MODEL_ID=${MODEL_ID:-llama_7b}
HARDWARE_KEY=${HARDWARE_KEY:-exp1/hw_hardware_1npu_2aim/sst2_rst2}
RUN_NAME=${RUN_NAME:-burstgpt_real_v2}
LATENCY_SLO_S=${LATENCY_SLO_S:-30.0}
PLAN_SWITCH_MARGIN=${PLAN_SWITCH_MARGIN:-0.02}
WORKLOAD_PATH="$ARTIFACT_DIR/workloads_${MODEL_ID}_${RUN_NAME}.jsonl"
DATASET_PATH="$ARTIFACT_DIR/dataset_${MODEL_ID}_${RUN_NAME}.csv"
MODEL_DIR="$ARTIFACT_DIR/model_${MODEL_ID}_${RUN_NAME}"
REAL_EVAL_DIR="$ARTIFACT_DIR/serving_eval_${MODEL_ID}_${RUN_NAME}_real_checkpoint"
SYNTHETIC_CHECKPOINT=${SYNTHETIC_CHECKPOINT:-$ARTIFACT_DIR/model_${MODEL_ID}_irregular_synthetic_v1/checkpoints/best.pt}
ZERO_SHOT_EVAL_DIR="$ARTIFACT_DIR/serving_eval_${MODEL_ID}_${RUN_NAME}_synthetic_checkpoint"

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

run_python -m unittest discover -s tests -v

run_python -m dops_serve.cli import-burstgpt \
  --input "$SOURCE_CSV" \
  --output "$WORKLOAD_PATH" \
  --model "$MODEL_ID" \
  --traces 16 \
  --requests-per-trace 100 \
  --target-median-rps 2.0 \
  --max-tokens-default 1024 \
  --prompt-min 128 \
  --prompt-max 2048 \
  --output-max 1024 \
  --source-model ChatGPT \
  --log-type "API log" \
  --selection mixed \
  --split-seed 7

run_python -m dops_serve.cli dataset \
  --workload "$WORKLOAD_PATH" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --output "$DATASET_PATH" \
  --hardware-key "$HARDWARE_KEY" \
  --max-decisions-per-trace 2000

run_python -m dops_serve.cli train \
  --dataset "$DATASET_PATH" \
  --output-dir "$MODEL_DIR" \
  --epochs 80 \
  --batch-size 512 \
  --seed 7

run_python -m dops_serve.cli serving-evaluate \
  --workload "$WORKLOAD_PATH" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --checkpoint "$SYNTHETIC_CHECKPOINT" \
  --output-dir "$ZERO_SHOT_EVAL_DIR" \
  --hardware-key "$HARDWARE_KEY" \
  --split test \
  --horizon 32 \
  --horizon 64 \
  --horizon 128 \
  --fixed-horizon 64 \
  --max-decisions 5000 \
  --latency-slo-s "$LATENCY_SLO_S" \
  --plan-switch-margin "$PLAN_SWITCH_MARGIN"

run_python -m dops_serve.cli serving-evaluate \
  --workload "$WORKLOAD_PATH" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --checkpoint "$MODEL_DIR/checkpoints/best.pt" \
  --output-dir "$REAL_EVAL_DIR" \
  --hardware-key "$HARDWARE_KEY" \
  --split test \
  --horizon 32 \
  --horizon 64 \
  --horizon 128 \
  --fixed-horizon 64 \
  --max-decisions 5000 \
  --latency-slo-s "$LATENCY_SLO_S" \
  --plan-switch-margin "$PLAN_SWITCH_MARGIN"
