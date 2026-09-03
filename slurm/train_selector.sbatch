#!/bin/bash
#SBATCH --job-name=dopsserve-train
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full}
MODEL_ID=${MODEL_ID:-llama_7b}
HARDWARE_KEY=${HARDWARE_KEY:-exp1/hw_hardware_1npu_2aim/sst2_rst2}
RUN_NAME=${RUN_NAME:-irregular_v1}
TRACES_PER_FAMILY=${TRACES_PER_FAMILY:-4}
REQUESTS_PER_TRACE=${REQUESTS_PER_TRACE:-100}
PROMPT_MIN=${PROMPT_MIN:-128}
PROMPT_MAX=${PROMPT_MAX:-2048}
LATENCY_SLO_S=${LATENCY_SLO_S:-30.0}
PLAN_SWITCH_MARGIN=${PLAN_SWITCH_MARGIN:-0.02}
WORKLOAD_PATH="$ARTIFACT_DIR/workloads_${MODEL_ID}_${RUN_NAME}.jsonl"
DATASET_PATH="$ARTIFACT_DIR/dataset_${MODEL_ID}_${RUN_NAME}.csv"
MODEL_DIR="$ARTIFACT_DIR/model_${MODEL_ID}_${RUN_NAME}"
EVAL_DIR="$ARTIFACT_DIR/serving_eval_${MODEL_ID}_${RUN_NAME}"

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

run_python -m dops_serve.cli generate-workloads \
  --output "$WORKLOAD_PATH" \
  --model "$MODEL_ID" \
  --traces-per-family "$TRACES_PER_FAMILY" \
  --requests-per-trace "$REQUESTS_PER_TRACE" \
  --base-rate 2.0 \
  --prompt-min "$PROMPT_MIN" \
  --prompt-max "$PROMPT_MAX" \
  --seed 7
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
  --checkpoint "$MODEL_DIR/checkpoints/best.pt" \
  --output-dir "$EVAL_DIR" \
  --hardware-key "$HARDWARE_KEY" \
  --split test \
  --horizon 32 \
  --horizon 64 \
  --horizon 128 \
  --fixed-horizon 64 \
  --max-decisions 5000 \
  --latency-slo-s "$LATENCY_SLO_S" \
  --plan-switch-margin "$PLAN_SWITCH_MARGIN"
