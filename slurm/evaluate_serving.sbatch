#!/bin/bash
#SBATCH --job-name=dopsserve-serving-eval
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full_trace_windows_parallel_v7}
MODEL_ID=${MODEL_ID:-llama_7b}
HARDWARE_KEY=${HARDWARE_KEY:-exp1/hw_hardware_1npu_2aim/sst2_rst2}
RUN_NAME=${RUN_NAME:-irregular_synthetic_v1}
LATENCY_SLO_S=${LATENCY_SLO_S:-30.0}
PLAN_SWITCH_MARGIN=${PLAN_SWITCH_MARGIN:-0.02}

module load pytorch/24.12-py3

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

singularity exec "$PYTORCH_SIF" python3 -m dops_serve.cli serving-evaluate \
  --workload "$ARTIFACT_DIR/workloads_${MODEL_ID}_${RUN_NAME}.jsonl" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --checkpoint "$ARTIFACT_DIR/model_${MODEL_ID}_${RUN_NAME}/checkpoints/best.pt" \
  --output-dir "$ARTIFACT_DIR/serving_eval_${MODEL_ID}_${RUN_NAME}" \
  --hardware-key "$HARDWARE_KEY" \
  --split test \
  --horizon 32 \
  --horizon 64 \
  --horizon 128 \
  --fixed-horizon 64 \
  --max-decisions 5000 \
  --latency-slo-s "$LATENCY_SLO_S" \
  --plan-switch-margin "$PLAN_SWITCH_MARGIN"
