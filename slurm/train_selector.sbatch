#!/bin/bash
#SBATCH --job-name=dopsserve-train
#SBATCH --partition=C064M0256G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full}
MODEL_ID=${MODEL_ID:-llama_7b}
HARDWARE_KEY=${HARDWARE_KEY:-exp1/hw_hardware_1npu_2aim/sst2_rst2}

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

run_python -m dops_serve.cli generate-workloads \
  --output "$ARTIFACT_DIR/workloads_${MODEL_ID}.jsonl" \
  --model "$MODEL_ID" \
  --traces-per-family 10 \
  --requests-per-trace 500 \
  --base-rate 2.0 \
  --seed 7
run_python -m dops_serve.cli dataset \
  --workload "$ARTIFACT_DIR/workloads_${MODEL_ID}.jsonl" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --output "$ARTIFACT_DIR/dataset_${MODEL_ID}.csv" \
  --hardware-key "$HARDWARE_KEY" \
  --max-decisions-per-trace 2000
run_python -m dops_serve.cli train \
  --dataset "$ARTIFACT_DIR/dataset_${MODEL_ID}.csv" \
  --output-dir "$ARTIFACT_DIR/model_${MODEL_ID}" \
  --epochs 80 \
  --batch-size 512 \
  --seed 7
