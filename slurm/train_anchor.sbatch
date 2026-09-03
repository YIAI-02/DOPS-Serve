#!/bin/bash
#SBATCH --job-name=dopsserve-anchor
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full}
ANCHOR_WORKERS=${ANCHOR_WORKERS:-8}

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src"

run_python -m dops_serve.cli anchor-dataset \
  --comparisons "$ARTIFACT_DIR/anchor_comparisons.csv" \
  --plan-sources "$ARTIFACT_DIR/plan_sources.csv" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --output "$ARTIFACT_DIR/anchor_dataset_windows.csv" \
  --decode-len 128 \
  --decode-len 256 \
  --decode-len 512 \
  --decode-len 1024 \
  --horizon 32 \
  --horizon 64 \
  --horizon 128 \
  --window-stride 32 \
  --workers "$ANCHOR_WORKERS"
run_python -m dops_serve.cli train \
  --dataset "$ARTIFACT_DIR/anchor_dataset_windows.csv" \
  --output-dir "$ARTIFACT_DIR/model_anchor_windows" \
  --epochs 80 \
  --batch-size 256 \
  --seed 7
