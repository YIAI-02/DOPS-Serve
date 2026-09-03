#!/bin/bash
#SBATCH --job-name=dopsserve-anchor
#SBATCH --partition=C064M0256G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full_trace_v4}

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
  --output "$ARTIFACT_DIR/anchor_dataset_128.csv" \
  --decode-len 128
run_python -m dops_serve.cli train \
  --dataset "$ARTIFACT_DIR/anchor_dataset_128.csv" \
  --output-dir "$ARTIFACT_DIR/model_anchor_128" \
  --epochs 80 \
  --batch-size 256 \
  --seed 7
