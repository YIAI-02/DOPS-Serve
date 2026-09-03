#!/bin/bash
#SBATCH --job-name=dopsserve-build
#SBATCH --partition=C064M0256G
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
DOPS_OUTPUT_ROOT=${DOPS_OUTPUT_ROOT:-/lustre/home/2501111916/workspace/DOPS_0330_merge/TriForm/algorithms/output}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/full}
INVENTORY_SOURCE=${INVENTORY_SOURCE:-}
REUSE_PLAN_BANK=${REUSE_PLAN_BANK:-0}
PLAN_BANK_WORKERS=${PLAN_BANK_WORKERS:-6}

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
mkdir -p "$ARTIFACT_DIR"
export PYTHONPATH="$REPO_ROOT/src"

if [[ -n "$INVENTORY_SOURCE" ]]; then
  cp "$INVENTORY_SOURCE/inventory.csv" "$ARTIFACT_DIR/inventory.csv"
  cp "$INVENTORY_SOURCE/excluded.csv" "$ARTIFACT_DIR/excluded.csv"
else
  run_python -m dops_serve.cli inventory \
    --output-root "$DOPS_OUTPUT_ROOT" \
    --artifact-dir "$ARTIFACT_DIR"
fi
run_python -m dops_serve.cli anchor-comparisons \
  --output-root "$DOPS_OUTPUT_ROOT" \
  --inventory "$ARTIFACT_DIR/inventory.csv" \
  --output "$ARTIFACT_DIR/anchor_comparisons.csv"
if [[ "$REUSE_PLAN_BANK" != "1" ]]; then
  run_python -m dops_serve.cli plan-bank \
    --inventory "$ARTIFACT_DIR/inventory.csv" \
    --artifact-dir "$ARTIFACT_DIR" \
    --policy attacc \
    --policy attn_on_pim \
    --policy facil \
    --policy heft \
    --policy hefthint \
    --policy ianus \
    --policy pd \
    --policy weights_on_pim \
    --max-plans-per-group 32 \
    --max-decode-steps 1024 \
    --workers "$PLAN_BANK_WORKERS"
fi
run_python -m dops_serve.cli profiles \
  --inventory "$ARTIFACT_DIR/inventory.csv" \
  --plan-sources "$ARTIFACT_DIR/plan_sources.csv" \
  --artifact-dir "$ARTIFACT_DIR"
