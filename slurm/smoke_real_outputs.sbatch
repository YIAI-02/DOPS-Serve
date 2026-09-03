#!/bin/bash
#SBATCH --job-name=dopsserve-smoke
#SBATCH --partition=C064M0256G
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --output=/lustre/home/2501111916/workspace/DOPS-Serve/logs/%x-%j.out

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/lustre/home/2501111916/workspace/DOPS-Serve}
DOPS_OUTPUT_ROOT=${DOPS_OUTPUT_ROOT:-/lustre/home/2501111916/workspace/DOPS_0330_merge/TriForm/algorithms/output}
ARTIFACT_DIR=${ARTIFACT_DIR:-$REPO_ROOT/artifacts/smoke/$SLURM_JOB_ID}
HARDWARE_KEY=${HARDWARE_KEY:-evaluate_single_test/hardware_1npu_2aim_evaluate_eta0p1}
SMOKE_PREFIX_B1=${SMOKE_PREFIX_B1:-$HARDWARE_KEY/llama_7b_fp16_b1_s2}
SMOKE_PREFIX_B4=${SMOKE_PREFIX_B4:-$HARDWARE_KEY/llama_7b_fp16_b4_s2}

module load pytorch/24.12-py3

run_python() {
  singularity exec "$PYTORCH_SIF" python3 "$@"
}

cd "$REPO_ROOT"
mkdir -p "$ARTIFACT_DIR"
export PYTHONPATH="$REPO_ROOT/src"

run_python -m unittest discover -s tests -v
run_python -m dops_serve.cli inventory \
  --output-root "$DOPS_OUTPUT_ROOT" \
  --artifact-dir "$ARTIFACT_DIR" \
  --path-prefix "$SMOKE_PREFIX_B1" \
  --path-prefix "$SMOKE_PREFIX_B4"
run_python -m dops_serve.cli plan-bank \
  --inventory "$ARTIFACT_DIR/inventory.csv" \
  --artifact-dir "$ARTIFACT_DIR" \
  --policy hefthint \
  --policy bifocal \
  --max-plans-per-group 8 \
  --max-decode-steps 128
run_python -m dops_serve.cli profiles \
  --inventory "$ARTIFACT_DIR/inventory.csv" \
  --plan-sources "$ARTIFACT_DIR/plan_sources.csv" \
  --artifact-dir "$ARTIFACT_DIR" \
  --include-trace-metrics
run_python -m dops_serve.cli generate-workloads \
  --output "$ARTIFACT_DIR/workloads.jsonl" \
  --model llama_7b \
  --traces-per-family 3 \
  --requests-per-trace 12 \
  --base-rate 1.0 \
  --seed 7
run_python -m dops_serve.cli dataset \
  --workload "$ARTIFACT_DIR/workloads.jsonl" \
  --profiles "$ARTIFACT_DIR/profiles.csv" \
  --signatures "$ARTIFACT_DIR/plan_signatures.csv" \
  --output "$ARTIFACT_DIR/dataset.csv" \
  --hardware-key "$HARDWARE_KEY" \
  --allow-extrapolation \
  --max-decisions-per-trace 64
run_python -m dops_serve.cli train \
  --dataset "$ARTIFACT_DIR/dataset.csv" \
  --output-dir "$ARTIFACT_DIR" \
  --epochs 10 \
  --batch-size 128 \
  --seed 7
