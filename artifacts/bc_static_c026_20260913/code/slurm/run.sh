#!/bin/bash
set -euo pipefail
module load pytorch/24.12-py3
BC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_static_c026_20260913
export PYTHONPATH="$BC_ROOT/code/src:$BC_ROOT/backend/DOPS/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIM_LATENCY_CACHE_FILE="$BC_ROOT/runtime/pim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.json"
mkdir -p "$BC_ROOT/runtime"
cd "$BC_ROOT/code"
STAGE="$1"
shift
if [[ "$STAGE" == preflight ]]; then
    singularity exec "$PYTORCH_SIF" python3 -m unittest discover -s tests -v
    singularity exec "$PYTORCH_SIF" python3 -m dops_serve.static_bc prepare --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS" --c026-root /lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bifocal_hparam_scan_20260909/full_results/C026 --other-root /lustre/home/2501111916/workspace/DOPS-Serve/artifacts/fast_bc_restart_20260908
    singularity exec "$PYTORCH_SIF" python3 -m dops_serve.static_bc import-teachers --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS"
elif [[ "$STAGE" == pilot ]]; then
    BC_CASE_ID=$((SLURM_ARRAY_TASK_ID * 64))
    singularity exec "$PYTORCH_SIF" python3 -m dops_serve.static_bc teacher --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS" --case-id "$BC_CASE_ID"
elif [[ "$STAGE" == teacher ]]; then
    if (( $# == 0 )); then
        set -- --case-id "$SLURM_ARRAY_TASK_ID"
    fi
    singularity exec "$PYTORCH_SIF" python3 -m dops_serve.static_bc teacher --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS" "$@"
elif [[ "$STAGE" == train || "$STAGE" == smoke ]]; then
    singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.static_bc "$STAGE" --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS" "$@"
else
    singularity exec "$PYTORCH_SIF" python3 -m dops_serve.static_bc "$STAGE" --root "$BC_ROOT" --backend "$BC_ROOT/backend/DOPS" "$@"
fi
