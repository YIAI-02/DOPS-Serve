#!/bin/bash
set -euo pipefail
module load pytorch/24.12-py3
BC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_phase_ab_20260913
BC_RUN="$BC_ROOT/fresh_b50_20260914"
export PYTHONPATH="$BC_ROOT/code/src:$BC_ROOT/backend/DOPS/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIM_LATENCY_CACHE_FILE="$BC_RUN/runtime/pim_${SLURM_JOB_ID}.json"
mkdir -p "$BC_RUN/runtime"
cd "$BC_ROOT/code"
BC_STAGE="$1"
shift
if [[ "$BC_STAGE" == preflight ]]; then
    singularity exec --nv "$PYTORCH_SIF" python3 -m unittest discover -s tests -v
    singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.static_bc_ab prepare --root "$BC_RUN" --backend "$BC_ROOT/backend/DOPS" --source-root /lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_static_c026_20260913
elif [[ "$BC_STAGE" == resume-preflight ]]; then
    singularity exec --nv "$PYTORCH_SIF" python3 -m unittest discover -s tests -v
fi
singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.static_bc_ab "$BC_STAGE" --root "$BC_RUN" --backend "$BC_ROOT/backend/DOPS" "$@"
