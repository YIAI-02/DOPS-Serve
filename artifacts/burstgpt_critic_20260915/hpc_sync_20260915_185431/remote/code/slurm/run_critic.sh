#!/bin/bash
set -euo pipefail
module load pytorch/24.12-py3
CRITIC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_critic_20260915
export PYTHONPATH="$CRITIC_ROOT/code/src:$CRITIC_ROOT/backend/DOPS/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIM_LATENCY_CACHE_FILE="$CRITIC_ROOT/runtime/pim_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.json"
mkdir -p "$CRITIC_ROOT/runtime"
cd "$CRITIC_ROOT/code"
CRITIC_STAGE="$1"
shift
if [[ "$CRITIC_STAGE" == check ]]; then
    singularity exec --nv "$PYTORCH_SIF" python3 -m unittest discover -s tests -p test_burstgpt_ppo.py -v
    singularity exec --nv "$PYTORCH_SIF" python3 -m unittest discover -s tests -p test_critic_pretrain.py -v
    exec singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo critic-check --root "$CRITIC_ROOT" --device cuda
elif [[ "$CRITIC_STAGE" == collect ]]; then
    exec singularity exec "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo collect-critic --root "$CRITIC_ROOT" --task "$SLURM_ARRAY_TASK_ID"
else
    exec singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo train-critic --root "$CRITIC_ROOT" "$@"
fi
