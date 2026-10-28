#!/bin/bash
set -euo pipefail
module load pytorch/24.12-py3
PPO_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/burstgpt_ppo_20260917
export PYTHONPATH="$PPO_ROOT/code/src:$PPO_ROOT/backend/DOPS/src"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export SINGULARITYENV_NVIDIA_TF32_OVERRIDE=0 SINGULARITYENV_TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
export TZ=Asia/Shanghai SINGULARITYENV_TZ=Asia/Shanghai
export PIM_LATENCY_CACHE_FILE="$PPO_ROOT/runtime/pim_${SLURM_JOB_ID}.pkl"
mkdir -p "$PPO_ROOT/runtime"
cd "$PPO_ROOT/code"
case "$1" in
    cpu-check)
        singularity exec "$PYTORCH_SIF" python3 - "$PPO_ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
protocol = json.loads((root / "protocol.json").read_text())
for split, key in (("train", "train_indices"), ("val", "quick_val_indices")):
    for index in protocol[key]:
        rows = json.loads((root / "data" / split / f"{index:06d}.json").read_text())
        assert len(rows) == 1000
        assert all(r["max_tokens"] == 7315 and 0 < r["output_len"] <= 7315 for r in rows)
print("Selected 8 train + 4 val complete traces verified; test unopened", flush=True)
PY
        singularity exec "$PYTORCH_SIF" python3 -m unittest discover -s tests -v
        singularity exec "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo preflight --root "$PPO_ROOT"
        singularity exec "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo orchestration-check --root "$PPO_ROOT"
        singularity exec "$PYTORCH_SIF" python3 -m dops_serve.cache_check --root "$PPO_ROOT" \
            --reference "$PPO_ROOT/validation_reference" \
            --output "$PPO_ROOT/hpc_validation/cache_full" --workers 16
        printf '{"passed":true,"job_id":"%s"}\n' "$SLURM_JOB_ID" > "$PPO_ROOT/hpc_cpu_check_passed.json"
        ;;
    cuda-check)
        test -f "$PPO_ROOT/exact_cache_check_passed.json"
        singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo worker-check --root "$PPO_ROOT" --device cuda
        ;;
    train)
        test -f "$PPO_ROOT/formal_budget.json"
        test -f "$PPO_ROOT/hpc_cpu_check_passed.json"
        test -f "$PPO_ROOT/worker_check_cuda_passed.json"
        test -f "$PPO_ROOT/exact_cache_check_passed.json"
        PPO_ROUTE="${2:?Missing model route}"
        PPO_SEED="${3:?Missing seed}"
        PPO_RESUME=()
        if [[ -f "$PPO_ROOT/training/$PPO_ROUTE/seed_$PPO_SEED/last.pt" ]]; then
            PPO_RESUME=(--resume)
        fi
        singularity exec --nv "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo train \
            --root "$PPO_ROOT" --route "$PPO_ROUTE" --seed "$PPO_SEED" --rounds 50 --device cuda "${PPO_RESUME[@]}"
        singularity exec "$PYTORCH_SIF" python3 -m dops_serve.burstgpt_ppo finalize \
            --root "$PPO_ROOT" --route "$PPO_ROUTE" --seed "$PPO_SEED"
        ;;
    *) echo "Unknown PPO stage: $1" >&2; exit 2 ;;
esac
