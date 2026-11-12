#!/usr/bin/env bash
# 通过 login 节点 sbatch 提交；实际安装、验收和训练都在计算节点执行。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
module load pytorch/24.12-py3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.mpl-cache"
# 容器默认 LD_LIBRARY_PATH 指向旧 torch/lib；CPU 实验必须同时隔离 Python 包和动态库。
python_cmd=(singularity exec "$PYTORCH_SIF" env -u PYTHONPATH -u LD_PRELOAD
    LD_LIBRARY_PATH="$PWD/.hpc-venv/lib/python3.12/site-packages/torch/lib" .hpc-venv/bin/python)
stage="${1:?setup-check, check, train, or compare}"
output="${2:?output directory}"
shift 2
if [[ "$stage" == setup-check ]]; then
    singularity exec "$PYTORCH_SIF" python3 -m venv .hpc-venv
    "${python_cmd[@]}" -m pip install \
        --no-index --find-links wheelhouse -r requirements.txt
    stage=check
fi
case "$stage" in
    check)
        "${python_cmd[@]}" -m pytest tests -q
        "${python_cmd[@]}" -m dops_sb3.train --smoke --out "$output" "$@"
        "${python_cmd[@]}" -m dops_sb3.verify_run "$output"
        ;;
    train)
        "${python_cmd[@]}" -m dops_sb3.train --out "$output" "$@"
        "${python_cmd[@]}" -m dops_sb3.verify_run "$output"
        ;;
    compare)
        "${python_cmd[@]}" -m dops_sb3.plot --compare "$output"
        ;;
    *)
        printf 'Unknown stage: %s\n' "$stage" >&2
        exit 2
        ;;
esac
