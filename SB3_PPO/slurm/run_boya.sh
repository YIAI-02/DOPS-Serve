#!/usr/bin/env bash
#SBATCH --account=lab-8-3090
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --exclude=node003
#SBATCH --qos=8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=sb3-ppo
#SBATCH --output=runs/boya_%j.out
# node003 在迁移验收时持续等待共享盘 RPC；节点恢复后可移除此排除。
# 博雅头节点只做同步、环境配置和提交；测试与训练仅在 Slurm 计算节点运行。
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/.mpl-cache"
python_cmd=(.boya-venv/bin/python)
stage="${1:?check, train, or compare}"
output="${2:?output directory}"
shift 2
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
