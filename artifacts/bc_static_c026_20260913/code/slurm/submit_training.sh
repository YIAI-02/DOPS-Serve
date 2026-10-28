#!/bin/bash
set -euo pipefail
BC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_static_c026_20260913
BC_RUN="$BC_ROOT/code/slurm/run.sh"
BC_ARRAY=$(cat "$BC_ROOT/code/slurm/teacher_array.txt")
BC_CHECK=${1:?Supply the corrected preflight job ID}
if [[ -e "$BC_ROOT/jobs.tsv" ]]; then
    printf 'Submission record already exists; inspect it before another submission.\n' >&2
    exit 1
fi
printf 'stage\tjob_id\npreflight\t%s\n' "$BC_CHECK" > "$BC_ROOT/jobs.tsv"
BC_PILOT=$(sbatch --parsable -A hpc2006191081 -p C064M0256G -q high -J bc26-pilot -c 4 -t 02:00:00 --array=0-2%3 --dependency="afterok:$BC_CHECK" -o "$BC_ROOT/logs/pilot_%A_%a.out" "$BC_RUN" pilot)
printf 'pilot\t%s\n' "$BC_PILOT" >> "$BC_ROOT/jobs.tsv"
BC_SMOKE=$(sbatch --parsable -A hpc2006191081 -p GPU80G -q high -J bc26-native-check -c 8 --gres=gpu:1 -t 02:00:00 --dependency="afterok:$BC_PILOT" -o "$BC_ROOT/logs/smoke_%j.out" "$BC_RUN" smoke)
printf 'smoke\t%s\n' "$BC_SMOKE" >> "$BC_ROOT/jobs.tsv"
BC_INFERENCE=$(sbatch --parsable -A hpc2006191081 -p C064M0256G -q high -J bc26-infer-check -c 4 -t 02:00:00 --dependency="afterok:$BC_SMOKE" -o "$BC_ROOT/logs/inference_%j.out" "$BC_RUN" inference-check)
printf 'inference_check\t%s\n' "$BC_INFERENCE" >> "$BC_ROOT/jobs.tsv"
BC_TEACHER=$(sbatch --parsable -A hpc2006191081 -p C064M0256G -q high -J bc26-rescore -c 4 -t 4-00:00:00 --array="$BC_ARRAY" --dependency="afterok:$BC_INFERENCE" -o "$BC_ROOT/logs/teacher_%A_%a.out" "$BC_RUN" teacher)
printf 'teachers\t%s\n' "$BC_TEACHER" >> "$BC_ROOT/jobs.tsv"
BC_COLLECT=$(sbatch --parsable -A hpc2006191081 -p C064M0256G -q high -J bc26-dataset -c 2 -t 00:30:00 --dependency="afterok:$BC_TEACHER:$BC_PILOT" -o "$BC_ROOT/logs/dataset_%j.out" "$BC_RUN" collect)
printf 'dataset\t%s\n' "$BC_COLLECT" >> "$BC_ROOT/jobs.tsv"
for BC_ROUTE in qwen_1.8b qwen_7b qwen_14b shared; do
    BC_CPUS=16
    BC_WORKERS=8
    if [[ "$BC_ROUTE" == shared ]]; then
        BC_CPUS=32
        BC_WORKERS=24
    fi
    BC_JOB=$(sbatch --parsable -A hpc2006191081 -p GPU80G -q high -J "bc26-$BC_ROUTE" -c "$BC_CPUS" --gres=gpu:1 -t 4-00:00:00 --dependency="afterok:$BC_COLLECT:$BC_SMOKE:$BC_INFERENCE" -o "$BC_ROOT/logs/train_${BC_ROUTE}_%j.out" "$BC_RUN" train --route "$BC_ROUTE" --workers "$BC_WORKERS")
    printf '%s\t%s\n' "$BC_ROUTE" "$BC_JOB" >> "$BC_ROOT/jobs.tsv"
done
cat "$BC_ROOT/jobs.tsv"
