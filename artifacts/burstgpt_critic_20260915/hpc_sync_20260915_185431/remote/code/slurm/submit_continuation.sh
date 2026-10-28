#!/bin/bash
set -euo pipefail
BC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_phase_ab_20260913
BC_RUN="$BC_ROOT/fresh_b50_20260914"
BC_CONTINUE="$BC_RUN/continuation_to_e100"
test -s "$BC_CONTINUE/preflight_passed.json"
mkdir -p "$BC_CONTINUE/logs"
if [[ ! -e "$BC_CONTINUE/jobs.tsv" ]]; then
    printf 'route\tjob_id\n' > "$BC_CONTINUE/jobs.tsv"
fi
BC_ROUTES=("$@")
if (( $# == 0 )); then
    BC_ROUTES=(qwen_1.8b qwen_7b)
fi
for BC_ROUTE in "${BC_ROUTES[@]}"; do
    if awk -F '\t' -v route="$BC_ROUTE" '$1 == route { found=1 } END { exit !found }' "$BC_CONTINUE/jobs.tsv"; then
        printf 'Already submitted: %s\n' "$BC_ROUTE" >&2
        exit 1
    fi
done
for BC_ROUTE in "${BC_ROUTES[@]}"; do
    BC_CPUS=16
    BC_WORKERS=8
    if [[ "$BC_ROUTE" == shared ]]; then
        BC_CPUS=32
        BC_WORKERS=24
    fi
    BC_JOB=$(sbatch --parsable -A hpc2006191081 -p GPU80G -q high -J "bc100-$BC_ROUTE" -c "$BC_CPUS" --gres=gpu:1 -t 12:00:00 -o "$BC_CONTINUE/logs/train_${BC_ROUTE}_%j.out" "$BC_ROOT/code/slurm/run.sh" train --route "$BC_ROUTE" --workers "$BC_WORKERS" --resume-epoch 50 --epochs 100)
    printf '%s\t%s\n' "$BC_ROUTE" "$BC_JOB" >> "$BC_CONTINUE/jobs.tsv"
done
cat "$BC_CONTINUE/jobs.tsv"
