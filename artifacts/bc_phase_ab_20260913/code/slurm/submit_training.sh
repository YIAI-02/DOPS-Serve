#!/bin/bash
set -euo pipefail
BC_ROOT=/lustre/home/2501111916/workspace/DOPS-Serve/artifacts/bc_phase_ab_20260913
BC_RUN="$BC_ROOT/fresh_b50_20260914"
test -s "$BC_RUN/preflight_passed.json"
test ! -e "$BC_RUN/jobs.tsv"
printf 'route\tjob_id\n' > "$BC_RUN/jobs.tsv"
for BC_ROUTE in qwen_1.8b qwen_7b qwen_14b shared; do
    BC_CPUS=16
    BC_WORKERS=8
    if [[ "$BC_ROUTE" == shared ]]; then
        BC_CPUS=32
        BC_WORKERS=24
    fi
    BC_JOB=$(sbatch --parsable -A hpc2006191081 -p GPU80G -q high -J "bc50-$BC_ROUTE" -c "$BC_CPUS" --gres=gpu:1 -t 24:00:00 -o "$BC_RUN/logs/train_${BC_ROUTE}_%j.out" "$BC_ROOT/code/slurm/run.sh" train --route "$BC_ROUTE" --workers "$BC_WORKERS")
    printf '%s\t%s\n' "$BC_ROUTE" "$BC_JOB" >> "$BC_RUN/jobs.tsv"
done
cat "$BC_RUN/jobs.tsv"
