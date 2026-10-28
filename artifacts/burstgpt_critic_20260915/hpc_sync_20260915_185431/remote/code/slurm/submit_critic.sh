#!/bin/bash
set -euo pipefail
CRITIC_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CHECK_JOB="$1"
test -f "$CRITIC_ROOT/critic_check_passed.json"
test ! -e "$CRITIC_ROOT/jobs.tsv"
mkdir -p "$CRITIC_ROOT/logs"
RUNNER="$CRITIC_ROOT/code/slurm/run_critic.sh"
ARRAY_JOB=$(sbatch --parsable --account=hpc2006191081 --qos=high --partition=C064M0256G \
    --cpus-per-task=1 --time=24:00:00 --array=0-107%18 --dependency="afterok:$CHECK_JOB" \
    --job-name=critic-data --output="$CRITIC_ROOT/logs/data_%A_%a.log" "$RUNNER" collect)
printf 'stage\troute\tjob_id\ncollect\tall\t%s\n' "$ARRAY_JOB" > "$CRITIC_ROOT/jobs.tsv"
for CRITIC_ROUTE in qwen_1.8b qwen_7b qwen_14b shared; do
    DEPENDENCY=$(awk -v route="$CRITIC_ROUTE" -v job="$ARRAY_JOB" 'NR > 1 && $2 == route {printf ":%s_%s", job, $1}' "$CRITIC_ROOT/tasks.tsv")
    TRAIN_JOB=$(sbatch --parsable --account=hpc2006191081 --qos=high --partition=GPU80G \
        --gres=gpu:1 --cpus-per-task=2 --time=08:00:00 --dependency="afterok$DEPENDENCY" \
        --job-name="critic-$CRITIC_ROUTE" --output="$CRITIC_ROOT/logs/train_${CRITIC_ROUTE}_%j.log" \
        "$RUNNER" train --route "$CRITIC_ROUTE" --device cuda)
    printf 'train\t%s\t%s\n' "$CRITIC_ROUTE" "$TRAIN_JOB" >> "$CRITIC_ROOT/jobs.tsv"
done
cat "$CRITIC_ROOT/jobs.tsv"
