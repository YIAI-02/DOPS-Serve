#!/usr/bin/env bash
# 在博雅 headnode01 提交十二次训练；汇总依赖全部训练成功及模型重载验收。
set -euo pipefail
root="${1:?new output root, e.g. runs/mixed_workload_20261113}"
workload_args=(--workloads "${2:-configs/mixed_workloads.json}")
if [[ -n "${3:-}" ]]; then
    workload_args+=(--fixed-baselines "$3")
fi
cd "$(dirname "$0")/.."
mkdir "$root"
mkdir -p "$root"/{A,B,C,D}
printf 'variant,seed,job_id\n' > "$root/jobs.csv"
job_ids=()
for variant in A B C D; do
    for seed in 0 1 2; do
        job=$(sbatch --parsable --nodelist=node001 --mem=2G --job-name="mix-${variant}-s${seed}" \
            slurm/run_boya.sh train "$root/$variant/seed$seed" \
            "${workload_args[@]}" --variant "$variant" --seed "$seed")
        job_ids+=("$job")
        printf '%s,%s,%s\n' "$variant" "$seed" "$job" >> "$root/jobs.csv"
    done
done
dependency=$(IFS=:; printf '%s' "${job_ids[*]}")
summary=$(sbatch --parsable --nodelist=node001 --mem=2G --time=01:00:00 --job-name=mix-summary \
    --dependency="afterok:$dependency" slurm/run_boya.sh compare "$root")
printf 'summary,,%s\n' "$summary" >> "$root/jobs.csv"
cat "$root/jobs.csv"
