# DOPS-Serve

DOPS-Serve is an independent, output-only pipeline for building serving-side
training data from precomputed [DOPS](https://github.com/YIAI-02/DOPS)
schedules. It never imports or invokes the DOPS scheduler.

## Boundary

- Input: an existing DOPS `output/` tree, opened read-only.
- Derived artifacts: inventory, plan bank, performance profiles, serving
  episodes, model checkpoints, and paired evaluation results.
- V1 action: `(plan_id, horizon)` with horizons 32, 64, and 128.
- Plan identity uses node-to-device mappings and decode transitions. Sampled
  `schedule: null` steps are recovered only from the linked operation trace,
  after every observed summary placement matches that trace exactly.
- Dataset generation rejects plans that remain incomplete after trace recovery,
  unless a smoke run explicitly uses `--allow-incomplete-plans`.
- Offline `output_len` may terminate replay, but it is never an online feature.
- Replay results are simulation artifacts, not physical NPU/PIM measurements.

The first target input tree is:

```text
/lustre/home/2501111916/workspace/DOPS_0330_merge/TriForm/algorithms/output
```

## Quick start

Run the standard-library stages directly from the checkout:

```bash
export PYTHONPATH="$PWD/src"
python -m dops_serve.cli inventory \
  --output-root /path/to/DOPS/output \
  --artifact-dir artifacts

python -m dops_serve.cli anchor-comparisons \
  --output-root /path/to/DOPS/output \
  --inventory artifacts/inventory.csv \
  --output artifacts/anchor_comparisons.csv

python -m dops_serve.cli plan-bank \
  --inventory artifacts/inventory.csv \
  --artifact-dir artifacts \
  --policy attacc --policy attn_on_pim --policy facil --policy heft \
  --policy hefthint --policy ianus --policy pd --policy weights_on_pim

python -m dops_serve.cli profiles \
  --inventory artifacts/inventory.csv \
  --plan-sources artifacts/plan_sources.csv \
  --artifact-dir artifacts

python -m dops_serve.cli anchor-dataset \
  --comparisons artifacts/anchor_comparisons.csv \
  --plan-sources artifacts/plan_sources.csv \
  --profiles artifacts/profiles.csv \
  --signatures artifacts/plan_signatures.csv \
  --output artifacts/anchor_dataset_windows.csv

python -m dops_serve.cli generate-workloads \
  --output artifacts/workloads.jsonl \
  --model llama_7b

python -m dops_serve.cli dataset \
  --workload artifacts/workloads.jsonl \
  --profiles artifacts/profiles.csv \
  --signatures artifacts/plan_signatures.csv \
  --output artifacts/dataset.csv \
  --hardware-key exp1/hw_hardware_1npu_2aim/sst2_rst2

python -m dops_serve.cli train \
  --dataset artifacts/dataset.csv \
  --output-dir artifacts/model_llama_7b
```

Use `slurm/smoke_real_outputs.sbatch` first. The full profile build and model
training are split into `slurm/build_artifacts.sbatch` and
`slurm/train_selector.sbatch`. Training requires PyTorch and all three scripts
must be submitted from a login node; compilation and execution stay on Slurm
compute nodes.

## Current output audit

The 2026-09-03 re-audit found 4,673 summaries and 711
`baseline_compare_*.json` files. Those comparison files contain 4,639 policy
outcomes; 579 files compare 2--8 policies under the same regular shape. They
are useful supervised labels for anchor-workload plan ranking.

The `anchor-comparisons` stage links those outcomes to the summary inventory,
checks timing agreement, and excludes unpaired records. The full run retained
577 paired groups and 4,495 ranking rows. It excluded 10 outcomes without a
valid inventory summary and two whose comparison and summary timings disagree.
The paired anchor labels do not replace replaying complete fixed plans under
the same irregular serving state. The first full Plan Bank run parsed all 4,669
valid summaries, retained 299 complete plans and 4,079 mapped profiles, and
excluded 590 source mappings only because of the 32-plan-per-target cap. Its
leave-one-profile-out validation produced 8,074 rows with 14.8% p95 absolute
percentage error; this is an interpolation diagnostic, not a serving result.

`anchor-dataset` maps only paired rows whose summaries belong to distinct,
complete retained plans. It reads measured `step_time` values from source
decode lengths 128, 256, 512, and 1024, creates a state every 32 generated
tokens, and labels actions with horizons 32, 64, and 128. A horizon that reaches
the end of an offline trace is truncated to the tokens actually generated.
Online state features contain only the current generated-token count and active
context; all `max_tokens` fields remain zero, and the final source decode length
is not a model feature. Plan extraction now validates placement through at most
1,024 decode steps and keeps the longest representative of an equivalent plan.

The completed 128-only anchor diagnostic contained 146 paired states and 1,005
candidate rows (703/123/179 train/validation/test). Its MLP reached 95.9% of the
offline oracle's mean throughput on 27 paired test states. This remains a
regular-anchor scorer diagnostic; serving claims still require counterfactual
replay on the same irregular request states.
