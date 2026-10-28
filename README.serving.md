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
- Closed-loop evaluation gives each policy the same request trace and an
  independent runtime state. Its latency metric is the global p95 over all
  completed requests, not the earlier mean of anchor-window labels.
- The MLP latency guardrail projects only already-arrived requests using their
  declared `max_tokens`; only the explicitly named myopic oracle may use
  simulator completion truth for one-step action selection.
- A closed-loop horizon is a maximum. If an observed completion moves the
  runtime state outside the selected plan's profile envelope, replay stops at
  that token boundary and selects again without extrapolating the missing shape.
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
  --policy hefthint --policy ianus --policy pd --policy weights_on_pim \
  --workers 6

python -m dops_serve.cli profiles \
  --inventory artifacts/inventory.csv \
  --plan-sources artifacts/plan_sources.csv \
  --artifact-dir artifacts

python -m dops_serve.cli anchor-dataset \
  --comparisons artifacts/anchor_comparisons.csv \
  --plan-sources artifacts/plan_sources.csv \
  --profiles artifacts/profiles.csv \
  --signatures artifacts/plan_signatures.csv \
  --output artifacts/anchor_dataset_windows.csv \
  --workers 8

python -m dops_serve.cli generate-workloads \
  --output artifacts/workloads.jsonl \
  --model llama_7b

python -m dops_serve.cli import-burstgpt \
  --input data/BurstGPT_without_fails_1.csv \
  --output artifacts/workloads_burstgpt.jsonl \
  --model llama_7b \
  --traces 16 \
  --requests-per-trace 100 \
  --target-median-rps 2.0 \
  --max-tokens-default 1024 \
  --source-model ChatGPT \
  --log-type "API log" \
  --selection mixed \
  --split-seed 7

python -m dops_serve.cli dataset \
  --workload artifacts/workloads.jsonl \
  --profiles artifacts/profiles.csv \
  --signatures artifacts/plan_signatures.csv \
  --output artifacts/dataset.csv \
  --hardware-key exp1/hw_hardware_1npu_2aim/sst2_rst2

python -m dops_serve.cli train \
  --dataset artifacts/dataset.csv \
  --output-dir artifacts/model_llama_7b

python -m dops_serve.cli serving-evaluate \
  --workload artifacts/workloads.jsonl \
  --profiles artifacts/profiles.csv \
  --signatures artifacts/plan_signatures.csv \
  --checkpoint artifacts/model_llama_7b/checkpoints/best.pt \
  --output-dir artifacts/serving_eval_llama_7b \
  --hardware-key exp1/hw_hardware_1npu_2aim/sst2_rst2 \
  --latency-slo-s 30 \
  --plan-switch-margin 0.02
```

Use `slurm/smoke_real_outputs.sbatch` first. The full anchor pipeline is split
into `slurm/build_artifacts.sbatch` and `slurm/train_anchor.sbatch`. The build
script requests one 64-core node because this cluster allocates memory per CPU,
then runs six independent model/hardware groups concurrently. All scripts must
be submitted from a login node; execution stays on Slurm compute nodes. Use
`slurm/evaluate_serving.sbatch` to rerun only closed-loop evaluation from an
existing irregular workload and checkpoint.
`slurm/run_burstgpt_real.sbatch` builds temporally separated representative and
peak windows from the official BurstGPT CSV, applies seeded trace-grouped
splits within each window class, and evaluates both the synthetic checkpoint
and the real-trace checkpoint on
the same held-out requests. The importer filters to the supported profile
envelope and assigns an independent constant `max_tokens`; `Response tokens`
is used only to terminate offline replay.

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

The 64-core, six-worker v7 build (`14733834`) completed in 2:22:29. It parsed
all 4,669 summaries and retained 299 plans; all 299 have complete placement,
including 215 plans covered through 1,024 decode steps. It mapped 3,993
profiles and excluded 676 source mappings only because of the 32-plan cap.
Leave-one-profile-out validation has 7,916 rows and 14.9% p95 absolute
percentage error.

The dependent eight-worker anchor run (`14733838`) completed in 17:38. It
retained 576 paired comparisons across source decode lengths 128, 256, 512,
and 1024, producing 8,600 states and 168,876 candidate-action rows
(117,432/27,240/24,204 train/validation/test). Every state compares 2--8 plans
and every plan has horizons 32, 64, and 128. On the same 1,284 paired test
states, the MLP reached 99.1% of the offline oracle's mean window throughput.
Its 358 action switches and 5.81 s mean window p95 latency are not evidence that
serving stability or latency guardrails are satisfied. These remain offline
regular-anchor timing labels, not tensor execution or irregular serving replay.

The first irregular synthetic run built 16 traces and 23,447 valid
counterfactual rows, then trained a 53,674-byte checkpoint. After changing
closed-loop horizons to stop and replan at an observed profile boundary,
evaluation-only high-QoS job `14740006` completed all four held-out traces and
400 requests with no exclusions. Against the paired robust-static baseline,
the MLP increased output-token throughput from 176.19 to 194.18 token/s
(+10.21%) and reduced global request p95 from 47.44 to 36.75 s (-22.53%). It
reached 96.22% of the myopic oracle's throughput, but still had 31 requests over
the 30 s SLO because 33 decisions had no projected action below that limit.

[The official BurstGPT v2.0 no-failure trace](https://github.com/HPMLL/BurstGPT/releases/tag/v2.0)
contains 1,404,294 requests. The
profile-compatible ChatGPT API subset has 706,125 requests; the v2 experiment
uses 16 non-overlapping, time-stratified windows (eight representative and eight
peak), with seeded grouped splits of 10/2/4 traces. Held-out replay covers 400
requests and 34,751 output tokens. Against the same paired robust-static
baseline, the real-trace MLP increased end-to-end output-token throughput from
109.729 to 110.741 token/s (+0.92%), reduced mean request latency from 2.634 to
2.253 s (-14.48%), and reduced global p95 latency from 9.330 to 9.247 s
(-0.89%). Its p95 queue wait increased from 2.526 to 2.772 s (+9.76%); no
request violated the 30 s SLO. It reached 99.20% of myopic-oracle throughput
with five plan switches and 20 horizon switches. The synthetic checkpoint had
nearly identical throughput and a lower 8.682 s p95, so this amount of real
training data does not yet establish an adaptation benefit.
Training loss continued down to 0.0050 at epoch 80, while validation MSE was
best at epoch 18 (0.318) and ended at 0.411; the saved checkpoint correctly
uses epoch 18 rather than the overfit final epoch.

Both experiments are profile-driven timing replay. They do not execute model
tensors or kernels and do not establish physical NPU/PIM serving performance.
