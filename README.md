# DOPS-Serve

DOPS-Serve is an independent, output-only pipeline for building serving-side
training data from precomputed [DOPS](https://github.com/YIAI-02/DOPS)
schedules. It never imports or invokes the DOPS scheduler.

## Boundary

- Input: an existing DOPS `output/` tree, opened read-only.
- Derived artifacts: inventory, plan bank, performance profiles, serving
  episodes, model checkpoints, and paired evaluation results.
- V1 action: `(plan_id, horizon)` with horizons 32, 64, and 128.
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

python -m dops_serve.cli plan-bank \
  --inventory artifacts/inventory.csv \
  --artifact-dir artifacts \
  --policy hefthint --policy bifocal

python -m dops_serve.cli profiles \
  --inventory artifacts/inventory.csv \
  --plan-sources artifacts/plan_sources.csv \
  --artifact-dir artifacts

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
