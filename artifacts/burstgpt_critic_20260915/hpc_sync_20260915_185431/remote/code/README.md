# Static-task BC, 2026-09-13

This directory is an independent copy of the project code. Run only `dops_serve.static_bc` for this experiment; legacy CLI modules are not the entrypoint.

## Flow

1. `prepare`: copy the recorded C026 fast parameters and per-case run configs and the 144/24/24 grouped split. Test remains locked.
2. `import-teachers` records the 168 train/val tasks' existing schedule paths; test schedules are not read. `teacher --case-id N`: reuse all nine existing native DOPS methods (generate only missing sources), normalize full trajectories, replay every decode token, select the maximum-throughput complete winner, and capture only its decision states. Raw and normalized candidates remain available.
3. `smoke`: native replay must reproduce all 27 pilot source totals; three model shapes must learn a fixed teacher on GPU, and a spawned CPU worker must finish autonomous full-task replay.
4. `collect`: require all 168 train/validation task results before producing the ready marker.
5. `train --route {qwen_1.8b,qwen_7b,qwen_14b,shared}`: fresh initialization; exactly ten epochs, autonomous validation each epoch; then stop for user review.

## Code

- `placement_compiler.py:StaticPlacementCompiler`: exact NPU/PIM0/PIM1 labels and common topological/small-operator rules.
- `placement_dataset.py:static_policy_inputs`: 47 state and 35 node features. The last six node features encode actual prefill and previous decode devices. Current target actions are absent.
- `placement_policy.py:StaticPlacementActor`: 4-layer state/node encoders; node GRU; shifted teacher-forcing history; three-device head; no value head.
- `static_bc.py`: data generation, full-task selection/replay, task-weighted training, validation, CSV and plots.
- `dops_fixed_replay.py`: calls native DOPS `schedule_with_plan`; no replacement latency simulator.

## Outputs

- `protocol.json`, `cases.json`: frozen parameters and split.
- `teachers/<case>/<method>/`: source configs, schedules, and DOPS timings.
- `normalized/<case>/<method>.json`: complete normalized trajectory, with decode plans at t=0,1,2,4,6,...
- `anchors/<case>.jsonl.gz`: only the winning teacher's states and actions.
- `results/<case>.json`, `teacher_summary.csv`: all candidate scores, winning source, PD reference.
- `training/<route>/epoch_*.pt`: actor, optimizer, RNG states; smoke weights are excluded.
- `training/<route>/training_progress.csv`, `epoch_*_cases.csv`, `report.md`, `figures/*.png|svg`: cumulative epoch metrics.

A decode horizon of 128/256/512/1024 yields 66/130/258/514 plans including prefill. Each independent route has 48 training tasks, 11,616 plans, and 726 gradient batches per epoch. Shared has 144 tasks, 34,848 plans, and 2,178 batches. This is conditional-NLL imitation of one whole-task winner; there is no PPO/reward mixing or BurstGPT sampling.

BC and PD decision timings use the same allocated CPU, one PyTorch thread. JIT warmup and shared backend initialization are outside per-decision timing. PD native device selection is remeasured once per route on that host and frozen for ten epochs. Simulator execution wall time is excluded. The primary teacher gap includes BC decision cost; per-case simulation and decision components are also retained.

Submission order is check -> three pilot cases -> GPU/native smoke -> remaining teachers -> dataset gate -> four independent Slurm training jobs. `jobs.tsv` records each submitted job. No automatic training beyond epoch 10, PPO, or test execution is included.

Raw per-operator CSV export is disabled for new generation calls because this pipeline uses the full native schedule JSON; source simulation and candidate costs are unchanged. A retry within this experiment reuses only methods with a completed `generated.json` in this new artifact root. The corrected run imports Bifocal from the C026 scan and the other eight methods from fast_bc_restart_20260908. Original source paths and configs are recorded in generated.json; unknown historic decision time remains null and PD is measured on the BC host.

Inference batches node-only affine terms while using the same native GRU equations and parameters. `inference-check` verifies that generated choices match native-GRU conditional logits on all pilot decision states and completes full replay; `inference_passed.json` is required before training.

The source C026 settings are authoritative: refresh stride 2, PIM load overlap 0.5, load/compute overlap 1.0. The discarded stride-32 run with swapped overlap values is not part of this experiment; see `../cleanup_20260913/README.md` for its cleanup record.
