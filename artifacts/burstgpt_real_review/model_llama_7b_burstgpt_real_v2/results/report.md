# DOPS-Serve Evaluation

All policies are evaluated on the same request-state rows. Values are
profile-driven replay results, not physical hardware measurements.

| Policy | States | Mean token throughput | vs. oracle | Mean window p95 latency | Mean window p95 wait | Switches |
|---|---:|---:|---:|---:|---:|---:|
| robust_static | 288 | 244.482575 | 0.9271 | 1.511963 | 0.146563 | 0 |
| nearest_shape | 288 | 125.015096 | 0.4741 | 1.616147 | 0.146563 | 0 |
| mlp_selector | 288 | 251.200941 | 0.9526 | 1.706190 | 0.146563 | 155 |
| offline_oracle | 288 | 263.713871 | 1.0000 | 1.480502 | 0.146563 | 244 |

Paired states: 288; excluded unpaired states: 0.
Batched CPU scorer timing: 284.815 us/state; this is not production online latency.

Static plans:

- `llama_7b|exp1/hw_hardware_1npu_2aim/sst2_rst2`: `llama_7b__exp1_hw_hardware_1npu_2aim_sst2_rst2__p04`
