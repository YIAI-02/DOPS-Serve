# DOPS-Serve Closed-Loop Evaluation

All policies replay the same paired request traces but advance independent
runtime states. Results use DOPS profile timing, not tensor execution.
The myopic oracle uses simulator truth for one-step action selection only.

| Policy | Traces | Requests | Throughput | vs. oracle | p95 E2E | p95 wait | SLO violations | Plan switches | Horizon switches | Boundary replans | Forced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| robust_static | 4 | 400 | 109.728960 | 0.9830 | 9.330398 | 2.525602 | 0 | 0 | 0 | 0 | 0 |
| nearest_shape | 4 | 400 | 109.728960 | 0.9830 | 9.330398 | 2.525602 | 0 | 0 | 0 | 0 | 0 |
| mlp_selector | 4 | 400 | 110.739627 | 0.9920 | 8.682067 | 2.893985 | 0 | 0 | 8 | 0 | 0 |
| myopic_oracle | 4 | 400 | 111.630681 | 1.0000 | 7.244200 | 1.726663 | 0 | 19 | 93 | 0 | 0 |

Paired traces: 4; excluded traces: 0.
MLP latency SLO: 30.000000 s; plan switch margin: 0.0200.
