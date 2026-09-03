import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.replay import ProfileEstimator, ReplayState, request_states
from dops_serve.serving_evaluation import (
    POLICIES,
    _project_action,
    _select_mlp,
    evaluate_serving,
)


def write_profiles(root: Path) -> tuple[Path, Path]:
    profiles = root / "profiles.csv"
    with profiles.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "plan_id",
                "model_id",
                "hardware_key",
                "batch",
                "prefill_len",
                "decode_len",
                "context_mid",
                "prefill_time_s",
                "decode_s_per_iteration",
            ],
        )
        writer.writeheader()
        for plan_id, scale in (("plan_fast", 1.0), ("plan_slow", 2.0)):
            for batch in (1, 4):
                writer.writerow(
                    {
                        "plan_id": plan_id,
                        "model_id": "llama_7b",
                        "hardware_key": "hw",
                        "batch": batch,
                        "prefill_len": 128,
                        "decode_len": 8,
                        "context_mid": 132,
                        "prefill_time_s": 0.1 * scale,
                        "decode_s_per_iteration": 0.01 * scale,
                    }
                )

    signatures = root / "plan_signatures.csv"
    with signatures.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "plan_id",
                "model_id",
                "hardware_key",
                "placement_complete",
                "incomplete_source_count",
                "profile_count",
                "mean_prefill_s_per_input_token",
                "mean_decode_s_per_iteration",
                "mean_output_token_throughput",
                "mean_total_time_s",
            ],
        )
        writer.writeheader()
        for plan_id, scale in (("plan_fast", 1.0), ("plan_slow", 2.0)):
            writer.writerow(
                {
                    "plan_id": plan_id,
                    "model_id": "llama_7b",
                    "hardware_key": "hw",
                    "placement_complete": True,
                    "incomplete_source_count": 0,
                    "profile_count": 2,
                    "mean_prefill_s_per_input_token": 0.001 * scale,
                    "mean_decode_s_per_iteration": 0.01 * scale,
                    "mean_output_token_throughput": 100.0 / scale,
                    "mean_total_time_s": scale,
                }
            )
    return profiles, signatures


def records(trace_id: str, output_len: int = 3) -> list[dict[str, object]]:
    return [
        {
            "trace_id": trace_id,
            "family": "bursty",
            "request_id": f"{trace_id}-r{index}",
            "model_id": "llama_7b",
            "arrival_time": index * 0.02,
            "prompt_len": 128,
            "max_tokens": 4,
            "output_len": min(output_len + index, 4),
        }
        for index in range(3)
    ]


class ServingEvaluationTest(unittest.TestCase):
    def test_projection_does_not_use_final_output_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, signatures = write_profiles(root)
            estimator = ProfileEstimator(profiles, signatures)
            short = ReplayState(request_states(records("short", output_len=1)), max_batch=4)
            long = ReplayState(request_states(records("long", output_len=3)), max_batch=4)
            short.prepare_epoch()
            long.prepare_epoch()
            self.assertEqual(
                _project_action(short, estimator, "plan_fast", 2),
                _project_action(long, estimator, "plan_fast", 2),
            )

    def test_latency_guardrail_precedes_plan_hysteresis(self) -> None:
        candidates = [
            {
                "plan_id": "p1",
                "horizon": 32,
                "feature_row": {"score": 100.0},
                "projected_max_age_s": 10.0,
            },
            {
                "plan_id": "p2",
                "horizon": 32,
                "feature_row": {"score": 101.0},
                "projected_max_age_s": 2.0,
            },
        ]
        chosen, diagnostics = _select_mlp(
            candidates,
            lambda rows: [float(row["score"]) for row in rows],
            previous_plan="p1",
            latency_slo_s=5.0,
            plan_switch_margin=0.02,
        )
        self.assertEqual(chosen["plan_id"], "p2")
        self.assertEqual(diagnostics["guardrail_filtered_actions"], 1)
        self.assertEqual(diagnostics["hysteresis_blocked_switches"], 0)

    def test_plan_hysteresis_requires_relative_gain(self) -> None:
        candidates = [
            {
                "plan_id": "p1",
                "horizon": 32,
                "feature_row": {"score": 100.0},
                "projected_max_age_s": 1.0,
            },
            {
                "plan_id": "p2",
                "horizon": 32,
                "feature_row": {"score": 101.0},
                "projected_max_age_s": 1.0,
            },
        ]
        chosen, diagnostics = _select_mlp(
            candidates,
            lambda rows: [float(row["score"]) for row in rows],
            previous_plan="p1",
            latency_slo_s=0.0,
            plan_switch_margin=0.02,
        )
        self.assertEqual(chosen["plan_id"], "p1")
        self.assertEqual(diagnostics["hysteresis_blocked_switches"], 1)

    def test_all_policies_complete_the_same_test_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, signatures = write_profiles(root)
            workload = root / "workload.jsonl"
            workload.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for trace_index in range(3)
                    for row in records(f"trace-{trace_index}")
                ),
                encoding="utf-8",
            )
            output = root / "evaluation"
            result = evaluate_serving(
                workload,
                profiles,
                signatures,
                root / "unused.pt",
                output,
                hardware_key="hw",
                horizons=[1, 2],
                fixed_horizon=1,
                max_batch=4,
                latency_slo_s=10.0,
                plan_switch_margin=0.02,
                score_rows=lambda rows: [
                    float(row["plan_mean_output_token_throughput"]) for row in rows
                ],
            )
            self.assertEqual(result["evaluation_scope"]["paired_traces"], 1)
            self.assertEqual(
                [row["policy"] for row in result["policies"]], list(POLICIES)
            )
            self.assertTrue(
                all(row["requests"] == 3 for row in result["policies"])
            )
            self.assertTrue(
                all(row["p95_request_latency_s"] > 0 for row in result["policies"])
            )
            self.assertTrue((output / "decisions.csv").is_file())
            self.assertTrue((output / "summary_by_family.csv").is_file())
            self.assertTrue((output / "report.md").is_file())


if __name__ == "__main__":
    unittest.main()
