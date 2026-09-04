import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.dataset import DATASET_FIELDS, _trace_splits, build_dataset
from dops_serve.replay import ProfileEstimator


class DatasetTest(unittest.TestCase):
    def test_declared_trace_split_overrides_lexical_order(self) -> None:
        traces = {
            "early": [{"family": "real", "split": "test"}],
            "late": [{"family": "real", "split": "train"}],
        }
        self.assertEqual(_trace_splits(traces), {"early": "test", "late": "train"})

    def test_robust_plan_prioritizes_profile_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles.csv"
            profiles.write_text(
                "plan_id,model_id,hardware_key,batch,prefill_len,decode_len,"
                "context_mid,prefill_time_s,decode_s_per_iteration\n"
                "covered,llama_7b,hw,1,128,128,192,1.0,0.1\n"
                "sparse,llama_7b,hw,1,128,128,192,0.5,0.05\n",
                encoding="utf-8",
            )
            signatures = root / "plan_signatures.csv"
            signatures.write_text(
                "plan_id,model_id,hardware_key,placement_complete,"
                "incomplete_source_count,profile_count,"
                "mean_prefill_s_per_input_token,mean_decode_s_per_iteration,"
                "mean_output_token_throughput,mean_total_time_s\n"
                "covered,llama_7b,hw,True,0,64,0.01,0.1,10.0,2.0\n"
                "sparse,llama_7b,hw,True,0,2,0.005,0.05,20.0,1.0\n",
                encoding="utf-8",
            )

            estimator = ProfileEstimator(profiles, signatures)
            self.assertEqual(estimator.robust_plan("llama_7b", "hw"), "covered")

    def test_incomplete_plan_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles.csv"
            profiles.write_text(
                "plan_id,model_id,hardware_key,batch,prefill_len,decode_len,"
                "context_mid,prefill_time_s,decode_s_per_iteration\n"
                "p1,llama_7b,hw,1,128,128,192,1.0,0.1\n",
                encoding="utf-8",
            )
            signatures = root / "plan_signatures.csv"
            signatures.write_text(
                "plan_id,model_id,hardware_key,placement_complete,"
                "incomplete_source_count,profile_count,"
                "mean_prefill_s_per_input_token,mean_decode_s_per_iteration,"
                "mean_output_token_throughput,mean_total_time_s\n"
                "p1,llama_7b,hw,False,1,1,0.01,0.1,10.0,2.0\n",
                encoding="utf-8",
            )

            self.assertEqual(ProfileEstimator(profiles, signatures).plan_ids("llama_7b", "hw"), [])
            opted_in = ProfileEstimator(
                profiles, signatures, allow_incomplete_plans=True
            )
            self.assertEqual(opted_in.plan_ids("llama_7b", "hw"), ["p1"])

    def test_counterfactual_rows_exclude_future_output_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles.csv"
            profile_fields = [
                "plan_id",
                "model_id",
                "hardware_key",
                "batch",
                "prefill_len",
                "decode_len",
                "context_mid",
                "prefill_time_s",
                "decode_s_per_iteration",
            ]
            with profiles.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=profile_fields)
                writer.writeheader()
                for plan_id, scale in (("plan_fast", 1.0), ("plan_slow", 2.0)):
                    for batch, prefill in ((1, 128), (4, 512)):
                        writer.writerow(
                            {
                                "plan_id": plan_id,
                                "model_id": "llama_7b",
                                "hardware_key": "exp1/hw/sst2",
                                "batch": batch,
                                "prefill_len": prefill,
                                "decode_len": 128,
                                "context_mid": prefill + 64,
                                "prefill_time_s": 0.1 * scale,
                                "decode_s_per_iteration": 0.01 * scale,
                            }
                        )

            signatures = root / "plan_signatures.csv"
            signature_fields = [
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
            ]
            with signatures.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=signature_fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "plan_id": "plan_fast",
                        "model_id": "llama_7b",
                        "hardware_key": "exp1/hw/sst2",
                        "placement_complete": True,
                        "incomplete_source_count": 0,
                        "profile_count": 2,
                        "mean_prefill_s_per_input_token": 0.001,
                        "mean_decode_s_per_iteration": 0.01,
                        "mean_output_token_throughput": 100,
                        "mean_total_time_s": 1,
                    }
                )
                writer.writerow(
                    {
                        "plan_id": "plan_slow",
                        "model_id": "llama_7b",
                        "hardware_key": "exp1/hw/sst2",
                        "placement_complete": True,
                        "incomplete_source_count": 0,
                        "profile_count": 2,
                        "mean_prefill_s_per_input_token": 0.002,
                        "mean_decode_s_per_iteration": 0.02,
                        "mean_output_token_throughput": 50,
                        "mean_total_time_s": 2,
                    }
                )

            workload = root / "workload.jsonl"
            records = [
                {
                    "trace_id": "poisson_llama_7b_000",
                    "family": "poisson",
                    "request_id": f"r{index}",
                    "model_id": "llama_7b",
                    "arrival_time": index * 0.1,
                    "prompt_len": 128,
                    "max_tokens": 128,
                    "output_len": 2,
                }
                for index in range(3)
            ]
            workload.write_text(
                "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
            )
            dataset = root / "dataset.csv"
            result = build_dataset(
                workload,
                profiles,
                signatures,
                dataset,
                hardware_key="exp1/hw/sst2",
                horizons=[2, 4],
                behavior_horizon=2,
                max_batch=4,
                allow_extrapolation=True,
            )
            self.assertGreater(result["rows"], 0)
            self.assertNotIn("output_len", DATASET_FIELDS)
            with dataset.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            first_state = [row for row in rows if row["state_id"] == rows[0]["state_id"]]
            self.assertEqual(len(first_state), 4)
            self.assertGreater(
                float(
                    next(row for row in first_state if row["plan_id"] == "plan_fast")[
                        "label_token_throughput"
                    ]
                ),
                float(
                    next(row for row in first_state if row["plan_id"] == "plan_slow")[
                        "label_token_throughput"
                    ]
                ),
            )


if __name__ == "__main__":
    unittest.main()
