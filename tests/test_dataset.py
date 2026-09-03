import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.dataset import DATASET_FIELDS, build_dataset


class DatasetTest(unittest.TestCase):
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
