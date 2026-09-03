import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.anchor_dataset import build_anchor_dataset
from dops_serve.dataset import DATASET_FIELDS


class AnchorDatasetTest(unittest.TestCase):
    def test_maps_paired_anchors_without_output_length_feature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparisons = root / "anchor_comparisons.csv"
            with comparisons.open("w", encoding="utf-8", newline="") as handle:
                fields = [
                    "comparison_id",
                    "model_id",
                    "hardware_key",
                    "batch",
                    "prefill_len",
                    "decode_len",
                    "summary_path",
                    "prefill_time_s",
                    "decode_time_s",
                    "total_time_s",
                    "output_token_throughput",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                source_rows = []
                for source_decode_len in (128, 256, 512, 1024):
                    for plan_id, step_time in (("p1", 1 / 128), ("p2", 3 / 128)):
                        decode_time = source_decode_len * step_time
                        total = 1.0 + decode_time
                        summary_path = root / f"{plan_id}_{source_decode_len}.json"
                        summary_path.write_text(
                            json.dumps(
                                {
                                    "decode_steps": [
                                        {
                                            "t": t,
                                            "step_time": step_time,
                                            "schedule": [],
                                        }
                                        for t in range(source_decode_len)
                                    ]
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        source_rows.append(
                            f"{summary_path},{plan_id},llama_7b,hw\n"
                        )
                        writer.writerow(
                            {
                                "comparison_id": f"shape-{source_decode_len}",
                                "model_id": "llama_7b",
                                "hardware_key": "hw",
                                "batch": 4,
                                "prefill_len": 512,
                                "decode_len": source_decode_len,
                                "summary_path": str(summary_path),
                                "prefill_time_s": 1.0,
                                "decode_time_s": decode_time,
                                "total_time_s": total,
                                "output_token_throughput": (
                                    4 * source_decode_len / total
                                ),
                            }
                        )

            (root / "plan_sources.csv").write_text(
                "summary_path,plan_id,model_id,hardware_key\n"
                + "".join(source_rows),
                encoding="utf-8",
            )
            profiles = root / "profiles.csv"
            profiles.write_text(
                "plan_id,model_id,hardware_key,batch,prefill_len,decode_len,"
                "context_mid,prefill_time_s,decode_s_per_iteration\n"
                "p1,llama_7b,hw,4,512,128,576,1.0,0.01\n"
                "p2,llama_7b,hw,4,512,128,576,2.0,0.02\n",
                encoding="utf-8",
            )
            signatures = root / "plan_signatures.csv"
            signatures.write_text(
                "plan_id,model_id,hardware_key,placement_complete,"
                "incomplete_source_count,profile_count,"
                "mean_prefill_s_per_input_token,mean_decode_s_per_iteration,"
                "mean_output_token_throughput,mean_total_time_s\n"
                "p1,llama_7b,hw,True,0,1,0.001,0.01,256,2\n"
                "p2,llama_7b,hw,True,0,1,0.002,0.02,128,4\n",
                encoding="utf-8",
            )

            output = root / "anchor_dataset.csv"
            result = build_anchor_dataset(
                comparisons,
                root / "plan_sources.csv",
                profiles,
                signatures,
                output,
            )
            self.assertEqual(result["comparisons"], 4)
            self.assertEqual(result["states"], 60)
            self.assertEqual(result["rows"], 360)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), DATASET_FIELDS)
            self.assertNotIn("decode_len", rows[0])
            self.assertEqual(len({row["state_id"] for row in rows}), 60)
            self.assertEqual(
                {row["trace_id"] for row in rows},
                {"shape-128", "shape-256", "shape-512", "shape-1024"},
            )
            self.assertEqual({row["plan_id"] for row in rows}, {"p1", "p2"})
            self.assertEqual({row["horizon"] for row in rows}, {"32", "64", "128"})
            self.assertEqual({row["max_tokens_mean"] for row in rows}, {"0.0"})
            prefix = next(
                row
                for row in rows
                if row["state_id"] == "anchor:shape-128:t0000"
                and row["plan_id"] == "p1"
                and row["horizon"] == "128"
            )
            self.assertEqual(float(prefix["plan_shape_distance"]), 0.0)
            self.assertEqual(float(prefix["label_token_throughput"]), 256.0)
            window = next(
                row
                for row in rows
                if row["state_id"] == "anchor:shape-128:t0032"
                and row["plan_id"] == "p1"
                and row["horizon"] == "32"
            )
            self.assertEqual(float(window["active_count"]), 4.0)
            self.assertEqual(float(window["generated_tokens_mean"]), 32.0)
            self.assertEqual(float(window["active_context_mean"]), 544.0)


if __name__ == "__main__":
    unittest.main()
