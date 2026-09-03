import csv
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
                    "total_time_s",
                    "output_token_throughput",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for plan_id, total in (("p1", 2.0), ("p2", 4.0)):
                    writer.writerow(
                        {
                            "comparison_id": "shape-1",
                            "model_id": "llama_7b",
                            "hardware_key": "hw",
                            "batch": 4,
                            "prefill_len": 512,
                            "decode_len": 128,
                            "summary_path": f"/{plan_id}.json",
                            "total_time_s": total,
                            "output_token_throughput": 4 * 128 / total,
                        }
                    )

            (root / "plan_sources.csv").write_text(
                "summary_path,plan_id,model_id,hardware_key\n"
                "/p1.json,p1,llama_7b,hw\n"
                "/p2.json,p2,llama_7b,hw\n",
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
            self.assertEqual(result["states"], 1)
            self.assertEqual(result["rows"], 2)
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), DATASET_FIELDS)
            self.assertEqual({row["state_id"] for row in rows}, {"anchor:shape-1"})
            self.assertEqual({row["plan_id"] for row in rows}, {"p1", "p2"})
            self.assertEqual({row["horizon"] for row in rows}, {"128"})
            self.assertEqual({row["max_tokens_mean"] for row in rows}, {"0.0"})
            self.assertEqual(float(rows[0]["plan_shape_distance"]), 0.0)


if __name__ == "__main__":
    unittest.main()
