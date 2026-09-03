import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.comparisons import build_anchor_comparisons
from dops_serve.inventory import build_inventory
from test_summary import write_summary


class AnchorComparisonsTest(unittest.TestCase):
    def test_builds_paired_anchor_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            model_dir = (
                output
                / "exp1"
                / "hw_hardware_1npu_2aim"
                / "sst2_rst2"
                / "llama_7b_fp16_b1_s2"
            )
            (output / "ops.csv").parent.mkdir(parents=True)
            (output / "ops.csv").write_text("phase,start,end\n", encoding="utf-8")
            (output / "comms.csv").write_text("phase,bytes\n", encoding="utf-8")

            policies = (("hefthint", 3.0), ("pd", 4.0))
            for policy, total_time in policies:
                summary_path = model_dir / f"algo_{policy}" / "best_summary_128x2.json"
                summary_path.parent.mkdir(parents=True)
                write_summary(summary_path)
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                payload["policy"] = f"algo:{policy}"
                payload["best_times"] = {
                    "prefill": 1.0,
                    "decode": total_time - 1.0,
                    "total": total_time,
                }
                summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            comparison = {
                "config": {
                    "result_dir": "output/exp1/hw_hardware_1npu_2aim/sst2_rst2/llama_7b_fp16_b1_s2",
                    "batch": 1,
                    "prefill_len": 128,
                    "decode_len": 2,
                },
                "results": [
                    {
                        "policy": f"algo:{policy}",
                        "prefill_time_s": 1.0,
                        "decode_time_s": total_time - 1.0,
                        "total_time_s": total_time,
                    }
                    for policy, total_time in policies
                ],
            }
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "baseline_compare_128x2.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )

            artifacts = root / "artifacts"
            build_inventory(output, artifacts)
            result = build_anchor_comparisons(
                output,
                artifacts / "inventory.csv",
                artifacts / "anchor_comparisons.csv",
            )
            self.assertEqual(result["paired_groups"], 1)
            self.assertEqual(result["rows"], 2)
            with (artifacts / "anchor_comparisons.csv").open(
                encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["policy"], "algo:hefthint")
            self.assertEqual(rows[0]["rank"], "1")
            self.assertEqual(rows[0]["is_best"], "True")
            self.assertEqual(rows[1]["policy"], "algo:pd")

    def test_excludes_single_candidate_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            (output / "baseline_compare_1x1.json").write_text(
                json.dumps({"config": {}, "results": [{"policy": "algo:pd"}]}),
                encoding="utf-8",
            )
            inventory = root / "inventory.csv"
            inventory.write_text("summary_path\n", encoding="utf-8")
            result = build_anchor_comparisons(
                output, inventory, root / "anchor_comparisons.csv"
            )
            self.assertEqual(result["paired_groups"], 0)
            self.assertEqual(result["excluded_groups"], 1)


if __name__ == "__main__":
    unittest.main()
