import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.inventory import build_inventory
from dops_serve.plan_bank import build_plan_bank
from test_summary import write_summary


class PlanBankTest(unittest.TestCase):
    def test_exact_plan_dedup_and_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            summary_dir = (
                output
                / "exp1"
                / "hw_hardware_1npu_2aim"
                / "sst2_rst2"
                / "llama_7b_fp16_b1_s2"
                / "algo_hefthint"
            )
            summary_dir.mkdir(parents=True)
            write_summary(summary_dir / "best_summary_128x2.json")
            (output / "ops.csv").write_text("phase,start,end\n", encoding="utf-8")
            (output / "comms.csv").write_text("phase,bytes\n", encoding="utf-8")
            artifacts = root / "artifacts"
            build_inventory(output, artifacts)

            result = build_plan_bank(
                artifacts / "inventory.csv",
                artifacts,
                policies=["hefthint"],
                max_decode_steps=2,
            )
            self.assertEqual(result["plans"], 1)
            manifest = json.loads(
                (artifacts / "plan_bank.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(manifest["decode_steps_covered"], 2)
            with (artifacts / "plan_sources.csv").open(encoding="utf-8") as handle:
                sources = list(csv.DictReader(handle))
            self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()
