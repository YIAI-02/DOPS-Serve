import csv
import tempfile
import unittest
from pathlib import Path

from dops_serve.inventory import build_inventory
from test_summary import write_summary


class InventoryTest(unittest.TestCase):
    def test_build_inventory(self) -> None:
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

            artifact_dir = root / "artifacts"
            valid, excluded = build_inventory(output, artifact_dir)
            self.assertEqual((valid, excluded), (1, 0))
            with (artifact_dir / "inventory.csv").open(encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["model_id"], "llama_7b")
            self.assertEqual(row["batch"], "1")
            self.assertEqual(row["policy"], "algo:hefthint")


if __name__ == "__main__":
    unittest.main()
