import csv
import tempfile
import unittest
from pathlib import Path

from dops_serve.inventory import build_inventory
from dops_serve.plan_bank import build_plan_bank
from dops_serve.profiles import build_profiles
from test_summary import write_summary


class ProfilesTest(unittest.TestCase):
    def test_profiles_and_leave_one_out_validation(self) -> None:
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
            write_summary(summary_dir / "best_summary_128x2.json", prefill_len=128)
            write_summary(summary_dir / "best_summary_256x2.json", prefill_len=256)
            (output / "ops.csv").write_text("phase,start,end\n", encoding="utf-8")
            (output / "comms.csv").write_text("phase,bytes\n", encoding="utf-8")
            artifacts = root / "artifacts"
            build_inventory(output, artifacts)
            build_plan_bank(
                artifacts / "inventory.csv",
                artifacts,
                policies=["hefthint"],
                max_decode_steps=2,
            )

            result = build_profiles(
                artifacts / "inventory.csv",
                artifacts / "plan_sources.csv",
                artifacts,
            )
            self.assertEqual(result, {"profiles": 2, "plans": 1})
            with (artifacts / "profile_validation.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)


if __name__ == "__main__":
    unittest.main()
