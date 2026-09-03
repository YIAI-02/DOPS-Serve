import csv
import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.inventory import build_inventory
from dops_serve.plan_bank import build_plan_bank
from test_summary import write_summary


class PlanBankTest(unittest.TestCase):
    def test_parallel_groups_match_serial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            for model_id in ("llama_7b", "qwen_7b"):
                summary_dir = (
                    output
                    / "exp1"
                    / "hw_hardware_1npu_2aim"
                    / "sst2_rst2"
                    / f"{model_id}_fp16_b1_s2"
                    / "algo_hefthint"
                )
                summary_dir.mkdir(parents=True)
                write_summary(summary_dir / "best_summary_128x2.json")
            (output / "ops.csv").write_text("phase,start,end\n", encoding="utf-8")
            (output / "comms.csv").write_text("phase,bytes\n", encoding="utf-8")

            serial = root / "serial"
            parallel = root / "parallel"
            build_inventory(output, serial)
            serial_result = build_plan_bank(
                serial / "inventory.csv",
                serial,
                policies=["hefthint"],
                max_decode_steps=2,
            )
            parallel_result = build_plan_bank(
                serial / "inventory.csv",
                parallel,
                policies=["hefthint"],
                max_decode_steps=2,
                workers=2,
            )

            self.assertEqual(serial_result, parallel_result)
            serial_manifest = [
                json.loads(line)
                for line in (serial / "plan_bank.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            parallel_manifest = [
                json.loads(line)
                for line in (parallel / "plan_bank.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for row in serial_manifest + parallel_manifest:
                row["plan_path"] = Path(row["plan_path"]).name
            self.assertEqual(serial_manifest, parallel_manifest)
            self.assertEqual(
                (serial / "plan_sources.csv").read_text(encoding="utf-8"),
                (parallel / "plan_sources.csv").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                sorted(path.name for path in (serial / "plans").iterdir()),
                sorted(path.name for path in (parallel / "plans").iterdir()),
            )

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
            self.assertEqual(
                manifest["placement_equivalence"],
                "prefill mapping and decode placement transitions",
            )
            with (artifacts / "plan_sources.csv").open(encoding="utf-8") as handle:
                sources = list(csv.DictReader(handle))
            self.assertEqual(len(sources), 1)

    def test_equivalent_plan_keeps_longest_decode_coverage(self) -> None:
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
            write_summary(summary_dir / "best_summary_128x2.json", decode_len=2)
            write_summary(summary_dir / "best_summary_128x4.json", decode_len=4)
            (output / "ops.csv").write_text("phase,start,end\n", encoding="utf-8")
            (output / "comms.csv").write_text("phase,bytes\n", encoding="utf-8")
            artifacts = root / "artifacts"
            build_inventory(output, artifacts)

            result = build_plan_bank(
                artifacts / "inventory.csv",
                artifacts,
                policies=["hefthint"],
                max_decode_steps=4,
            )
            self.assertEqual(result["plans"], 1)
            manifest = json.loads(
                (artifacts / "plan_bank.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(manifest["decode_steps_covered"], 4)
            self.assertEqual(len(manifest["source_shapes"]), 2)


if __name__ == "__main__":
    unittest.main()
