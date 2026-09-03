import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.summary import extract_plan, iter_summary_array, read_summary_metadata


def write_summary(
    path: Path, batch: int = 1, prefill_len: int = 128, decode_len: int = 2
) -> None:
    payload = {
        "policy": "algo:hefthint",
        "config": {
            "batch": batch,
            "prefill_len": prefill_len,
            "decode_len": decode_len,
            "dtype": "fp16",
        },
        "best_times": {"prefill": 1.0, "decode": 2.0, "total": 3.0},
        "plan_label": {
            "trace_ops_csv": "output/ops.csv",
            "trace_comms_csv": "output/comms.csv",
        },
        "prefill_schedule": [
            {"node_id": "A", "device": "NPU0", "phase": "prefill"},
            {"node_id": "B", "device": "PIM0", "phase": "prefill"},
        ],
        "decode_steps": [
            {
                "t": 0,
                "seq_len": 128,
                "schedule": [
                    {"node_id": "A", "device": "NPU0"},
                    {"node_id": "B", "device": "PIM0"},
                ],
            },
            {
                "t": 1,
                "seq_len": 129,
                "schedule": [
                    {"node_id": "A", "device": "NPU0"},
                    {"node_id": "B", "device": "PIM0"},
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SummaryTest(unittest.TestCase):
    def test_metadata_and_streamed_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_summary_128x2.json"
            write_summary(path)
            metadata = read_summary_metadata(path)
            self.assertEqual(metadata["policy"], "algo:hefthint")
            self.assertNotIn("prefill_schedule", metadata)
            self.assertEqual(len(list(iter_summary_array(path, "prefill_schedule"))), 2)

    def test_extract_plan_rle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_summary_128x2.json"
            write_summary(path)
            plan = extract_plan(path)
            self.assertEqual(plan["decode_steps_covered"], 2)
            self.assertEqual(len(plan["decode_segments"]), 1)
            self.assertEqual(plan["decode_segments"][0]["count"], 2)
            self.assertTrue(plan["placement_complete"])

    def test_estimated_step_preserves_missing_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_summary_128x2.json"
            write_summary(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decode_steps"][1]["estimated"] = True
            payload["decode_steps"][1]["schedule"] = None
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            plan = extract_plan(path)
            self.assertFalse(plan["placement_complete"])
            self.assertEqual(plan["decode_steps_observed"], 1)
            self.assertEqual(plan["decode_steps_estimated"], 1)
            self.assertIsNone(plan["decode_segments"][1]["placement"])


if __name__ == "__main__":
    unittest.main()
