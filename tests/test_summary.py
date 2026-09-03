import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.summary import (
    canonical_plan_text,
    extract_plan,
    iter_summary_array,
    read_decode_step_times,
    read_summary_metadata,
)


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
                "t": t,
                "seq_len": prefill_len + t,
                "step_time": 2.0 / decode_len,
                "schedule": [
                    {"node_id": "A", "device": "NPU0"},
                    {"node_id": "B", "device": "PIM0"},
                ],
            }
            for t in range(decode_len)
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

    def test_reads_decode_step_times_without_schedules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_summary_128x2.json"
            write_summary(path)
            self.assertEqual(read_decode_step_times(path), [1.0, 1.0])

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
            self.assertEqual(plan["decode_steps_recovered"], 0)
            self.assertEqual(plan["decode_steps_missing"], 1)
            self.assertIsNone(plan["decode_segments"][1]["placement"])

    def test_ops_trace_recovers_sampled_decode_placement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampled.json"
            trace = Path(directory) / "ops.csv"
            write_summary(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decode_steps"][1]["estimated"] = True
            payload["decode_steps"][1]["schedule"] = None
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            trace.write_text(
                "phase,node_id,device\n"
                "decode,A,NPU0\n"
                "decode,B,PIM0\n"
                "decode,A,NPU0\n"
                "decode,B,PIM0\n",
                encoding="utf-8",
            )

            plan = extract_plan(path, ops_trace_path=trace)
            self.assertTrue(plan["placement_complete"])
            self.assertEqual(plan["decode_steps_recovered"], 1)
            self.assertEqual(plan["decode_steps_missing"], 0)
            self.assertEqual(plan["decode_segments"][0]["count"], 2)

    def test_ops_trace_must_match_observed_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sampled.json"
            trace = Path(directory) / "ops.csv"
            write_summary(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["decode_steps"][1]["estimated"] = True
            payload["decode_steps"][1]["schedule"] = None
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            trace.write_text(
                "phase,node_id,device\n"
                "decode,A,NPU1\n"
                "decode,B,PIM0\n"
                "decode,A,NPU0\n"
                "decode,B,PIM0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "disagree"):
                extract_plan(path, ops_trace_path=trace)

    def test_sampling_pattern_does_not_change_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            complete_path = Path(directory) / "complete.json"
            sampled_path = Path(directory) / "sampled.json"
            write_summary(complete_path)
            write_summary(sampled_path)
            payload = json.loads(sampled_path.read_text(encoding="utf-8"))
            payload["decode_steps"][1]["estimated"] = True
            payload["decode_steps"][1]["schedule"] = None
            sampled_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            complete = extract_plan(complete_path)
            sampled = extract_plan(sampled_path)
            self.assertEqual(canonical_plan_text(complete), canonical_plan_text(sampled))

    def test_observed_decode_transition_changes_plan_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            write_summary(first_path)
            write_summary(second_path)
            payload = json.loads(second_path.read_text(encoding="utf-8"))
            payload["decode_steps"][1]["schedule"][1]["device"] = "NPU0"
            second_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            first = extract_plan(first_path)
            second = extract_plan(second_path)
            self.assertNotEqual(canonical_plan_text(first), canonical_plan_text(second))

    def test_record_order_does_not_change_placement_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            write_summary(first_path)
            write_summary(second_path)
            payload = json.loads(second_path.read_text(encoding="utf-8"))
            payload["prefill_schedule"].reverse()
            for step in payload["decode_steps"]:
                step["schedule"].reverse()
            second_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            first = extract_plan(first_path)
            second = extract_plan(second_path)
            self.assertEqual(canonical_plan_text(first), canonical_plan_text(second))

    def test_metadata_stops_after_plan_label_without_schedule_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best_summary_128x2.json"
            payload = {
                "policy": "algo:baseline",
                "config": {"batch": 1, "prefill_len": 128, "decode_len": 2},
                "best_times": {"prefill": 1.0, "decode": 2.0, "total": 3.0},
                "plan_label": {
                    "trace_ops_csv": "output/ops.csv",
                    "trace_comms_csv": "output/comms.csv",
                },
                "other_large_array": list(range(1000)),
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            metadata = read_summary_metadata(path)
            self.assertEqual(metadata["policy"], "algo:baseline")
            self.assertNotIn("other_large_array", metadata)


if __name__ == "__main__":
    unittest.main()
