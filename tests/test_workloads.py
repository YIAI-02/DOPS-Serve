import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.workloads import (
    generate_synthetic_workloads,
    import_burstgpt_windows,
    import_trace_csv,
)


class WorkloadsTest(unittest.TestCase):
    def test_synthetic_output_length_is_not_the_online_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "requests.jsonl"
            result = generate_synthetic_workloads(
                output,
                model_ids=["llama_7b"],
                families=["poisson"],
                traces_per_family=1,
                requests_per_trace=40,
                seed=3,
                prompt_min=256,
                prompt_max=512,
                max_token_choices=[64, 128],
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(result["requests"], 40)
            self.assertTrue(all(row["output_len"] <= row["max_tokens"] for row in rows))
            self.assertTrue(any(row["output_len"] != row["max_tokens"] for row in rows))
            self.assertTrue(all(256 <= row["prompt_len"] <= 512 for row in rows))
            self.assertEqual({row["max_tokens"] for row in rows}, {64, 128})

    def test_import_rejects_output_as_max_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "trace.csv"
            source.write_text("arrival,prompt,output\n0,128,32\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot be reused"):
                import_trace_csv(
                    source,
                    Path(directory) / "requests.jsonl",
                    model_id="llama_7b",
                    trace_id="trace",
                    arrival_column="arrival",
                    prompt_column="prompt",
                    output_column="output",
                    max_tokens_column="output",
                )

    def test_burstgpt_windows_are_grouped_and_do_not_leak_output_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "BurstGPT.csv"
            rows = [
                "Timestamp,Model,Request tokens,Response tokens,Log Type",
            ]
            timestamps = [
                block * 100 + offset
                for block in range(8)
                for offset in (0, 1, 1.1, 10)
            ]
            rows.extend(
                f"{timestamp},ChatGPT,{128 + index},{1 + index % 8},API log"
                for index, timestamp in enumerate(timestamps)
            )
            rows.append("400,GPT-4,256,8,API log")
            source.write_text("\n".join(rows) + "\n", encoding="utf-8")
            output = Path(directory) / "requests.jsonl"

            result = import_burstgpt_windows(
                source,
                output,
                model_id="llama_7b",
                traces=8,
                requests_per_trace=2,
                target_median_rps=2.0,
                max_tokens_default=32,
                prompt_min=128,
                prompt_max=256,
                output_max=16,
                source_model="ChatGPT",
                log_type="API log",
                selection="mixed",
            )

            records = [json.loads(line) for line in output.read_text().splitlines()]
            trace_ids = sorted({row["trace_id"] for row in records})
            self.assertEqual(result["traces"], 8)
            self.assertEqual(result["requests"], 16)
            self.assertEqual(result["eligible_requests"], 32)
            self.assertEqual(result["representative_traces"], 4)
            self.assertEqual(result["peak_traces"], 4)
            self.assertEqual(result["train_traces"], 4)
            self.assertEqual(result["validation_traces"], 2)
            self.assertEqual(result["test_traces"], 2)
            self.assertEqual(len(trace_ids), 8)
            self.assertEqual(
                {row["split"] for row in records}, {"train", "val", "test"}
            )
            self.assertEqual(
                {row["family"] for row in records},
                {"burstgpt_peak", "burstgpt_representative"},
            )
            self.assertTrue(all(row["max_tokens"] == 32 for row in records))
            self.assertTrue(
                all(row["output_len"] < row["max_tokens"] for row in records)
            )
            self.assertTrue(
                all(
                    min(
                        row["arrival_time"]
                        for row in records
                        if row["trace_id"] == trace_id
                    )
                    == 0.0
                    for trace_id in trace_ids
                )
            )


if __name__ == "__main__":
    unittest.main()
