import json
import tempfile
import unittest
from pathlib import Path

from dops_serve.workloads import generate_synthetic_workloads, import_trace_csv


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


if __name__ == "__main__":
    unittest.main()
