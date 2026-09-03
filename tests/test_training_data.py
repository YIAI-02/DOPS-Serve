import unittest

from dops_serve.dataset import PLAN_FEATURES, STATE_FEATURES
from dops_serve.training import _encode_rows, _fit_preprocessor, _ranking_pairs


def training_row(state_id: str, plan_id: str, throughput: float) -> dict[str, str]:
    row = {name: "1.0" for name in [*STATE_FEATURES, *PLAN_FEATURES]}
    row.update(
        {
            "state_id": state_id,
            "model_id": "llama_7b",
            "hardware_key": "exp1/hw/sst2",
            "plan_id": plan_id,
            "horizon": "64",
            "label_token_throughput": str(throughput),
        }
    )
    return row


class TrainingDataTest(unittest.TestCase):
    def test_preprocessor_and_ranking_pairs(self) -> None:
        rows = [
            training_row("s0", "fast", 10.0),
            training_row("s0", "slow", 5.0),
            training_row("s1", "fast", 12.0),
            training_row("s1", "slow", 6.0),
        ]
        preprocessor = _fit_preprocessor(rows)
        features, targets = _encode_rows(rows, preprocessor)
        self.assertEqual(len(features), 4)
        self.assertEqual(len(features[0]), len(features[1]))
        self.assertAlmostEqual(sum(targets), 0.0)
        self.assertEqual(_ranking_pairs(rows), [(0, 1), (2, 3)])


if __name__ == "__main__":
    unittest.main()
