"""Exact per-execution caching on the real ragged cost model."""
from pathlib import Path
import unittest
from unittest.mock import patch

from dops_serve.burstgpt_ppo import make_evaluator
from dops_serve.placement_evaluator import RequestShape


class CostCacheContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = make_evaluator(Path(__file__).resolve().parents[2], "qwen_1.8b")
        cls.node = next(n for n in cls.evaluator.compiler.dag.nodes.values() if n.name.upper() == "QK")

    def setUp(self):
        self.cost = self.evaluator.initial.clone().scheduler.cost
        self.shapes = [RequestShape("a", 1, 64), RequestShape("b", 1, 127)]
        self.cost.configure_requests("decode", self.shapes)

    def test_repeated_activation_query_reuses_exact_value_and_next_token_invalidates(self):
        from cost_model import CostModel
        original = CostModel.estimate_activation_bytes
        calls = []

        def count(instance, *args):
            calls.append(1)
            return original(instance, *args)

        with patch.object(CostModel, "estimate_activation_bytes", count):
            first = self.cost.estimate_activation_bytes(self.node, 2, 128, "decode")
            count_first = len(calls)
            self.assertEqual(first, self.cost.estimate_activation_bytes(self.node, 2, 128, "decode"))
            self.assertEqual(len(calls), count_first)
            self.cost.configure_requests("decode", [RequestShape("a", 1, 65), RequestShape("b", 1, 128)])
            second = self.cost.estimate_activation_bytes(self.node, 2, 129, "decode")
            self.assertNotEqual(first, second)
            self.assertGreater(len(calls), count_first)

    def test_flops_reuse_and_batch_change(self):
        from cost_model import CostModel
        original = CostModel.estimate_flops
        calls = []

        def count(instance, *args):
            calls.append(1)
            return original(instance, *args)

        with patch.object(CostModel, "estimate_flops", count):
            first = self.cost.estimate_flops(self.node, 2, 128, "decode")
            count_first = len(calls)
            self.assertEqual(first, self.cost.estimate_flops(self.node, 2, 128, "decode"))
            self.assertEqual(len(calls), count_first)
            self.cost.configure_requests("decode", self.shapes[:1])
            self.assertNotEqual(first, self.cost.estimate_flops(self.node, 1, 65, "decode"))

    def test_primitive_cost_bypasses_ragged_cache(self):
        from cost_model import CostModel
        self.cost.estimate_activation_bytes(self.node, 2, 128, "decode")
        expected = CostModel.estimate_activation_bytes(self.cost, self.node, 2, 128, "decode")
        with self.cost.regular_primitive():
            actual = self.cost.estimate_activation_bytes(self.node, 2, 128, "decode")
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
