import unittest
import tempfile
from collections import Counter
from pathlib import Path

import torch

from dops_serve import static_bc as bc
from dops_serve.static_bc_ab import GROUPS, add_phase_weights, initialize, restore, teacher_metrics, update_batch


class KnownLossActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, state, nodes, actions, masks, learned):
        return state[:, 0] * self.scale, torch.zeros_like(masks, dtype=torch.float32)


def known_task(case_id, model, scale=1):
    values = [(-1, 3), (0, 6), (1, 6), (2, 6), (4, 9), (6, 9)]
    return [dict(case_id=case_id, model=model, phase='prefill' if t == -1 else 'decode', token_index=t,
        task_weight=1/6, state=torch.tensor([float(loss*scale)]), nodes=torch.zeros(2, 1),
        actions=torch.zeros(2, dtype=torch.long), masks=torch.ones(2, 3, dtype=torch.bool), learned=torch.ones(2))
        for t, loss in values]


class FreshPhaseBCContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_phase_balance_preserves_equal_tasks_at_every_decode_length(self):
        rows = []
        for case_id, length in enumerate((128, 256, 512, 1024)):
            indices = [-1, *[t for t in range(length) if bc.refresh(t)]]
            rows.extend(dict(case_id=case_id, phase='prefill' if t == -1 else 'decode',
                token_index=t, task_weight=1/len(indices)) for t in indices)
        add_phase_weights(rows)
        totals = Counter()
        for row in rows:
            totals[row['case_id'], row['group']] += row['phase_weight']
        for case_id in range(4):
            for group in GROUPS:
                self.assertAlmostEqual(totals[case_id, group], 1/3)
            startup = [row['token_index'] for row in rows if row['case_id'] == case_id and row['group'] == 'startup_decode']
            self.assertEqual(startup, [0, 1, 2])

    def test_shared_metrics_keep_models_separate_and_report_actual_B_loss(self):
        rows = add_phase_weights(known_task(0, 'qwen_1.8b') + known_task(64, 'qwen_7b', 2) + known_task(65, 'qwen_7b', 2))
        metrics = teacher_metrics(KnownLossActor(), rows, 'cpu', 4)
        for model, scale in (('qwen_1.8b', 1), ('qwen_7b', 2)):
            self.assertAlmostEqual(metrics[model]['loss'], 6*scale)
            self.assertAlmostEqual(metrics[model]['common_nll'], 6.5*scale)
            for group, expected in zip(GROUPS, (3, 6, 9)):
                self.assertAlmostEqual(metrics[model][group+'_nll'], expected*scale)

    def test_update_uses_group_means_instead_of_plan_mean(self):
        rows = add_phase_weights(known_task(0, 'qwen_7b'))
        actor = KnownLossActor()
        optimizer = torch.optim.SGD(actor.parameters(), lr=.1)
        loss = update_batch(actor, optimizer, rows, 'cpu', 6, 1, 100.)
        self.assertAlmostEqual(loss, 6, places=5)
        self.assertAlmostEqual(actor.scale.item(), .4, places=5)

    def test_new_route_discards_all_previous_network_optimizer_and_rng_state(self):
        protocol = dict(seed=7, state_dim=1, node_dim=1, hidden=8, depth=1, learning_rate=1e-4)
        first, optimizer, rng = initialize(protocol, 'cpu')
        initial_weights = {key: value.clone() for key, value in first.state_dict().items()}
        rows = add_phase_weights(known_task(0, 'qwen_7b'))
        update_batch(first, optimizer, rows, 'cpu', 6, 1, 1.)
        rng.random()
        self.assertTrue(optimizer.state)
        fresh, fresh_optimizer, fresh_rng = initialize(protocol, 'cpu')
        self.assertFalse(fresh_optimizer.state)
        torch.testing.assert_close(fresh.state_dict(), initial_weights, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(first.state_dict()[key], value) for key, value in initial_weights.items()))
        self.assertEqual(fresh_rng.random(), __import__('random').Random(7).random())

    def test_resume_reproduces_next_batch_update_optimizer_and_rng(self):
        protocol = dict(seed=7, state_dim=1, node_dim=1, hidden=8, depth=1, learning_rate=1e-4)
        actor, optimizer, rng = initialize(protocol, 'cpu')
        rows = add_phase_weights(known_task(0, 'qwen_1.8b') + known_task(64, 'qwen_7b', 2) + known_task(128, 'qwen_14b', 3))
        for batch in bc.batches(rows, 4, rng):
            update_batch(actor, optimizer, batch, 'cpu', len(rows), 3, 1.)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'checkpoint.pt'
            torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(), dimensions=actor.dimensions,
                epoch=50, random_state=rng.getstate(), torch_rng=torch.get_rng_state(), cuda_rng=[]), checkpoint)
            expected_random = torch.rand(5)
            resumed, resumed_optimizer, resumed_rng, epoch = restore(checkpoint, 'cpu')
            self.assertEqual(epoch, 50)
            torch.testing.assert_close(torch.rand(5), expected_random, rtol=0, atol=0)
            torch.testing.assert_close(resumed_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
            expected_batches = bc.batches(rows, 4, rng)
            actual_batches = bc.batches(rows, 4, resumed_rng)
            self.assertEqual([[(row['case_id'], row['token_index']) for row in batch] for batch in actual_batches],
                             [[(row['case_id'], row['token_index']) for row in batch] for batch in expected_batches])
            update_batch(actor, optimizer, expected_batches[0], 'cpu', len(rows), 3, 1.)
            update_batch(resumed, resumed_optimizer, actual_batches[0], 'cpu', len(rows), 3, 1.)
            torch.testing.assert_close(resumed.state_dict(), actor.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(resumed_optimizer.state_dict(), optimizer.state_dict(), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
