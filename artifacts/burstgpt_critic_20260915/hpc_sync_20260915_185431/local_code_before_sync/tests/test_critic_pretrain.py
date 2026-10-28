import tempfile
import unittest
from pathlib import Path

import torch

from dops_serve.placement_policy import StaticPlacementActor, StaticPlacementCritic
from dops_serve.placement_training import complete_critic_returns, critic_plan_mse
from dops_serve.burstgpt_ppo import evaluate_critic, critic_baselines


class CriticPretrainContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_complete_mc_crosses_chunks_and_rejects_truncation(self):
        durations = torch.arange(1., 36., dtype=torch.float64)
        terminal = torch.zeros(35, dtype=torch.bool)
        terminal[-1] = True
        chunks = [durations[:16], durations[16:32], durations[32:]]
        actual = complete_critic_returns(torch.cat(chunks), terminal, 2.)
        expected = torch.tensor([-sum(range(i+1, 36))/2 for i in range(35)], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            complete_critic_returns(durations[:16], terminal[:16], 2.)

    def test_zero_head_fits_returns_without_changing_actor(self):
        actor = StaticPlacementActor(56, 35, hidden=16, depth=2)
        actor.requires_grad_(False)
        original = {key: value.clone() for key, value in actor.state_dict().items()}
        critic = StaticPlacementCritic(actor)
        batch = dict(state=torch.randn(4, 56), nodes=torch.randn(4, 8, 35),
            actions=torch.randint(0, 3, (4, 8)), learned=torch.ones(4, 8, dtype=torch.bool))
        target = torch.tensor([-.1, -.2, -.3, -.4])
        self.assertEqual(int(torch.count_nonzero(critic(batch['state'], batch['nodes'], batch['actions']))), 0)
        optimizer = torch.optim.Adam(critic.parameters(), lr=.01)
        before = float(critic_plan_mse(critic, batch, target).mean().detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            critic_plan_mse(critic, batch, target).mean().backward()
            optimizer.step()
        self.assertLess(float(critic_plan_mse(critic, batch, target).mean().detach()), before*.1)
        self.assertTrue(all(torch.equal(value, actor.state_dict()[key]) for key, value in original.items()))
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        self.assertTrue(any(not torch.equal(value, critic.state_encoder.state_dict()[key])
                            for key, value in actor.state_encoder.state_dict().items()))
        # Prefix causality remains true after a nonzero head has been learned.
        changed = batch['actions'].clone()
        changed[:, 4:] = (changed[:, 4:]+1) % 3
        torch.testing.assert_close(critic(batch['state'], batch['nodes'], batch['actions'])[:, :5],
                                   critic(batch['state'], batch['nodes'], changed)[:, :5])

    def test_validation_is_trace_equal_and_constant_does_not_pass(self):
        class Constant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.value = torch.nn.Parameter(torch.tensor(-2.))

            def forward(self, state, nodes, actions):
                return self.value.expand(actions.shape)

        with tempfile.TemporaryDirectory() as directory:
            traces = []
            for index, returns in enumerate(([-1., -3.], [-1., -2., -3., -4.])):
                path = Path(directory) / str(index)
                path.mkdir()
                n = len(returns)
                raw = dict(state=torch.zeros(n, 56), nodes=torch.zeros(n, 3, 35),
                    actions=torch.zeros(n, 3, dtype=torch.long), learned=torch.tensor([[1,0,1]]*n).bool(),
                    phase=torch.arange(n) % 2)
                torch.save(raw, path/'chunk.pt')
                traces.append(dict(model='qwen_7b', load=1., path=path,
                    record=dict(chunks=[dict(file='chunk.pt', offset=0, plans=n)]),
                    labels=dict(returns=torch.tensor(returns, dtype=torch.float64), phase=raw['phase'])))
            result = evaluate_critic(Constant(), traces, critic_baselines(traces), 2)
            cell = result['cells'][0]['all']
            self.assertAlmostEqual(cell['mse'], 1.25)
            self.assertAlmostEqual(cell['mse_constant'], 1.1875)
            self.assertAlmostEqual(cell['mse_zero'], 6.25)
            self.assertAlmostEqual(cell['ev'], 0.)
            self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
