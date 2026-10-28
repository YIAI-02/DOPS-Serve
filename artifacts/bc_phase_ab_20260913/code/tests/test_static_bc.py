import tempfile
import unittest
from pathlib import Path
import torch
from dops_serve.placement_policy import StaticPlacementActor
from dops_serve.static_bc import MODELS, prepare, refresh, read, write, batches
import itertools

class StaticBCContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        self.actor = StaticPlacementActor(47, 35)
        self.state = torch.randn(1, 47)
        self.nodes = torch.randn(1, 6, 35)
        self.actions = torch.tensor([[0, 1, 2, 0, 1, 2]])
        self.mask = torch.ones(1, 6, 3, dtype=torch.bool)
        self.learned = torch.ones(1, 6)

    def test_current_answer_does_not_leak_but_previous_answer_conditions_later_nodes(self):
        _, a = self.actor(self.state, self.nodes, self.actions, self.mask, self.learned)
        changed = self.actions.clone()
        changed[0, 2] = 0
        _, b = self.actor(self.state, self.nodes, changed, self.mask, self.learned)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=0)
        self.assertGreater((a[:, 3:]-b[:, 3:]).abs().max().item(), 1e-6)

    def test_generate_matches_conditionals_and_fixed_rules(self):
        fixed = torch.tensor([-2, -1, 1, -1, 3, -1])
        actor = torch.jit.script(self.actor.eval())
        with torch.inference_mode():
            generated = actor.generate(self.state, self.nodes, self.mask, fixed)
            _, logits = actor(self.state, self.nodes, generated, self.mask, self.learned)
        self.assertEqual(generated[0, 0].item(), 0)
        for i, parent in enumerate(fixed.tolist()):
            if parent >= 0:
                self.assertEqual(generated[0, i].item(), generated[0, parent].item())
            elif parent == -1:
                self.assertEqual(generated[0, i].item(), logits[0, i].argmax().item())

    def test_prefill_and_previous_decode_features_affect_predictions(self):
        with torch.no_grad():
            _, before = self.actor(self.state, self.nodes, self.actions, self.mask, self.learned)
            for offset in (-6, -3):
                nodes = self.nodes.clone()
                nodes[:, :, offset] += 3
                _, after = self.actor(self.state, nodes, self.actions, self.mask, self.learned)
                self.assertGreater((after-before).abs().max().item(), 1e-6)

    def test_split_and_refresh_and_equal_task_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'C026'
            parameters = {'SCHED_JOINT_LK_H': 2, 'SCHED_JOINT_LK_GAMMA': 0.4}
            write(source / 'F000/result.json', {'candidate': {'parameters': parameters}})
            for i, (model, batch, prompt, decode) in enumerate(itertools.product(MODELS, [1,4,8,16], [128,512,1024,2048], [128,256,512,1024])):
                write(source / f'F{i:03d}/run_config.json', dict(model_variant=model.removeprefix('qwen_'),
                    batch=batch, prefill_len=prompt, decode_len=decode, decode_sample_stride=2,
                    decode_plan_refresh_stride=2, pim_weight_load_overlap_ratio=0.5,
                    weight_load_compute_overlap_ratio=1.0, algo=['Bifocal'], baselines=[], result_dir='old'))
            protocol = prepare(root, root / 'backend', source, root / 'others')
            self.assertEqual(protocol['counts'], {'train': 144, 'val': 24, 'test': 24})
            self.assertTrue(protocol['test_locked'])
            self.assertEqual(protocol['epochs'], 10)
            self.assertEqual(protocol['parameters'], parameters)
            cfg = read(root / 'configs/000.json')
            self.assertEqual(cfg['pim_weight_load_overlap_ratio'], 0.5)
            self.assertEqual(cfg['weight_load_compute_overlap_ratio'], 1.0)
            self.assertEqual(cfg['decode_plan_refresh_stride'], 2)
        self.assertEqual([t for t in range(128) if refresh(t)], [0, 1, *range(2,128,2)])
        lengths = [66, 130, 258, 514]
        for length in lengths:
            self.assertAlmostEqual(sum([1/length]*length), 1.0)

    def test_gradient_update_reduces_fixed_teacher_nll(self):
        optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        first = self.actor(self.state, self.nodes, self.actions, self.mask, self.learned)[0].item()
        for _ in range(3):
            optimizer.zero_grad()
            loss = self.actor(self.state, self.nodes, self.actions, self.mask, self.learned)[0].mean()
            loss.backward()
            optimizer.step()
        last = self.actor(self.state, self.nodes, self.actions, self.mask, self.learned)[0].item()
        self.assertLess(last, first)

if __name__ == '__main__':
    unittest.main()
