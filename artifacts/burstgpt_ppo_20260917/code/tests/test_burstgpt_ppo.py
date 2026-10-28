import csv
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from dops_serve.placement_policy import StaticPlacementActor, StaticPlacementCritic, migrate_bc_actor
from dops_serve.placement_training import (node_returns, static_tensors, update_static_ppo,
                                         critic_bootstrap, group_rollout_fragments, normalize_fragments)
from dops_serve.workloads import import_full_burstgpt, load_full_trace


class BurstPPOContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.actor = StaticPlacementActor(56, 35, hidden=16, depth=2)
        self.critic = StaticPlacementCritic(self.actor)
        self.inputs = dict(state=torch.randn(56).tolist(), nodes=torch.randn(6, 35).tolist(),
            masks=[[True, True, True] for _ in range(6)], fixed_from=[-2, -1, 1, -1, -1, -1],
            learned=[False, True, False, True, True, True])

    def rows(self):
        state, nodes, masks, fixed, learned = static_tensors(self.inputs)
        rows = []
        with torch.no_grad():
            for reward in (-.2, -.1):
                actions = self.actor.sample(state, nodes, masks, fixed)
                _, logits = self.actor(state, nodes, actions, masks, learned)
                logdist = logits.log_softmax(-1)
                values = self.critic(state, nodes, actions)
                rows.append(dict(inputs=self.inputs, actions=actions[0].tolist(),
                    old_logp=logdist.gather(-1, actions.unsqueeze(-1)).squeeze(-1)[0].tolist(),
                    old_logdist=logdist[0].tolist(), values=values[0].tolist(), reward=reward, terminal=False))
        node_returns(rows, 0.)
        normalize_fragments([rows])
        return rows

    def test_migration_preserves_old_weights_and_zero_columns(self):
        old = StaticPlacementActor(47, 35, hidden=16, depth=2)
        new = migrate_bc_actor(dict(actor=old.state_dict(), dimensions=old.dimensions))
        state, nodes, masks, fixed, learned = static_tensors(self.inputs)
        actions = old.generate(state[:, :47], nodes, masks, fixed)
        old_logits = old(state[:, :47], nodes, actions, masks, learned)[1]
        new_logits = new(state, nodes, actions, masks, learned)[1]
        torch.testing.assert_close(old_logits, new_logits)
        self.assertEqual(int(torch.count_nonzero(new.state_encoder[0].weight[:, 47:])), 0)

    def test_sampling_conditional_ratio_and_fixed_nodes(self):
        compiled = torch.jit.script(self.actor)
        state, nodes, masks, fixed, learned = static_tensors(self.inputs)
        masks[:, 3, 2] = False
        with torch.no_grad():
            actions = compiled.sample(state, nodes, masks, fixed)
            self.assertEqual(actions[0, 0], 0)
            self.assertEqual(actions[0, 2], actions[0, 1])
            self.assertNotEqual(actions[0, 3], 2)
            before = compiled(state, nodes, actions, masks, learned)[1]
            after = self.actor(state, nodes, actions, masks, learned)[1]
            torch.testing.assert_close(before[torch.isfinite(before)], after[torch.isfinite(after)])
            logp = before.log_softmax(-1).gather(-1, actions.unsqueeze(-1))
            torch.testing.assert_close((logp-logp).exp(), torch.ones_like(logp))

    def test_terminal_and_truncated_node_returns(self):
        rows = [dict(actions=[0, 1], reward=-2., terminal=False),
                dict(actions=[1, 2], reward=-3., terminal=False)]
        self.assertEqual([r["returns"] for r in node_returns(rows, 7.)], [[2., 2.], [4., 4.]])
        rows[-1]["terminal"] = True
        self.assertEqual([r["returns"] for r in node_returns(rows, 7.)], [[-5., -5.], [-3., -3.]])

    def test_critic_bootstrap_and_gradient_separation(self):
        state, nodes, masks, fixed, learned = static_tensors(self.inputs)
        actions = self.actor.generate(state, nodes, masks, fixed)
        values = self.critic(state, nodes, actions)
        self.assertAlmostEqual(critic_bootstrap(self.critic, self.inputs), float(values[0, 1]), places=6)
        values.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in self.actor.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.critic.parameters()))
        self.assertFalse({p.data_ptr() for p in self.actor.parameters()} & {p.data_ptr() for p in self.critic.parameters()})

    def test_ppo_updates_both_networks_and_resume_reproduces_next_update(self):
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-5)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        settings = dict(clip=.1, ppo_epochs=2, target_kl=.01, entropy=.001, max_grad_norm=.5, mini_plans=2)
        rng = random.Random(7)
        groups = [self.rows(), []]
        before = {k: v.clone() for k, v in self.actor.state_dict().items()}
        result = update_static_ppo(self.actor, self.critic, actor_opt, critic_opt, groups, settings, rng)
        self.assertLess(abs(result["initial_kl"]), 1e-6)
        self.assertTrue(any(not torch.equal(v, self.actor.state_dict()[k]) for k, v in before.items()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.pt"
            torch.save(dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                a=actor_opt.state_dict(), c=critic_opt.state_dict(), torch_rng=torch.get_rng_state(),
                python_rng=rng.getstate()), path)
            sampled = self.rows()
            update_static_ppo(self.actor, self.critic, actor_opt, critic_opt, [sampled, []], settings, rng)
            expected = {k: v.clone() for k, v in self.actor.state_dict().items()}
            saved = torch.load(path, weights_only=False)
            self.actor.load_state_dict(saved["actor"])
            self.critic.load_state_dict(saved["critic"])
            actor_opt.load_state_dict(saved["a"])
            critic_opt.load_state_dict(saved["c"])
            torch.set_rng_state(saved["torch_rng"])
            rng.setstate(saved["python_rng"])
            repeated = self.rows()
            self.assertEqual([r["actions"] for r in sampled], [r["actions"] for r in repeated])
            update_static_ppo(self.actor, self.critic, actor_opt, critic_opt, [repeated, []], settings, rng)
            for k, v in expected.items():
                torch.testing.assert_close(v, self.actor.state_dict()[k], rtol=0, atol=0)

    def test_full_import_retains_types_duplicates_tails_and_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Timestamp", "Model", "Request tokens", "Response tokens", "Total tokens", "Log Type"])
                for index in range(23):
                    writer.writerow([index//2, "GPT-4", 29665, 7315, 36980, "Conversation log"])
            manifest = import_full_burstgpt(source, root / "data", requests_per_trace=5)
            self.assertEqual(manifest["requests"], 23)
            self.assertEqual([len(v["traces"]) for v in manifest["splits"].values()], [4, 1, 1])
            train = [r for i in range(4) for r in load_full_trace(root / "data", "train", i)]
            self.assertEqual([r["source_row"] for r in train], list(range(16)))
            self.assertTrue(all(r["prompt_len"] == 29665 and r["output_len"] == 7315 for r in train))

    def test_fragment_normalization_removes_independent_bootstrap_offsets(self):
        original = [self.rows() for _ in range(12)]
        shifted = copy.deepcopy(original)
        for i, rows in enumerate(shifted):
            for row in rows:
                row["returns"] = [v + 10*(i+1) for v in row["returns"]]
        loads = [("model", .5), ("model", 1.), ("model", 1.5)]
        first, _ = group_rollout_fragments(original, loads*4, loads)
        second, _ = group_rollout_fragments(shifted, loads*4, loads)
        self.assertEqual(list(map(len, first)), [8, 8, 8])
        for rows, repeated in zip(first, second):
            for a, b in zip(rows, repeated):
                mask = torch.tensor(a["inputs"]["learned"])
                torch.testing.assert_close(torch.tensor(a["advantages"])[mask],
                                           torch.tensor(b["advantages"])[mask], atol=1e-4, rtol=1e-4)

    def test_kl_stops_after_first_minibatch_not_after_full_pass(self):
        actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-2)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        settings = dict(clip=.1, ppo_epochs=4, target_kl=1e-12, entropy=.001,
                        max_grad_norm=.5, mini_plans=1)
        result = update_static_ppo(self.actor, self.critic, actor_opt, critic_opt,
                                  [self.rows() for _ in range(3)], settings, random.Random(4))
        self.assertTrue(result["kl_stopped"])
        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["epochs"], 1)

    def test_final_selection_keeps_e0_and_e50_without_opening_test(self):
        from dops_serve.burstgpt_ppo import finalize, evaluate_with_workers
        from dops_serve.io import write, read
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "training/qwen_1.8b/seed_4"
            write(base / "complete.json", dict(epochs=50, history=[
                dict(epoch=0, score=1., checkpoint="e0"),
                dict(epoch=50, score=.9, checkpoint="e50")]))
            finalize(root, "qwen_1.8b", 4)
            result = read(base / "selection.json")
            self.assertEqual(result["selected"]["epoch"], 0)
            self.assertTrue(result["final_checkpoint"].endswith("epoch_050.pt"))
            self.assertTrue(result["test_locked"])
            self.assertFalse((base / "test").exists())
            with self.assertRaisesRegex(ValueError, "test locked"):
                evaluate_with_workers(root, None, None, [], "test", {})


if __name__ == "__main__":
    unittest.main()
