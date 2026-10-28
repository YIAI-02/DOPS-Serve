"""Fixed-order Bifocal using the pinned DOPS device-scoring heuristics."""
from __future__ import annotations

import itertools
import math

from scheduler import BifocalScheduler
from scheduler import scheduler_bifocal as bifocal

from .placement_cost import ContinuousScheduler
from .placement_evaluator import RequestShape


class FixedOrderBifocalScheduler(ContinuousScheduler, BifocalScheduler):
    def candidates(self, node_id):
        compiler = self.compiler
        if node_id in compiler.comm_nodes:
            return [self.cost.get_host_device()]
        if node_id in compiler.fixed_devices:
            return [self.cluster.devices[compiler.fixed_devices[node_id]]]
        return [self.cluster.devices[name] for name, allowed in zip(
            (compiler.npu, compiler.pim_owner[node_id]), compiler.mask[node_id]
        ) if allowed]

    def _node_allowed_on(self, node, device):
        if not super()._node_allowed_on(node, device):
            return False
        if node.id in self.compiler.fixed_devices:
            return device.name == self.compiler.fixed_devices[node.id]
        return device.type != "pim" or device.name == self.compiler.pim_owner.get(node.id)

    def _lookahead_window_estimate(self, g, chain, first_dev, first_eft, phase):
        # Future nodes obey the same individual PIM ownership as current nodes.
        options = [self.candidates(node_id) for node_id in chain[1:]]
        best = (float("inf"), {})
        for future in itertools.product(*options):
            finish, assignments = self._simulate_chain_finish(
                g, chain, [first_dev, *future], first_eft, phase,
                apply_consistency_penalty=True,
            )
            if finish < best[0]:
                best = (finish, assignments)
        return best

    def placement(self, phase):
        compiler, graph = self.compiler, self.compiler.dag
        self.reset_state(clear_caches=False)
        self.stats.set_phase(phase)
        self.peak_runtime_bytes = {}
        index = self._get_graph_index(graph)
        gamma, horizon = bifocal.SCHED_JOINT_LK_GAMMA, bifocal.SCHED_JOINT_LK_H
        lookahead = bifocal.SCHED_JOINT_LK_ENABLE and horizon > 1 and gamma > 0
        if lookahead:
            self._upward_rank(graph, phase)
        ranks = index.rank_u_by_phase.get(phase, {})
        devices, scheduled = {}, set()
        for node_id in compiler.order:
            chain = self._build_lookahead_chain(graph, node_id, index, ranks, phase, horizon) if lookahead else [node_id]
            best = None
            for device in self.candidates(node_id):
                _, finish = self._earliest_finish_on_device(graph, node_id, device, self.label, phase, commit=False)
                if not math.isfinite(finish):
                    continue
                window, hints = self._lookahead_window_estimate(graph, chain, device, finish, phase) if lookahead else (finish, {})
                score = (1 - gamma) * finish + gamma * window + self._weight_reuse_bias(graph, index, node_id, device, phase)
                # The pinned backend's exact-score ties are seeded random ties.
                if best is None or score < best[0] or (abs(score - best[0]) < 1e-9 and (
                    finish < best[1] or (abs(finish - best[1]) < 1e-9 and self._rng.random() < 0.5)
                )):
                    best = (score, finish, device, hints)
            if best is None:
                raise ValueError(f"Bifocal has no feasible binary placement for {node_id}")
            _, _, device, hints = best
            self._earliest_finish_on_device(graph, node_id, device, self.label, phase, commit=True)
            self._after_commit_consume_predecessors(graph, node_id)
            scheduled.add(node_id)
            self._update_weight_hint_after_commit(graph, node_id, device.type)
            self._update_plan_hints(hints, scheduled)
            if node_id not in compiler.comm_nodes:
                devices[node_id] = device.name
        return compiler.compile(compiler.project(devices))


class FixedOrderBifocalPlacement:
    def __init__(self, evaluator, snapshot, *, max_batch=16, seed=7):
        self.evaluator, self.snapshot = evaluator, snapshot
        self.max_batch, self.seed = max_batch, seed

    def _plan(self, snapshot, phase, shapes):
        shadow = snapshot.clone().scheduler
        # Initialize Bifocal-only hint fields, then restore the copied live state.
        scheduler = FixedOrderBifocalScheduler(
            shadow.cluster, shadow.cost, shadow.label, len(shapes),
            max(r.query_len + r.kv_len for r in shapes), None,
        )
        scheduler.__dict__.update(shadow.__dict__)
        scheduler.compiler = self.evaluator.compiler
        scheduler._rng.seed(self.seed)
        scheduler.clear_decode_context()  # Actual remaining output length is unavailable online.
        scheduler.cost.configure_requests(phase, shapes)
        scheduler.batch = len(shapes)
        scheduler.seq_len = max(r.query_len + r.kv_len for r in shapes)
        for device in scheduler.avail:
            scheduler.avail[device] = max(scheduler.avail[device], snapshot.now)
        for link in scheduler.comm.timeline_end:
            scheduler.comm.timeline_end[link] = max(scheduler.comm.timeline_end[link], snapshot.now)
        return scheduler.placement(phase)

    def __call__(self, observation, current_plans):
        snapshot = self.snapshot().clone()
        admitted = []
        for row in observation["waiting"][:max(0, self.max_batch - len(observation["active"]))]:
            if not self.evaluator.can_reserve(snapshot, row["prompt_len"], row["max_tokens"]):
                break
            self.evaluator.reserve(snapshot, row["request_id"], row["prompt_len"], row["max_tokens"])
            admitted.append(row)
        compiler = self.evaluator.compiler
        plans = dict(current_plans) if current_plans else {
            phase: compiler.compile(compiler.uniform(0)) for phase in ("prefill", "decode")
        }
        if admitted:
            shapes = [RequestShape(r["request_id"], r["prompt_len"], 0) for r in admitted]
            plans["prefill"] = self._plan(snapshot, "prefill", shapes)
            # Decode planning inherits the chosen prefill's cache/KV state.
            snapshot = self.evaluator.evaluate(snapshot, plans["prefill"], "prefill", shapes).snapshot
        requests = [*observation["active"], *admitted]
        if requests:
            shapes = [RequestShape(r["request_id"], 1, snapshot.kv_written_tokens[r["request_id"]]) for r in requests]
            plans["decode"] = self._plan(snapshot, "decode", shapes)
        return plans
