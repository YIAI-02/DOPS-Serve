"""DOPS cost extensions, imported only after the backend source path is installed."""
from __future__ import annotations

import copy
import math
from collections import Counter
from contextlib import contextmanager

from comm_primitives import _reserve_p2p
from cost_model import CostModel
from scheduler import NaiveTopoScheduler


class RaggedCostModel(CostModel):
    def configure_requests(self, phase, shapes):
        self.request_shapes = tuple(shapes)
        self.request_phase = phase
        self._primitive_depth = 0

    @contextmanager
    def regular_primitive(self):
        self._primitive_depth += 1
        try:
            yield
        finally:
            self._primitive_depth -= 1

    def primitive_groups(self, node, phase):
        if node.name.upper() in {"QK", "SOFTMAX", "SV"}:
            groups = Counter((r.query_len, r.kv_len) for r in self.request_shapes)
            for (query, past), batch in sorted(groups.items()):
                primitive = copy.copy(node)
                primitive.attrs = {**node.attrs, "kv_len": past + query}
                yield primitive, batch, query if phase == "prefill" else past + query
        elif phase == "prefill":
            yield node, 1, sum(r.query_len for r in self.request_shapes)
        else:
            yield node, len(self.request_shapes), 1

    def estimate_flops(self, node, batch, seq_len, phase):
        if self._primitive_depth:
            return super().estimate_flops(node, batch, seq_len, phase)
        with self.regular_primitive():
            return sum(super(RaggedCostModel, self).estimate_flops(n, b, s, phase) for n, b, s in self.primitive_groups(node, phase))

    def estimate_activation_bytes(self, node, batch, seq_len, phase):
        if self._primitive_depth:
            return super().estimate_activation_bytes(node, batch, seq_len, phase)
        with self.regular_primitive():
            values = [super(RaggedCostModel, self).estimate_activation_bytes(n, b, s, phase) for n, b, s in self.primitive_groups(node, phase)]
        return sum(v[0] for v in values), sum(v[1] for v in values)

    def estimate_kv_cache_read_bytes(self, node, batch, seq_len, phase):
        if phase == "prefill":
            return 0
        # The current query's K/V already travels on the original K/V DAG edges.
        # Only previously written KV is loaded from the persistent KV cache.
        return int(sum(r.kv_len for r in self.request_shapes) * node.attrs["n_kv_heads"] * node.attrs["head_dim"] * self._kv_dtype_bytes(node, phase))

    def node_device_cost(self, node, dev, label, batch, seq_len, phase):
        if self._primitive_depth:
            return super().node_device_cost(node, dev, label, batch, seq_len, phase)
        with self.regular_primitive():
            return sum(super(RaggedCostModel, self).node_device_cost(n, dev, label, b, s, phase) for n, b, s in self.primitive_groups(node, phase))

    def weighted_compute_stage(self, node, dev, label, batch, seq_len, phase, *, resident_weight_fmt):
        if self._primitive_depth:
            return super().weighted_compute_stage(node, dev, label, batch, seq_len, phase, resident_weight_fmt=resident_weight_fmt)
        primitive, merged_batch, merged_seq = next(self.primitive_groups(node, phase))
        with self.regular_primitive():
            return super().weighted_compute_stage(primitive, dev, label, merged_batch, merged_seq, phase, resident_weight_fmt=resident_weight_fmt)


class ContinuousScheduler(NaiveTopoScheduler):
    """Retain DOPS scheduling semantics while reserving FC collective links."""

    def _earliest_finish_allreduce(self, g, nid, label, phase, commit):
        if self.cluster.topology != "fc":
            raise ValueError("placement v1 requires the specified FC interconnect")
        node = g.nodes[nid]
        rd, wr = self.cost.estimate_activation_bytes(node, self.batch, self.seq_len, phase)
        tensor_bytes = max(rd, wr)
        predecessors = list(g.predecessors(nid))
        ring = sorted({self._node_placement[u] for u in predecessors})
        start = max(max(self._node_finish_time[u] for u in predecessors), max(self.avail[d] for d in ring))
        end = start
        if len(ring) > 1:
            chunk = math.ceil(tensor_bytes / len(ring))
            local_timeline = {}
            for step in range(2 * (len(ring) - 1)):
                finishes = []
                for i, source in enumerate(ring):
                    _, finish = _reserve_p2p(
                        comm=self.comm, cost=self.cost, cluster=self.cluster,
                        src=source, dst=ring[(i + 1) % len(ring)], bytes_amount=chunk,
                        earliest=end, commit=commit, tag="allreduce", local_tl=local_timeline,
                        extra={"node_id": nid, "step": step, "tensor_bytes": tensor_bytes},
                    )
                    finishes.append(finish)
                end = max(finishes)
        if commit:
            self._node_finish_time[nid] = end
            self._node_placement[nid] = ring[0]
            self._node_out_fmt[nid] = "ND"
            self._collective_output_devs[nid] = set(ring)
            for device in ring:
                self.avail[device] = max(self.avail[device], end)
        return start, end

    def _after_commit_consume_predecessors(self, g, nid):
        for device, state in self.buffer.pim_state.items():
            used = self.buffer.pim_used_bytes(device)[-1]
            self.peak_runtime_bytes[device] = max(self.peak_runtime_bytes.get(device, 0), used)
            if used > state.limit_bytes:
                raise ValueError(f"{nid}: {device} runtime capacity exceeded")
        super()._after_commit_consume_predecessors(g, nid)
