from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PlacementBlock:
    layer: int
    kind: str
    node_ids: tuple[str, ...]


class PlacementCompiler:
    """Compile independent NPU/PIM node actions into the existing Fixed Plan ABI."""

    def __init__(self, dag: Any, scheduler: Any) -> None:
        self.dag = dag
        self.scheduler = scheduler
        npus = scheduler.cluster.devices_by_type("npu")
        pims = sorted(scheduler.cluster.devices_by_type("pim"), key=lambda d: d.name)
        if len(npus) != 1 or [d.name for d in pims] != ["PIM0", "PIM1"]:
            raise ValueError("placement v1 requires one NPU and PIM0/PIM1")
        self.npu = npus[0].name
        self.order = tuple(dag.topological())
        self.pim_owner: dict[str, str] = {}
        self.mask: dict[str, tuple[bool, bool]] = {}
        self.fixed_devices: dict[str, str] = {}
        self.comm_nodes: set[str] = set()
        grouped: dict[tuple[int, str], list[str]] = {}
        for node_id in self.order:
            node = dag.nodes[node_id]
            if scheduler._is_comm_node(node):
                self.comm_nodes.add(node_id)
                continue
            attrs = node.attrs
            layer = int(attrs["layer"])
            if "head_shard" in attrs:
                if int(attrs["tp_qkv"]) != 2:
                    raise ValueError(f"{node_id}: effective tp_qkv must be 2")
                owners = {
                    scheduler.label.kv_head_to_pim[head]
                    for head in range(int(attrs["kv_head_start"]), int(attrs["kv_head_end"]))
                }
                if len(owners) != 1:
                    raise ValueError(f"{node_id}: KV head shard must belong to one PIM")
                owner = owners.pop()
            elif "ffn_shard" in attrs:
                if int(attrs["tp_ffn"]) != 2:
                    raise ValueError(f"{node_id}: effective tp_ffn must be 2")
                owner = pims[int(attrs["ffn_shard"])].name
            elif node.name.upper() in {"LN", "ADD"}:
                owner = pims[layer % 2].name
            else:
                raise ValueError(f"{node_id}: expected an explicitly sharded Qwen node")
            self.pim_owner[node_id] = owner
            if node.name.upper() in {"K_WRITE", "V_WRITE"}:
                pinned = scheduler._preferred_kv_write_device(dag, node_id)
                if pinned is None or pinned.name != owner:
                    raise ValueError(f"{node_id}: KV write binding differs from compiler owner")
                self.fixed_devices[node_id] = owner
                continue
            legal = tuple(
                scheduler._node_allowed_on(node, scheduler.cluster.devices[name])
                for name in (self.npu, owner)
            )
            if not any(legal):
                raise ValueError(f"{node_id}: no legal NPU/PIM action")
            self.mask[node_id] = legal
            is_ffn = "ffn_shard" in attrs or node_id in {f"L{layer}_LN2", f"L{layer}_Add2"}
            grouped.setdefault((layer, "ffn" if is_ffn else "attention"), []).append(node_id)
        layers = sorted({layer for layer, _ in grouped})
        self.blocks = tuple(
            PlacementBlock(layer, kind, tuple(grouped[(layer, kind)]))
            for layer in layers
            for kind in ("attention", "ffn")
        )

    def compile(self, program: Sequence[Sequence[int]]) -> dict[str, Any]:
        if len(program) != len(self.blocks):
            raise ValueError("program must contain attention and FFN actions for every layer")
        devices = dict(self.fixed_devices)
        for block, actions in zip(self.blocks, program):
            if len(actions) != len(block.node_ids):
                raise ValueError(f"layer {block.layer} {block.kind}: wrong node action count")
            for node_id, action in zip(block.node_ids, actions):
                if action not in (0, 1) or not self.mask[node_id][action]:
                    raise ValueError(f"{node_id}: illegal binary action {action}")
                devices[node_id] = self.npu if action == 0 else self.pim_owner[node_id]
        return {"device_by_node": devices, "order": self.order}

    def project(self, mapping: Mapping[str, str]) -> tuple[tuple[int, ...], ...]:
        """Project historical free-node PIM identity; reject incompatible fixed nodes."""
        expected = set(self.dag.nodes) - self.comm_nodes
        missing = sorted(expected - mapping.keys())
        unknown = sorted(mapping.keys() - self.dag.nodes.keys())
        if missing or unknown:
            raise ValueError(f"DAG mapping mismatch: missing={missing[:8]}, unknown={unknown[:8]}")
        for node_id, owner in self.fixed_devices.items():
            if mapping[node_id] != owner:
                raise ValueError(f"{node_id}: historical KV binding {mapping[node_id]} != {owner}")
        actions: list[tuple[int, ...]] = []
        for block in self.blocks:
            values = []
            for node_id in block.node_ids:
                device = mapping[node_id]
                if device == self.npu:
                    values.append(0)
                elif device in {"PIM0", "PIM1"}:
                    values.append(1)
                else:
                    raise ValueError(f"{node_id}: unsupported historical device {device}")
            actions.append(tuple(values))
        program = tuple(actions)
        self.compile(program)
        return program

    def uniform(self, action: int) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(action if self.mask[n][action] else 1 - action for n in block.node_ids)
            for block in self.blocks
        )


class StaticPlacementCompiler:
    """Exact three-device actions under the agreed static-task execution rules."""

    def __init__(self, dag, scheduler):
        self.dag, self.scheduler = dag, scheduler
        self.devices = (scheduler.cluster.devices_by_type("npu")[0].name, "PIM0", "PIM1")
        self.order = tuple(dag.topological())
        self.fixed_devices = {}
        nodes = []
        for nid in self.order:
            node = dag.nodes[nid]
            if scheduler._is_comm_node(node):
                continue
            if node.name.upper() in {"K_WRITE", "V_WRITE", "KV_WRITE"}:
                self.fixed_devices[nid] = scheduler._preferred_kv_write_device(dag, nid).name
            else:
                nodes.append(nid)
        self.node_ids = tuple(nodes)
        positions = {nid: i for i, nid in enumerate(nodes)}
        self.masks, self.fixed_from = [], []
        for i, nid in enumerate(nodes):
            node = dag.nodes[nid]
            layer = int(node.attrs["layer"])
            mask = [scheduler._node_allowed_on(node, scheduler.cluster.devices[d]) for d in self.devices]
            if not any(mask):
                raise ValueError(f"{nid}: no legal three-device action")
            parent = -1
            if nid == f"L{layer}_LN":
                parent = -2 if layer == 0 else positions[f"L{layer-1}_Add2"]
            elif nid == f"L{layer}_LN2":
                parent = positions[f"L{layer}_Add1"]
            elif node.name.upper() == "SOFTMAX":
                parent = positions[f"L{layer}_QK_s{node.attrs['head_shard']}"]
            if parent >= i:
                raise ValueError(f"{nid}: fixed device rule does not follow the node order")
            self.masks.append(mask)
            self.fixed_from.append(parent)
        self.learned = [parent == -1 for parent in self.fixed_from]

    def normalize(self, plan):
        mapping = dict(plan["device_by_node"])
        for nid, device in self.fixed_devices.items():
            if mapping[nid] != device:
                raise ValueError(f"{nid}: source KV placement differs from this task's KV policy")
        actions = []
        for i, nid in enumerate(self.node_ids):
            parent = self.fixed_from[i]
            action = 0 if parent == -2 else actions[parent] if parent >= 0 else self.devices.index(mapping[nid])
            if not self.masks[i][action]:
                raise ValueError(f"{nid}: fixed rule or source device violates the legal mask")
            actions.append(action)
        return self.compile(actions)

    def actions(self, plan):
        return [self.devices.index(plan["device_by_node"][nid]) for nid in self.node_ids]

    def compile(self, actions):
        if len(actions) != len(self.node_ids):
            raise ValueError("static plan must cover every compute node")
        mapping = dict(self.fixed_devices)
        for i, (nid, action) in enumerate(zip(self.node_ids, actions)):
            action = int(action)
            parent = self.fixed_from[i]
            if parent == -2 and action != 0 or parent >= 0 and action != int(actions[parent]):
                raise ValueError(f"{nid}: fixed small-operator rule violated")
            if action not in (0, 1, 2) or not self.masks[i][action]:
                raise ValueError(f"{nid}: illegal exact device action {action}")
            mapping[nid] = self.devices[action]
        return {"order": self.order, "device_by_node": mapping}
