"""Compile exact NPU/PIM0/PIM1 node actions into a fixed plan."""
from __future__ import annotations


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
