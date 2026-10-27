"""九类决策展开，以及复用原后端内核的可暂停执行器。"""
from pathlib import Path

from .timing.placement_evaluator import FixedPlacementEvaluator, RequestShape

ROOT = Path(__file__).resolve().parents[1]
OPERATORS = ("Q", "K", "V", "QK", "SV", "O", "FFN_W1", "FFN_W3", "FFN_W2")
REQUEST_ID = "single_request"
PROMPT_LENGTH = 64
DECODE_LENGTH = 64
REFRESH_TOKENS = 4
SEGMENTS = 1 + DECODE_LENGTH // REFRESH_TOKENS
EPISODE_STEPS = SEGMENTS * len(OPERATORS)


def make_evaluator():
    return FixedPlacementEvaluator(ROOT / "backend", ROOT / "configs/environment.json", {}, 0)


def cold_snapshot(evaluator):
    # 不能只 reset_state：设备/链路时钟、权重缓存和 KV 都必须回到未执行过的状态。
    snapshot = evaluator.initial.clone()
    evaluator.reserve(snapshot, REQUEST_ID, PROMPT_LENGTH, DECODE_LENGTH)
    return snapshot


def request_shape(snapshot, phase):
    return (RequestShape(REQUEST_ID, PROMPT_LENGTH if phase == "prefill" else 1,
                         snapshot.kv_written_tokens[REQUEST_ID]),)


class SharedPlacement:
    def __init__(self, evaluator):
        self.compiler = evaluator.compiler
        self.npu = self.compiler.devices[0]
        self.rules = {}
        for nid in self.compiler.node_ids:
            node = self.compiler.dag.nodes[nid]
            op = node.name.upper()
            shard = int(node.attrs.get("head_shard", node.attrs.get("ffn_shard", 0)))
            if op in OPERATORS:
                source = op
            elif op == "SOFTMAX":
                source = "QK"
            elif op == "SWIGLU":
                source = "FFN_W1"
            elif nid.endswith("_Add1") or nid.endswith("_LN2"):
                source, shard = "O", 0
            elif nid.endswith("_Add2") or op == "LN":
                source, shard = "FFN_W2", 0
            else:
                raise ValueError(f"Unspecified follower: {nid}")
            self.rules[nid] = (source, shard)

    def partial(self, choices):
        mapping = dict(self.compiler.fixed_devices)
        mapping["L0_LN"] = self.npu
        for nid, (source, shard) in self.rules.items():
            if nid == "L0_LN":
                continue
            if source in choices:
                mapping[nid] = self.npu if choices[source] == 0 else f"PIM{shard}"
        return mapping

    def plan(self, actions):
        if len(actions) != len(OPERATORS) or any(a not in (0, 1) for a in actions):
            raise ValueError("A segment requires nine binary decisions")
        mapping = self.partial(dict(zip(OPERATORS, actions)))
        plan = {"order": self.compiler.order, "device_by_node": mapping}
        if self.compiler.normalize(plan) != plan:
            raise ValueError("Shared placement violates backend constraints")
        return plan


class PrefixExecution:
    """与 schedule_with_plan 使用同一单节点内核，不在动作之间重置时钟。"""

    def __init__(self, evaluator, snapshot, phase):
        self.evaluator, self.snapshot, self.phase = evaluator, snapshot, phase
        self.graph, self.order = evaluator.compiler.dag, evaluator.compiler.order
        self.cursor = 0
        self.schedule = []
        self.shapes = request_shape(snapshot, phase)
        scheduler = snapshot.scheduler
        scheduler.cost.configure_requests(phase, self.shapes)
        scheduler.batch = 1
        scheduler.seq_len = self.shapes[0].query_len + self.shapes[0].kv_len
        scheduler.stats.reset()
        scheduler.stats.set_phase(phase)
        scheduler.peak_runtime_bytes = {}
        scheduler.peak_activation_bytes = {}
        # 仅在一个完整 forward 开始时应用自回归边界；每个 RL step 不加 barrier。
        for device in scheduler.avail:
            scheduler.avail[device] = max(scheduler.avail[device], snapshot.now)
        for link in scheduler.comm.timeline_end:
            scheduler.comm.timeline_end[link] = max(scheduler.comm.timeline_end[link], snapshot.now)
        scheduler.reset_state(clear_caches=False)
        scheduler._get_graph_index(self.graph)

    @property
    def done(self):
        return self.cursor == len(self.order)

    def horizon(self):
        scheduler = self.snapshot.scheduler
        return max(self.snapshot.now, *(t.finish for t in self.schedule),
                   *scheduler.avail.values(), *scheduler.comm.timeline_end.values())

    def advance(self, device_by_node):
        from scheduler import ScheduledTask

        scheduler = self.snapshot.scheduler
        while not self.done:
            nid = self.order[self.cursor]
            node = self.graph.nodes[nid]
            is_comm = scheduler._is_comm_node(node)
            if not is_comm and nid not in device_by_node:
                break
            device = (scheduler.cost.get_host_device() if is_comm else
                      scheduler.cluster.devices[device_by_node[nid]])
            start, finish = scheduler._earliest_finish_on_device(
                self.graph, nid, device, scheduler.label, self.phase, commit=True)
            self.schedule.append(ScheduledTask(nid, "COMM" if is_comm else device.name,
                                               float(start), float(finish)))
            scheduler._after_commit_consume_predecessors(self.graph, nid)
            scheduler._log_scheduled_op_trace(
                nid=nid, op=node.attrs.get("op") or node.name,
                device="COMM" if is_comm else device.name,
                device_type="comm" if is_comm else device.type,
                start=float(start), end=float(finish),
                mode="FIXED_COMM" if is_comm else "FIXED_PLAN")
            self.cursor += 1

        horizon = self.horizon()
        if self.done:
            self.snapshot.now = horizon
            self.snapshot.kv_written_tokens[REQUEST_ID] += self.shapes[0].query_len
            scheduler.reset_state(clear_caches=False)
        return horizon


def full_replay(evaluator, segment_actions):
    """原完整执行路径：作为增量执行的参照，也用于固定基线。"""
    if len(segment_actions) != SEGMENTS:
        raise ValueError(f"Expected {SEGMENTS} segment plans")
    snapshot = cold_snapshot(evaluator)
    placement = SharedPlacement(evaluator)
    for segment, actions in enumerate(segment_actions):
        phase = "prefill" if segment == 0 else "decode"
        plan = placement.plan(actions)
        for _ in range(1 if segment == 0 else REFRESH_TOKENS):
            evaluator.evaluate(snapshot, plan, phase, request_shape(snapshot, phase), inplace=True)
    return snapshot
