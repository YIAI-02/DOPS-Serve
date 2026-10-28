from __future__ import annotations

import json
import copy
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .placement_compiler import PlacementCompiler, StaticPlacementCompiler


MODEL_KEYS = ("qwen_1.8b", "qwen_7b", "qwen_14b")
HARDWARE_KEY = "exp1/hw_hardware_1npu_2aim/sst2_rst2"


def build_backend(
    backend_root: str | Path,
    artifact_dir: str | Path,
    model_key: str,
    config: dict[str, Any],
    *,
    ragged: bool = False,
) -> tuple[Any, Any, PlacementCompiler]:
    """Construct the pinned DOPS primitives; no historical profile is consulted."""
    if model_key not in MODEL_KEYS:
        raise ValueError(f"unsupported model: {model_key}")
    root = Path(backend_root).resolve()
    destination = Path(artifact_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / "src"))
    import config as dops_config
    from buffer_manager import GlobalMemoryManager
    from cost_model import CostModel
    from hardware import demo_cluster
    from model_parser import build_graph
    from plan_label import PlanLabel
    from scheduler import NaiveTopoScheduler

    # The first integration gate uses the complete analytical backend path.
    # Other modes require their own coverage gate before enabling training.
    if config["npu_backend"] != "fast" or config["pim_fast_mode"] is not True:
        raise ValueError("initial placement integration currently requires explicit fast/fast")
    if config["weight_source"] != "host" or config["weight_format"] != "ND":
        raise ValueError("initial placement integration requires fixed host ND weights")
    for name in ("PIM_RUNTIME_LRU_THRESHOLD", "PIM_STATIC_ALLOC_RATIO"):
        if getattr(dops_config, name) != config[name]:
            raise ValueError(f"backend {name} differs from the experiment configuration")
    actual_launch = {
        name: item["enabled"]
        for name, item in dops_config.KERNEL_LAUNCH_OVERHEAD["by_device_name"].items()
    }
    if actual_launch != config["kernel_launch_enabled"]:
        raise ValueError("backend kernel launch switches differ from experiment configuration")
    hardware = json.loads((root / "src/examples/hardware_1npu_2aim.json").read_text())
    cfg = {
        **hardware,
        "model_family": "qwen",
        "model_variant": model_key.removeprefix("qwen_"),
        "dtype": "fp16",
        "batch": 1,
        "prefill_len": 128,
        "decode_len": 1024,
        "tp_qkv": 2,
        "tp_ffn": 2,
    }
    graph, shape = build_graph(cfg)
    cluster = demo_cluster(cfg)
    heads_per_pim = int(shape.n_kv_heads) // 2
    head_map = {head: f"PIM{head // heads_per_pim}" for head in range(shape.n_kv_heads)}
    label = PlanLabel(
        kv_in_pim=True,
        kv_place="pim",
        kv_head_to_pim=head_map,
        kv_heads_by_pim={p: [h for h, owner in head_map.items() if owner == p] for p in ("PIM0", "PIM1")},
        kv_bytes_by_pim={"PIM0": 0, "PIM1": 0},
    )
    cost_type, scheduler_type = CostModel, NaiveTopoScheduler
    if ragged:
        from .placement_cost import RaggedCostModel, ContinuousScheduler

        cost_type, scheduler_type = RaggedCostModel, ContinuousScheduler
    cost = cost_type(
        cluster, dtype="fp16", npu_backend=config["npu_backend"],
        pim_fast_mode=config["pim_fast_mode"], tp_qkv=2, tp_ffn=2,
        simulation_log_file=destination / f"{model_key}_backend.log",
    )
    scheduler = scheduler_type(cluster, cost, label, batch=1, seq_len=128, buffer=GlobalMemoryManager())
    compiler = PlacementCompiler(graph, scheduler)
    scheduler.set_storage_format_map({n.weight_id: "ND" for n in graph.nodes.values() if n.weight_id})
    return scheduler, shape, compiler


@dataclass(frozen=True)
class RequestShape:
    request_id: str
    query_len: int
    kv_len: int


@dataclass
class PlacementSnapshot:
    scheduler: Any
    now: float = 0.0
    kv_reserved_tokens: dict[str, int] = field(default_factory=dict)
    kv_written_tokens: dict[str, int] = field(default_factory=dict)

    def clone(self):
        # The diagnostic log handle is not simulated state and cannot be copied.
        return copy.deepcopy(self, {id(self.scheduler.cost.logger): self.scheduler.cost.logger})


@dataclass
class StepResult:
    duration_s: float
    snapshot: PlacementSnapshot
    schedule: list[Any]
    comm_events: list[dict[str, Any]]
    host_evaluator_s: float
    peak_runtime_bytes: dict[str, int]


class PlacementEvaluator:
    def __init__(self, backend_root, artifact_dir, model_key, config):
        scheduler, self.shape, self.compiler = build_backend(backend_root, artifact_dir, model_key, config, ragged=True)
        self.initial = PlacementSnapshot(scheduler)
        self.kv_bytes_per_token_per_pim = 2 * (self.shape.n_kv_heads // 2) * self.shape.head_dim * self.shape.layer_num * 2

    def can_reserve(self, snapshot, prompt_len, max_tokens):
        total_tokens = sum(snapshot.kv_reserved_tokens.values()) + prompt_len + max_tokens
        needed = total_tokens * self.kv_bytes_per_token_per_pim
        return all(needed <= snapshot.scheduler.buffer.pim_state[p].limit_bytes - getattr(self, "kv_headroom_bytes", 0)
                   for p in ("PIM0", "PIM1"))

    def reserve(self, snapshot, request_id, prompt_len, max_tokens):
        if request_id in snapshot.kv_reserved_tokens:
            raise ValueError(f"duplicate KV reservation: {request_id}")
        if not self.can_reserve(snapshot, prompt_len, max_tokens):
            raise ValueError(f"KV capacity exceeded for {request_id}")
        snapshot.kv_reserved_tokens[request_id] = prompt_len + max_tokens
        snapshot.kv_written_tokens[request_id] = 0
        self._update_budgets(snapshot)

    def release(self, snapshot, request_id):
        del snapshot.kv_reserved_tokens[request_id]
        del snapshot.kv_written_tokens[request_id]
        self._update_budgets(snapshot)

    def _update_budgets(self, snapshot):
        scheduler = snapshot.scheduler
        reserved = sum(snapshot.kv_reserved_tokens.values()) * self.kv_bytes_per_token_per_pim
        scheduler.label.kv_total_bytes = reserved * 2
        scheduler.label.kv_bytes_by_pim = {"PIM0": reserved, "PIM1": reserved}
        for device in ("PIM0", "PIM1"):
            state = scheduler.buffer.pim_state[device]
            scheduler.buffer.register_pim_device(
                device, phy_bytes=state.phy_bytes, runtime_limit_bytes=state.limit_bytes,
                kv_reserved_bytes=reserved, kv_in_pim=True,
                weight_cache_capacity_bytes=state.limit_bytes - reserved,
            )

    def evaluate(self, snapshot, fixed_plan, phase, request_shapes, *, inplace=False):
        begin = time.perf_counter()
        if phase not in {"prefill", "decode"} or not request_shapes:
            raise ValueError("evaluate requires a phase and non-empty request shapes")
        if tuple(fixed_plan["order"]) != self.compiler.order:
            raise ValueError("placement v1 requires the compiler's fixed topological order")
        canonical = (self.compiler.normalize(fixed_plan) if isinstance(self.compiler, StaticPlacementCompiler)
                     else self.compiler.compile(self.compiler.project(fixed_plan["device_by_node"])))
        if canonical["device_by_node"] != fixed_plan["device_by_node"]:
            raise ValueError("fixed plan differs from the canonical PIM ownership")
        shapes = tuple(request_shapes)
        if len({r.request_id for r in shapes}) != len(shapes):
            raise ValueError("duplicate request IDs in one execution step")
        for r in shapes:
            if r.query_len <= 0 or r.kv_len != snapshot.kv_written_tokens[r.request_id]:
                raise ValueError(f"{r.request_id}: request shape disagrees with actual KV ledger")
            if (phase == "prefill" and r.kv_len != 0) or (phase == "decode" and r.query_len != 1):
                raise ValueError("v1 uses complete prefill and single-token decode")
            if r.kv_len + r.query_len > snapshot.kv_reserved_tokens[r.request_id]:
                raise ValueError(f"{r.request_id}: KV write exceeds declared reservation")
        successor = snapshot if inplace else snapshot.clone()
        start_time = snapshot.now
        scheduler = successor.scheduler
        scheduler.cost.configure_requests(phase, shapes)
        scheduler.batch = len(shapes)
        scheduler.seq_len = max(r.query_len + r.kv_len for r in shapes)
        scheduler.stats.reset()
        scheduler.peak_runtime_bytes = {}
        scheduler.peak_activation_bytes = {}
        for device in scheduler.avail:
            scheduler.avail[device] = max(scheduler.avail[device], snapshot.now)
        for link in scheduler.comm.timeline_end:
            scheduler.comm.timeline_end[link] = max(scheduler.comm.timeline_end[link], snapshot.now)
        schedule = scheduler.schedule_with_plan(self.compiler.dag, phase, fixed_plan)
        if tuple(t.node_id for t in schedule) != tuple(fixed_plan["order"]):
            raise ValueError("backend changed the fixed order")
        if {t.node_id: t.device for t in schedule if t.device != "COMM"} != fixed_plan["device_by_node"]:
            raise ValueError("backend changed the fixed placement")
        if any(not math.isfinite(t.finish) or t.start < snapshot.now for t in schedule):
            raise ValueError("non-finite execution or violated replay time lower bound")
        successor.now = max(snapshot.now, scheduler.makespan(schedule), *scheduler.avail.values(), *scheduler.comm.timeline_end.values())
        for request in shapes:
            successor.kv_written_tokens[request.request_id] += request.query_len
        result = StepResult(
            successor.now - start_time, successor, schedule,
            list(scheduler.stats.comm_events), time.perf_counter() - begin,
            dict(scheduler.peak_runtime_bytes),
        )
        # A phase executes the whole DAG. Its temporary activations have no
        # consumers in the next phase; weight/KV/link/device state remains live.
        scheduler.reset_state(clear_caches=False)
        return result


class BurstPlacementEvaluator(PlacementEvaluator):
    """Current BC hardware/configuration with ragged exact three-device replay."""

    def __init__(self, backend_root, config_path, parameters, kv_headroom_bytes):
        sys.path.insert(0, str(Path(backend_root) / "src"))
        import config as native_config
        for name, value in parameters.items():
            setattr(native_config, name, value)
        from mainlib.cli import _apply_runtime_config_overrides, _load_cfg_from_json
        from mainlib.kv_policy import _apply_kv_place_constraints, _make_label_given_kv_place
        from mainlib.shared import GlobalMemoryManager, build_graph, demo_cluster
        from .placement_cost import RaggedCostModel, ContinuousScheduler

        cfg = _load_cfg_from_json(str(config_path))
        _apply_runtime_config_overrides(cfg)
        cluster = demo_cluster(cfg)
        graph, self.shape = build_graph(cfg)
        label, feasible = _make_label_given_kv_place(cfg=cfg, cluster=cluster, graph=graph,
                                                   shape=self.shape, kv_place="pim")
        if not feasible:
            raise ValueError("BC configuration cannot place KV on PIM")
        graph = _apply_kv_place_constraints(graph, "pim")
        label.kv_total_bytes = label.kv_total_bytes_all = label.kv_total_bytes_on_pim = 0
        label.kv_bytes_by_pim = {"PIM0": 0, "PIM1": 0}
        cost = RaggedCostModel(cluster, dtype=cfg["dtype"], npu_backend=cfg["npu_backend"],
                               pim_fast_mode=cfg["pim_fast_mode"], tp_qkv=cfg["tp_qkv"],
                               tp_ffn=cfg["tp_ffn"], simulation_log_file="/dev/null")
        scheduler = ContinuousScheduler(cluster, cost, label, batch=1, seq_len=cfg["prefill_len"],
                                        buffer=GlobalMemoryManager())
        scheduler.reset_state()
        scheduler.set_storage_format_map({n.weight_id: cfg["weight_format"]
                                          for n in graph.nodes.values() if n.weight_id})
        self.compiler = StaticPlacementCompiler(graph, scheduler)
        self.initial = PlacementSnapshot(scheduler)
        self.kv_bytes_per_token_per_pim = 2 * (self.shape.n_kv_heads // 2) * self.shape.head_dim * self.shape.layer_num * 2
        self.kv_headroom_bytes = int(kv_headroom_bytes)
        self.config = cfg

    def pd_plan(self, snapshot, phase, shapes):
        from mainlib.baselines import _apply_policy_on_graph
        from mainlib.kv_policy import _apply_kv_place_constraints
        graph = _apply_kv_place_constraints(_apply_policy_on_graph(self.compiler.dag, "PD", phase=phase), "pim")
        scheduler = snapshot.clone().scheduler
        scheduler.cost.configure_requests(phase, shapes)
        scheduler.batch = len(shapes)
        scheduler.seq_len = max(r.query_len + r.kv_len for r in shapes)
        scheduler.peak_runtime_bytes, scheduler.peak_activation_bytes = {}, {}
        for device in scheduler.avail:
            scheduler.avail[device] = max(scheduler.avail[device], snapshot.now)
        for link in scheduler.comm.timeline_end:
            scheduler.comm.timeline_end[link] = max(scheduler.comm.timeline_end[link], snapshot.now)
        schedule = scheduler.schedule(graph, phase)
        return self.compiler.normalize(scheduler.export_fixed_plan(schedule))
