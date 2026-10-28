"""BC-compatible static features plus visible BurstGPT serving features."""
from __future__ import annotations


def static_policy_inputs(scheduler, compiler, shape, case, phase, token_index, now,
                         prefill_plan=None, previous_decode_plan=None):
    """Features at a static decision boundary, with explicit physical-device history."""
    import math

    def scale(value):
        value = float(value)
        return math.copysign(math.log1p(abs(value)), value) / 10.0

    generated = max(0, token_index)
    context = 0 if phase == "prefill" else case["prompt"] + generated
    state = [scale(v) for v in (case["batch"], case["prompt"], case["decode"], generated, context)]
    state += [float(phase == "prefill"), float(phase == "decode")]
    state += [scale(v) for v in (shape.layer_num, shape.dim, shape.ffn_dim, shape.n_heads,
                                shape.n_kv_heads, shape.head_dim,
                                sum(n.weight_size for n in compiler.dag.nodes.values()) / 2**30)]
    host = scheduler.cost.get_host_device().name
    all_devices = (*compiler.devices, host)
    for device in compiler.devices:
        cache = scheduler.buffer.device_cache.get(device)
        pim = scheduler.buffer.pim_state.get(device)
        state += [scale(scheduler.avail[device] - now),
                  scale(cache.used / 2**30 if cache else 0),
                  scale(cache.capacity / 2**30 if cache else 0),
                  scale(pim.kv_reserved_bytes / 2**30 if pim else 0),
                  scale(pim.kv_used_bytes / 2**30 if pim else 0),
                  scale(pim.limit_bytes / 2**30 if pim else 0)]
    for source in all_devices:
        for destination in all_devices:
            if source != destination:
                state.append(scale(scheduler.comm.timeline_end.get((source, destination), 0.0) - now))
    state += [float(scheduler.label.kv_place == place) for place in ("pim", "host", "npu")]

    roles = ("LN", "Q", "K", "V", "QK", "SOFTMAX", "SV", "O", "ADD", "W1", "W3", "SWIGLU", "W2")
    tokens = case["batch"] * (case["prompt"] if phase == "prefill" else 1)
    pairs = case["batch"] * (case["prompt"] * (case["prompt"] + 1) / 2 if phase == "prefill" else context + 1)
    nodes = []
    for index, nid in enumerate(compiler.node_ids):
        node = compiler.dag.nodes[nid]
        role = node.name.upper().removeprefix("FFN_")
        attrs = node.attrs
        values = [float(role == name) for name in roles]
        values += [attrs["layer"] / shape.layer_num, index / len(compiler.node_ids),
                   float(attrs.get("head_shard", -1)), float(attrs.get("ffn_shard", -1)),
                   scale(node.weight_size / 2**20), scale(tokens), scale(pairs),
                   scale(attrs.get("q_heads", shape.n_heads)),
                   scale(attrs.get("kv_heads", shape.n_kv_heads)), scale(attrs.get("ffn_dim", shape.ffn_dim))]
        values += [float(scheduler.buffer.is_cached(device, node.weight_id)) for device in compiler.devices]
        values += [float(v) for v in compiler.masks[index]]
        for plan in (prefill_plan, previous_decode_plan):
            chosen = plan["device_by_node"][nid] if plan is not None else None
            values += [float(chosen == device) for device in compiler.devices]
        nodes.append(values)
    return {"state": state, "nodes": nodes, "masks": compiler.masks,
            "fixed_from": compiler.fixed_from, "learned": compiler.learned}


def serving_policy_inputs(scheduler, compiler, shape, phase, requests, waiting,
                          admitted_count, decode_feature, now,
                          prefill_plan=None, previous_decode_plan=None):
    """BC-compatible features from visible, actually admitted requests only."""
    import math
    from statistics import mean, pstdev

    def scale(value):
        return math.copysign(math.log1p(abs(value)), value) / 10.0

    prompts = [r["prompt_len"] for r in requests]
    kv = [r["kv_len"] for r in requests] if phase == "decode" else [0]
    progress = mean(r["generated"] - 1 for r in requests) if phase == "decode" else 0
    case = dict(batch=len(requests), prompt=mean(prompts), decode=0)
    inputs = static_policy_inputs(scheduler, compiler, shape, case, phase, progress, now,
                                  prefill_plan, previous_decode_plan)
    inputs["state"][2] = decode_feature  # Already scaled on BC train tasks.
    inputs["state"][4] = scale(mean(kv))
    ages = [r["age_s"] for r in waiting] or [0]
    inputs["state"] += [scale(v) for v in (
        7315, len(waiting), admitted_count, mean(ages), max(ages),
        max(prompts), pstdev(prompts), max(kv), pstdev(kv))]
    queries = sum(prompts) if phase == "prefill" else len(requests)
    pairs = (sum(p * (p + 1) / 2 for p in prompts) if phase == "prefill"
             else sum(k + 1 for k in kv))
    for node in inputs["nodes"]:
        node[18], node[19] = scale(queries), scale(pairs)
    return inputs
