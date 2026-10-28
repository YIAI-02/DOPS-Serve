from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical


OP_TYPES = ("LN", "Q", "K", "V", "QK", "SOFTMAX", "SV", "O", "ADD", "W1", "W3", "SWIGLU", "W2")


def _log(value):
    return math.copysign(math.log1p(abs(float(value))), float(value)) / 10.0


def state_features(observation, shape, npu, phase):
    """Only information available at the decision boundary enters the network."""
    features = []
    for queue in (observation["waiting"], observation["active"]):
        features.append(_log(len(queue)))
        for key, bounds in (("prompt_len", (256, 512, 1024, 2048)), ("kv_len", (256, 512, 1024, 2048)), ("age_s", (0.1, 1, 10, 100))):
            values = [row[key] for row in queue]
            features.extend((_log(sum(values) / len(values)) if values else 0.0, _log(max(values)) if values else 0.0))
            features.extend(sum(value <= edge for value in values) / len(values) if values else 0.0 for edge in bounds)
        features.extend(_log(sum(row[key] for row in queue) / len(queue)) if queue else 0.0 for key in ("max_tokens", "generated"))
    for device in (npu, "PIM0", "PIM1"):
        features.extend((
            _log(observation["device_available_s"].get(device, observation["now"]) - observation["now"]),
            _log(observation["weight_cache_bytes"].get(device, 0) / 2**30),
            _log(observation["weight_cache_capacity_bytes"].get(device, 0) / 2**30),
        ))
    for device in ("PIM0", "PIM1"):
        features.extend(_log(observation[key].get(device, 0) / 2**30) for key in ("kv_reserved_bytes_by_pim", "kv_actual_bytes_by_pim"))
    features.extend(_log(observation.get(key, 0)) for key in ("recent_arrivals_1s", "recent_arrivals_10s", "recent_arrivals_60s", "queue_growth_per_s", "recent_service_tokens_per_s"))
    features.extend(_log(value) for value in (shape.layer_num, shape.dim, shape.n_heads, shape.n_kv_heads, shape.head_dim, 2 * (shape.n_kv_heads // 2) * shape.head_dim * shape.layer_num * 2))
    features.extend((float(phase == "prefill"), float(phase == "decode")))
    return features


def policy_inputs(observation, evaluator, phase, current_program=None):
    compiler, shape = evaluator.compiler, evaluator.shape
    state = state_features(observation, shape, compiler.npu, phase)
    admitted = observation["waiting"][:max(0, 16 - len(observation["active"]))]
    queue = admitted if phase == "prefill" else [*observation["active"], *admitted]
    tokens = sum(row["prompt_len"] if phase == "prefill" else 1 for row in queue)
    attention_pairs = sum(row["prompt_len"] * (row["prompt_len"] + 1) / 2 if phase == "prefill" else (row["kv_len"] if row["generated"] else row["prompt_len"]) + 1 for row in queue)
    nodes, masks = [], []
    for block_index, block in enumerate(compiler.blocks):
        block_features, block_masks = [], []
        for node_index, node_id in enumerate(block.node_ids):
            node = compiler.dag.nodes[node_id]
            # The semantic position is shared across layers and model sizes.
            role = node_id.split("_", 1)[1].split("_s")[0].upper()
            role = role.removeprefix("FFN_")
            if role.startswith("LN"):
                role = "LN"
            if role.startswith("ADD"):
                role = "ADD"
            role_index = OP_TYPES.index(role)
            owner = compiler.pim_owner[node_id]
            cached = observation.get("weight_cache_keys", {})
            values = [float(i == role_index) for i in range(len(OP_TYPES))]
            values.extend((
                block.layer / shape.layer_num, float(block.kind == "ffn"),
                node.attrs.get("head_shard", node.attrs.get("ffn_shard", -1)),
                float(owner == "PIM1"), _log(node.weight_size / 2**20),
                _log(tokens), _log(attention_pairs),
                float(node.weight_id in cached.get(compiler.npu, [])),
                float(node.weight_id in cached.get(owner, [])),
                float(current_program[block_index][node_index]) if current_program is not None else 0.5,
            ))
            block_features.append(values)
            block_masks.append(compiler.mask[node_id])
        nodes.append(block_features)
        masks.append(block_masks)
    return {"state": state, "nodes": nodes, "masks": masks}


class PlacementActor(nn.Module):
    """Shared 128-wide block GRU; one categorical decision per free node."""

    def __init__(self, state_dim, node_dim, hidden=128):
        super().__init__()
        self.dimensions = {"state_dim": state_dim, "node_dim": node_dim, "hidden": hidden}
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh())
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.Tanh())
        self.action_embedding = nn.Embedding(2, hidden)
        self.gru = nn.GRUCell(hidden, hidden)
        self.actor_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))
        self.value_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def sequence(self, inputs, actions=None, *, deterministic=False, preceding_phase=None, value_features=False):
        device = next(self.parameters()).device
        hidden = self.state_encoder(torch.as_tensor(inputs["state"], dtype=torch.float32, device=device))
        preceding = torch.zeros_like(hidden) if preceding_phase is None else preceding_phase
        program, log_probs, entropies, values, summaries, features = [], [], [], [], [], []
        for index, (node_rows, mask_rows) in enumerate(zip(inputs["nodes"], inputs["masks"])):
            nodes = self.node_encoder(torch.as_tensor(node_rows, dtype=torch.float32, device=device))
            mask = torch.as_tensor(mask_rows, dtype=torch.bool, device=device)
            logits = self.actor_head(torch.cat((nodes, hidden.expand_as(nodes)), dim=-1)).masked_fill(~mask, -torch.inf)
            distribution = Categorical(logits=logits)
            chosen = (logits.argmax(-1) if deterministic else distribution.sample()) if actions is None else torch.as_tensor(actions[index], dtype=torch.long, device=device)
            if not mask.gather(1, chosen[:, None]).all():
                raise ValueError("teacher-forced action violates the node mask")
            value_input = torch.cat((hidden, nodes.mean(0), preceding))
            values.append(self.value_head(value_input).squeeze(-1))
            if value_features:
                features.append(value_input)
            log_probs.append(distribution.log_prob(chosen).sum())
            entropies.append(distribution.entropy().sum())
            program.append(chosen.detach().cpu().tolist())
            summary = (nodes + self.action_embedding(chosen)).mean(0)
            summaries.append(summary)
            hidden = self.gru(summary, hidden)
        return {
            "program": program, "log_probs": torch.stack(log_probs),
            "entropies": torch.stack(entropies), "values": torch.stack(values),
            "summary": torch.stack(summaries).mean(0),
            **({"value_features": torch.stack(features)} if value_features else {}),
        }

    def pair(self, inputs, actions=None, *, deterministic=False, value_features=False):
        prefill = self.sequence(inputs["prefill"], None if actions is None else actions["prefill"], deterministic=deterministic, value_features=value_features)
        decode = self.sequence(inputs["decode"], None if actions is None else actions["decode"], deterministic=deterministic, preceding_phase=prefill["summary"], value_features=value_features)
        return {
            "programs": {"prefill": prefill["program"], "decode": decode["program"]},
            **{key: torch.cat((prefill[key], decode[key])) for key in ("log_probs", "entropies", "values")},
            **({"value_features": torch.cat((prefill["value_features"], decode["value_features"]))} if value_features else {}),
        }


class LearnedPlacement:
    def __init__(self, actor, evaluator, *, deterministic=True):
        self.actor, self.evaluator, self.deterministic = actor, evaluator, deterministic
        self.last = None

    def __call__(self, observation, current_plans):
        compiler = self.evaluator.compiler
        inputs = {
            phase: policy_inputs(observation, self.evaluator, phase, compiler.project(current_plans[phase]["device_by_node"]) if current_plans else None)
            for phase in ("prefill", "decode")
        }
        with torch.no_grad():
            result = self.actor.pair(inputs, deterministic=self.deterministic)
        self.last = {"inputs": inputs, "programs": result["programs"], "log_probs": result["log_probs"].cpu().tolist(), "values": result["values"].cpu().tolist()}
        return {phase: compiler.compile(program) for phase, program in result["programs"].items()}


class StaticPlacementActor(nn.Module):
    """Node-autoregressive BC actor; full GRU sequences accelerate teacher forcing."""

    def __init__(self, state_dim, node_dim, hidden=256, depth=4):
        super().__init__()
        self.dimensions = {"state_dim": state_dim, "node_dim": node_dim, "hidden": hidden, "depth": depth}

        def encoder(input_dim):
            layers = []
            for index in range(depth):
                layers.extend((nn.Linear(input_dim if index == 0 else hidden, hidden), nn.Tanh()))
            return nn.Sequential(*layers)

        self.state_encoder = encoder(state_dim)
        self.node_encoder = encoder(node_dim)
        self.gru = nn.GRU(hidden + 3, hidden, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 3))

    def forward(self, state, nodes, actions, masks, learned):
        initial = self.state_encoder(state)
        vectors = self.node_encoder(nodes)
        history = torch.cat((vectors, torch.nn.functional.one_hot(actions, 3).to(vectors.dtype)), dim=-1)
        after, _ = self.gru(history, initial.unsqueeze(0))
        # Current action must not enter its own conditional prediction.
        before = torch.cat((initial.unsqueeze(1), after[:, :-1]), dim=1)
        logits = self.actor_head(torch.cat((vectors, before), dim=-1)).masked_fill(~masks, -torch.inf)
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), actions, reduction="none")
        losses = (nll * learned).sum(1) / learned.sum(1)
        return losses, logits

    @torch.jit.export
    def generate(self, state: torch.Tensor, nodes: torch.Tensor, masks: torch.Tensor,
                 fixed_from: torch.Tensor) -> torch.Tensor:
        return self.sample(state, nodes, masks, fixed_from, True)

    @torch.jit.export
    def sample(self, state: torch.Tensor, nodes: torch.Tensor, masks: torch.Tensor,
               fixed_from: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        hidden = self.state_encoder(state)
        vectors = self.node_encoder(nodes)
        width = vectors.size(-1)
        # Batch the node-only affine terms; retain the same GRU/head parameters and equations.
        input_terms = torch.nn.functional.linear(vectors, self.gru.weight_ih_l0[:, :width], self.gru.bias_ih_l0)
        action_terms = self.gru.weight_ih_l0[:, width:].transpose(0, 1)
        head_terms = torch.nn.functional.linear(vectors, self.actor_head[0].weight[:, :width], self.actor_head[0].bias)
        actions = torch.zeros((nodes.size(0), nodes.size(1)), dtype=torch.long, device=nodes.device)
        for index in range(nodes.size(1)):
            parent = int(fixed_from[index])
            if parent == -2:
                chosen = torch.zeros(nodes.size(0), dtype=torch.long, device=nodes.device)
            elif parent >= 0:
                chosen = actions[:, parent]
            else:
                head = head_terms[:, index] + torch.nn.functional.linear(hidden, self.actor_head[0].weight[:, width:])
                logits = self.actor_head[2](torch.tanh(head))
                logits = logits.masked_fill(~masks[:, index], -torch.inf)
                if deterministic:
                    chosen = logits.argmax(-1)
                else:
                    chosen = torch.multinomial(logits.softmax(-1), 1).squeeze(-1)
            actions[:, index] = chosen
            incoming = input_terms[:, index] + action_terms[chosen]
            recurrent = torch.nn.functional.linear(hidden, self.gru.weight_hh_l0, self.gru.bias_hh_l0)
            ir, iz, inn = incoming.chunk(3, dim=-1)
            hr, hz, hn = recurrent.chunk(3, dim=-1)
            reset = torch.sigmoid(ir + hr)
            update = torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = candidate + update * (hidden - candidate)
        return actions


def migrate_bc_actor(checkpoint):
    dimensions = {**checkpoint["dimensions"], "state_dim": 56}
    actor = StaticPlacementActor(**dimensions)
    weights = dict(checkpoint["actor"])
    first = weights["state_encoder.0.weight"]
    weights["state_encoder.0.weight"] = torch.cat((first, first.new_zeros(first.size(0), 9)), dim=1)
    actor.load_state_dict(weights)
    return actor


class StaticPlacementCritic(nn.Module):
    """Separate BC-initialized encoder/GRU, predicting value before each action."""

    def __init__(self, actor):
        super().__init__()
        import copy
        self.state_encoder = copy.deepcopy(actor.state_encoder)
        self.node_encoder = copy.deepcopy(actor.node_encoder)
        self.gru = copy.deepcopy(actor.gru)
        self.value_head = nn.Linear(2 * actor.dimensions["hidden"], 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        self.requires_grad_(True)

    def forward(self, state, nodes, actions):
        initial = self.state_encoder(state)
        vectors = self.node_encoder(nodes)
        history = torch.cat((vectors, torch.nn.functional.one_hot(actions, 3).to(vectors.dtype)), dim=-1)
        after, _ = self.gru(history, initial.unsqueeze(0))
        before = torch.cat((initial.unsqueeze(1), after[:, :-1]), dim=1)
        return self.value_head(torch.cat((vectors, before), dim=-1)).squeeze(-1)
