"""BC-initialized node actor and independent critic for BurstGPT."""
from __future__ import annotations

import torch
from torch import nn


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
