from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    A_norm = D^-1/2 (A + I) D^-1/2
    adj: (n,n) float
    """
    n = adj.size(0)
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)
    deg = a.sum(dim=1)  # (n,)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    D = torch.diag(deg_inv_sqrt)
    return D @ a @ D


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, a_norm: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # (n,n) @ (n,in) -> (n,in) -> linear -> (n,out)
        h = a_norm @ x
        return self.lin(h)


class GraphEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, hid_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.gcn1 = DenseGCNLayer(in_dim, hid_dim)
        self.gcn2 = DenseGCNLayer(hid_dim, out_dim)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        adj: (n,n) float
        return: node_emb (n, d)
        """
        n = adj.size(0)
        deg = adj.sum(dim=1)  # (n,)
        deg_norm = deg / deg.clamp(min=1.0).max().clamp(min=1.0)
        x = torch.stack([torch.ones(n, device=adj.device), deg_norm], dim=1)  # (n,2)

        a_norm = normalize_adj(adj)
        h = F.relu(self.gcn1(a_norm, x))
        h = self.gcn2(a_norm, h)
        return h


class PointerDecoder(nn.Module):
    """
    Decoder tipo 'pointer' com estado recorrente:
      state_{t+1} = GRU(state_t, emb[action_t])
      logits_i = v^T tanh(W_h emb_i + W_s state_t)
    """
    def __init__(self, emb_dim: int = 64, state_dim: int = 64):
        super().__init__()
        self.gru = nn.GRUCell(input_size=emb_dim, hidden_size=state_dim)

        self.W_h = nn.Linear(emb_dim, state_dim, bias=False)
        self.W_s = nn.Linear(state_dim, state_dim, bias=True)
        self.v = nn.Linear(state_dim, 1, bias=False)

        self.state0 = nn.Parameter(torch.zeros(state_dim))

    def init_state(self, device: torch.device) -> torch.Tensor:
        return self.state0.to(device)

    def logits(self, node_emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # node_emb: (n,d), state: (d,)
        n = node_emb.size(0)
        s = state.unsqueeze(0).expand(n, -1)  # (n,d)
        z = torch.tanh(self.W_h(node_emb) + self.W_s(s))
        return self.v(z).squeeze(-1)  # (n,)

    def step_state(self, state: torch.Tensor, chosen_emb: torch.Tensor) -> torch.Tensor:
        return self.gru(chosen_emb.unsqueeze(0), state.unsqueeze(0)).squeeze(0)


@dataclass
class PolicyOutput:
    order: List[int]
    logprob_sum: torch.Tensor
    entropy_sum: torch.Tensor


class GraphOrderingPolicy(nn.Module):
    def __init__(self, emb_dim: int = 64, state_dim: int = 64):
        super().__init__()
        self.encoder = GraphEncoder(in_dim=2, hid_dim=emb_dim, out_dim=emb_dim)
        self.decoder = PointerDecoder(emb_dim=emb_dim, state_dim=state_dim)

    @torch.no_grad()
    def greedy_order(self, adj: torch.Tensor) -> List[int]:
        self.eval()
        node_emb = self.encoder(adj)
        n = adj.size(0)
        mask = torch.ones(n, dtype=torch.bool, device=adj.device)
        state = self.decoder.init_state(adj.device)

        out: List[int] = []
        for _ in range(n):
            logits = self.decoder.logits(node_emb, state)
            logits = logits.masked_fill(~mask, -1e9)
            a = int(torch.argmax(logits).item())
            out.append(a)
            mask[a] = False
            state = self.decoder.step_state(state, node_emb[a])
        return out

    def sample_order(self, adj: torch.Tensor) -> PolicyOutput:
        """
        Amostra uma permutação completa.
        Retorna soma de logprobs e entropias (útil no REINFORCE).
        """
        self.train()
        node_emb = self.encoder(adj)
        n = adj.size(0)
        mask = torch.ones(n, dtype=torch.bool, device=adj.device)
        state = self.decoder.init_state(adj.device)

        order: List[int] = []
        logprob_sum = torch.zeros((), device=adj.device)
        entropy_sum = torch.zeros((), device=adj.device)

        for _ in range(n):
            logits = self.decoder.logits(node_emb, state)
            logits = logits.masked_fill(~mask, -1e9)

            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logprob_sum = logprob_sum + dist.log_prob(a)
            entropy_sum = entropy_sum + dist.entropy()

            ai = int(a.item())
            order.append(ai)

            mask[ai] = False
            state = self.decoder.step_state(state, node_emb[ai])

        return PolicyOutput(order=order, logprob_sum=logprob_sum, entropy_sum=entropy_sum)
