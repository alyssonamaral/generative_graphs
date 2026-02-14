from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlod.sequences.actions import ActionType


@dataclass
class DGMGConfig:
    hidden_dim: int = 64
    node_init_dim: int = 64


class DGMGMinimal(nn.Module):
    """
    DGMG minimalista (teacher forcing) para sequências:
      ADD_NODE / ADD_EDGE / CHOOSE_DEST / STOP_EDGE / STOP_NODE

    Não gera nós explicitamente (ADD_NODE sempre acontece),
    mas atualiza estado do grafo e dos nós ao longo da sequência.
    """
    def __init__(self, cfg: DGMGConfig):
        super().__init__()
        self.cfg = cfg

        H = cfg.hidden_dim

        # embedding inicial de um novo nó
        self.node_init = nn.Parameter(torch.zeros(cfg.node_init_dim))
        self.node_gru = nn.GRUCell(input_size=H, hidden_size=H)

        # projeções para estado global
        self.graph_gru = nn.GRUCell(input_size=H, hidden_size=H)

        # projeta node_init_dim -> H
        self.node_init_proj = nn.Linear(cfg.node_init_dim, H)

        # Edge head: decide ADD_EDGE vs STOP_EDGE
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.ReLU(),
            nn.Linear(H, 2),
        )

        # Dest head (pointer): escolhe destino entre nós anteriores
        self.dest_q = nn.Linear(2 * H, H)   # query
        self.dest_k = nn.Linear(H, H)       # keys

        # Inicial state
        self.graph0 = nn.Parameter(torch.zeros(H))

    def forward_nll(self, actions: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
        """
        Calcula NLL total do batch por teacher forcing.

        actions: (L,) int64
        args: (L,) int64
        retorna: nll escalar (somatório)
        """
        device = actions.device
        H = self.cfg.hidden_dim

        # estado global do grafo
        g = self.graph0.to(device)

        # embeddings dos nós construídos até agora (lista de tensores H)
        node_embs = []

        cur_node = -1
        pending_edge = False

        nll = torch.zeros((), device=device)

        for a, arg in zip(actions.tolist(), args.tolist()):
            if a == -1:
                break

            at = ActionType(int(a))

            if at == ActionType.ADD_NODE:
                # cria novo nó
                cur_node += 1
                h0 = self.node_init_proj(self.node_init.to(device))  # (H,)
                node_embs.append(h0)

                # atualiza estado global (entrada = emb do novo nó)
                g = self.graph_gru(h0.unsqueeze(0), g.unsqueeze(0)).squeeze(0)

                pending_edge = False

            elif at == ActionType.ADD_EDGE:
                # Edge decision: deve prever "ADD_EDGE" (classe 0)
                if cur_node < 0:
                    raise ValueError("ADD_EDGE antes de ADD_NODE")

                cur = node_embs[cur_node]
                logits = self.edge_mlp(torch.cat([g, cur], dim=0))
                # classe 0 = ADD_EDGE, classe 1 = STOP_EDGE
                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([0], device=device))
                pending_edge = True

            elif at == ActionType.STOP_EDGE:
                # Edge decision: deve prever "STOP_EDGE" (classe 1)
                if cur_node < 0:
                    raise ValueError("STOP_EDGE antes de ADD_NODE")

                cur = node_embs[cur_node]
                logits = self.edge_mlp(torch.cat([g, cur], dim=0))
                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([1], device=device))
                pending_edge = False

            elif at == ActionType.CHOOSE_DEST:
                if not pending_edge:
                    raise ValueError("CHOOSE_DEST sem ADD_EDGE anterior.")
                if cur_node <= 0:
                    raise ValueError("CHOOSE_DEST com cur_node <= 0")

                dest = int(arg)
                if dest < 0 or dest >= cur_node:
                    raise ValueError(f"dest inválido {dest} (cur_node={cur_node})")

                cur = node_embs[cur_node]

                # pointer logits para destinos 0..cur_node-1
                prev = torch.stack(node_embs[:cur_node], dim=0)  # (cur_node, H)
                q = self.dest_q(torch.cat([g, cur], dim=0))       # (H,)
                k = self.dest_k(prev)                             # (cur_node, H)
                logits = (k @ q)                                  # (cur_node,)

                nll = nll + F.cross_entropy(logits.unsqueeze(0), torch.tensor([dest], device=device))

                # atualiza embeddings dos nós envolvidos (mensagem simples)
                # (isso aqui é propositalmente simples; melhora depois)
                chosen = prev[dest]
                new_cur = self.node_gru(chosen.unsqueeze(0), cur.unsqueeze(0)).squeeze(0)
                node_embs[cur_node] = new_cur

                # atualiza estado global com o novo emb do nó atual
                g = self.graph_gru(new_cur.unsqueeze(0), g.unsqueeze(0)).squeeze(0)

                pending_edge = False

            elif at == ActionType.STOP_NODE:
                break

            else:
                raise ValueError(f"Ação desconhecida: {a}")

        return nll

    def batch_nll(self, actions_batch: torch.Tensor, args_batch: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        actions_batch: (B, L)
        args_batch: (B, L)
        lengths: (B,)
        retorna nll_total (somatório no batch)
        """
        B = actions_batch.size(0)
        nll = torch.zeros((), device=actions_batch.device)
        for i in range(B):
            L = int(lengths[i].item())
            nll = nll + self.forward_nll(actions_batch[i, :L], args_batch[i, :L])
        return nll
