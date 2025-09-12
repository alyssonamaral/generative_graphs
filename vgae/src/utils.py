import math
import os
import random
from typing import Dict
import numpy as np
import torch
import networkx as nx


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def normalize_adj(A: torch.Tensor) -> torch.Tensor:
    """A_hat = D^{-1/2} (A + I) D^{-1/2}.
    A: [N,N] (0/1), simétrica.
    """
    N = A.size(0)
    I = torch.eye(N, device=A.device)
    A_tilde = A + I
    deg = A_tilde.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(deg.clamp(min=1.0), -0.5))

    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

def cycle_adj(n: int) -> torch.Tensor:
    A = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A

def is_cycle_adj(A: torch.Tensor) -> bool:
    """Checa se A representa um ciclo simples (invariante a rótulos)."""
    A = (A > 0.5).float()
    N = A.size(0)
    # sem loops
    if torch.any(torch.diag(A) != 0):
        return False
    deg = A.sum(dim=1)
    if not torch.allclose(deg, torch.full((N,), 2.0), atol=1e-5):
        return False
    # conectado e |E|==N
    G = nx.from_numpy_array(A.cpu().numpy())
    return nx.is_connected(G) and (G.number_of_edges() == N)

def ring_positions(n: int) -> torch.Tensor:
    """s_i = i / n, i=0..n-1."""
    return torch.arange(n, dtype=torch.float32) / float(n)

def save_report(path: str, stats: Dict[str, float]):
    lines = [f"{k}\t{(f'{v:.4f}' if isinstance(v, float) else v)}" for k, v in stats.items()]
    with open(path, "w") as f:
        f.write("\n".join(lines))