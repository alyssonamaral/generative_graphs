import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import utils as U

class SimpleGCNEncoder(nn.Module):
    """GCN simples (2 camadas) + pooling para latente global."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int, z_g_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.mu_g = nn.Linear(out_dim, z_g_dim)
        self.logvar_g = nn.Linear(out_dim, z_g_dim)

def forward(self, X: torch.Tensor, A: torch.Tensor):
    # A: [N,N], X: [N,in_dim]
    A_hat = U.normalize_adj(A)
    h = F.relu(A_hat @ self.lin1(X))
    h = A_hat @ self.lin2(h) # [N, out_dim]
    g = h.mean(dim=0, keepdim=True) # [1, out_dim]
    mu_g = self.mu_g(g) # [1, z_g_dim]
    logvar_g = self.logvar_g(g)
    return h, mu_g, logvar_g