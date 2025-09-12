import math
import torch
import torch.nn as nn
from .positional import ring_pe

class GraphonDecoder(nn.Module):
    """Decoder posicional (size-agnostic) condicionado em z_g e nas posições s."""
    def __init__(self, pe_dim: int, z_g_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
        nn.Linear(2*pe_dim + 2 + z_g_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)
        )
        self.pe_dim = pe_dim
        self.z_g_dim = z_g_dim


    def forward(self, z_g: torch.Tensor, s: torch.Tensor, K: int = 2) -> torch.Tensor:
        """
        z_g: [1, z_g_dim]
        s: [N]
        return: logits [N,N]
        """
        N = s.size(0)
        pe = ring_pe(s, K=K) # [N, 2K]
        pe_i = pe[:, None, :].expand(N, N, -1)
        pe_j = pe[None, :, :].expand(N, N, -1)
        diff = s[:, None] - s[None, :]
        diff = diff - diff.floor() # wrap para [0,1)
        diff = torch.minimum(diff, 1.0 - diff)
        diff_pe = torch.stack([torch.cos(2*math.pi*diff), torch.sin(2*math.pi*diff)], dim=-1) # [N,N,2]
        zrep = z_g.view(1,1,-1).expand(N, N, -1)
        x = torch.cat([pe_i, pe_j, diff_pe, zrep], dim=-1) # [N,N, 2K+2K+2+z]
        logits = self.mlp(x).squeeze(-1) # [N,N]
        # opcional: simetrizar
        logits = 0.5*(logits + logits.T)
        logits.fill_diagonal_(-1e9) # evita laço
        return logits