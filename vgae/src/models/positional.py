import math
import torch

def ring_pe(s: torch.Tensor, K: int = 2) -> torch.Tensor:
    """Positional encoding periódico para o anel.
    s: [N] em [0,1)
    return: [N, 2K]
    """
    feats = [torch.cos(2*math.pi*s), torch.sin(2*math.pi*s)]
    for k in range(2, K+1):
        feats += [torch.cos(k*2*math.pi*s), torch.sin(k*2*math.pi*s)]
    return torch.stack(feats, dim=1)