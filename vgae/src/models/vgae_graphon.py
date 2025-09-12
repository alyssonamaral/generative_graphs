import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import SimpleGCNEncoder
from .decoder import GraphonDecoder

class VGAEGraphon(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, z_g_dim: int, pe_K: int = 2, dec_hidden: int = 64):
        super().__init__()
        self.encoder = SimpleGCNEncoder(in_dim, hidden, out_dim, z_g_dim)
        self.decoder = GraphonDecoder(pe_dim=2*pe_K, z_g_dim=z_g_dim, hidden=dec_hidden)
        self.pe_K = pe_K

    def encode(self, X, A):
        h, mu_g, logvar_g = self.encoder(X, A)
        return h, mu_g, logvar_g

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode_logits(self, z_g, s):
        return self.decoder(z_g, s, K=self.pe_K)


    @staticmethod
    def bce_logits_masked(logits, A):
        """BCE com máscara na diagonal; média por entrada."""
        N = A.size(0)
        mask = torch.ones((N,N), device=A.device, dtype=torch.bool)
        mask.fill_diagonal_(False)
        loss = F.binary_cross_entropy_with_logits(logits[mask], A[mask], reduction='mean')
        return loss


    @staticmethod
    def kl_standard_normal(mu, logvar):
        # KL(q||p) com p=N(0,I) para latente global (batch=1)
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar)


    def forward(self, X, A, s, beta_kl: float = 1.0, lambda_deg: float = 0.0):
        # Encode
        _, mu_g, logvar_g = self.encode(X, A)
        z_g = self.reparameterize(mu_g, logvar_g) # [1, z]
        # Decode
        logits = self.decode_logits(z_g, s) # [N,N]
        # Losses
        recon = self.bce_logits_masked(logits, A)
        kl = self.kl_standard_normal(mu_g, logvar_g)
        loss = recon + beta_kl * kl
        if lambda_deg > 0:
            probs = torch.sigmoid(logits).detach() # evitar puxar grad direto daqui
            deg = probs.sum(dim=1)  
            loss = loss + lambda_deg * ((deg - 2.0) ** 2).mean()
        return loss, recon.detach(), kl.detach(), logits