# vgae_mlp_lappe.py
import math
import random
import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    negative_sampling,
    remove_self_loops,
    coalesce,
    to_dense_adj,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# --------------------------
# 1) Dataset de exemplo (ciclos) + Laplacian PE
# --------------------------

def cycle_graph(n: int) -> Data:
    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges.append([i, j])
        edges.append([j, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    g = Data(edge_index=edge_index, num_nodes=n)
    g.edge_index, _ = remove_self_loops(g.edge_index)
    g.edge_index = coalesce(g.edge_index, num_nodes=n)
    return g

def make_cycle_dataset(num_graphs=800, n_min=5, n_max=20) -> List[Data]:
    ds = []
    for _ in range(num_graphs):
        n = random.randint(n_min, n_max)
        ds.append(cycle_graph(n))
    return ds

def laplacian_positional_encoding(data: Data, k: int = 8) -> torch.Tensor:
    """
    Retorna k autovetores (menores não-triviais) do Laplaciano normalizado.
    Para grafos pequenos, usamos eigendecomp denso (rápido o suficiente).
    """
    n = data.num_nodes
    A = to_dense_adj(data.edge_index, max_num_nodes=n).squeeze(0).cpu().numpy()
    # Graus
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-8, None)))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt  # Laplaciano normalizado

    # Autovetores: menores autovalores
    evals, evecs = np.linalg.eigh(L)  # simétrico
    idx = np.argsort(evals)
    evecs = evecs[:, idx]

    # ignorar o primeiro autovetor constante se quiser (aqui pegamos os k seguintes)
    start = 1 if n > 1 else 0
    k_eff = min(k, n - start)
    X = evecs[:, start:start + k_eff]
    # se n < k+1, completa com zeros
    if X.shape[1] < k:
        pad = np.zeros((n, k - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)
    return torch.tensor(X, dtype=torch.float)

def add_node_features_lappe(ds: List[Data], k: int = 8) -> List[Data]:
    out = []
    for g in ds:
        x = laplacian_positional_encoding(g, k=k)
        g.x = x
        out.append(g)
    return out

# ---- arestas únicas (i<j) para não duplicar perda em grafos não-dirigidos
def undirected_unique(edge_index: torch.Tensor) -> torch.Tensor:
    ei = coalesce(edge_index, num_nodes=int(edge_index.max()) + 1)
    src, dst = ei
    mask = src < dst
    return ei[:, mask]

# --------------------------
# 2) Modelo (Encoder + Decoder MLP)
# --------------------------

class Encoder(nn.Module):
    def __init__(self, in_dim=8, hidden=64, z_dim=32, dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv_mu = GCNConv(hidden, z_dim)
        self.conv_logvar = GCNConv(hidden, z_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        mu = self.conv_mu(h, edge_index)
        logvar = self.conv_logvar(h, edge_index)
        return mu, logvar

class PairMLP(nn.Module):
    def __init__(self, z_dim, hidden=128, use_bn: bool = True):
        super().__init__()
        layers = [
            nn.Linear(2 * z_dim, hidden),
            nn.ReLU(),
        ]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ]
        self.net = nn.Sequential(*layers)

    def forward_on_pairs(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        pairs = torch.cat([z[src], z[dst]], dim=-1)  # [E, 2d]
        logits = self.net(pairs).squeeze(-1)
        return logits

    def forward_dense_logits(self, z: torch.Tensor) -> torch.Tensor:
        N, d = z.shape
        Zi = z.unsqueeze(1).expand(N, N, d)
        Zj = z.unsqueeze(0).expand(N, N, d)
        pairs = torch.cat([Zi, Zj], dim=-1).reshape(N * N, 2 * d)
        logits = self.net(pairs).reshape(N, N)
        return logits

class VGAE_MLP(nn.Module):
    def __init__(self, encoder: Encoder, decoder: PairMLP):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, edge_index):
        mu, logvar = self.encoder(x, edge_index)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

# --------------------------
# 3) Perdas
# --------------------------

def recon_bce_on_edge_pairs(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    decoder: PairMLP,
    neg_ratio: float = 1.0,
    undirected_unique_pos: bool = True,
) -> torch.Tensor:
    if undirected_unique_pos:
        pos_ei = undirected_unique(pos_edge_index)
    else:
        pos_ei = pos_edge_index
    Epos = pos_ei.size(1)
    if Epos == 0:
        return torch.tensor(0.0, device=z.device)

    num_neg = max(1, int(math.ceil(Epos * neg_ratio)))
    neg_ei = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        force_undirected=True,
        method="sparse",
    )
    pos_logits = decoder.forward_on_pairs(z, pos_ei)
    neg_logits = decoder.forward_on_pairs(z, neg_ei)

    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return pos_loss + neg_loss

def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

# --------------------------
# 4) Treino
# --------------------------

def train(
    model: VGAE_MLP,
    loader: DataLoader,
    epochs: int = 60,
    lr: float = 3e-3,
    neg_ratio: float = 2.0,
    beta_kl: float = 0.01,      # KL mais fraco inicialmente
    beta_anneal_to: float = 0.1,  # opcional: anneal up
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        # annealing linear simples
        beta = beta_kl + (beta_anneal_to - beta_kl) * (ep - 1) / max(1, epochs - 1)

        for g in loader:
            g = g.to(device)
            opt.zero_grad()

            z, mu, logvar = model(g.x, g.edge_index)
            rec = recon_bce_on_edge_pairs(
                z=z,
                pos_edge_index=g.edge_index,
                num_nodes=g.num_nodes,
                decoder=model.decoder,
                neg_ratio=neg_ratio,
                undirected_unique_pos=True,
            )
            kl = kl_loss(mu, logvar)
            loss = rec + beta * kl
            loss.backward()
            opt.step()
            total += float(loss.detach())

        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:03d} | loss {total/len(loader):.4f} | beta={beta:.3f}")

# --------------------------
# 5) Amostragem (genérica, sem projetor)
# --------------------------

@torch.no_grad()
def sample_graph(
    model: VGAE_MLP,
    n_nodes: int,
    threshold: float = 0.55,
    symmetrize: bool = True,
    remove_self_loops_flag: bool = True,
) -> torch.Tensor:
    model.eval()
    z_dim = model.encoder.conv_mu.out_channels
    z = torch.randn(n_nodes, z_dim, device=device)

    logits = model.decoder.forward_dense_logits(z)
    probs = torch.sigmoid(logits)

    if remove_self_loops_flag:
        probs.fill_diagonal_(0.0)

    A = (probs >= threshold).float()
    if symmetrize:
        A = torch.triu(A, diagonal=1)
        A = A + A.t()

    src, dst = A.nonzero(as_tuple=True)
    edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edges

# --------------------------
# 6) Exemplo de uso
# --------------------------

if __name__ == "__main__":
    # 1) Dataset
    ds = make_cycle_dataset(num_graphs=4000, n_min=5, n_max=20)
    ds = add_node_features_lappe(ds, k=8)   # <— chave para quebrar a simetria
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # 2) Modelo
    enc = Encoder(in_dim=8, hidden=64, z_dim=32, dropout=0.0).to(device)
    dec = PairMLP(z_dim=32, hidden=128, use_bn=True).to(device)
    model = VGAE_MLP(enc, dec).to(device)

    # 3) Treino
    train(model, loader, epochs=60, lr=3e-3, neg_ratio=2.0, beta_kl=0.01, beta_anneal_to=0.1)

    # 4) Amostra (sem projetor)
    for n in [8, 12, 16]:
        ei = sample_graph(model, n_nodes=n, threshold=0.55)
        print(f"\nAmostra n={n} | arestas não-dirigidas ≈ {ei.size(1)//2}")