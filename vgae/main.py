# main.py
# VGAE com decoder MLP, Laplacian PE, treino conservador,
# amostragem com top-k EXATO (densidade-alvo), avaliação de ciclos
# e salvamento de imagens das amostras rápidas em runs/YYYYmmdd_HHMMSS.

import math
import random
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    negative_sampling,
    remove_self_loops,
    coalesce,
    to_dense_adj,
)

import networkx as nx
import matplotlib.pyplot as plt

# -----------------------------
# Configuração global
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# =============================
# 1) Dataset 
# =============================

def cycle_graph(n: int) -> Data:
    """Constrói C_n (não-dirigido representado com duas direções em edge_index)."""
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


def make_cycle_dataset(num_graphs=4000, n_min=5, n_max=20) -> List[Data]:
    ds = []
    for _ in range(num_graphs):
        n = random.randint(n_min, n_max)
        ds.append(cycle_graph(n))
    return ds


def laplacian_positional_encoding(data: Data, k: int = 8) -> torch.Tensor:
    """
    Retorna k autovetores (menores não-triviais) do Laplaciano normalizado.
    Para grafos pequenos, eigendecomp denso é ok.
    """
    n = data.num_nodes
    A = to_dense_adj(data.edge_index, max_num_nodes=n).squeeze(0).cpu().numpy()
    d = A.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(d, 1e-8, None)))
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    evals, evecs = np.linalg.eigh(L)
    idx = np.argsort(evals)
    evecs = evecs[:, idx]

    start = 1 if n > 1 else 0  # ignora o autovetor constante
    k_eff = min(k, n - start)
    X = evecs[:, start:start + k_eff]
    if X.shape[1] < k:
        pad = np.zeros((n, k - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)
    return torch.tensor(X, dtype=torch.float)


def add_node_features_lappe(ds: List[Data], k: int = 8) -> List[Data]:
    out = []
    for g in ds:
        g.x = laplacian_positional_encoding(g, k=k)
        out.append(g)
    return out


def undirected_unique(edge_index: torch.Tensor) -> torch.Tensor:
    """Retorna apenas uma direção (i<j) das arestas não-dirigidas."""
    ei = coalesce(edge_index, num_nodes=int(edge_index.max()) + 1)
    src, dst = ei
    mask = src < dst
    return ei[:, mask]


# =============================
# 2) Modelo (Encoder + Decoder MLP)
# =============================

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
    """Decoder MLP que recebe [z_i || z_j] e retorna logit_ij."""
    def __init__(self, z_dim, hidden=128, use_bn: bool = True):
        super().__init__()
        layers = [nn.Linear(2 * z_dim, hidden), nn.ReLU()]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden))
        layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward_on_pairs(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        pairs = torch.cat([z[src], z[dst]], dim=-1)  # [E, 2d]
        logits = self.net(pairs).squeeze(-1)
        return logits

    def forward_dense_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Computa logits densos para TODOS os pares (i,j): [N,N]."""
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


# =============================
# 3) Perdas
# =============================

def recon_bce_on_edge_pairs(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    decoder: PairMLP,
    neg_ratio: float = 4.0,
    undirected_unique_pos: bool = True,
) -> torch.Tensor:
    """BCE nos pares positivos e negativos amostrados (genérico, não força ciclo)."""
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


# =============================
# 4) Treino 
# =============================

def train(
    model: VGAE_MLP,
    loader: DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    neg_ratio: float = 4.0,
    beta_kl: float = 0.01,
    beta_anneal_to: float = 0.1,
    weight_decay: float = 1e-5,
    lambda_deg: float = 0.05,  # força suavemente grau esperado ~2 (ajuste 0.02–0.1)
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    for ep in range(1, epochs + 1):
        total = 0.0
        beta = beta_kl + (beta_anneal_to - beta_kl) * (ep - 1) / max(1, epochs - 1)

        for g in loader:
            g = g.to(device)
            opt.zero_grad()

            z, mu, logvar = model(g.x, g.edge_index)

            # reconstrução (pares + negativos)
            rec = recon_bce_on_edge_pairs(
                z=z,
                pos_edge_index=g.edge_index,
                num_nodes=g.num_nodes,
                decoder=model.decoder,
                neg_ratio=neg_ratio,
                undirected_unique_pos=True,
            )

            # KL
            kl = kl_loss(mu, logvar)

            # regularizador de grau (suave e genérico)
            logits_dense = model.decoder.forward_dense_logits(z)
            probs = torch.sigmoid(logits_dense)
            probs.fill_diagonal_(0.0)
            probs = torch.maximum(probs, probs.t())  # simetriza
            deg_exp = probs.sum(dim=1)
            deg_reg = F.mse_loss(deg_exp, torch.full_like(deg_exp, 2.0))

            loss = rec + beta * kl + lambda_deg * deg_reg
            loss.backward()
            opt.step()
            total += float(loss.detach())

        if ep == 1 or ep % 5 == 0:
            print(f"epoch {ep:03d} | loss {total/len(loader):.4f} | beta={beta:.3f}")


# =============================
# 5) Amostragem 
# =============================

@torch.no_grad()
def sample_graph(
    model: VGAE_MLP,
    n_nodes: int,
    threshold: Optional[float] = None,
    expected_undirected_edges: Optional[int] = None,
    temperature: float = 1.0,
    symmetrize: bool = True,
    remove_self_loops_flag: bool = True,
) -> torch.Tensor:
    """
    - Se expected_undirected_edges for dado, seleciona EXATAMENTE k pares (i<j) com maior prob.
    - Caso contrário, usa 'threshold' (default 0.55).
    Retorna edge_index [2, E] com duas direções (não-dirigido).
    """
    model.eval()
    z_dim = model.encoder.conv_mu.out_channels
    z = torch.randn(n_nodes, z_dim, device=device)

    logits = model.decoder.forward_dense_logits(z) / float(max(1e-8, temperature))
    probs = torch.sigmoid(logits)

    if remove_self_loops_flag:
        probs.fill_diagonal_(0.0)

    if symmetrize:
        probs = torch.maximum(probs, probs.t())

    if expected_undirected_edges is not None:
        iu = torch.triu_indices(n_nodes, n_nodes, offset=1, device=probs.device)
        vals = probs[iu[0], iu[1]]  # [M], M = N*(N-1)/2
        k = int(max(1, min(expected_undirected_edges, vals.numel())))
        _, top_idx = torch.topk(vals, k)  # EXATAMENTE k pares
        A = torch.zeros((n_nodes, n_nodes), device=probs.device)
        sel_i = iu[0][top_idx]
        sel_j = iu[1][top_idx]
        A[sel_i, sel_j] = 1.0
        A[sel_j, sel_i] = 1.0
    else:
        tau = torch.tensor(0.55 if threshold is None else threshold, device=probs.device)
        A = (probs >= tau).float()
        A = torch.triu(A, diagonal=1)
        A = A + A.t()

    src, dst = A.nonzero(as_tuple=True)
    edges = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edges


# =============================
# 6) Avaliação
# =============================

def undirected_unique_from_edge_index(edge_index: torch.Tensor, n: int):
    """Deduplica (u,v) ~ (v,u) e remove self-loops. Retorna lista de pares (u<v)."""
    src, dst = edge_index
    pairs = []
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        pairs.append((a, b))
    pairs = sorted(set(pairs))
    return pairs


def is_connected_undirected(n: int, undirected_edges: List[Tuple[int, int]]) -> bool:
    if n == 0:
        return True
    adj = [[] for _ in range(n)]
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)
    # começa por um nó com grau > 0
    start = None
    for i in range(n):
        if len(adj[i]) > 0:
            start = i
            break
    if start is None:
        return n == 1  # grafo trivial
    seen = [False] * n
    stack = [start]
    seen[start] = True
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if not seen[w]:
                seen[w] = True
                stack.append(w)
    # todos os nós devem ter grau > 0 e serem alcançáveis
    return all(len(adj[i]) > 0 for i in range(n)) and all(seen[i] for i in range(n))


def is_simple_cycle(n: int, undirected_edges: List[Tuple[int, int]]) -> bool:
    if len(undirected_edges) != n:
        return False
    deg = [0] * n
    for u, v in undirected_edges:
        deg[u] += 1
        deg[v] += 1
    if any(d != 2 for d in deg):
        return False
    if not is_connected_undirected(n, undirected_edges):
        return False
    return True


@torch.no_grad()
def evaluate_cycles(
    model: VGAE_MLP,
    num_samples: int = 1000,
    n_min: int = 5,
    n_max: int = 20,
    use_expected_density: bool = True,
    temperature: float = 1.2,
    threshold: float = 0.55,
):
    cycles = 0
    stats = []
    for _ in range(num_samples):
        n = random.randint(n_min, n_max)
        if use_expected_density:
            ei = sample_graph(
                model,
                n_nodes=n,
                expected_undirected_edges=n,  # alvo ~n (ciclo-like)
                temperature=temperature,
            )
        else:
            ei = sample_graph(
                model,
                n_nodes=n,
                threshold=threshold,
                temperature=temperature,
            )
        undirected_edges = undirected_unique_from_edge_index(ei, n)
        ok = is_simple_cycle(n, undirected_edges)
        if ok:
            cycles += 1
        stats.append((n, len(undirected_edges), ok))

    total = len(stats)
    pct = 100.0 * cycles / total if total > 0 else 0.0
    print(f"\nAmostras: {total} | Ciclos detectados: {cycles} ({pct:.1f}%)")

    from collections import defaultdict
    by_n = defaultdict(lambda: {"tot": 0, "cycles": 0, "edges": 0})
    for n, e, ok in stats:
        by_n[n]["tot"] += 1
        by_n[n]["edges"] += e
        by_n[n]["cycles"] += int(ok)

    print("\nResumo por n (nós) [média de arestas não-dirigidas | % ciclos]:")
    for n in sorted(by_n):
        tot = by_n[n]["tot"]
        mean_edges = by_n[n]["edges"] / tot
        pcycle = 100.0 * by_n[n]["cycles"] / tot
        print(f"  n={n:2d} -> {mean_edges:5.2f} arestas | {pcycle:5.1f}% ciclos")


def save_graph_image(edge_index: torch.Tensor, n_nodes: int, out_path: Path, title: str = ""):
    """
    Constrói um grafo não-dirigido a partir de edge_index (com duas direções),
    e salva uma imagem (layout circular) em out_path.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # pegar pares únicos i<j
    undirected = undirected_unique_from_edge_index(edge_index, n_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(undirected)

    plt.figure(figsize=(4.5, 4.5), dpi=150)
    pos = nx.circular_layout(G)  
    nx.draw_networkx_nodes(G, pos, node_size=220, linewidths=0.5, edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis("off")
    if title:
        plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(str(out_path))
    plt.close()


# =============================
# 8) Execução
# =============================

if __name__ == "__main__":
    # 0) pasta de saída com timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / ts
    os.makedirs(run_dir, exist_ok=True)

    # 1) Dataset
    ds = make_cycle_dataset(num_graphs=4000, n_min=5, n_max=20)
    ds = add_node_features_lappe(ds, k=8)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    # 2) Modelo
    enc = Encoder(in_dim=8, hidden=64, z_dim=32, dropout=0.0).to(device)
    dec = PairMLP(z_dim=32, hidden=128, use_bn=True).to(device)
    model = VGAE_MLP(enc, dec).to(device)

    # 3) Treino
    train(
        model,
        loader,
        epochs=60,
        lr=1e-3,
        neg_ratio=4.0,
        beta_kl=0.01,
        beta_anneal_to=0.1,
        weight_decay=1e-5,
        lambda_deg=0.05,  # ajuste se quiser puxar mais/menos para grau≈2
    )

    # 4) Amostras rápidas 
    for n in [8, 12, 16]:
        # a) densidade-alvo (top-k exato)
        ei_a = sample_graph(
            model,
            n_nodes=n,
            expected_undirected_edges=n,
            temperature=1.2,
        )
        print(f"\n[n={n}] densidade-alvo ~{n} | arestas não-dirigidas ≈ {ei_a.size(1)//2}")
        out_a = run_dir / f"sample_n{n}_topk.png"
        save_graph_image(ei_a, n, out_a, title=f"Top-k exato | n={n}")

        # b) threshold fixo + temperatura
        ei_b = sample_graph(
            model,
            n_nodes=n,
            threshold=0.55,
            temperature=1.5,
        )
        print(f"[n={n}] threshold=0.55, T=1.5 | arestas não-dirigidas ≈ {ei_b.size(1)//2}")
        out_b = run_dir / f"sample_n{n}_threshold.png"
        save_graph_image(ei_b, n, out_b, title=f"Threshold 0.55, T=1.5 | n={n}")

    # 5) Avaliação em larga escala 
    evaluate_cycles(
        model,
        num_samples=1000,
        n_min=5,
        n_max=20,
        use_expected_density=True,
        temperature=1.2,
    )

    evaluate_cycles(
        model,
        num_samples=1000,
        n_min=5,
        n_max=20,
        use_expected_density=False,  # threshold fixo
        temperature=1.5,
        threshold=0.55,
    )