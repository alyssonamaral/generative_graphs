import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from .models.vgae_graphon import VGAEGraphon
from .models.positional import ring_pe
from .utils import ring_positions

def load_model(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt['args']
    model = VGAEGraphon(in_dim=a['in_dim'], hidden=a['hidden'], out_dim=a['hidden'],
    z_g_dim=a['z_g'], pe_K=a['pe_K'], dec_hidden=a['dec_hidden']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, a

def sample_graph(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, a = load_model(args.ckpt, device)
    N = args.num_nodes
    s = ring_positions(N).to(device)

    # amostra z_g ~ N(0,I)
    z = torch.randn(1, a['z_g'], device=device)
    logits = model.decode_logits(z, s)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    # Amostra Bernoulli simétrico, zera diagonal
    P = np.triu(probs, k=1)
    A = (np.random.rand(N, N) < P).astype(np.float32)
    A = A + A.T

    # Plot opcional
    if args.save_plot:
        G = nx.from_numpy_array(A)
        pos = {i: (np.cos(2*np.pi*i/N), np.sin(2*np.pi*i/N)) for i in range(N)}
        plt.figure(figsize=(6,6))
        nx.draw(G, pos=pos, node_size=80, with_labels=False)
        plt.tight_layout()
        os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
        plt.savefig(args.save_plot)
        plt.close()
        print(f"Saved plot to {args.save_plot}")

    # salva A se quiser
    if args.save_adj:
        np.save(args.save_adj, A)
        print(f"Saved adjacency to {args.save_adj}.npy")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--num-nodes', type=int, required=True)
    p.add_argument('--save-plot', type=str, default='')
    p.add_argument('--save-adj', type=str, default='')
    args = p.parse_args()
    sample_graph(args)