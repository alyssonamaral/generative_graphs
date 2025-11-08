# dgmg/compare_eval_ba.py
import argparse
import glob
import os
import math
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from cycles import simplegraph_to_nx  # já existe no seu repo

def gini(x: np.ndarray) -> float:
    # Gini do vetor de graus (não-negativos)
    if len(x) == 0: return 0.0
    x = np.sort(x.astype(float))
    n = len(x)
    if x.sum() == 0: return 0.0
    cum = np.cumsum(x)
    return (n + 1 - 2*(cum.sum()/cum[-1])) / n

def eval_model(model, vmin, vmax, nsamples=2000, save_dir=None, save_limit=24):
    model.eval()
    count = 0
    valid_size = 0
    connected = 0

    sizes = []
    gini_list = []
    hubiness = []  # max_deg / (N-1)

    os.makedirs(save_dir, exist_ok=True) if save_dir else None
    saved = 0

    with torch.no_grad():
        for i in range(nsamples):
            g = model()
            if isinstance(g, list): g = g[0]
            nx_g = simplegraph_to_nx(g)

            N = nx_g.number_of_nodes()
            sizes.append(N)
            if vmin <= N <= vmax: valid_size += 1

            if N > 0 and nx.is_connected(nx_g):
                connected += 1

            deg = np.array([d for _, d in nx_g.degree()])
            if N > 1 and len(deg) > 0:
                gini_list.append(gini(deg))
                hubiness.append(deg.max() / (N-1))
            else:
                gini_list.append(0.0)
                hubiness.append(0.0)

            # plots opcionais (sem graphviz)
            if save_dir and saved < save_limit and N > 0:
                pos = nx.spring_layout(nx_g, seed=7)
                plt.figure(figsize=(3.2, 3.2))
                nx.draw(nx_g, pos=pos, node_size=20, width=0.4, with_labels=False)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{i:04d}.png"), dpi=160)
                plt.close()
                saved += 1

    res = {
        "average_size": float(np.mean(sizes)) if sizes else 0.0,
        "valid_size_ratio": valid_size / nsamples,
        "connected_ratio": connected / nsamples,
        "gini_degree": float(np.mean(gini_list)) if gini_list else 0.0,
        "hubiness": float(np.mean(hubiness)) if hubiness else 0.0,
    }
    return res

def eval_dir(dirpath, vmin, vmax, nsamples=2000):
    ckpt = os.path.join(dirpath, "model.pth")
    from model import DGMG  # garante que a classe existe no import path
    model = torch.load(ckpt, map_location="cpu", weights_only=False)
    return eval_model(model, vmin, vmax, nsamples=nsamples,
                      save_dir=os.path.join(dirpath, "samples_eval"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logroot", type=str, default=None,
                    help="Se vazio, pega a pasta mais recente em ./runs_ba/")
    ap.add_argument("--vmin", type=int, default=20)
    ap.add_argument("--vmax", type=int, default=60)
    ap.add_argument("--nsamples", type=int, default=2000)
    args = ap.parse_args()

    root = args.logroot
    if root is None:
        candidates = sorted(glob.glob("./runs_ba/train_orders_ba_*"))
        if not candidates:
            raise RuntimeError("Nenhuma pasta encontrada em ./runs_ba/")
        root = candidates[-1]

    print(f"\nUsando: {root}\n")
    print(f"{'ordering':12s} {'average_size':>14s} {'valid_size_ratio':>17s} {'connected':>10s} {'gini_deg':>10s} {'hubiness':>10s}")

    for o in sorted(next(os.walk(root))[1]):
        d = os.path.join(root, o)
        if not os.path.isfile(os.path.join(d, "model.pth")):
            continue
        res = eval_dir(d, args.vmin, args.vmax, nsamples=args.nsamples)
        print(f"{o:12s} {res['average_size']:14.4f} {res['valid_size_ratio']:17.4f} {res['connected_ratio']:10.4f} {res['gini_degree']:10.4f} {res['hubiness']:10.4f}")

if __name__ == "__main__":
    main()
