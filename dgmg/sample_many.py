# sample_many.py  (coloque na pasta dgmg/)
import argparse
import os
import math
import torch
import networkx as nx
import matplotlib.pyplot as plt

from cycles import simplegraph_to_nx  # só se você quiser salvar PNGs

def load_model(path):
    # PyTorch 2.6+ muda default p/ weights_only=True. Força pickle completo (é seu checkpoint).
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def sample_once(model, v_max):
    model.v_max = v_max
    model.eval()
    with torch.no_grad():
        g = model()  # forward_inference()
    return g

def maybe_save_png(g, outpath):
    nx_g = simplegraph_to_nx(g)
    pos = nx.circular_layout(nx_g)
    plt.figure(figsize=(6,6))
    nx.draw(nx_g, pos=pos, node_size=40, width=0.6, with_labels=False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=180)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="caminho do model.pth salvo pelo main.py")
    p.add_argument("--trials", type=int, default=400, help="quantas amostras gerar")
    p.add_argument("--v-max", type=int, default=100, help="teto de nós na geração")
    p.add_argument("--threshold", type=int, default=20, help="contar quantos grafos excedem este tamanho")
    p.add_argument("--save-dir", type=str, default="", help="se setado, salva PNGs dos grafos > threshold")
    p.add_argument("--save-limit", type=int, default=20, help="máximo de PNGs para salvar")
    args = p.parse_args()

    model = load_model(args.ckpt)

    sizes = []
    over = 0
    saved = 0
    max_n = 0

    for i in range(1, args.trials + 1):
        g = sample_once(model, args.v_max)
        n = g.num_nodes()
        sizes.append(n)
        max_n = max(max_n, n)
        if n > args.threshold:
            over += 1
            if args.save_dir and saved < args.save_limit:
                out = os.path.join(args.save_dir, f"sample_{i:04d}_N{n}.png")
                maybe_save_png(g, out)
                saved += 1
        if i % 50 == 0:
            print(f"[{i}/{args.trials}] avg={sum(sizes)/len(sizes):.2f} | max={max_n} | >{args.threshold} = {over}")

    avg = sum(sizes)/len(sizes)
    print("----- resumo -----")
    print(f"trials: {args.trials}")
    print(f"v_max: {args.v_max}")
    print(f"avg size: {avg:.2f}")
    print(f"max size: {max_n}")
    print(f"count > {args.threshold}: {over} ({over/args.trials:.2%})")

if __name__ == "__main__":
    main()
