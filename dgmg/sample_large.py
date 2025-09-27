import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt

from model import DGMG
from cycles import simplegraph_to_nx

def sample_once(model, num_nodes):
    model.v_max = num_nodes
    model.eval()
    with torch.no_grad():
        g = model()  # forward_inference()
    return g

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="./model.pth", help="checkpoint salvo pelo main.py (torch.save(model, ...))")
    p.add_argument("--num-nodes", type=int, required=True, help="N desejado (ex.: 100)")
    p.add_argument("--save-plot", type=str, default="", help="caminho para salvar PNG opcional")
    args = p.parse_args()

    model = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    g = sample_once(model, args.num_nodes)

    nx_g = simplegraph_to_nx(g)
    print(f"Nodes: {nx_g.number_of_nodes()} | Edges: {nx_g.number_of_edges()}")

    if args.save_plot:
        pos = nx.circular_layout(nx_g)
        plt.figure(figsize=(6,6))
        nx.draw(nx_g, pos=pos, node_size=60, width=0.6, with_labels=False)
        plt.tight_layout()
        plt.savefig(args.save_plot, dpi=180)
        plt.close()
        print(f"Saved plot to {args.save_plot}")

if __name__ == "__main__":
    main()