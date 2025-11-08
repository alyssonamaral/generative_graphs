# dgmg/build_ba_datasets.py
import argparse
import os
import random
import pickle
from typing import List

import numpy as np
import networkx as nx

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gds.get_decision import decisions_from_adj  

def make_ba_adj(n: int, m: int, seed: int) -> np.ndarray:
    # BA conectado para m >= 1
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    A = nx.to_numpy_array(G, dtype=int)
    # garante simetria / diag zero
    A = np.triu(A, 1)
    A = A + A.T
    np.fill_diagonal(A, 0)
    return A.astype(np.int64)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-size", type=int, default=20)
    p.add_argument("--max-size", type=int, default=60)
    p.add_argument("--m", type=int, default=2, help="nº de arestas por novo nó em BA")
    p.add_argument("--num-graphs", type=int, default=6000)
    p.add_argument("--orderings", type=str, default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral")
    p.add_argument("--outdir", type=str, default="./runs_ba/datasets")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    orderings = [o.strip() for o in args.orderings.split(",") if o.strip()]

    # acumuladores por ordenação
    buckets = {o: [] for o in orderings}

    print(f"Gerando {args.num_graphs} grafos BA com N∈[{args.min_size},{args.max_size}], m={args.m}…")
    for k in range(args.num_graphs):
        n = random.randint(args.min_size, args.max_size)
        A = make_ba_adj(n, args.m, seed=args.seed + k)
        for o in orderings:
            seq = decisions_from_adj(A, ordering=o, seed=args.seed + 13*k)
            buckets[o].append(seq)

        if (k+1) % 500 == 0:
            print(f"  [{k+1}/{args.num_graphs}]")

    # salva um .p por ordenação
    for o, seqs in buckets.items():
        fname = os.path.join(args.outdir, f"ba_{o}.p")
        with open(fname, "wb") as f:
            pickle.dump(seqs, f)
        print(f"Salvo: {fname}  (#seq={len(seqs)})")

if __name__ == "__main__":
    main()
