# build_trees_datasets.py
# Gera árvores e converte para sequências DGMG por ordenação.
# Salva um .p por ordenação em runs/datasets_trees/

import os
import sys
import argparse
import pickle
import random
import numpy as np
import networkx as nx

# permite importar gds/get_decision.py rodando a partir de dgmg/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gds.get_decision import decisions_from_adj, ORDERINGS

def make_tree_adj(n: int, seed: int) -> np.ndarray:
    # árvore aleatória conexa com n nós (n>=2) — se n==1, retorna grafo trivial
    if n <= 1:
        A = np.zeros((1, 1), dtype=np.int64)
        return A
    G = nx.random_tree(n, seed=seed)
    A = nx.to_numpy_array(G, dtype=np.int64)
    return A

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-size", type=int, default=5)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--num-graphs", type=int, default=4000)
    p.add_argument("--orderings", type=str,
                   default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral",
                   help="lista separada por vírgula; deve existir em ORDERINGS do get_decision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./runs/datasets_trees")
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) gera adjacências de árvores
    adjs = []
    sizes = []
    for k in range(args.num-graphs):
        n = rng.randint(args.min_size, args.max_size)
        A = make_tree_adj(n, seed=args.seed + k)  # muda seed por grafo p/ diversidade
        adjs.append(A)
        sizes.append(n)
    print(f"Geradas {len(adjs)} árvores. Tamanho médio: {sum(sizes)/len(sizes):.2f}")

    # 2) por ordenação, converte para sequências e salva .p
    orders = [s.strip() for s in args.orderings.split(",") if s.strip()]
    for ordname in orders:
        if ordname not in ORDERINGS:
            print(f" [!] ignorando '{ordname}' (não existe em ORDERINGS)")
            continue
        seqs = [decisions_from_adj(A, ordering=ordname, seed=args.seed) for A in adjs]
        outp = os.path.join(args.outdir, f"trees_{ordname}.p")
        with open(outp, "wb") as f:
            pickle.dump(seqs, f)
        print(f" [+] salvo {len(seqs)} seqs em {outp}")

if __name__ == "__main__":
    main()
