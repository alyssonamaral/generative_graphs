# build_star_datasets.py
import argparse, os, pickle, random, numpy as np
import networkx as nx
import sys

# garante import do gds.get_decision
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from gds.get_decision import decisions_from_adj

def make_star_adj(n: int, seed: int = 0) -> np.ndarray:
    """
    Retorna a adjacência de uma estrela com n nós.
    networkx.star_graph(k) cria k+1 nós (0..k); aqui queremos n nós totais.
    """
    assert n >= 2
    # centro 0 conectado a 1..n-1
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for v in range(1, n):
        G.add_edge(0, v)
    # opcional: aleatorizar rótulos para não fixar centro=0
    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    mapping = {i: perm[i] for i in range(n)}
    G = nx.relabel_nodes(G, mapping, copy=True)
    A = nx.to_numpy_array(G, dtype=np.int64)
    return A

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--max-size', type=int, default=20)
    p.add_argument('--num-graphs', type=int, default=4000)
    p.add_argument('--orderings', type=str, default='degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral')
    p.add_argument('--outdir', type=str, default='./runs_star/datasets')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    orderings = [o.strip() for o in args.orderings.split(',') if o.strip()]
    buckets = {o: [] for o in orderings}

    rng = random.Random(args.seed)
    for k in range(args.num_graphs):
        n = rng.randint(args.min_size, args.max_size)
        A = make_star_adj(n, seed=args.seed + 101*k)
        for o in orderings:
            seq = decisions_from_adj(A, ordering=o, seed=args.seed + 7*k)
            buckets[o].append(seq)
        if (k+1) % 200 == 0:
            print(f'[{k+1}/{args.num_graphs}]')

    for o in orderings:
        path = os.path.join(args.outdir, f'star_{o}.p')
        with open(path, 'wb') as f:
            pickle.dump(buckets[o], f)
        print(f'Salvo: {path} (samples={len(buckets[o])})')

if __name__ == '__main__':
    main()
