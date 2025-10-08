# build_binary_trees_datasets.py
import argparse
import os
import pickle
import random
import numpy as np
import sys 

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gds.get_decision import decisions_from_adj, ORDERINGS

def make_random_binary_tree_adj(n: int, seed: int) -> np.ndarray:
    """
    Gera uma árvore binária (não-direcional) aleatória com n nós.
    Regra: começa no nó 0; para k=1..n-1, conecta k a um nó aleatório com grau < 3.
    Isso garante grau <= 3 para todos (raiz pode ter 2, internos até 3 contando o pai).
    """
    rng = random.Random(seed)
    if n <= 1:
        return np.zeros((1,1), dtype=np.int64)

    # Graus iniciam em 0; adj 0/1 simétrica
    deg = [0]*n
    A = np.zeros((n,n), dtype=np.int64)

    # conjunto de candidatos com grau < 3
    candidates = {0}
    for k in range(1, n):
        if not candidates:
            # fallback impossível teoricamente, mas por segurança, recomeça
            return make_random_binary_tree_adj(n, seed+1)
        parent = rng.choice(list(candidates))
        # liga parent - k
        A[parent, k] = 1
        A[k, parent] = 1
        deg[parent] += 1
        deg[k] += 1
        # atualiza candidatos
        if deg[parent] >= 3 and parent in candidates:
            candidates.remove(parent)
        # recém-adicionado pode receber até 2 filhos ainda
        if deg[k] < 3:
            candidates.add(k)
    return A

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min-size', type=int, default=5)
    p.add_argument('--max-size', type=int, default=20)
    p.add_argument('--num-graphs', type=int, default=4000)
    p.add_argument('--orderings', type=str, default='degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral')
    p.add_argument('--outdir', type=str, default='./runs_bin/datasets')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    orderings = [o.strip() for o in args.orderings.split(',') if o.strip()]

    # cria um "bucket" de listas por ordenação
    buckets = {o: [] for o in orderings}

    for k in range(args.num_graphs):
        n = random.Random(args.seed + 31*k).randint(args.min_size, args.max_size)
        A = make_random_binary_tree_adj(n, seed=args.seed + 101*k)
        for o in orderings:
            seq = decisions_from_adj(A, ordering=o, seed=args.seed + 7*k)
            buckets[o].append(seq)
        if (k+1) % 200 == 0:
            print(f'[{k+1}/{args.num_graphs}]')

    # salva um .p por ordenação
    for o in orderings:
        path = os.path.join(args.outdir, f'bin_trees_{o}.p')
        with open(path, 'wb') as f:
            pickle.dump(buckets[o], f)
        print(f'Salvo: {path}  (samples={len(buckets[o])})')

if __name__ == '__main__':
    main()
