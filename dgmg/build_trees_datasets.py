# build_trees_datasets.py
# Gera árvores e converte para sequências DGMG por ordenação.
# Salva um .p por ordenação em runs/datasets_trees/

import os
import sys
import argparse
import pickle
import random
import numpy as np

# permite importar gds/get_decision.py rodando a partir de dgmg/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gds.get_decision import decisions_from_adj, ORDERINGS


# ---------- Gerador de árvore aleatória via sequência de Prüfer ----------
def prufer_random_tree_edges(n: int, rng: random.Random):
    """
    Gera uma árvore rotulada uniforme em {0,...,n-1} via sequência de Prüfer.
    Retorna lista de arestas (u, v).
    Complexidade: O(n log n) com heap/min-set; aqui usamos abordagem simples O(n^2) aceitável para n <= 20.
    """
    if n <= 1:
        return []

    # sequência de prufer de comprimento n-2, com rótulos em [0, n-1]
    prufer = [rng.randrange(0, n) for _ in range(n - 2)]

    degree = [1] * n
    for v in prufer:
        degree[v] += 1

    # conjunto de folhas (grau==1)
    leaves = sorted([i for i in range(n) if degree[i] == 1])

    edges = []
    for v in prufer:
        # escolhe a menor folha (ou qualquer regra determinística)
        u = leaves.pop(0)
        edges.append((u, v))
        degree[u] -= 1  # vira 0
        degree[v] -= 1
        if degree[v] == 1:
            # insere mantendo ordenação
            # (para n pequeno não compensa usar heap)
            import bisect
            bisect.insort(leaves, v)

    # sobram duas folhas
    u, w = leaves[0], leaves[1]
    edges.append((u, w))
    return edges


def make_tree_adj(n: int, seed: int) -> np.ndarray:
    """Adjacência 0/1 simétrica de uma árvore aleatória com n nós."""
    if n <= 1:
        return np.zeros((1, 1), dtype=np.int64)
    rng = random.Random(seed)
    edges = prufer_random_tree_edges(n, rng)
    A = np.zeros((n, n), dtype=np.int64)
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    return A
# ------------------------------------------------------------------------


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
    for k in range(args.num_graphs):
        n = rng.randint(args.min_size, args.max_size)
        A = make_tree_adj(n, seed=args.seed + k)  # diversidade por amostra
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