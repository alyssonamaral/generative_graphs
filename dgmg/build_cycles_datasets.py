# build_cycles_datasets.py
# Gera ciclos (adjacências) e converte para sequências DGMG por ordenação.
# Salva um .p por ordenação em runs/datasets/

import os
import sys
import argparse
import pickle
import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gds.get_decision import decisions_from_adj, ORDERINGS

def make_cycle_adj(n: int) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        j = (i + 1) % n
        A[i, j] = 1
        A[j, i] = 1
    return A

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-size", type=int, default=5)
    p.add_argument("--max-size", type=int, default=20)
    p.add_argument("--num-graphs", type=int, default=4000)
    p.add_argument("--orderings", type=str, default="degree,bfs,dfs,degeneracy,random,mcs,lexbfs,spectral",
                   help="lista separada por vírgula; deve existir em ORDERINGS do get_decision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="./runs/datasets")
    args = p.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # 1) gera adjacências de ciclos
    adjs = []
    sizes = []
    for _ in range(args.num_graphs):
        n = rng.randint(args.min_size, args.max_size)
        A = make_cycle_adj(n)
        adjs.append(A)
        sizes.append(n)
    print(f"Gerados {len(adjs)} ciclos. Tamanho médio: {sum(sizes)/len(sizes):.2f}")

    # 2) por ordenação, converte para sequências e salva .p
    orders = [s.strip() for s in args.orderings.split(",") if s.strip()]
    for ordname in orders:
        if ordname not in ORDERINGS:
            print(f" [!] ignorando '{ordname}' (não existe em ORDERINGS)")
            continue
        seqs = [decisions_from_adj(A, ordering=ordname, seed=args.seed) for A in adjs]
        outp = os.path.join(args.outdir, f"cycles_{ordname}.p")
        with open(outp, "wb") as f:
            pickle.dump(seqs, f)
        print(f" [+] salvo {len(seqs)} seqs em {outp}")

if __name__ == "__main__":
    main()
