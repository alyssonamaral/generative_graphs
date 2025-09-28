
import numpy as np
import networkx as nx

from get_decision import decisions_from_adj, ORDERINGS  

SEED = 42
PRINT_SEQS = True  

def seq_len_expected(A: np.ndarray) -> int:
    """
    Para grafo não-dirigido (adj 0/1 simétrica):
    comprimento = 2N + 2m + 1
    onde m = número de arestas (contadas uma vez).
    """
    N = A.shape[0]
    m = int(np.triu(A, 1).sum())
    return 2 * N + 2 * m + 1

def is_perm(order, N) -> bool:
    """Checa se 'order' é uma permutação de 0..N-1."""
    return len(order) == N and len(set(order)) == N and set(order) == set(range(N))

def order_from_seq(seq):
    """
    Extrai a ordem implícita pela sequência:
    toda vez que aparece um AddNode (0) começando um bloco, é um novo nó.
    (útil para debugar se a ordem que o conversor está usando é válida)
    """
    order_indices = []
    i = 0
    node_idx = 0
    while i < len(seq):
        if seq[i] != 0:
            # esperamos 0 para AddNode
            i += 1
            continue
        # AddNode
        order_indices.append(node_idx)
        i += 1
        # consumir possíveis pares (0, dest) até StopEdge (1)
        while i < len(seq) and seq[i] == 0:
            i += 2  # pula AddEdge + dest
        if i < len(seq) and seq[i] == 1:
            i += 1  # StopEdge
        node_idx += 1
    # o último 1 (StopNode) fica no final
    return order_indices

def build_cycle_adj(N: int) -> np.ndarray:
    A = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        j = (i + 1) % N
        A[i, j] = 1
        A[j, i] = 1
    return A

def run_checks_on_adj(A: np.ndarray, name: str):
    print(f"\n=== Teste em '{name}' ===")
    N = A.shape[0]
    Lexp = seq_len_expected(A)
    print(f"N={N} | m={int(np.triu(A,1).sum())} | len esperado = {Lexp}")

    for ordname in ["degree","bfs","dfs","degeneracy","random","mcs","lexbfs","spectral"]:
        if ordname not in ORDERINGS:
            continue
        try:
            seq = decisions_from_adj(A, ordering=ordname, seed=SEED)
        except Exception as e:
            print(f"{ordname:10s} -> ERRO: {e}")
            continue

        L = len(seq)
        ok = (L == Lexp)
        print(f"{ordname:10s} -> len={L:2d} | ok_len={ok}")

        if not ok and PRINT_SEQS:
            print("  seq =", seq)

        ord_impl = order_from_seq(seq)
        ord_ok = (len(ord_impl) == N)
        if not ord_ok:
            print(f"  atenção: ordem implícita tem {len(ord_impl)} nós (esperado {N})")

def main():
    # 1) ciclo N=6 → comprimento deve ser 25 para QUALQUER ordenação
    A_cycle6 = build_cycle_adj(6)
    run_checks_on_adj(A_cycle6, "ciclo_N6")

    # 2) grafo aleatório pequeno (simétrico 0/1), só para sanidade
    rng = np.random.default_rng(SEED)
    N = 8
    P = 0.25
    M = rng.random((N, N))
    M = np.triu((M < P).astype(np.int64), 1)
    A_rand = M + M.T
    run_checks_on_adj(A_rand, "random_N8_p0.25")

if __name__ == "__main__":
    main()
