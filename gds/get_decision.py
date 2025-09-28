# decisions.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Callable, Optional
import random
import numpy as np
import networkx as nx


# ===============================
# Helpers
# ===============================
def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(seed) if seed is not None else random.Random()


def _ensure_undirected_adj(A: np.ndarray) -> None:
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "Adjacência deve ser NxN."
    assert np.allclose(A, A.T), "Adjacência deve ser simétrica (grafo não-direcional)."
    assert np.all((A == 0) | (A == 1)), "Adjacência deve ser binária (0/1) para este conversor."

# ===============================

def order_by_degree(G: nx.Graph, seed: Optional[int] = 42) -> List[int]:
    """
    Ordenação por GRAU MÁXIMO com desempate aleatório.
    Como funciona:
      - Enquanto houver nós remanescentes, escolhe o(s) de maior grau no subgrafo atual.
      - Se houver empate, escolhe aleatoriamente entre os empatados (com seed).
    Boa para:
      - Grafos heterogêneos (scale-free, redes sociais) onde hubs “explicam” muitas arestas cedo.
    Observação:
      - Concentra muitas AddEdge no início (pode aumentar variância para a política de arestas).
    Complexidade:
      - O(n^2).
    """
    rng = _rng(seed)
    remaining = set(G.nodes())
    order: List[int] = []
    while remaining:
        # grau restrito aos remaining
        degs = {u: sum(1 for v in G.neighbors(u) if v in remaining) for u in remaining}
        max_deg = max(degs.values())
        candidates = [u for u, d in degs.items() if d == max_deg]
        u = rng.choice(candidates)
        order.append(u)
        remaining.remove(u)
    return order


def order_bfs_from_maxdeg(G: nx.Graph, seed: Optional[int] = 42) -> List[int]:
    """
    Ordenação por BFS (BUSCA EM LARGURA) a partir de um nó de maior grau.
    Como funciona:
      - Escolhe como raiz um nó de maior grau (desempate aleatório com seed) e faz BFS.
      - Visita por camadas (distância crescente da raiz). Em grafos desconexos, repete por componente.
    Boa para:
      - Árvores, grafos com comunidades, malhas/grades — decisões de aresta ficam mais “locais”.
    Complexidade:
      - O(n + m).
    """
    rng = _rng(seed)

    def bfs_component(start: int, visited: set) -> List[int]:
        order_local: List[int] = []
        Q = [start]
        visited.add(start)
        while Q:
            u = Q.pop(0)
            order_local.append(u)
            nbrs = [v for v in G.neighbors(u) if v not in visited]
            rng.shuffle(nbrs)  # desempate estável com seed
            for v in nbrs:
                visited.add(v)
                Q.append(v)
        return order_local

    # raiz: entre os de maior grau
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    max_deg = degs[0][1]
    roots = [u for u, d in degs if d == max_deg]
    rng.shuffle(roots)

    visited: set = set()
    order: List[int] = []

    # tenta BFS começando pelos candidatos “mais promissores”
    for r in roots:
        if r not in visited:
            order += bfs_component(r, visited)

    # cobre componentes remanescentes (se houver)
    for u in G.nodes():
        if u not in visited:
            order += bfs_component(u, visited)

    # garante unicidade/ordem final
    seen = set()
    uniq: List[int] = []
    for u in order:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    for u in G.nodes():
        if u not in seen:
            uniq.append(u)
    return uniq


def order_dfs_from_maxdeg(G: nx.Graph, seed: Optional[int] = 42) -> List[int]:
    """
    Ordenação por DFS (BUSCA EM PROFUNDIDADE) a partir de um nó de maior grau.
    Como funciona:
      - Escolhe raiz entre os nós de maior grau (empate aleatório com seed).
      - Executa DFS iterativa (stack), marcando visitados para evitar duplicatas.
      - Cobre componentes desconexas repetindo o processo para nós ainda não visitados.
    Boa para:
      - Árvores profundas, cadeias; quando uma linearização por caminhos é desejada.
    Complexidade:
      - O(n + m).
    """
    rng = random.Random(seed)
    # escolher raiz entre maiores graus
    degs = sorted(G.degree, key=lambda x: x[1], reverse=True)
    max_deg = degs[0][1] if degs else 0
    roots = [u for u, d in degs if d == max_deg] or list(G.nodes())
    start = rng.choice(roots)

    visited: set = set()
    order: List[int] = []

    def dfs_from(s: int):
        stack = [s]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            order.append(u)
            nbrs = [v for v in G.neighbors(u) if v not in visited]
            rng.shuffle(nbrs)           # desempate estável
            stack.extend(nbrs)          # LIFO => profundidade

    # componente do start
    dfs_from(start)
    # cobre componentes restantes (se houver)
    for u in G.nodes():
        if u not in visited:
            dfs_from(u)

    # sanity: deve ser permutação
    assert len(order) == G.number_of_nodes() and len(set(order)) == G.number_of_nodes(), \
        "DFS ordering must be a permutation of nodes"
    return order


def order_degeneracy_smallest_last(G: nx.Graph) -> List[int]:
    """
    Ordenação SMALLest-LAST (degeneracy ordering).
    Como funciona:
      - Remove iterativamente o nó de MENOR grau, empilhando; no final, inverte a pilha.
      - Relacionada ao k-core / k-degeneracy; nós “difíceis” tendem a aparecer primeiro na ordem final.
    Boa para:
      - Grafos esparsos em geral; costuma produzir sequências estáveis e “bem comportadas”.
    Complexidade:
      - Com estruturas de apoio adequadas, O(n + m).
    """
    Gm = G.copy()
    stack: List[int] = []
    while Gm.number_of_nodes() > 0:
        u = min(Gm.nodes(), key=lambda x: Gm.degree[x])
        stack.append(u)
        Gm.remove_node(u)
    return list(reversed(stack))


def order_random(G: nx.Graph, seed: Optional[int] = 42) -> List[int]:
    """
    Ordenação ALEATÓRIA com semente.
    Como funciona:
      - Permutação aleatória dos nós.
    Boa para:
      - Baseline/controle; quando você quer medir o ganho de outras ordenações.
    Complexidade:
      - O(n).
    """
    rng = _rng(seed)
    order = list(G.nodes())
    rng.shuffle(order)
    return order


def order_mcs(G: nx.Graph, seed: Optional[int] = 42) -> List[int]:
    """
    Maximum Cardinality Search (MCS).
    Como funciona:
      - Iterativamente escolhe o nó com MAIOR nº de vizinhos já selecionados (empates aleatórios).
      - Clássica para grafos chordais (encontra ordem perfeita em chordal).
    Boa para:
      - Grafos chordais ou próximos; estrutura de cliques/triangulação.
    Complexidade:
      - O(n^2).
    """
    rng = _rng(seed)
    selected: List[int] = []
    remaining = set(G.nodes())
    scores: Dict[int, int] = {u: 0 for u in G.nodes()}

    while remaining:
        max_score = max(scores[u] for u in remaining)
        candidates = [u for u in remaining if scores[u] == max_score]
        u = rng.choice(candidates)
        selected.append(u)
        remaining.remove(u)
        # atualiza scores: +1 para vizinhos de u que ainda não foram selecionados
        for v in G.neighbors(u):
            if v in remaining:
                scores[v] += 1

    return selected


def order_lex_bfs(G: nx.Graph) -> List[int]:
    """
    Lexicographic BFS (Lex-BFS).
    Como funciona:
      - Variante determinística da BFS usando rótulos lexicográficos de conjuntos.
      - Também encontra ordens perfeitas para grafos chordais.
    Boa para:
      - Estruturas de clique e separadores mínimos; ordens muito estáveis.
    Complexidade:
      - O(n + m).
    """
    # Implementação simples via networkx (quando disponível)
    # Fallback manual mínimo:
    try:
        return list(nx.lexicographical_bfs_ordering(G))
    except AttributeError:
        # Fallback determinístico básico
        labels: Dict[int, List[int]] = {u: [] for u in G.nodes()}
        order: List[int] = []
        remaining = set(G.nodes())
        while remaining:
            # escolhe com maior rótulo lexicográfico
            u = max(remaining, key=lambda x: labels[x])
            order.append(u)
            remaining.remove(u)
            for v in G.neighbors(u):
                if v in remaining:
                    labels[v].append(len(order))  # aumenta “prioridade” de vizinhos visitados
        return order


def order_spectral_fiedler(G: nx.Graph) -> List[int]:
    """
    Ordenação ESPECTRAL pelo vetor de Fiedler (2º menor autovetor do Laplaciano).
    Como funciona:
      - Constrói L = D - A em NumPy (denso), calcula autovetores com np.linalg.eigh.
      - Se desconexo, usa a maior componente conectada para evitar multiplicidade de zero.
    Boa para:
      - Grafos com geometria/comunidades; grades, malhas, redes com clusters.
    Complexidade:
      - Decomposição densa: O(n^3) — ok para grafos pequenos/médios.
    """
    import numpy as np

    if G.number_of_nodes() <= 2:
        return list(G.nodes())

    # usar maior componente conectada para definir Fiedler com mais estabilidade
    if not nx.is_connected(G):
        C = max(nx.connected_components(G), key=len)
        H = G.subgraph(C).copy()
    else:
        H = G

    # Laplaciano denso: L = D - A
    A = nx.to_numpy_array(H, dtype=float)
    deg = A.sum(axis=1)
    L = np.diag(deg) - A

    try:
        vals, vecs = np.linalg.eigh(L)
        # índices ordenados por autovalor crescente
        order_idx = np.argsort(vals)
        # pega o 2º menor se existir; caso contrário, o 1º
        idx = order_idx[1] if len(order_idx) > 1 else order_idx[0]
        f = vecs[:, idx]
        order_sub = [node for _, node in sorted(zip(f, H.nodes()), key=lambda x: x[0])]
    except Exception:
        # fallback simples: grau crescente
        order_sub = sorted(H.nodes(), key=lambda u: H.degree[u])

    if H.number_of_nodes() != G.number_of_nodes():
        rest = [u for u in G.nodes() if u not in H.nodes()]
        order_sub += rest

    # sanity
    assert len(order_sub) == G.number_of_nodes() and len(set(order_sub)) == G.number_of_nodes(), \
        "Spectral ordering must be a permutation"
    return order_sub


# Map de nomes -> função
ORDERINGS: Dict[str, Callable[..., List[int]]] = {
    "degree": order_by_degree,
    "bfs": order_bfs_from_maxdeg,
    "dfs": order_dfs_from_maxdeg,
    "degeneracy": order_degeneracy_smallest_last,
    "random": order_random,
    "mcs": order_mcs,
    "lexbfs": order_lex_bfs,
    "spectral": order_spectral_fiedler,
}


# ===============================
# Conversão: adjacência -> sequência DGMG
# ===============================
def decisions_from_adj(
    A: np.ndarray,
    ordering: str = "degree",
    seed: Optional[int] = 42,
    neighbor_order: str = "as_is",
) -> List[int]:
    """
    Converte uma adjacência (0/1, simétrica) na SEQUÊNCIA de decisões no estilo DGMG.

    Protocolo (para cada nó i na ordem π):
      - 0          -> AddNode
      - para cada vizinho v com posição pos[v] < i (v já inserido antes):
            0      -> AddEdge
            pos[v] -> ChooseDest (índice do destino dentre {0..i-1})
      - 1          -> Stop AddEdge
    Ao final: 1     -> Stop AddNode

    Parâmetros:
      A: NxN (0/1, simétrica)
      ordering: nome de uma ordenação em ORDERINGS
      seed: controla desempates nas ordenações estocásticas
      neighbor_order: "as_is" | "by_degree_desc"
                      (a ordem em que conectamos aos vizinhos anteriores)

    Retorna:
      Lista de inteiros representando as ações (para o seu DGMG).
    """
    _ensure_undirected_adj(A)
    N = A.shape[0]
    G = nx.from_numpy_array(A)

    if ordering not in ORDERINGS:
        raise ValueError(f"ordering '{ordering}' não suportado. Opções: {list(ORDERINGS.keys())}")
    order_fn = ORDERINGS[ordering]
    pi = order_fn(G, seed=seed) if order_fn.__code__.co_argcount >= 2 else order_fn(G)

    pos = {node: i for i, node in enumerate(pi)}

    seq: List[int] = []
    for i, u in enumerate(pi):
        # AddNode
        seq.append(0)

        # vizinhos que já apareceram (pos[v] < i)
        prev_neighbors = [v for v in G.neighbors(u) if pos[v] < i]
        if neighbor_order == "by_degree_desc":
            prev_neighbors = sorted(prev_neighbors, key=lambda x: G.degree[x], reverse=True)

        for v in prev_neighbors:
            seq.append(0)            # AddEdge
            seq.append(pos[v])       # ChooseDest (índice do destino entre {0..i-1})

        seq.append(1)                # Stop AddEdge

    seq.append(1)                    # Stop AddNode
    return seq


def build_sequences_from_adjs(
    list_of_A: List[np.ndarray],
    ordering: str = "degree",
    seed: Optional[int] = 42,
    neighbor_order: str = "as_is",
) -> List[List[int]]:
    """
    Constrói uma lista de sequências de decisões a partir de várias adjacências.
    Útil para gerar um dataset (ex.: salvar via pickle).
    """
    return [decisions_from_adj(A, ordering=ordering, seed=seed, neighbor_order=neighbor_order)
            for A in list_of_A]
