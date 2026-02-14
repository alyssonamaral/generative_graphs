from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple
import numpy as np


@dataclass(frozen=True)
class GraphSample:
    """
    Formato padrão do projeto para um grafo "raw" (antes de ordenação/seq DGMG).

    - n_nodes: número de nós
    - edges: lista de arestas não-direcionadas únicas (u, v) com u < v
    - adj: matriz de adjacência (uint8 0/1), simétrica, sem self-loop
    - root: nó raiz (se fizer sentido na família; para árvores binárias, faz)
    - meta: dicionário de metadados (família, parâmetros do gerador, etc.)
    """
    n_nodes: int
    edges: np.ndarray                 # shape (m, 2), dtype=int64, u < v
    adj: np.ndarray                   # shape (n, n), dtype=uint8
    root: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Para serialização estável (pickle/json-friendly)
        return {
            "n_nodes": int(self.n_nodes),
            "edges": np.asarray(self.edges, dtype=np.int64),
            "adj": np.asarray(self.adj, dtype=np.uint8),
            "root": int(self.root),
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GraphSample":
        return GraphSample(
            n_nodes=int(d["n_nodes"]),
            edges=np.asarray(d["edges"], dtype=np.int64),
            adj=np.asarray(d["adj"], dtype=np.uint8),
            root=int(d.get("root", 0)),
            meta=dict(d.get("meta", {})),
        )


class GraphSource(Protocol):
    """
    Interface (protocolo) de uma fonte de grafos.
    Implementações devem ser determinísticas dado um RNG/seed.
    """
    def sample(self) -> GraphSample:
        ...

    def sample_batch(self, k: int) -> List[GraphSample]:
        ...
