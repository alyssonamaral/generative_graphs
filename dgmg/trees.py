# trees.py
# Processo de decisão (por nó i):
# AddNode(0=add) -> 
#   se i==0: AddEdge(1=stop)
#   se i>0:  AddEdge(0=add), ChooseDest(parent in [0..i-1]), AddEdge(1=stop)
# ... repetir até decidir AddNode(1=stop) no final.

import os
import pickle
import random
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset

# -------------------- Helpers --------------------

def simplegraph_to_nx(g):
    nx_g = nx.Graph()
    nx_g.add_nodes_from(g.node_states.keys())
    # seu SimpleGraph armazena arestas duplicadas (u,v) e (v,u); vamos deduplicar
    nx_g.add_edges_from({tuple(sorted(e)) for e in g.edges})
    return nx_g

def is_tree(g) -> bool:
    size = g.num_nodes()
    if size == 0:
        return False              # evita NetworkXPointlessConcept
    nx_g = simplegraph_to_nx(g)
    return nx.is_tree(nx_g)       # True inclusive para 1 nó

# -------------------- Sequência de decisões --------------------

def get_tree_decisions(size: int, parents: List[int] | None = None) -> List[int]:
    """
    Gera a sequência de decisões para uma árvore com 'size' nós.
    Nós chegam na ordem 0..size-1. Para i>0, conecta i a um pai em [0..i-1].
    Se 'parents' for dado (len=size), usa esses pais; caso contrário, sorteia.
    Convenções de ação:
      AddNode: 0=add, 1=stop
      AddEdge: 0=add, 1=stop
      ChooseDest: índice do nó destino (pai)
    """
    assert size >= 1
    seq = []
    if parents is None:
        parents = [-1] + [random.randint(0, i-1) for i in range(1, size)]
    else:
        assert len(parents) == size and parents[0] == -1
        for i in range(1, size):
            assert 0 <= parents[i] <= i-1, "parent inválido para nó i"

    for i in range(size):
        seq.append(0)  # AddNode (add)
        if i == 0:
            # raiz: nenhuma aresta
            seq.append(1)  # AddEdge (stop)
        else:
            seq.append(0)         # AddEdge (add)
            seq.append(parents[i])# ChooseDest (pai de i)
            seq.append(1)         # AddEdge (stop)

    seq.append(1)  # AddNode (stop)
    return seq

# -------------------- Geração de dataset --------------------

def generate_dataset(v_min: int, v_max: int, n_samples: int, fname: str, seed: int | None = None):
    """
    Gera árvores aleatórias no modelo de 'random recursive tree' (RRT):
    cada nó i>0 escolhe pai uniformemente em [0..i-1].
    """
    if seed is not None:
        random.seed(seed)

    samples = []
    for _ in range(n_samples):
        size = random.randint(v_min, v_max)
        samples.append(get_tree_decisions(size))
    with open(fname, "wb") as f:
        pickle.dump(samples, f)

# -------------------- Dataset --------------------

class TreeDataset(Dataset):
    def __init__(self, fname: str):
        super().__init__()
        with open(fname, "rb") as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_single(self, batch):
        assert len(batch) == 1
        return batch[0]

    def collate_batch(self, batch):
        return batch

# -------------------- Avaliação e plots --------------------

class TreeModelEvaluation(object):
    def __init__(self, v_min: int, v_max: int, dir: str):
        super().__init__()
        self.v_min = v_min
        self.v_max = v_max
        self.dir = dir
        os.makedirs(os.path.join(self.dir, "samples"), exist_ok=True)

    def rollout_and_examine(self, model, num_samples: int):
        assert not model.training, "Chame model.eval()."

        num_total_size = 0
        num_valid_size = 0
        num_trees = 0
        num_valid = 0
        plot_times = 0
        buf = []

        for k in range(num_samples):
            sampled_graph = model()
            nx_g = simplegraph_to_nx(sampled_graph)
            graph_size = sampled_graph.num_nodes()

            num_total_size += graph_size
            valid_size = self.v_min <= graph_size <= self.v_max
            tree_ok = is_tree(sampled_graph)

            if valid_size:
                num_valid_size += 1
            if tree_ok:
                num_trees += 1
            if valid_size and tree_ok:
                num_valid += 1

            buf.append(nx_g)

            if len(buf) >= 4:
                plot_times += 1
                fig, axs = plt.subplots(2, 2, figsize=(6, 6))
                axes = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
                for j, G in enumerate(buf[:4]):
                    nx.draw_networkx(
                            G,
                            pos=nx.spring_layout(G, seed=42),  # layout padrão, reprodutível
                            with_labels=True, ax=axes[j]
                        )

                    axes[j].set_axis_off()
                plt.tight_layout()
                plt.savefig(os.path.join(self.dir, "samples", f"{plot_times:03d}.png"))
                plt.close()
                buf = []

        if buf:
            plot_times += 1
            fig, axs = plt.subplots(1, len(buf), figsize=(3*len(buf), 3))
            if len(buf) == 1:
                axs = [axs]
            for j, G in enumerate(buf):
                nx.draw_networkx(
                    G,
                    pos=nx.spring_layout(G, seed=42),  # layout padrão, reprodutível
                    with_labels=True, ax=axes[j]
                )

                axs[j].set_axis_off()
            plt.tight_layout()
            plt.savefig(os.path.join(self.dir, "samples", f"{plot_times:03d}.png"))
            plt.close()

        self.num_samples_examined = num_samples
        self.average_size = num_total_size / num_samples
        self.valid_size_ratio = num_valid_size / num_samples
        self.tree_ratio = num_trees / num_samples
        self.valid_ratio = num_valid / num_samples

    def write_summary(self):
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else f"{v}"
        stats = {
            "num_samples": self.num_samples_examined,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "average_size": self.average_size,
            "valid_size_ratio": self.valid_size_ratio,
            "tree_ratio": self.tree_ratio,
            "valid_ratio": self.valid_ratio,
        }
        path = os.path.join(self.dir, "model_eval.txt")
        with open(path, "w") as f:
            for k, v in stats.items():
                f.write(f"{k}\t{fmt(v)}\n")
        print(f"Saved model evaluation statistics to {path}")
