# binary_trees.py
import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset

from cycles import simplegraph_to_nx  # você já tem isso

class BinaryTreeDataset(Dataset):
    def __init__(self, fname):
        super().__init__()
        with open(fname, 'rb') as f:
            self.dataset = pickle.load(f)

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def collate_single(self, batch):
        assert len(batch) == 1
        return batch[0]
    def collate_batch(self, batch):
        return batch

def is_binary_tree(nx_g: nx.Graph) -> bool:
    """
    Árvore binária (não-direcional):
    - deve ser árvore (conexa, acíclica, |E|=|V|-1)
    - grau(v) <= 3 para todo v (pai+até dois filhos)
    """
    if nx_g.number_of_nodes() == 0:
        return False
    if not nx.is_tree(nx_g):
        return False
    max_deg = max(dict(nx_g.degree()).values()) if nx_g.number_of_nodes() > 0 else 0
    return max_deg <= 3

class BinaryTreeModelEvaluation(object):
    def __init__(self, v_min, v_max, dir):
        self.v_min = v_min
        self.v_max = v_max
        self.dir = dir

    def rollout_and_examine(self, model, num_samples):
        assert not model.training
        num_total_size = num_valid_size = 0
        num_bin = num_valid = 0
        plot_times = 0
        adj_lists_to_plot = []

        for i in range(num_samples):
            sampled_graph = model()
            if isinstance(sampled_graph, list):
                sampled_graph = sampled_graph[0]

            nx_g = simplegraph_to_nx(sampled_graph)
            sampled_adj_list = {n: list(nx_g.neighbors(n)) for n in nx_g.nodes()}
            adj_lists_to_plot.append(sampled_adj_list)

            graph_size = nx_g.number_of_nodes()
            valid_size = self.v_min <= graph_size <= self.v_max
            bin_ok = is_binary_tree(nx_g)

            num_total_size += graph_size
            if valid_size: num_valid_size += 1
            if bin_ok: num_bin += 1
            if valid_size and bin_ok: num_valid += 1

            if len(adj_lists_to_plot) >= 4:
                plot_times += 1
                fig, axes = plt.subplots(2,2, figsize=(6,6))
                axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
                for j in range(4):
                    Gplot = nx.from_dict_of_lists(adj_lists_to_plot[j])
                    try:
                        from networkx.drawing.nx_agraph import graphviz_layout
                        pos = graphviz_layout(Gplot, prog="dot") if Gplot.number_of_nodes() > 1 else None
                    except Exception:
                        pos = nx.spring_layout(Gplot, seed=0) if Gplot.number_of_nodes() > 1 else None

                    nx.draw_networkx(Gplot, pos=pos, with_labels=False, node_size=40, ax=axes[j])

                    axes[j].set_axis_off()
                plt.tight_layout()
                os.makedirs(self.dir + "/samples", exist_ok=True)
                plt.savefig(self.dir + f"/samples/{plot_times}")
                plt.close()
                adj_lists_to_plot = []

        self.num_samples_examined = num_samples
        self.average_size = num_total_size / num_samples
        self.valid_size_ratio = num_valid_size / num_samples
        self.binary_tree_ratio = num_bin / num_samples
        self.valid_ratio = num_valid / num_samples

    def write_summary(self):
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        stats = {
            "num_samples": self.num_samples_examined,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "average_size": self.average_size,
            "valid_size_ratio": self.valid_size_ratio,
            "binary_tree_ratio": self.binary_tree_ratio,
            "valid_ratio": self.valid_ratio,
        }
        path = os.path.join(self.dir, "model_eval.txt")
        with open(path, "w") as f:
            for k,v in stats.items():
                f.write(f"{k}\t{fmt(v)}\n")
        print(f"Saved model evaluation statistics to {path}")
