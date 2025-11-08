# ba.py
import os, math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch.utils.data import Dataset
from cycles import simplegraph_to_nx 

class BADataset(Dataset):
    def __init__(self, fname):
        super().__init__()
        import pickle
        with open(fname, 'rb') as f:
            self.dataset = pickle.load(f)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx): return self.dataset[idx]
    def collate_single(self, batch):
        assert len(batch) == 1
        return batch[0]
    def collate_batch(self, batch):
        return batch

def _powerlaw_mle_alpha(deg_vals, kmin=1):
    """
    Estimador MLE do expoente alpha para graus >= kmin (discreto aproximado).
    Retorna alpha_hat e os dados filtraods.
    """
    tail = np.array([k for k in deg_vals if k >= kmin], dtype=np.float64)
    if tail.size == 0:
        return None, None
    # contínuo aproximado: alpha_hat = 1 + n / sum(log(k/kmin))
    z = np.log(tail / float(kmin))
    s = z.sum()
    if s <= 0:  # todos == kmin
        return None, tail
    alpha = 1.0 + tail.size / s
    return alpha, tail

def _ks_to_plaw(tail_vals, alpha, kmin):
    """
    KS entre a CDF empírica dos dados >= kmin e a CDF teórica ~ k^(-alpha+1)
    (contínuo aproximado). Retorna KS.
    """
    x = np.sort(tail_vals)
    n = x.size
    if n == 0: return 1.0
    # CDF empírica
    ecdf = np.arange(1, n+1) / n
    # CDF teórica contínua para k>=kmin: F(k) = 1 - (k/kmin)^(-alpha+1)
    # (para alpha>1)
    if alpha <= 1.0: return 1.0
    tcdf = 1.0 - (x / float(kmin))**(-alpha + 1.0)
    ks = np.max(np.abs(ecdf - tcdf))
    return ks

def is_scale_free(nx_g: nx.Graph, kmin: int = 2, alpha_range=(2.0, 3.5), ks_thresh: float = 0.15) -> bool:
    """
    Heurística: estima alpha na cauda (k >= kmin) e calcula KS.
    Aceita se alpha in [2.0, 3.5] e KS <= 0.15.
    """
    if nx_g.number_of_nodes() < 5:
        return False
    degs = np.array([d for _, d in nx_g.degree()], dtype=np.int64)
    alpha, tail = _powerlaw_mle_alpha(degs, kmin=kmin)
    if alpha is None: 
        return False
    if not (alpha_range[0] <= alpha <= alpha_range[1]):
        return False
    ks = _ks_to_plaw(tail, alpha, kmin=kmin)
    return ks <= ks_thresh

class BAModelEvaluation(object):
    def __init__(self, v_min, v_max, dir, kmin=2):
        self.v_min = v_min
        self.v_max = v_max
        self.dir = dir
        self.kmin = kmin

    def rollout_and_examine(self, model, num_samples):
        assert not model.training
        num_total_size = num_valid_size = 0
        num_sf = num_valid = 0
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
            sf_ok = is_scale_free(nx_g, kmin=self.kmin)

            num_total_size += graph_size
            if valid_size: num_valid_size += 1
            if sf_ok: num_sf += 1
            if valid_size and sf_ok: num_valid += 1

            if len(adj_lists_to_plot) >= 4:
                plot_times += 1
                fig, axes = plt.subplots(2,2, figsize=(6,6))
                axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
                for j in range(4):
                    Gplot = nx.from_dict_of_lists(adj_lists_to_plot[j])
                    # fallback de layout sem pygraphviz
                    try:
                        from networkx.drawing.nx_agraph import graphviz_layout
                        pos = graphviz_layout(Gplot, prog="dot") if Gplot.number_of_nodes()>1 else None
                    except Exception:
                        pos = nx.spring_layout(Gplot, seed=0) if Gplot.number_of_nodes()>1 else None
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
        self.scale_free_ratio = num_sf / num_samples
        self.valid_ratio = num_valid / num_samples

    def write_summary(self):
        def fmt(v): return f"{v:.4f}" if isinstance(v, float) else str(v)
        stats = {
            "num_samples": self.num_samples_examined,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "average_size": self.average_size,
            "valid_size_ratio": self.valid_size_ratio,
            "scale_free_ratio": self.scale_free_ratio,
            "valid_ratio": self.valid_ratio,
        }
        path = os.path.join(self.dir, "model_eval.txt")
        with open(path, "w") as f:
            for k,v in stats.items(): f.write(f"{k}\t{fmt(v)}\n")
        print(f"Saved model evaluation statistics to {path}")
