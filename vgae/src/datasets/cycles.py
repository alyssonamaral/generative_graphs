import random
from typing import Dict
import torch
from torch.utils.data import Dataset
from .. import utils as U

class CyclesDataset(Dataset):
    """Gera ciclos com n aleatório em [min_size, max_size].
    Retorna dict com A (adj), X (features), s (posições).
    """
    def __init__(self, num_graphs: int, min_size: int, max_size: int, in_dim: int = 1, seed: int = 42):
        super().__init__()
        random.seed(seed)
        self.items = []
        for _ in range(num_graphs):
            n = random.randint(min_size, max_size)
            A = U.cycle_adj(n)
            X = torch.ones((n, in_dim), dtype=torch.float32) # feature constante
            s = U.ring_positions(n)
            self.items.append({"A": A, "X": X, "s": s})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.items[idx]


    @staticmethod
    def collate_single(batch):
        assert len(batch) == 1
        return batch[0]