from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import torch

from rlod.graphs.types import GraphSample


class RewardProvider(ABC):
    @abstractmethod
    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        """
        Retorna reward escalar (torch tensor) para (grafo, ordenação).
        """
        ...


@dataclass
class DummyReward(RewardProvider):
    value: float = 0.0

    def __call__(self, sample: GraphSample, order: List[int]) -> torch.Tensor:
        return torch.tensor(float(self.value))
