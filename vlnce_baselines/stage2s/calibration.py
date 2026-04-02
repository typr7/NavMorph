from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self, initial_temperature: float = 1.0, min_temperature: float = 1e-3) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self._log_temperature = nn.Parameter(torch.log(torch.tensor(float(initial_temperature))))
        self.min_temperature = float(min_temperature)

    @property
    def temperature(self) -> torch.Tensor:
        return self._log_temperature.exp().clamp_min(self.min_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


@dataclass
class CalibratedBranchAggregator:
    semantic_weight: float = 1.0
    exec_weight: float = 1.0
    progress_weight: float = 1.0
    blocking_weight: float = 1.0
    uncertainty_weight: float = 1.0

    def __call__(
        self,
        *,
        exec_prob: float,
        progress: float,
        blocking: float,
        uncertainty: float,
        semantic: float,
    ) -> float:
        return float(
            self.semantic_weight * semantic
            + self.exec_weight * exec_prob
            + self.progress_weight * progress
            - self.blocking_weight * blocking
            - self.uncertainty_weight * uncertainty
        )
