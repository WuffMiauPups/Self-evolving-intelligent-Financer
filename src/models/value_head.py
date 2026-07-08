from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RiskHead(nn.Module):
    """Predicts raw-fraction MFE / MAE / volatility / path_drawdown.

    Outputs are NOT sigmoid-clamped — the regression targets (in
    ``src/training/targets.py``) are raw fractions (e.g. 0.025 = 2.5%) and
    MAE is typically negative. Sigmoid was preventing the head from fitting
    its targets and blocked using the outputs as EV gate signals in the
    risk manager. Consumers interpret ``mfe`` as expected favorable
    excursion (fraction) and ``mae`` as expected adverse excursion
    (typically negative fraction).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mfe = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.mae = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.volatility = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.path_drawdown = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "mfe": self.mfe(x),
            "mae": self.mae(x),
            "volatility": self.volatility(x),
            "path_drawdown": self.path_drawdown(x),
        }

