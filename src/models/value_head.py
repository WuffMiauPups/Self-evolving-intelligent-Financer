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


class ReturnHeads(nn.Module):
    """One small regression head per forward horizon (Phase 2 §6.1).

    Replaces the old design where every ``return_h{H}`` target was
    regressed onto the single ``target_fraction`` scalar — one scalar
    cannot fit three horizons, so the gradient pulled it toward an
    uninterpretable blend. Keys follow the training target names
    (``return_h15`` etc.); outputs are raw return fractions.
    """

    def __init__(self, input_dim: int, horizons: list[int], hidden_dim: int = 64) -> None:
        super().__init__()
        self.horizons = sorted(int(h) for h in horizons)
        if not self.horizons:
            raise ValueError("ReturnHeads requires at least one horizon")
        self.heads = nn.ModuleDict(
            {
                f"return_h{horizon}": nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for horizon in self.horizons
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


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

