from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import torch


@dataclass(frozen=True)
class MarketBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    spread: float = 0.0


def bars_to_tensor(
    bars: Sequence[MarketBar],
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    rows = [
        [bar.open, bar.high, bar.low, bar.close, bar.volume, bar.spread]
        for bar in bars
    ]
    return torch.tensor(rows, dtype=dtype, device=device)

