from __future__ import annotations

import torch


def compute_volatility_features(bars: torch.Tensor, windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    features: dict[str, torch.Tensor] = {}
    if close.numel() < 2:
        return {f"volatility_{window}": torch.zeros((), dtype=bars.dtype, device=bars.device) for window in windows}

    log_returns = torch.log((close[1:] / close[:-1]).clamp_min(epsilon))
    for window in windows:
        if log_returns.numel() >= window:
            features[f"volatility_{window}"] = log_returns[-window:].std(unbiased=False)
        else:
            features[f"volatility_{window}"] = torch.zeros((), dtype=bars.dtype, device=bars.device)
    return features

