from __future__ import annotations

import torch


def compute_regime_features(
    bars: torch.Tensor,
    *,
    short_window: int,
    medium_window: int,
    long_window: int,
    epsilon: float,
) -> dict[str, torch.Tensor]:
    close = bars[:, 3]
    volume = bars[:, 4]
    short_window = min(short_window, close.numel())
    medium_window = min(medium_window, close.numel())
    long_window = min(long_window, close.numel())
    short_close = close[-short_window:]
    medium_close = close[-medium_window:]
    long_close = close[-long_window:]
    short_return = (short_close[-1] / (short_close[0] + epsilon)) - 1.0 if short_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    medium_return = (medium_close[-1] / (medium_close[0] + epsilon)) - 1.0 if medium_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    long_return = (long_close[-1] / (long_close[0] + epsilon)) - 1.0 if long_close.numel() > 1 else torch.zeros((), dtype=bars.dtype, device=bars.device)
    short_vol = short_close.std(unbiased=False)
    long_vol = long_close.std(unbiased=False)
    short_volume = volume[-short_window:].mean() if volume.numel() >= short_window else volume.mean()
    long_volume = volume[-long_window:].mean() if volume.numel() >= long_window else volume.mean()
    return {
        "regime_trend_strength": torch.abs(medium_return) / (short_vol + epsilon),
        "regime_volatility_ratio": short_vol / (long_vol + epsilon),
        "regime_volume_ratio": short_volume / (long_volume + epsilon) - 1.0,
        "regime_return_alignment": torch.tanh((short_return + medium_return + long_return) / 3.0),
    }

