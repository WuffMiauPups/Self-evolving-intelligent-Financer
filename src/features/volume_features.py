from __future__ import annotations

import torch

from src.features.normalization import safe_zscore
from src.features.rolling import rolling_mean, rolling_std


def compute_volume_features(bars: torch.Tensor, windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    volume = bars[:, 4]
    last_volume = volume[-1]
    # ``volume_latest`` (raw share count, ~10K-100K scale) was removed:
    # it dominated LayerNorm at the model input and squashed every other
    # feature to ~-1, causing the brain to collapse to constant outputs
    # (val_acc pinned to majority baseline). All useful volume signal is
    # already captured by ``volume_z_W`` and ``volume_change_W`` below in
    # properly normalized form.
    features: dict[str, torch.Tensor] = {}
    for window in windows:
        if volume.numel() >= window:
            reference = volume[-window:]
            features[f"volume_z_{window}"] = safe_zscore(last_volume, reference, epsilon)
            features[f"volume_change_{window}"] = (last_volume / (reference.mean() + epsilon)) - 1.0
        else:
            zero = torch.zeros((), dtype=bars.dtype, device=bars.device)
            features[f"volume_z_{window}"] = zero
            features[f"volume_change_{window}"] = zero
    return features


def compute_volume_features_rolling(
    bars: torch.Tensor, windows: list[int], epsilon: float
) -> dict[str, torch.Tensor]:
    """Vectorized rolling variant of ``compute_volume_features``.

    Shares the rolling mean/std computation across windows instead of
    re-slicing + re-reducing per bar.
    """
    volume = bars[:, 4]
    n = volume.numel()
    zero_full = torch.zeros(n, dtype=bars.dtype, device=bars.device)
    # ``volume_latest`` removed — see compute_volume_features() for the
    # full rationale. All useful volume signal is in volume_z_* / volume_change_*.
    features: dict[str, torch.Tensor] = {}
    for window in windows:
        if n >= window:
            means = rolling_mean(volume, window)
            stds = rolling_std(volume, window, unbiased=False)
            z = (volume - means) / (stds + epsilon)
            if window > 1:
                z[: window - 1] = 0.0
            features[f"volume_z_{window}"] = z
            change = (volume / (means + epsilon)) - 1.0
            if window > 1:
                change[: window - 1] = 0.0
            features[f"volume_change_{window}"] = change
        else:
            features[f"volume_z_{window}"] = zero_full
            features[f"volume_change_{window}"] = zero_full
    return features

