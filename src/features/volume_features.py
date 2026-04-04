from __future__ import annotations

import torch

from src.features.normalization import safe_zscore


def compute_volume_features(bars: torch.Tensor, windows: list[int], epsilon: float) -> dict[str, torch.Tensor]:
    volume = bars[:, 4]
    last_volume = volume[-1]
    features: dict[str, torch.Tensor] = {"volume_latest": last_volume}
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

