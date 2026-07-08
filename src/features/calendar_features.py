"""Calendar features -- hour-of-day and day-of-week sin/cos encoding."""
from __future__ import annotations

import math
import torch
from datetime import datetime


def compute_calendar_features(
    timestamp: datetime,
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Compute calendar sin/cos features from a bar timestamp.

    Returns 4 scalar tensors: hour_sin, hour_cos, dow_sin, dow_cos.
    """
    hour = timestamp.hour + timestamp.minute / 60.0
    dow = timestamp.weekday()  # 0=Mon, 4=Fri

    return {
        "calendar_hour_sin": torch.tensor(math.sin(2.0 * math.pi * hour / 24.0), device=device),
        "calendar_hour_cos": torch.tensor(math.cos(2.0 * math.pi * hour / 24.0), device=device),
        "calendar_dow_sin": torch.tensor(math.sin(2.0 * math.pi * dow / 5.0), device=device),
        "calendar_dow_cos": torch.tensor(math.cos(2.0 * math.pi * dow / 5.0), device=device),
    }


def compute_calendar_features_rolling(
    timestamps: list[datetime],
    *,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorized calendar features for batch processing.

    Args:
        timestamps: list of datetime objects, one per bar

    Returns:
        dict of feature_name -> (N,) tensor
    """
    n = len(timestamps)
    if n == 0:
        return {
            "calendar_hour_sin": torch.zeros(0, device=device),
            "calendar_hour_cos": torch.zeros(0, device=device),
            "calendar_dow_sin": torch.zeros(0, device=device),
            "calendar_dow_cos": torch.zeros(0, device=device),
        }
    hours = torch.tensor(
        [t.hour + t.minute / 60.0 for t in timestamps], dtype=torch.float32, device=device
    )
    dows = torch.tensor(
        [float(t.weekday()) for t in timestamps], dtype=torch.float32, device=device
    )

    return {
        "calendar_hour_sin": torch.sin(2.0 * math.pi * hours / 24.0),
        "calendar_hour_cos": torch.cos(2.0 * math.pi * hours / 24.0),
        "calendar_dow_sin": torch.sin(2.0 * math.pi * dows / 5.0),
        "calendar_dow_cos": torch.cos(2.0 * math.pi * dows / 5.0),
    }
