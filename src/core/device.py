from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device
    model_dtype: torch.dtype
    accounting_dtype: torch.dtype
    use_amp: bool


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)


def build_device_context(preferred: str = "auto", use_amp: bool = False) -> DeviceContext:
    device = resolve_device(preferred)
    amp_enabled = bool(use_amp and device.type == "cuda")
    return DeviceContext(
        device=device,
        model_dtype=torch.float32,
        accounting_dtype=torch.float64,
        use_amp=amp_enabled,
    )


def maybe_autocast(device: torch.device, enabled: bool) -> nullcontext | torch.amp.autocast_mode.autocast:
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

