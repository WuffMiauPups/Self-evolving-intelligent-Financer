from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch

from src.core.exceptions import ConfigurationError


@dataclass(frozen=True)
class DeviceContext:
    device: torch.device
    model_dtype: torch.dtype
    accounting_dtype: torch.dtype
    use_amp: bool


def resolve_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "cuda" or preferred.startswith("cuda:"):
        # An explicit CUDA request must fail loudly when CUDA is missing.
        # Falling through to torch.device("cuda") used to defer the error
        # to the first tensor op (or silently run CPU-slow paths elsewhere).
        if not torch.cuda.is_available():
            raise ConfigurationError(
                f"device.preferred={preferred!r} but CUDA is not available in this "
                "environment. Install a CUDA-enabled torch build / fix the driver, "
                "or set device.preferred to 'auto' or 'cpu'."
            )
        return torch.device(preferred)
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

