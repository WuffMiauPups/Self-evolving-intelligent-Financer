from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def to_tensor(
    value: Any,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if device is not None and tensor.device != device:
            tensor = tensor.to(device=device)
        return tensor
    return torch.as_tensor(value, dtype=dtype, device=device)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def ensure_batch_dimension(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 1:
        return tensor.unsqueeze(0)
    return tensor

