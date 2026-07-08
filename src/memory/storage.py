from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

from src.memory.memory_bank import MemoryBank


def save_memory_bank(path: str | Path, bank: MemoryBank) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.to_payload(), path)


def load_memory_bank(path: str | Path, *, map_location: str | torch.device = "cpu") -> MemoryBank:
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        logger.warning("cache_load_legacy_format path=%s", str(path))
        payload = torch.load(path, map_location=map_location, weights_only=False)
    return MemoryBank.from_payload(payload)
