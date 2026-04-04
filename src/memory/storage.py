from __future__ import annotations

from pathlib import Path

import torch

from src.memory.memory_bank import MemoryBank


def save_memory_bank(path: str | Path, bank: MemoryBank) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.to_payload(), path)


def load_memory_bank(path: str | Path, *, map_location: str | torch.device = "cpu") -> MemoryBank:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return MemoryBank.from_payload(payload)
