from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(path: str | Path, *, model: torch.nn.Module, optimizer: torch.optim.Optimizer, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a crash mid-save must never leave a truncated file at
    # the canonical path.  When overwriting an existing checkpoint the
    # previous version is kept as .bak so a bad run can be rolled back.
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            backup.unlink(missing_ok=True)
            path.replace(backup)
        except OSError:
            logger.warning("checkpoint_backup_failed path=%s", str(path))
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "meta": meta,
        },
        tmp_path,
    )
    tmp_path.replace(path)


def load_checkpoint_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        logger.warning("cache_load_legacy_format path=%s", str(path))
        return torch.load(path, map_location=map_location, weights_only=False)


