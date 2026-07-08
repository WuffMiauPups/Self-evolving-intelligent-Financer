"""Checkpoint registry inspection + gated promotion.

The GUI must NEVER bypass the existing PromotionGate. We surface the gate
results to the operator and only call promote_checkpoint() if the gate is
satisfied. Promotion always goes through src/training/checkpoint_promotion.py.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

from src.gui_services.config_service import load_app_config
from src.gui_services.state import CheckpointInfo, ServiceError
from src.training.checkpoint_promotion import (
    PromotionDecision,
    PromotionGate,
    evaluate_promotion,
    get_active_checkpoint,
    get_checkpoint_history,
    promote_checkpoint as _promote_checkpoint,
)

REGISTRY_PATH = Path("checkpoints/registry.json")


def get_active() -> CheckpointInfo | None:
    entry = get_active_checkpoint(REGISTRY_PATH)
    if entry is None:
        return None
    return CheckpointInfo.from_registry_entry(entry)


def list_history() -> list[CheckpointInfo]:
    return [CheckpointInfo.from_registry_entry(e) for e in get_checkpoint_history(REGISTRY_PATH)]


def validate_checkpoint(checkpoint_path: str | Path) -> dict[str, object]:
    """Best-effort load + introspection of a checkpoint.

    We do not load the model into a runtime; we only inspect the saved meta.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise ServiceError(f"Checkpoint not found: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            logger.warning("cache_load_legacy_format", path=str(path))
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Failed to read checkpoint: {exc}") from exc

    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    extra = meta.get("extra", {}) if isinstance(meta, dict) else {}
    summary = {
        "path": str(path),
        "model_kind": meta.get("model_kind", "unknown"),
        "feature_contract_id": meta.get("feature_contract_id", ""),
        "val_macro_f1": float(extra.get("val_macro_f1", 0.0)) if isinstance(extra, dict) else 0.0,
        "val_balanced_accuracy": float(extra.get("val_balanced_accuracy", 0.0)) if isinstance(extra, dict) else 0.0,
        "test_balanced_accuracy": float(extra.get("test_balanced_accuracy", 0.0)) if isinstance(extra, dict) else 0.0,
        "data_meta": extra.get("data_meta", {}) if isinstance(extra, dict) else {},
    }
    return summary


def evaluate_for_promotion(
    *,
    candidate_metrics: dict[str, float],
    parent_metrics: dict[str, float] | None = None,
    gate: PromotionGate | None = None,
) -> PromotionDecision:
    return evaluate_promotion(
        candidate_metrics=candidate_metrics,
        parent_metrics=parent_metrics,
        gate=gate or PromotionGate(),
    )


def promote_if_eligible(
    *,
    source_path: str | Path,
    target_path: str | Path,
    version_id: str,
    fingerprint: str,
    candidate_metrics: dict[str, float],
    parent_metrics: dict[str, float] | None = None,
    gate: PromotionGate | None = None,
    notes: str = "",
) -> CheckpointInfo:
    """Evaluate the gate, then promote ONLY if eligible."""
    decision = evaluate_for_promotion(
        candidate_metrics=candidate_metrics,
        parent_metrics=parent_metrics,
        gate=gate,
    )
    if not decision.eligible:
        raise ServiceError(
            "Promotion blocked by gate:\n  - " + "\n  - ".join(decision.reasons)
        )
    from datetime import UTC, datetime

    from src.training.checkpoint_promotion import CheckpointVersion

    version = CheckpointVersion(
        version_id=version_id,
        checkpoint_path=str(source_path),
        fingerprint=fingerprint,
        parent_fingerprint=None,
        created_at=datetime.now(UTC).isoformat(),
        retrain_id=None,
        metrics=dict(candidate_metrics),
        status="candidate",
        notes=notes,
    )
    promoted = _promote_checkpoint(
        source_path=source_path,
        target_path=target_path,
        version=version,
        registry_path=REGISTRY_PATH,
    )
    return CheckpointInfo(
        version_id=promoted.version_id,
        checkpoint_path=promoted.checkpoint_path,
        fingerprint=promoted.fingerprint,
        parent_fingerprint=promoted.parent_fingerprint,
        created_at=promoted.created_at,
        status=promoted.status,
        metrics=dict(promoted.metrics),
        notes=promoted.notes,
    )


def get_active_checkpoint_path(*, config_paths: list[str] | None = None) -> str:
    """Resolve which checkpoint is currently configured for the runtime."""
    config_paths = config_paths or ["configs/default.yaml"]
    cfg = load_app_config(*config_paths)
    return cfg.model.checkpoint_path or ""
