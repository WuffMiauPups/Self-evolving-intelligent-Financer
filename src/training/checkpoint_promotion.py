"""Checkpoint promotion and deployment rules.

Provides explicit versioning, comparison against the current production
checkpoint, and safe promotion gating based on configurable metric thresholds.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointVersion:
    """Immutable descriptor of a checkpoint version."""

    version_id: str
    checkpoint_path: str
    fingerprint: str
    parent_fingerprint: str | None
    created_at: str
    retrain_id: str | None
    metrics: dict[str, float]
    status: str  # "candidate", "promoted", "active", "retired"
    notes: str = ""


@dataclass
class PromotionGate:
    """Configurable promotion thresholds."""

    min_val_balanced_accuracy: float = 0.40
    min_val_macro_f1: float = 0.30
    min_long_recall: float = 0.15
    min_short_recall: float = 0.15
    max_val_loss_regression: float = 0.10  # max allowed val loss increase vs parent
    require_improvement_over_parent: bool = True
    improvement_metric: str = "val_report.macro_f1"
    min_improvement_delta: float = 0.0  # >=0 means must not degrade


@dataclass
class PromotionDecision:
    """Result of evaluating a checkpoint for promotion."""

    eligible: bool
    reasons: list[str]
    candidate_metrics: dict[str, float]
    parent_metrics: dict[str, float]
    delta: dict[str, float] = field(default_factory=dict)


def evaluate_promotion(
    *,
    candidate_metrics: dict[str, float],
    parent_metrics: dict[str, float] | None,
    gate: PromotionGate,
) -> PromotionDecision:
    """Evaluate whether a candidate checkpoint should be promoted.

    Returns a PromotionDecision with eligibility and reasons.
    """
    reasons: list[str] = []
    eligible = True

    # Absolute metric floors
    bal_acc = candidate_metrics.get("val_report.balanced_accuracy", 0.0)
    if bal_acc < gate.min_val_balanced_accuracy:
        reasons.append(f"val_balanced_accuracy={bal_acc:.4f} < {gate.min_val_balanced_accuracy}")
        eligible = False

    macro_f1 = candidate_metrics.get("val_report.macro_f1", 0.0)
    if macro_f1 < gate.min_val_macro_f1:
        reasons.append(f"val_macro_f1={macro_f1:.4f} < {gate.min_val_macro_f1}")
        eligible = False

    long_recall = candidate_metrics.get("val_report.per_class_metrics.long.recall", 0.0)
    if long_recall < gate.min_long_recall:
        reasons.append(f"long_recall={long_recall:.4f} < {gate.min_long_recall}")
        eligible = False

    short_recall = candidate_metrics.get("val_report.per_class_metrics.short.recall", 0.0)
    if short_recall < gate.min_short_recall:
        reasons.append(f"short_recall={short_recall:.4f} < {gate.min_short_recall}")
        eligible = False

    # Relative improvement over parent
    delta: dict[str, float] = {}
    if parent_metrics is not None and gate.require_improvement_over_parent:
        improvement_key = gate.improvement_metric
        candidate_val = candidate_metrics.get(improvement_key, 0.0)
        parent_val = parent_metrics.get(improvement_key, 0.0)
        improvement = candidate_val - parent_val
        delta[improvement_key] = improvement
        if improvement < gate.min_improvement_delta:
            reasons.append(
                f"{improvement_key} delta={improvement:.4f} < {gate.min_improvement_delta} "
                f"(candidate={candidate_val:.4f}, parent={parent_val:.4f})"
            )
            eligible = False

        # Check loss regression
        candidate_loss = candidate_metrics.get("val.loss", 0.0)
        parent_loss = parent_metrics.get("val.loss", 0.0)
        loss_regression = candidate_loss - parent_loss
        delta["val.loss"] = loss_regression
        if loss_regression > gate.max_val_loss_regression:
            reasons.append(
                f"val.loss regression={loss_regression:.4f} > {gate.max_val_loss_regression} "
                f"(candidate={candidate_loss:.4f}, parent={parent_loss:.4f})"
            )
            eligible = False

    if eligible:
        reasons.append("all promotion gates passed")

    return PromotionDecision(
        eligible=eligible,
        reasons=reasons,
        candidate_metrics=dict(candidate_metrics),
        parent_metrics=dict(parent_metrics or {}),
        delta=delta,
    )


def promote_checkpoint(
    *,
    source_path: str | Path,
    target_path: str | Path,
    version: CheckpointVersion,
    registry_path: str | Path = "checkpoints/registry.json",
) -> CheckpointVersion:
    """Copy checkpoint to the active deployment location and update registry.

    Returns updated CheckpointVersion with status="active".
    """
    source = Path(source_path)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint file
    shutil.copy2(source, target)

    # Update the registry
    promoted = CheckpointVersion(
        version_id=version.version_id,
        checkpoint_path=str(target),
        fingerprint=version.fingerprint,
        parent_fingerprint=version.parent_fingerprint,
        created_at=version.created_at,
        retrain_id=version.retrain_id,
        metrics=dict(version.metrics),
        status="active",
        notes=version.notes,
    )

    _update_registry(promoted, registry_path=registry_path)

    logger.info(
        "checkpoint_promoted version_id=%s source=%s target=%s fingerprint=%s",
        promoted.version_id, str(source), str(target), promoted.fingerprint,
    )
    return promoted


def _update_registry(version: CheckpointVersion, *, registry_path: str | Path) -> None:
    """Append or update a checkpoint version in the registry file."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    registry: list[dict[str, Any]] = []
    if path.exists():
        with open(path) as f:
            registry = json.load(f)

    # Retire any currently active versions
    for entry in registry:
        if entry.get("status") == "active":
            entry["status"] = "retired"
            entry["retired_at"] = datetime.now(UTC).isoformat()

    # Add new version
    registry.append({
        "version_id": version.version_id,
        "checkpoint_path": version.checkpoint_path,
        "fingerprint": version.fingerprint,
        "parent_fingerprint": version.parent_fingerprint,
        "created_at": version.created_at,
        "retrain_id": version.retrain_id,
        "metrics": version.metrics,
        "status": version.status,
        "notes": version.notes,
        "registered_at": datetime.now(UTC).isoformat(),
    })

    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)


def load_registry(registry_path: str | Path = "checkpoints/registry.json") -> list[dict[str, Any]]:
    """Load the full checkpoint registry."""
    path = Path(registry_path)
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def get_active_checkpoint(registry_path: str | Path = "checkpoints/registry.json") -> dict[str, Any] | None:
    """Get the currently active checkpoint from the registry."""
    registry = load_registry(registry_path)
    for entry in reversed(registry):
        if entry.get("status") == "active":
            return entry
    return None


def get_checkpoint_history(
    registry_path: str | Path = "checkpoints/registry.json",
) -> list[dict[str, Any]]:
    """Get all checkpoint versions, newest first."""
    registry = load_registry(registry_path)
    return list(reversed(registry))
