"""Checkpoint promotion and deployment rules.

Provides explicit versioning, comparison against the current production
checkpoint, and safe promotion gating based on configurable metric thresholds.

The gate judges three layers (MASTER_PLAN §4.5.4-7):
  1. classification floors + improvement-over-parent (always),
  2. scorecard criteria when benchmark rows are supplied (candidate must
     beat the incumbent's balanced accuracy AND expectancy, and beat
     buy-and-hold),
  3. a backward-transfer floor when held-out slice metrics are supplied.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric-key schema adapter
# ---------------------------------------------------------------------------
# The gate reads DOTTED keys (matching the nested run-summary/eval-report
# layout); checkpoint meta ``extra`` and gui_services.checkpoint_service
# produce FLAT keys (``val_balanced_accuracy``). This single adapter maps
# flat aliases onto the dotted canonical keys so both schemas satisfy the
# gate — the flat-vs-dotted mismatch is fixed here and nowhere else.

#: Every metric key evaluate_promotion reads.
GATE_METRIC_KEYS: tuple[str, ...] = (
    "val_report.balanced_accuracy",
    "val_report.macro_f1",
    "val_report.per_class_metrics.long.recall",
    "val_report.per_class_metrics.short.recall",
    "val.loss",
)

FLAT_TO_DOTTED_METRIC_KEYS: dict[str, str] = {
    "val_balanced_accuracy": "val_report.balanced_accuracy",
    "val_macro_f1": "val_report.macro_f1",
    "val_long_recall": "val_report.per_class_metrics.long.recall",
    "val_short_recall": "val_report.per_class_metrics.short.recall",
    "val_loss": "val.loss",
}


def normalize_promotion_metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    """Map flat metric keys onto the dotted keys the gate reads.

    Existing dotted keys always win; flat aliases only fill gaps.
    """
    if metrics is None:
        return {}
    normalized = dict(metrics)
    for flat_key, dotted_key in FLAT_TO_DOTTED_METRIC_KEYS.items():
        if flat_key in normalized and dotted_key not in normalized:
            normalized[dotted_key] = float(normalized[flat_key])
    return normalized


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
    """Configurable promotion thresholds.

    Scorecard fields apply only when :func:`evaluate_promotion` receives
    benchmark rows; ``backward_transfer_floor`` applies only when it
    receives per-slice metrics.  Producing the frozen held-out slices is
    a later workstream (MASTER_PLAN §4.5.7) — the gate logic ships now so
    slice producers have a fixed contract to target.
    """

    min_val_balanced_accuracy: float = 0.40
    min_val_macro_f1: float = 0.30
    min_long_recall: float = 0.15
    min_short_recall: float = 0.15
    max_val_loss_regression: float = 0.10  # max allowed val loss increase vs parent
    require_improvement_over_parent: bool = True
    improvement_metric: str = "val_report.macro_f1"
    min_improvement_delta: float = 0.0  # >=0 means must not degrade
    # --- Scorecard criteria (§4.5.5) — applied when benchmark rows given ---
    min_benchmark_balanced_accuracy_delta: float = 0.0  # candidate must beat incumbent by more
    min_benchmark_expectancy_delta: float = 0.0
    min_excess_vs_buy_hold: float = 0.0  # candidate must beat buy&hold by more
    # --- Backward-transfer floor (§4.5.7) — applied when slices given ---
    backward_transfer_floor: float = 0.05  # max per-slice balanced-accuracy degradation


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
    candidate_benchmark: Any | None = None,
    incumbent_benchmark: Any | None = None,
    slice_metrics: Mapping[str, tuple[float, float]] | None = None,
) -> PromotionDecision:
    """Evaluate whether a candidate checkpoint should be promoted.

    ``candidate_benchmark`` / ``incumbent_benchmark`` are
    ``BenchmarkRow``-shaped objects (duck-typed: ``balanced_accuracy``,
    ``expectancy``, ``excess_vs_buy_hold``) from the OOS scorecard
    protocol; when supplied, the candidate must beat the incumbent on
    balanced accuracy AND expectancy, and beat buy-and-hold.

    ``slice_metrics`` maps ``slice_name -> (candidate_balanced_accuracy,
    incumbent_balanced_accuracy)`` on frozen held-out regime slices;
    promotion is refused if any slice degrades by more than
    ``gate.backward_transfer_floor``.  Producing the slices is a later
    workstream — pass ``None`` until they exist.

    Returns a PromotionDecision with eligibility and reasons.
    """
    candidate_metrics = normalize_promotion_metrics(candidate_metrics)
    parent_metrics = normalize_promotion_metrics(parent_metrics) if parent_metrics is not None else None
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

    # Scorecard criteria (§4.5.5): trading metrics, not just classification.
    if candidate_benchmark is not None:
        excess = float(candidate_benchmark.excess_vs_buy_hold)
        delta["benchmark.excess_vs_buy_hold"] = excess
        if excess <= gate.min_excess_vs_buy_hold:
            reasons.append(
                f"benchmark excess_vs_buy_hold={excess:+.4f} <= {gate.min_excess_vs_buy_hold} "
                f"(candidate does not beat buy-and-hold)"
            )
            eligible = False
        if incumbent_benchmark is not None:
            bal_acc_delta = float(candidate_benchmark.balanced_accuracy) - float(
                incumbent_benchmark.balanced_accuracy
            )
            delta["benchmark.balanced_accuracy"] = bal_acc_delta
            if bal_acc_delta <= gate.min_benchmark_balanced_accuracy_delta:
                reasons.append(
                    f"benchmark balanced_accuracy delta={bal_acc_delta:+.4f} <= "
                    f"{gate.min_benchmark_balanced_accuracy_delta} "
                    f"(candidate={float(candidate_benchmark.balanced_accuracy):.4f}, "
                    f"incumbent={float(incumbent_benchmark.balanced_accuracy):.4f})"
                )
                eligible = False
            expectancy_delta = float(candidate_benchmark.expectancy) - float(
                incumbent_benchmark.expectancy
            )
            delta["benchmark.expectancy"] = expectancy_delta
            if expectancy_delta <= gate.min_benchmark_expectancy_delta:
                reasons.append(
                    f"benchmark expectancy delta={expectancy_delta:+.4f} <= "
                    f"{gate.min_benchmark_expectancy_delta} "
                    f"(candidate={float(candidate_benchmark.expectancy):+.2f}, "
                    f"incumbent={float(incumbent_benchmark.expectancy):+.2f})"
                )
                eligible = False

    # Backward-transfer floor (§4.5.7): no old regime may be forgotten.
    for slice_name, (candidate_acc, incumbent_acc) in (slice_metrics or {}).items():
        degradation = float(incumbent_acc) - float(candidate_acc)
        delta[f"slice.{slice_name}.balanced_accuracy"] = -degradation
        if degradation > gate.backward_transfer_floor:
            reasons.append(
                f"backward_transfer[{slice_name}] degradation={degradation:.4f} > "
                f"{gate.backward_transfer_floor} "
                f"(candidate={float(candidate_acc):.4f}, incumbent={float(incumbent_acc):.4f})"
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


def _promotion_lock_path(registry_path: Path) -> Path:
    return registry_path.with_suffix(registry_path.suffix + ".lock")


def promote_checkpoint(
    *,
    source_path: str | Path,
    target_path: str | Path,
    version: CheckpointVersion,
    registry_path: str | Path = "checkpoints/registry.json",
) -> CheckpointVersion:
    """Copy checkpoint to the active deployment location and update registry.

    Transactional (§4.5.4):
      - a lock file next to the registry refuses concurrent promotions;
      - the target is staged to a tmp file and swapped via ``os.replace``
        (never torn), with the previous target preserved as ``.bak``;
      - the registry is written atomically (tmp + replace) and a corrupt
        registry is backed up and rebuilt instead of crashing promotion.

    A crash between the file swap and the registry write leaves the new
    checkpoint live with a stale-but-valid registry and a ``.bak`` for
    rollback; re-running the promotion converges both.

    Returns updated CheckpointVersion with status="active".
    """
    source = Path(source_path)
    target = Path(target_path)
    registry_file = Path(registry_path)
    registry_file.parent.mkdir(parents=True, exist_ok=True)

    lock_path = _promotion_lock_path(registry_file)
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"concurrent promotion in progress (lock file exists: {lock_path}); "
            f"remove it only if you are sure no other promotion is running"
        ) from None
    try:
        with os.fdopen(lock_fd, "w") as lock_fp:
            lock_fp.write(str(os.getpid()))

        target.parent.mkdir(parents=True, exist_ok=True)
        # Stage next to the target, then swap atomically; the previous
        # checkpoint survives as .bak for rollback.
        tmp_target = target.with_suffix(target.suffix + f".promote_tmp{os.getpid()}")
        shutil.copy2(source, tmp_target)
        if target.exists():
            shutil.copy2(target, target.with_suffix(target.suffix + ".bak"))
        os.replace(tmp_target, target)

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

        _update_registry(promoted, registry_path=registry_file)

        logger.info(
            "checkpoint_promoted version_id=%s source=%s target=%s fingerprint=%s",
            promoted.version_id, str(source), str(target), promoted.fingerprint,
        )
        return promoted
    finally:
        lock_path.unlink(missing_ok=True)


def _update_registry(version: CheckpointVersion, *, registry_path: str | Path) -> None:
    """Append or update a checkpoint version in the registry file (atomic)."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    registry: list[dict[str, Any]] = []
    if path.exists():
        try:
            with open(path) as f:
                loaded = json.load(f)
            if not isinstance(loaded, list):
                raise ValueError("registry root must be a JSON list")
            registry = loaded
        except (json.JSONDecodeError, ValueError) as exc:
            # A corrupt registry must not block promotion: preserve the
            # bytes for forensics and rebuild from this promotion onward.
            corrupt_backup = path.with_suffix(
                path.suffix + f".corrupt.{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            )
            shutil.copy2(path, corrupt_backup)
            logger.error(
                "registry_corrupt_rebuilt path=%s backup=%s error=%s",
                str(path), str(corrupt_backup), exc,
            )
            registry = []

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

    # Atomic write: a crash mid-dump must never leave a torn registry.
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp_path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    os.replace(tmp_path, path)


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
