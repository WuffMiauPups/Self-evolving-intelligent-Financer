"""Offline retrain cycle: trigger evaluation + lineage metadata.

Provides metadata tracking for retrain cycles (parent checkpoint lineage,
retrain reasons, data ranges) plus the drift trigger that decides WHEN a
retrain is due (MASTER_PLAN §4.5.3).  Wired into
``scripts/run_train.py --check-retrain-trigger``, which evaluates the
trigger against the persisted per-decision outcome stream
(``src.training.replay_buffer``).
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetainMetadata:
    """Immutable record of a retrain cycle."""

    retrain_id: str
    parent_checkpoint_path: str | None
    parent_checkpoint_fingerprint: str | None
    retrain_reason: str
    data_start: str
    data_end: str
    symbols: list[str]
    paper_observation_count: int
    config_snapshot: dict[str, Any]
    started_at: str
    completed_at: str | None = None
    output_checkpoint_path: str | None = None
    output_checkpoint_fingerprint: str | None = None
    validation_metrics: dict[str, float] = field(default_factory=dict)
    promotion_eligible: bool = False
    notes: str = ""


def build_retrain_id(parent_fingerprint: str | None, timestamp: datetime) -> str:
    """Generate a unique retrain cycle ID."""
    seed = f"{parent_fingerprint or 'genesis'}:{timestamp.isoformat()}"
    return f"retrain-{sha1(seed.encode()).hexdigest()[:20]}"


def build_retrain_metadata(
    *,
    parent_checkpoint_path: str | None,
    parent_checkpoint_fingerprint: str | None,
    retrain_reason: str,
    data_start: str,
    data_end: str,
    symbols: list[str],
    paper_observation_count: int,
    config_snapshot: dict[str, Any],
) -> RetainMetadata:
    """Build metadata for a new retrain cycle (before training starts)."""
    now = datetime.now(UTC)
    retrain_id = build_retrain_id(parent_checkpoint_fingerprint, now)
    return RetainMetadata(
        retrain_id=retrain_id,
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_fingerprint=parent_checkpoint_fingerprint,
        retrain_reason=retrain_reason,
        data_start=data_start,
        data_end=data_end,
        symbols=symbols,
        paper_observation_count=paper_observation_count,
        config_snapshot=config_snapshot,
        started_at=now.isoformat(),
    )


def finalize_retrain_metadata(
    meta: RetainMetadata,
    *,
    output_checkpoint_path: str,
    output_checkpoint_fingerprint: str,
    validation_metrics: dict[str, float],
    promotion_eligible: bool,
) -> RetainMetadata:
    """Return a new metadata record with training results filled in."""
    return RetainMetadata(
        retrain_id=meta.retrain_id,
        parent_checkpoint_path=meta.parent_checkpoint_path,
        parent_checkpoint_fingerprint=meta.parent_checkpoint_fingerprint,
        retrain_reason=meta.retrain_reason,
        data_start=meta.data_start,
        data_end=meta.data_end,
        symbols=list(meta.symbols),
        paper_observation_count=meta.paper_observation_count,
        config_snapshot=dict(meta.config_snapshot),
        started_at=meta.started_at,
        completed_at=datetime.now(UTC).isoformat(),
        output_checkpoint_path=output_checkpoint_path,
        output_checkpoint_fingerprint=output_checkpoint_fingerprint,
        validation_metrics=dict(validation_metrics),
        promotion_eligible=promotion_eligible,
    )


def save_retrain_metadata(meta: RetainMetadata, *, runs_dir: str = "runs") -> Path:
    """Save retrain metadata as JSON."""
    path = Path(runs_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"{meta.retrain_id}.json"
    payload = {
        "retrain_id": meta.retrain_id,
        "parent_checkpoint_path": meta.parent_checkpoint_path,
        "parent_checkpoint_fingerprint": meta.parent_checkpoint_fingerprint,
        "retrain_reason": meta.retrain_reason,
        "data_start": meta.data_start,
        "data_end": meta.data_end,
        "symbols": meta.symbols,
        "paper_observation_count": meta.paper_observation_count,
        "config_snapshot": meta.config_snapshot,
        "started_at": meta.started_at,
        "completed_at": meta.completed_at,
        "output_checkpoint_path": meta.output_checkpoint_path,
        "output_checkpoint_fingerprint": meta.output_checkpoint_fingerprint,
        "validation_metrics": meta.validation_metrics,
        "promotion_eligible": meta.promotion_eligible,
        "notes": meta.notes,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("retrain_metadata_saved path=%s retrain_id=%s", str(filepath), meta.retrain_id)
    return filepath


def load_retrain_metadata(path: str | Path) -> RetainMetadata:
    """Load retrain metadata from JSON."""
    with open(path) as f:
        payload = json.load(f)
    return RetainMetadata(
        retrain_id=str(payload["retrain_id"]),
        parent_checkpoint_path=payload.get("parent_checkpoint_path"),
        parent_checkpoint_fingerprint=payload.get("parent_checkpoint_fingerprint"),
        retrain_reason=str(payload["retrain_reason"]),
        data_start=str(payload["data_start"]),
        data_end=str(payload["data_end"]),
        symbols=list(payload.get("symbols", [])),
        paper_observation_count=int(payload.get("paper_observation_count", 0)),
        config_snapshot=dict(payload.get("config_snapshot", {})),
        started_at=str(payload["started_at"]),
        completed_at=payload.get("completed_at"),
        output_checkpoint_path=payload.get("output_checkpoint_path"),
        output_checkpoint_fingerprint=payload.get("output_checkpoint_fingerprint"),
        validation_metrics=dict(payload.get("validation_metrics", {})),
        promotion_eligible=bool(payload.get("promotion_eligible", False)),
        notes=str(payload.get("notes", "")),
    )


def list_retrain_history(runs_dir: str = "runs") -> list[RetainMetadata]:
    """List all retrain metadata files, newest first."""
    path = Path(runs_dir)
    if not path.exists():
        return []
    results = []
    for filepath in sorted(path.glob("retrain-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            results.append(load_retrain_metadata(filepath))
        except Exception:
            continue
    return results


def get_checkpoint_lineage(
    retrain_id: str,
    *,
    runs_dir: str = "runs",
    max_depth: int = 50,
) -> list[RetainMetadata]:
    """Trace the lineage of retrains from a given retrain_id back to genesis."""
    all_meta = {m.retrain_id: m for m in list_retrain_history(runs_dir)}
    lineage: list[RetainMetadata] = []
    current_id: str | None = retrain_id
    depth = 0
    while current_id and current_id in all_meta and depth < max_depth:
        meta = all_meta[current_id]
        lineage.append(meta)
        # Find parent by matching output fingerprint to parent fingerprint
        parent_fp = meta.parent_checkpoint_fingerprint
        if parent_fp is None:
            break
        current_id = None
        for candidate in all_meta.values():
            if candidate.output_checkpoint_fingerprint == parent_fp:
                current_id = candidate.retrain_id
                break
        depth += 1
    return lineage


def validate_retrain_prerequisites(
    *,
    parent_checkpoint_path: str | None,
    min_paper_observations: int = 100,
    paper_observation_count: int = 0,
) -> tuple[bool, list[str]]:
    """Check whether an offline retrain should proceed.

    Returns (can_proceed, list_of_warnings).
    """
    warnings: list[str] = []
    can_proceed = True

    if parent_checkpoint_path is not None and not Path(parent_checkpoint_path).exists():
        warnings.append(f"Parent checkpoint not found: {parent_checkpoint_path}")
        can_proceed = False

    if paper_observation_count < min_paper_observations:
        warnings.append(
            f"Only {paper_observation_count} paper observations "
            f"(minimum {min_paper_observations} recommended)"
        )

    return can_proceed, warnings


# Per-metric drift thresholds. The old shared 0.15 conflated two very
# different alarms: a 0.15 Sharpe drop is noise while a 0.15 win-rate
# drop is catastrophe. min_observations gates the check on sample size.
DEFAULT_RETRAIN_TRIGGERS: dict[str, float] = {
    "sharpe_drop": 0.5,
    "win_rate_drop": 0.05,
    "min_observations": 200,
}


def check_retrain_trigger(
    *,
    recent_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    thresholds: Mapping[str, float] | None = None,
    observation_count: int | None = None,
) -> tuple[bool, str]:
    """Check if performance has drifted enough to trigger a retrain.

    ``thresholds`` carries per-metric drops (``training.retrain_triggers``);
    missing keys fall back to ``DEFAULT_RETRAIN_TRIGGERS``.  When
    ``observation_count`` is provided, the check refuses to fire below
    ``min_observations`` — drift measured on a handful of trades is noise.

    Returns (should_retrain, reason).
    """
    merged = {**DEFAULT_RETRAIN_TRIGGERS, **dict(thresholds or {})}
    min_observations = int(merged["min_observations"])
    if observation_count is not None and observation_count < min_observations:
        return False, f"insufficient_observations_{observation_count}<{min_observations}"
    sharpe_drop = baseline_metrics.get("sharpe_ratio", 0.0) - recent_metrics.get("sharpe_ratio", 0.0)
    if sharpe_drop > merged["sharpe_drop"]:
        return True, f"sharpe_drift_{sharpe_drop:.3f}"
    win_rate_drop = baseline_metrics.get("win_rate", 0.0) - recent_metrics.get("win_rate", 0.0)
    if win_rate_drop > merged["win_rate_drop"]:
        return True, f"win_rate_drift_{win_rate_drop:.3f}"
    return False, "no_drift"


def compute_outcome_metrics(observations: Sequence[Any]) -> dict[str, float]:
    """Trigger metrics from a slice of the per-decision outcome stream.

    ``observations`` are ``DecisionObservation``-shaped records (duck-typed:
    ``executed_action`` / ``realized_pnl``).  FLAT decisions place no trade
    and are excluded.  ``sharpe_ratio`` is per-trade mean/std with no
    annualization — the trigger only compares it against its own baseline,
    so the scale cancels.
    """
    pnls = [float(obs.realized_pnl) for obs in observations if int(obs.executed_action) != 0]
    if not pnls:
        return {"sharpe_ratio": 0.0, "win_rate": 0.0, "trade_count": 0.0}
    mean = sum(pnls) / len(pnls)
    if len(pnls) > 1:
        variance = sum((pnl - mean) ** 2 for pnl in pnls) / (len(pnls) - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return {
        "sharpe_ratio": mean / std if std > 0 else 0.0,
        "win_rate": sum(1 for pnl in pnls if pnl > 0) / len(pnls),
        "trade_count": float(len(pnls)),
    }


def check_retrain_trigger_from_observations(
    observations: Sequence[Any],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> tuple[bool, str]:
    """Evaluate the drift trigger over a persisted outcome stream.

    The stream is ordered by timestamp and split at the midpoint: the
    older half is the baseline, the newer half is "recent".  Returns
    (should_retrain, reason).
    """
    ordered = sorted(observations, key=lambda obs: obs.timestamp)
    count = len(ordered)
    midpoint = count // 2
    baseline_metrics = compute_outcome_metrics(ordered[:midpoint])
    recent_metrics = compute_outcome_metrics(ordered[midpoint:])
    return check_retrain_trigger(
        recent_metrics=recent_metrics,
        baseline_metrics=baseline_metrics,
        thresholds=thresholds,
        observation_count=count,
    )


def _file_fingerprint(path: str | Path) -> str:
    digest = sha1()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_triggered_retrain(
    *,
    observations: Sequence[Any],
    train_fn: Callable[[], Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
    parent_checkpoint_path: str | None = None,
    parent_checkpoint_fingerprint: str | None = None,
    config_snapshot: dict[str, Any] | None = None,
    runs_dir: str = "runs",
) -> RetainMetadata | None:
    """Evaluate the trigger; if it fires, run ``train_fn`` with lineage recorded.

    ``train_fn`` is the retrain entry point (a ``run_supervised_training``
    invocation) returning its result dict; its ``"checkpoint"`` key becomes
    the lineage output.  Returns the finalized :class:`RetainMetadata`
    (persisted under ``runs_dir``) or ``None`` when the trigger did not fire.
    """
    should_retrain, reason = check_retrain_trigger_from_observations(
        observations, thresholds=thresholds
    )
    if not should_retrain:
        logger.info("retrain_trigger_not_fired reason=%s", reason)
        return None
    ordered = sorted(observations, key=lambda obs: obs.timestamp)
    meta = build_retrain_metadata(
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_fingerprint=parent_checkpoint_fingerprint,
        retrain_reason=reason,
        data_start=ordered[0].timestamp.isoformat(),
        data_end=ordered[-1].timestamp.isoformat(),
        symbols=sorted({str(obs.symbol) for obs in ordered}),
        paper_observation_count=len(ordered),
        config_snapshot=dict(config_snapshot or {}),
    )
    logger.info(
        "retrain_trigger_fired retrain_id=%s reason=%s observations=%d",
        meta.retrain_id, reason, len(ordered),
    )
    result = train_fn()
    output_checkpoint_path = str(result.get("checkpoint", "")) if result else ""
    output_fingerprint = ""
    if output_checkpoint_path and Path(output_checkpoint_path).exists():
        output_fingerprint = _file_fingerprint(output_checkpoint_path)
    validation_metrics: dict[str, float] = {}
    reports = result.get("evaluation_reports", {}) if result else {}
    val_report = reports.get("val", {}) if isinstance(reports, Mapping) else {}
    for key in ("balanced_accuracy", "macro_f1", "accuracy"):
        if isinstance(val_report, Mapping) and key in val_report:
            validation_metrics[f"val_{key}"] = float(val_report[key])
    finalized = finalize_retrain_metadata(
        meta,
        output_checkpoint_path=output_checkpoint_path,
        output_checkpoint_fingerprint=output_fingerprint,
        validation_metrics=validation_metrics,
        promotion_eligible=False,  # eligibility is the promotion gate's call
    )
    save_retrain_metadata(finalized, runs_dir=runs_dir)
    return finalized
