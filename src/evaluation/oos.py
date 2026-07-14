"""Out-of-sample enforcement for checkpoint evaluation.

Default backtests used to load the identical date range the checkpoint
was trained on (~70% training data, ~20% of it the val set that drove
early stopping and promotion), so every headline number was dominated by
memorization.  Checkpoint meta persists ``training_split_ranges`` — the
helpers here turn that into a hard evaluation boundary.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger("evaluation.oos")


def checkpoint_training_end(checkpoint_path: str | Path) -> datetime | None:
    """Latest train/val end timestamp recorded in the checkpoint meta.

    Returns None for checkpoints without recorded ranges (pre-fix
    artifacts) — callers should treat those as unbounded-overlap and
    warn loudly.
    """
    from src.training.checkpoints import load_checkpoint_payload

    try:
        meta = load_checkpoint_payload(checkpoint_path).get("meta", {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("oos_checkpoint_meta_unreadable", path=str(checkpoint_path), error=str(exc))
        return None
    ranges = meta.get("training_split_ranges") or {}
    latest: datetime | None = None
    for split_name in ("train", "val"):
        for payload in (ranges.get(split_name) or {}).values():
            raw = payload.get("end_timestamp")
            if not raw:
                continue
            try:
                end = datetime.fromisoformat(str(raw))
            except ValueError:
                continue
            if latest is None or end > latest:
                latest = end
    return latest


def filter_out_of_sample(
    bars_by_symbol: dict[str, list],
    cutoff: datetime,
) -> tuple[dict[str, list], int]:
    """Keep only bars strictly after ``cutoff``; drop emptied symbols.

    Returns (filtered, dropped_bar_count).
    """
    filtered: dict[str, list] = {}
    dropped = 0
    for symbol, bars in bars_by_symbol.items():
        keep = [bar for bar in bars if bar.timestamp > cutoff]
        dropped += len(bars) - len(keep)
        if keep:
            filtered[symbol] = keep
    return filtered, dropped


def enforce_out_of_sample(
    bars_by_symbol: dict[str, list],
    *,
    checkpoint_path: str | Path | None,
    allow_in_sample: bool = False,
    cutoff_override: datetime | None = None,
) -> tuple[dict[str, list], bool]:
    """Apply the OOS boundary for an evaluation of ``checkpoint_path``.

    Returns (bars, in_sample_tagged).  Hard-fails (RuntimeError) when the
    checkpoint's training range covers every requested bar and in-sample
    evaluation was not explicitly allowed.

    ``cutoff_override`` replaces the checkpoint's own training end as the
    boundary.  Comparisons between two checkpoints (the promotion gate)
    need both evaluated on the *same* window — bars after the max of both
    training ends — or their metric deltas are apples-to-oranges.
    """
    if cutoff_override is not None:
        filtered, dropped = filter_out_of_sample(bars_by_symbol, cutoff_override)
        if not filtered:
            raise RuntimeError(
                f"No out-of-sample bars after the shared cutoff "
                f"{cutoff_override.isoformat()}. Fetch newer data."
            )
        if dropped:
            logger.info(
                "oos_bars_dropped",
                cutoff=cutoff_override.isoformat(),
                dropped=dropped,
                remaining={s: len(b) for s, b in filtered.items()},
                source="cutoff_override",
            )
        return filtered, False
    if not checkpoint_path:
        return bars_by_symbol, False
    cutoff = checkpoint_training_end(checkpoint_path)
    if cutoff is None:
        logger.warning(
            "oos_no_training_ranges",
            checkpoint=str(checkpoint_path),
            note="checkpoint records no training_split_ranges; results may be in-sample",
        )
        return bars_by_symbol, True
    if allow_in_sample:
        logger.warning("oos_override_in_sample", checkpoint=str(checkpoint_path), cutoff=cutoff.isoformat())
        return bars_by_symbol, True
    filtered, dropped = filter_out_of_sample(bars_by_symbol, cutoff)
    if not filtered:
        raise RuntimeError(
            f"No out-of-sample bars: every requested bar is inside the checkpoint's "
            f"training/validation range (ends {cutoff.isoformat()}). Fetch newer data "
            f"or pass --allow-in-sample to evaluate in-sample (results will be tagged)."
        )
    if dropped:
        logger.info(
            "oos_bars_dropped",
            cutoff=cutoff.isoformat(),
            dropped=dropped,
            remaining={s: len(b) for s, b in filtered.items()},
        )
    return filtered, False
