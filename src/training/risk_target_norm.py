"""Per-symbol risk-target normalization (§7.0.4 fix).

The EV-input audit measured the risk head's MFE/MAE predictions as a
cross-sectional volatility prior: per-symbol rank correlation with
realized excursions ≈ 0, i.e. the head learned each symbol's typical
excursion LEVEL (trivially available from feature scale) and nothing
about timing.  Dividing the mfe/mae targets by the symbol's train-split
median |excursion| removes the level answer: in normalized units every
symbol's typical excursion is ~1.0, so loss reduction requires
modulating AROUND 1.0 — timing.

This is a TRAINER-TIME transform, applied after the sample cache loads
(mirroring feature z-scoring in ``_finalize_training_bundle``): raw
cached samples stay unchanged, so no cache fingerprint churn.  The
scales persist in checkpoint meta (``risk_target_scales``) and inference
multiplies the head's outputs back per symbol — the EV gate keeps
consuming raw fractions, old checkpoints (no scales in meta) are
untouched.
"""
from __future__ import annotations

import torch

from src.core.logging import get_logger

RISK_NORM_TARGETS: tuple[str, ...] = ("mfe", "mae")

# Median |excursion| below this is degenerate (dead symbol / synthetic
# fixture); the scale floors here so division can never explode targets.
_SCALE_FLOOR = 1.0e-5


def compute_risk_target_scales(train_columns) -> dict[str, dict[str, float]]:
    """{symbol: {target: median |train value|}} for mfe/mae, floored.

    Train split only — val/test reuse the train-derived scale exactly like
    feature z-scoring (no leakage; scales are per-symbol constants).
    """
    scales: dict[str, dict[str, float]] = {}
    symbols = train_columns.symbols
    for target in RISK_NORM_TARGETS:
        column = train_columns.regression_targets.get(target)
        if column is None:
            continue
        by_symbol: dict[str, list[int]] = {}
        for index, symbol in enumerate(symbols):
            by_symbol.setdefault(symbol, []).append(index)
        for symbol, indices in by_symbol.items():
            values = column[torch.tensor(indices, dtype=torch.long)]
            scale = float(values.abs().median().item())
            scales.setdefault(symbol, {})[target] = max(scale, _SCALE_FLOOR)
    return scales


def apply_risk_target_scales(columns, scales: dict[str, dict[str, float]]) -> None:
    """Divide mfe/mae target columns IN PLACE by each sample's symbol scale.

    Symbols absent from ``scales`` (cannot happen for same-universe splits;
    guarded anyway) divide by 1.0.
    """
    for target in RISK_NORM_TARGETS:
        column = columns.regression_targets.get(target)
        if column is None:
            continue
        divisors = torch.tensor(
            [scales.get(symbol, {}).get(target, 1.0) for symbol in columns.symbols],
            dtype=column.dtype,
        )
        column /= divisors


def normalize_risk_targets(train_columns, *other_splits) -> dict[str, dict[str, float]]:
    """Compute train-split scales, apply to every split, return the scales."""
    logger = get_logger("risk_target_norm")
    scales = compute_risk_target_scales(train_columns)
    if not scales:
        logger.warning("risk_target_norm_no_targets", note="no mfe/mae columns present")
        return scales
    for columns in (train_columns, *other_splits):
        apply_risk_target_scales(columns, scales)
    logger.info(
        "risk_targets_per_symbol_normalized",
        symbols=len(scales),
        targets=list(RISK_NORM_TARGETS),
        example={s: {t: round(v, 6) for t, v in ts.items()} for s, ts in list(scales.items())[:2]},
    )
    return scales
