"""Platt scaling for confidence calibration.

Transforms raw confidence scores into calibrated probabilities.
P(chosen direction profitable) = sigmoid(a * raw_confidence + b)
Parameters a, b fitted by gradient descent on historical predictions.

The fitted artifact is bound to the checkpoint it calibrates: it is written
next to the checkpoint file (``<checkpoint dir>/platt_calibration.json``)
and records that checkpoint's ``encoder_fingerprint``.  Loaders resolve the
artifact relative to the checkpoint being loaded and refuse a fingerprint
mismatch (falling back to the uncalibrated passthrough — never crashing the
runtime).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical artifact filename. The calibrator always sits NEXT TO the
# checkpoint it was fitted for, so isolated experiment runs keep their
# calibrator inside their run dir and never clobber the canonical one.
PLATT_CALIBRATION_FILENAME = "platt_calibration.json"

# Direction-label indices (mirrors src/training/targets.py).
_DIRECTION_FLAT = 1
_DIRECTION_LONG = 2


def platt_calibration_path_for_checkpoint(checkpoint_path: Path | str) -> Path:
    """Resolve the calibrator artifact path for a given checkpoint."""
    return Path(checkpoint_path).parent / PLATT_CALIBRATION_FILENAME


def direction_aware_outcomes(
    predicted_actions: np.ndarray,
    primary_returns: np.ndarray,
    *,
    per_trade_cost: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-trade net outcomes for direction-aware Platt labels.

    Calibrated confidence must approximate P(chosen direction wins), not
    P(price up) — a correct SHORT call is a win even though the price fell.
    Outcome = predicted_sign x realized_return - per_trade_cost.  FLAT
    predictions place no trade and are masked out of the fit.

    Args:
        predicted_actions: (N,) direction classes (0=short, 1=flat, 2=long).
        primary_returns: (N,) realized primary-horizon returns.
        per_trade_cost: fractional round-trip cost (``transaction_cost_bps / 10_000``).

    Returns:
        ``(traded_mask, net_outcomes)`` — both (N,); only entries where
        ``traded_mask`` is True belong in the fit.
    """
    predicted_actions = np.asarray(predicted_actions)
    primary_returns = np.asarray(primary_returns, dtype=np.float64)
    traded_mask = predicted_actions != _DIRECTION_FLAT
    predicted_sign = np.where(predicted_actions == _DIRECTION_LONG, 1.0, -1.0)
    return traded_mask, predicted_sign * primary_returns - per_trade_cost


def load_calibrator_for_checkpoint(
    checkpoint_path: Path | str | None,
    *,
    expected_fingerprint: str | None,
) -> PlattCalibrator:
    """Load the calibrator bound to ``checkpoint_path``.

    Resolves ``platt_calibration.json`` next to the checkpoint and verifies
    the stored encoder fingerprint against the loaded checkpoint's.  Any
    problem degrades to an unfitted passthrough calibrator (never raises —
    a stale calibration artifact must not take down the runtime):

    * missing file / no checkpoint path → passthrough
    * fingerprint mismatch              → passthrough with a warning (refit required)
    * legacy file without a fingerprint → loads, with a warning (unverifiable)
    """
    calibrator = PlattCalibrator()
    if not checkpoint_path:
        logger.warning("platt_load_skipped reason=no_checkpoint_path")
        return calibrator
    path = platt_calibration_path_for_checkpoint(checkpoint_path)
    if not path.exists():
        logger.info(
            "platt_calibrator_missing path=%s (uncalibrated passthrough)", path
        )
        return calibrator
    calibrator.load(path)
    if not calibrator.is_fitted:
        return calibrator
    if calibrator.encoder_fingerprint is None:
        logger.warning(
            "platt_calibrator_legacy path=%s has no encoder_fingerprint; cannot "
            "verify it matches checkpoint %s — loading anyway (refit to bind it)",
            path,
            checkpoint_path,
        )
        return calibrator
    if expected_fingerprint is None:
        logger.warning(
            "platt_calibrator_unverified path=%s: checkpoint %s has no "
            "encoder_fingerprint — loading anyway",
            path,
            checkpoint_path,
        )
        return calibrator
    if calibrator.encoder_fingerprint != expected_fingerprint:
        logger.warning(
            "platt_calibrator_fingerprint_mismatch path=%s stored=%s checkpoint=%s "
            "— refusing stale calibration, falling back to uncalibrated passthrough",
            path,
            calibrator.encoder_fingerprint,
            expected_fingerprint,
        )
        return PlattCalibrator()
    return calibrator


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator. Passthrough when not fitted."""

    a: float = 1.0
    b: float = 0.0
    is_fitted: bool = False
    n_samples: int = 0
    # Provenance: the checkpoint this calibrator was fitted for. ``None`` on
    # legacy artifacts written before fingerprint binding.
    encoder_fingerprint: str | None = None
    checkpoint_filename: str | None = None

    def calibrate(self, raw_confidence: float) -> float:
        """Apply calibration. Passthrough if not fitted."""
        if not self.is_fitted:
            return float(np.clip(raw_confidence, 0.0, 1.0))
        logit = self.a * raw_confidence + self.b
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))

    def fit(self, confidences: np.ndarray, outcomes: np.ndarray, epochs: int = 2000, lr: float = 0.01) -> None:
        """Fit Platt parameters via gradient descent on binary cross-entropy.

        Args:
            confidences: Raw confidence scores (N,).
            outcomes: ROI values (N,) — positive = profitable.
            epochs: GD iterations.
            lr: Learning rate.
        """
        confidences = np.asarray(confidences, dtype=np.float64)
        labels = (np.asarray(outcomes, dtype=np.float64) > 0.0).astype(np.float64)

        if len(confidences) < 20:
            return

        a, b = 1.0, 0.0
        for _ in range(epochs):
            logits = a * confidences + b
            p = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            err = p - labels
            grad_a = float(np.clip(np.mean(err * confidences), -1.0, 1.0))
            grad_b = float(np.clip(np.mean(err), -1.0, 1.0))
            a -= lr * grad_a
            b -= lr * grad_b

        self.a = float(a)
        self.b = float(b)
        self.is_fitted = True
        self.n_samples = len(confidences)

    def expected_calibration_error(self, confidences: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
        """ECE: lower is better calibrated."""
        cal_probs = np.array([self.calibrate(float(c)) for c in confidences])
        labels = (np.asarray(outcomes) > 0.0).astype(float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(cal_probs)
        for i in range(n_bins):
            mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
            if mask.sum() == 0:
                continue
            acc = float(labels[mask].mean())
            conf = float(cal_probs[mask].mean())
            ece += (mask.sum() / n) * abs(acc - conf)
        return float(ece)

    def brier_score(self, confidences: np.ndarray, outcomes: np.ndarray) -> float:
        """Brier score: lower is better."""
        cal_probs = np.array([self.calibrate(float(c)) for c in confidences])
        labels = (np.asarray(outcomes) > 0.0).astype(float)
        return float(np.mean((cal_probs - labels) ** 2))

    def to_dict(self) -> dict:
        return {
            "a": self.a,
            "b": self.b,
            "is_fitted": self.is_fitted,
            "n_samples": self.n_samples,
            "encoder_fingerprint": self.encoder_fingerprint,
            "checkpoint_filename": self.checkpoint_filename,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PlattCalibrator:
        fingerprint = d.get("encoder_fingerprint")
        checkpoint_filename = d.get("checkpoint_filename")
        return cls(
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
            is_fitted=bool(d.get("is_fitted", False)),
            n_samples=int(d.get("n_samples", 0)),
            encoder_fingerprint=None if fingerprint is None else str(fingerprint),
            checkpoint_filename=None if checkpoint_filename is None else str(checkpoint_filename),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path: Path | str) -> None:
        path = Path(path)
        if not path.exists():
            return
        with open(path) as f:
            d = json.load(f)
        loaded = PlattCalibrator.from_dict(d)
        self.a = loaded.a
        self.b = loaded.b
        self.is_fitted = loaded.is_fitted
        self.n_samples = loaded.n_samples
        self.encoder_fingerprint = loaded.encoder_fingerprint
        self.checkpoint_filename = loaded.checkpoint_filename
