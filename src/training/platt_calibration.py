"""Platt scaling for confidence calibration.

Transforms raw confidence scores into calibrated probabilities.
P(profitable) = sigmoid(a * raw_confidence + b)
Parameters a, b fitted by gradient descent on historical predictions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PlattCalibrator:
    """Platt scaling calibrator. Passthrough when not fitted."""

    a: float = 1.0
    b: float = 0.0
    is_fitted: bool = False
    n_samples: int = 0

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
        return {"a": self.a, "b": self.b, "is_fitted": self.is_fitted, "n_samples": self.n_samples}

    @classmethod
    def from_dict(cls, d: dict) -> PlattCalibrator:
        return cls(
            a=float(d.get("a", 1.0)),
            b=float(d.get("b", 0.0)),
            is_fitted=bool(d.get("is_fitted", False)),
            n_samples=int(d.get("n_samples", 0)),
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
        self.a = float(d.get("a", 1.0))
        self.b = float(d.get("b", 0.0))
        self.is_fitted = bool(d.get("is_fitted", False))
        self.n_samples = int(d.get("n_samples", 0))
