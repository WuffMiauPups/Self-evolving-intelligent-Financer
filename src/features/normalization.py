from __future__ import annotations

from dataclasses import dataclass, field

import torch


def safe_zscore(value: torch.Tensor, reference: torch.Tensor, epsilon: float) -> torch.Tensor:
    std = reference.std(unbiased=False)
    return (value - reference.mean()) / (std + epsilon)


@dataclass(frozen=True)
class NormalizationStats:
    """Train-set z-score statistics shared by training and inference.

    Training normalizes every model input in-place with train-split
    mean/std (``normalize_columns``).  These statistics were previously
    discarded, so backtest and live inference fed raw-scale inputs to a
    model trained and validated exclusively on z-scored inputs — the
    train/serve skew behind the observed live degeneracy.  The stats are
    now returned by ``normalize_columns``, persisted in checkpoint meta,
    and re-applied by every inference path through ``apply_*`` below so
    the transform literally cannot diverge between train and serve.

    Shapes: features ``(F,)``, regime ``(R,)``, temporal per-channel
    ``(C,)``, branches ``{name: (D,)}``.  ``clamp`` mirrors the training
    clamp so AMP-safe bounds match too.
    """

    features_mean: torch.Tensor
    features_std: torch.Tensor
    regime_mean: torch.Tensor
    regime_std: torch.Tensor
    temporal_mean: torch.Tensor
    temporal_std: torch.Tensor
    branch_means: dict[str, torch.Tensor] = field(default_factory=dict)
    branch_stds: dict[str, torch.Tensor] = field(default_factory=dict)
    clamp: float = 10.0

    @staticmethod
    def _apply(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, clamp: float) -> torch.Tensor:
        mean = mean.to(device=value.device, dtype=value.dtype)
        std = std.to(device=value.device, dtype=value.dtype)
        return ((value - mean) / std).clamp_(-clamp, clamp)

    def apply_features(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(F,)`` or ``(B, F)`` flat-feature tensor."""
        return self._apply(value, self.features_mean, self.features_std, self.clamp)

    def apply_regime(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(R,)`` or ``(B, R)`` regime tensor."""
        return self._apply(value, self.regime_mean, self.regime_std, self.clamp)

    def apply_temporal(self, value: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(C, T)`` or ``(B, C, T)`` temporal tensor per channel."""
        return self._apply(
            value,
            self.temporal_mean.reshape(-1, 1),
            self.temporal_std.reshape(-1, 1),
            self.clamp,
        )

    def apply_branch(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Normalize a ``(D,)`` or ``(B, D)`` branch-input tensor."""
        mean = self.branch_means.get(name)
        std = self.branch_stds.get(name)
        if mean is None or std is None:
            return value
        return self._apply(value, mean, std, self.clamp)

    def to_dict(self) -> dict[str, object]:
        return {
            "features_mean": self.features_mean.tolist(),
            "features_std": self.features_std.tolist(),
            "regime_mean": self.regime_mean.tolist(),
            "regime_std": self.regime_std.tolist(),
            "temporal_mean": self.temporal_mean.reshape(-1).tolist(),
            "temporal_std": self.temporal_std.reshape(-1).tolist(),
            "branch_means": {name: t.tolist() for name, t in self.branch_means.items()},
            "branch_stds": {name: t.tolist() for name, t in self.branch_stds.items()},
            "clamp": float(self.clamp),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> NormalizationStats:
        def _t(key: str) -> torch.Tensor:
            return torch.tensor(payload[key], dtype=torch.float32)

        return cls(
            features_mean=_t("features_mean"),
            features_std=_t("features_std"),
            regime_mean=_t("regime_mean"),
            regime_std=_t("regime_std"),
            temporal_mean=_t("temporal_mean"),
            temporal_std=_t("temporal_std"),
            branch_means={
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in dict(payload.get("branch_means", {})).items()
            },
            branch_stds={
                name: torch.tensor(values, dtype=torch.float32)
                for name, values in dict(payload.get("branch_stds", {})).items()
            },
            clamp=float(payload.get("clamp", 10.0)),
        )

    @classmethod
    def from_meta(cls, meta: dict[str, object] | None) -> NormalizationStats | None:
        """Parse stats from checkpoint meta; None for pre-fix checkpoints."""
        if not meta:
            return None
        payload = meta.get("normalization")
        if not isinstance(payload, dict):
            return None
        return cls.from_dict(payload)


def normalization_stats_max_rel_diff(
    a: NormalizationStats,
    b: NormalizationStats,
    *,
    denom_floor: float = 1.0e-3,
) -> dict[str, float]:
    """Per-tensor max relative difference between two stats objects.

    Used to detect train/serve (or train/eval) normalization skew: e.g.
    ``evaluate_checkpoint`` re-derives stats from the *current* data range
    while the checkpoint carries the stats it was trained with — if they
    differ materially, evaluation metrics are not comparable across runs.

    For each tensor pair the elementwise relative difference is
    ``|a - b| / max(|a|, |b|, denom_floor)`` and the tensor's score is the
    max over elements.  ``denom_floor`` keeps numerically-zero entries from
    turning fp noise into false alarms.  Shape mismatches (different
    feature contracts) and branch keys present on only one side score
    ``inf``.  Keys: ``features_mean``, ``features_std``, ``regime_mean``,
    ``regime_std``, ``temporal_mean``, ``temporal_std``,
    ``branch_mean[<name>]``, ``branch_std[<name>]``.
    """

    def _rel(x: torch.Tensor, y: torch.Tensor) -> float:
        x = x.reshape(-1).to(torch.float64)
        y = y.reshape(-1).to(torch.float64)
        if x.shape != y.shape:
            return float("inf")
        if x.numel() == 0:
            return 0.0
        denom = torch.maximum(x.abs(), y.abs()).clamp(min=denom_floor)
        return float(((x - y).abs() / denom).max())

    diffs: dict[str, float] = {
        "features_mean": _rel(a.features_mean, b.features_mean),
        "features_std": _rel(a.features_std, b.features_std),
        "regime_mean": _rel(a.regime_mean, b.regime_mean),
        "regime_std": _rel(a.regime_std, b.regime_std),
        "temporal_mean": _rel(a.temporal_mean, b.temporal_mean),
        "temporal_std": _rel(a.temporal_std, b.temporal_std),
    }
    for name in sorted(set(a.branch_means) | set(b.branch_means)):
        if name in a.branch_means and name in b.branch_means:
            diffs[f"branch_mean[{name}]"] = _rel(a.branch_means[name], b.branch_means[name])
        else:
            diffs[f"branch_mean[{name}]"] = float("inf")
    for name in sorted(set(a.branch_stds) | set(b.branch_stds)):
        if name in a.branch_stds and name in b.branch_stds:
            diffs[f"branch_std[{name}]"] = _rel(a.branch_stds[name], b.branch_stds[name])
        else:
            diffs[f"branch_std[{name}]"] = float("inf")
    return diffs

