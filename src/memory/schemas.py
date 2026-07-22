from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class OutcomeSummary:
    future_returns: dict[int, float]
    max_favorable_excursion: float
    max_adverse_excursion: float
    realized_volatility: float
    drawdown: float
    path_label: str
    direction_class: str
    time_to_move: int | None = None

    @property
    def primary_horizon(self) -> int:
        return max(self.future_returns)

    @property
    def primary_return(self) -> float:
        return self.future_returns[self.primary_horizon]

    def to_payload(self) -> dict[str, Any]:
        return {
            "future_returns": {str(horizon): value for horizon, value in self.future_returns.items()},
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "realized_volatility": self.realized_volatility,
            "drawdown": self.drawdown,
            "path_label": self.path_label,
            "direction_class": self.direction_class,
            "time_to_move": self.time_to_move,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "OutcomeSummary":
        return cls(
            future_returns={int(horizon): float(value) for horizon, value in payload["future_returns"].items()},
            max_favorable_excursion=float(payload["max_favorable_excursion"]),
            max_adverse_excursion=float(payload["max_adverse_excursion"]),
            realized_volatility=float(payload["realized_volatility"]),
            drawdown=float(payload["drawdown"]),
            path_label=str(payload["path_label"]),
            direction_class=str(payload["direction_class"]),
            time_to_move=None if payload.get("time_to_move") is None else int(payload["time_to_move"]),
        )


@dataclass
class RiskSummary:
    realized_volatility: float
    max_adverse_excursion: float
    drawdown: float

    def to_payload(self) -> dict[str, float]:
        return {
            "realized_volatility": self.realized_volatility,
            "max_adverse_excursion": self.max_adverse_excursion,
            "drawdown": self.drawdown,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RiskSummary":
        return cls(
            realized_volatility=float(payload["realized_volatility"]),
            max_adverse_excursion=float(payload["max_adverse_excursion"]),
            drawdown=float(payload["drawdown"]),
        )


@dataclass
class MemoryItem:
    item_id: str
    symbol: str
    timestamp: datetime
    latent: torch.Tensor
    feature_contract_id: str
    regime_weights: torch.Tensor | None
    branch_confidences: dict[str, float]
    branch_norms: dict[str, float]
    outcome: OutcomeSummary
    risk: RiskSummary
    cluster_id: str | None = None
    action_annotation: str | None = None
    encoder_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "latent": self.latent.detach().cpu(),
            "feature_contract_id": self.feature_contract_id,
            "regime_weights": None if self.regime_weights is None else self.regime_weights.detach().cpu(),
            "branch_confidences": dict(self.branch_confidences),
            "branch_norms": dict(self.branch_norms),
            "outcome": self.outcome.to_payload(),
            "risk": self.risk.to_payload(),
            "cluster_id": self.cluster_id,
            "action_annotation": self.action_annotation,
            "encoder_fingerprint": self.encoder_fingerprint,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryItem":
        return cls(
            item_id=str(payload["item_id"]),
            symbol=str(payload["symbol"]),
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            latent=payload["latent"].detach().cpu().to(dtype=torch.float32),
            feature_contract_id=str(payload["feature_contract_id"]),
            regime_weights=None
            if payload.get("regime_weights") is None
            else payload["regime_weights"].detach().cpu().to(dtype=torch.float32),
            branch_confidences={str(name): float(value) for name, value in payload["branch_confidences"].items()},
            branch_norms={str(name): float(value) for name, value in payload["branch_norms"].items()},
            outcome=OutcomeSummary.from_payload(payload["outcome"]),
            risk=RiskSummary.from_payload(payload["risk"]),
            cluster_id=None if payload.get("cluster_id") is None else str(payload["cluster_id"]),
            action_annotation=None
            if payload.get("action_annotation") is None
            else str(payload["action_annotation"]),
            encoder_fingerprint=payload.get("encoder_fingerprint"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class RetrievalHealth:
    """Running statistics from recent retrievals — the bank's homeostatic state.

    Inspired by neural homeostasis: the system tracks its own retrieval
    distribution and adapts gating sensitivity dynamically.  No fixed
    thresholds — similarity quality is measured relative to the bank's
    own observed distribution via z-score.

    Updated incrementally on each retrieval via EMA (exponential moving
    average) with rate ``adaptation_rate`` (default 0.01 = slow, stable).
    """

    similarity_ema: float = 0.5
    similarity_sq_ema: float = 0.26  # E[sim^2], for variance = E[sim^2] - E[sim]^2
    conflict_ema: float = 0.5
    entropy_ema: float = 1.0
    retrieval_count: int = 0
    trade_rate_ema: float = 0.5
    outcome_ema: float = 0.0
    warmup: int = 100
    outcome_similarity_adjustment: float = -0.1

    @property
    def similarity_var(self) -> float:
        """Variance of observed similarities."""
        return max(self.similarity_sq_ema - self.similarity_ema ** 2, 1e-6)

    @property
    def similarity_std(self) -> float:
        return self.similarity_var ** 0.5

    def update(self, *, similarity: float, conflict: float, entropy: float, alpha: float = 0.01) -> None:
        """Update running statistics with a new retrieval observation."""
        self.similarity_ema = alpha * similarity + (1 - alpha) * self.similarity_ema
        self.similarity_sq_ema = alpha * (similarity ** 2) + (1 - alpha) * self.similarity_sq_ema
        self.conflict_ema = alpha * conflict + (1 - alpha) * self.conflict_ema
        self.entropy_ema = alpha * entropy + (1 - alpha) * self.entropy_ema
        self.retrieval_count += 1

    def update_trade_outcome(self, *, traded: bool, outcome: float = 0.0, alpha: float = 0.01) -> None:
        """Update trade rate and outcome statistics after a risk decision."""
        self.trade_rate_ema = alpha * float(traded) + (1 - alpha) * self.trade_rate_ema
        if traded:
            self.outcome_ema = alpha * outcome + (1 - alpha) * self.outcome_ema

    def z_score(self, similarity: float) -> float:
        """How many standard deviations above/below the running mean."""
        z = (similarity - self.similarity_ema) / self.similarity_std
        # Clamp to prevent extreme values when std is tiny (early session).
        return max(-10.0, min(10.0, z))

    def retrieval_quality(self, similarity: float) -> float:
        """Sigmoid of z-score: continuous [0, 1] quality measure.

        Above-average similarity → quality > 0.5 → support.
        Below-average → quality < 0.5 → dampen.
        Self-calibrating: adapts to whatever similarity range the bank produces.
        """
        import math
        z = self.z_score(similarity)
        if math.isnan(z) or math.isinf(z):
            return 0.5
        return 1.0 / (1.0 + math.exp(-z))

    def adaptive_min_similarity(self, base: float = 0.75) -> float:
        """Adjust min_similarity based on outcome history.

        Positive outcomes -> lower threshold (trade more).
        Negative outcomes -> higher threshold (be more selective).

        Requires at least ``warmup`` retrievals before adjusting to avoid
        premature adaptation during cold start.
        """
        if self.retrieval_count < self.warmup:
            return base
        adjustment = self.outcome_similarity_adjustment * self.outcome_ema
        return max(0.5, min(0.9, base + adjustment))


@dataclass
class PendingMemoryState:
    symbol: str
    timestamp: datetime
    feature_contract_id: str
    latent: torch.Tensor
    regime_weights: torch.Tensor | None
    branch_confidences: dict[str, float]
    branch_norms: dict[str, float]
    action_annotation: str | None = None
    encoder_fingerprint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


