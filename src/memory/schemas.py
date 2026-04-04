from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import torch


class UnitStatus(str, Enum):
    """Lifecycle status for memory units."""
    CANDIDATE = "candidate"
    MATURE = "mature"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"

    @classmethod
    def from_str(cls, value: str) -> "UnitStatus":
        try:
            return cls(value)
        except ValueError:
            return cls.CANDIDATE


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
class PathStatistics:
    sample_count: int
    horizon_return_mean: dict[int, float]
    horizon_return_p25: dict[int, float]
    horizon_return_p75: dict[int, float]
    max_favorable_mean: float
    max_adverse_mean: float
    max_adverse_p10: float
    realized_volatility_mean: float
    drawdown_mean: float
    drawdown_p10: float
    path_label_frequencies: dict[str, float]
    direction_probabilities: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "horizon_return_mean": {str(horizon): value for horizon, value in self.horizon_return_mean.items()},
            "horizon_return_p25": {str(horizon): value for horizon, value in self.horizon_return_p25.items()},
            "horizon_return_p75": {str(horizon): value for horizon, value in self.horizon_return_p75.items()},
            "max_favorable_mean": self.max_favorable_mean,
            "max_adverse_mean": self.max_adverse_mean,
            "max_adverse_p10": self.max_adverse_p10,
            "realized_volatility_mean": self.realized_volatility_mean,
            "drawdown_mean": self.drawdown_mean,
            "drawdown_p10": self.drawdown_p10,
            "path_label_frequencies": dict(self.path_label_frequencies),
            "direction_probabilities": dict(self.direction_probabilities),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PathStatistics":
        return cls(
            sample_count=int(payload["sample_count"]),
            horizon_return_mean={int(horizon): float(value) for horizon, value in payload["horizon_return_mean"].items()},
            horizon_return_p25={int(horizon): float(value) for horizon, value in payload["horizon_return_p25"].items()},
            horizon_return_p75={int(horizon): float(value) for horizon, value in payload["horizon_return_p75"].items()},
            max_favorable_mean=float(payload["max_favorable_mean"]),
            max_adverse_mean=float(payload["max_adverse_mean"]),
            max_adverse_p10=float(payload["max_adverse_p10"]),
            realized_volatility_mean=float(payload["realized_volatility_mean"]),
            drawdown_mean=float(payload["drawdown_mean"]),
            drawdown_p10=float(payload["drawdown_p10"]),
            path_label_frequencies={str(name): float(value) for name, value in payload["path_label_frequencies"].items()},
            direction_probabilities={
                str(name): float(value) for name, value in payload["direction_probabilities"].items()
            },
        )


@dataclass
class RiskStatistics:
    sample_count: int
    realized_volatility_mean: float
    realized_volatility_p75: float
    max_adverse_mean: float
    max_adverse_p10: float
    drawdown_mean: float
    drawdown_p10: float

    def to_payload(self) -> dict[str, float]:
        return {
            "sample_count": self.sample_count,
            "realized_volatility_mean": self.realized_volatility_mean,
            "realized_volatility_p75": self.realized_volatility_p75,
            "max_adverse_mean": self.max_adverse_mean,
            "max_adverse_p10": self.max_adverse_p10,
            "drawdown_mean": self.drawdown_mean,
            "drawdown_p10": self.drawdown_p10,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RiskStatistics":
        return cls(
            sample_count=int(payload["sample_count"]),
            realized_volatility_mean=float(payload["realized_volatility_mean"]),
            realized_volatility_p75=float(payload["realized_volatility_p75"]),
            max_adverse_mean=float(payload["max_adverse_mean"]),
            max_adverse_p10=float(payload["max_adverse_p10"]),
            drawdown_mean=float(payload["drawdown_mean"]),
            drawdown_p10=float(payload["drawdown_p10"]),
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
class PrototypeCluster:
    cluster_id: str
    centroid: torch.Tensor
    member_indices: list[int]
    member_count: int
    average_similarity: float
    min_similarity: float
    max_similarity: float
    similarity_std: float
    path_stats: PathStatistics
    risk_stats: RiskStatistics
    regime_composition: dict[str, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": self.centroid.detach().cpu(),
            "member_indices": list(self.member_indices),
            "member_count": self.member_count,
            "average_similarity": self.average_similarity,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
            "similarity_std": self.similarity_std,
            "path_stats": self.path_stats.to_payload(),
            "risk_stats": self.risk_stats.to_payload(),
            "regime_composition": dict(self.regime_composition),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "PrototypeCluster":
        return cls(
            cluster_id=str(payload["cluster_id"]),
            centroid=payload["centroid"].detach().cpu().to(dtype=torch.float32),
            member_indices=[int(index) for index in payload["member_indices"]],
            member_count=int(payload["member_count"]),
            average_similarity=float(payload["average_similarity"]),
            min_similarity=float(payload["min_similarity"]),
            max_similarity=float(payload["max_similarity"]),
            similarity_std=float(payload["similarity_std"]),
            path_stats=PathStatistics.from_payload(payload["path_stats"]),
            risk_stats=RiskStatistics.from_payload(payload["risk_stats"]),
            regime_composition={str(name): float(value) for name, value in payload["regime_composition"].items()},
        )


@dataclass
class MemoryUnit:
    unit_id: str
    status: str
    prototype: torch.Tensor
    support_count: int
    member_indices: list[int]
    average_similarity: float
    min_similarity: float
    max_similarity: float
    similarity_std: float
    path_stats: PathStatistics
    risk_stats: RiskStatistics
    regime_composition: dict[str, float]
    symbol_composition: dict[str, float]
    session_composition: dict[str, float]
    utility_score: float
    maturity_score: float
    latent_dispersion: float
    outcome_dispersion: float
    created_at: datetime
    updated_at: datetime
    created_step: int
    updated_step: int
    calibration_confidence: float = 0.0
    retrieval_hits: int = 0
    conflict_rate: float = 0.0
    # --- SatSOM-inspired saturation / plasticity fields ---
    saturation_score: float = 0.0
    plasticity_factor: float = 1.0
    cumulative_updates: int = 0
    # --- Conflict history tracking ---
    conflict_history: list[float] = field(default_factory=list)
    recent_conflict_rate: float = 0.0
    # --- GNG-inspired topology fields ---
    edge_error: float = 0.0
    local_error_accumulator: float = 0.0
    insertion_candidate_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status_enum(self) -> UnitStatus:
        return UnitStatus.from_str(self.status)

    @status_enum.setter
    def status_enum(self, value: UnitStatus) -> None:
        self.status = value.value

    def to_payload(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "prototype": self.prototype.detach().cpu(),
            "support_count": self.support_count,
            "member_indices": list(self.member_indices),
            "average_similarity": self.average_similarity,
            "min_similarity": self.min_similarity,
            "max_similarity": self.max_similarity,
            "similarity_std": self.similarity_std,
            "path_stats": self.path_stats.to_payload(),
            "risk_stats": self.risk_stats.to_payload(),
            "regime_composition": dict(self.regime_composition),
            "symbol_composition": dict(self.symbol_composition),
            "session_composition": dict(self.session_composition),
            "utility_score": self.utility_score,
            "maturity_score": self.maturity_score,
            "latent_dispersion": self.latent_dispersion,
            "outcome_dispersion": self.outcome_dispersion,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_step": self.created_step,
            "updated_step": self.updated_step,
            "calibration_confidence": self.calibration_confidence,
            "retrieval_hits": self.retrieval_hits,
            "conflict_rate": self.conflict_rate,
            "saturation_score": self.saturation_score,
            "plasticity_factor": self.plasticity_factor,
            "cumulative_updates": self.cumulative_updates,
            "conflict_history": list(self.conflict_history),
            "recent_conflict_rate": self.recent_conflict_rate,
            "edge_error": self.edge_error,
            "local_error_accumulator": self.local_error_accumulator,
            "insertion_candidate_score": self.insertion_candidate_score,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryUnit":
        return cls(
            unit_id=str(payload["unit_id"]),
            status=str(payload["status"]),
            prototype=payload["prototype"].detach().cpu().to(dtype=torch.float32),
            support_count=int(payload["support_count"]),
            member_indices=[int(index) for index in payload["member_indices"]],
            average_similarity=float(payload["average_similarity"]),
            min_similarity=float(payload["min_similarity"]),
            max_similarity=float(payload["max_similarity"]),
            similarity_std=float(payload["similarity_std"]),
            path_stats=PathStatistics.from_payload(payload["path_stats"]),
            risk_stats=RiskStatistics.from_payload(payload["risk_stats"]),
            regime_composition={str(name): float(value) for name, value in payload["regime_composition"].items()},
            symbol_composition={str(name): float(value) for name, value in payload["symbol_composition"].items()},
            session_composition={str(name): float(value) for name, value in payload["session_composition"].items()},
            utility_score=float(payload["utility_score"]),
            maturity_score=float(payload["maturity_score"]),
            latent_dispersion=float(payload["latent_dispersion"]),
            outcome_dispersion=float(payload["outcome_dispersion"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
            updated_at=datetime.fromisoformat(str(payload["updated_at"])),
            created_step=int(payload["created_step"]),
            updated_step=int(payload["updated_step"]),
            calibration_confidence=float(payload.get("calibration_confidence", 0.0)),
            retrieval_hits=int(payload.get("retrieval_hits", 0)),
            conflict_rate=float(payload.get("conflict_rate", 0.0)),
            saturation_score=float(payload.get("saturation_score", 0.0)),
            plasticity_factor=float(payload.get("plasticity_factor", 1.0)),
            cumulative_updates=int(payload.get("cumulative_updates", 0)),
            conflict_history=[float(v) for v in payload.get("conflict_history", [])],
            recent_conflict_rate=float(payload.get("recent_conflict_rate", 0.0)),
            edge_error=float(payload.get("edge_error", 0.0)),
            local_error_accumulator=float(payload.get("local_error_accumulator", 0.0)),
            insertion_candidate_score=float(payload.get("insertion_candidate_score", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class RetrievalCalibrationBucket:
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_similarity: float
    direction_agreement: float
    outcome_dispersion: float
    conflict_rate: float
    calibrated_confidence: float

    def to_payload(self) -> dict[str, float]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "sample_count": self.sample_count,
            "mean_similarity": self.mean_similarity,
            "direction_agreement": self.direction_agreement,
            "outcome_dispersion": self.outcome_dispersion,
            "conflict_rate": self.conflict_rate,
            "calibrated_confidence": self.calibrated_confidence,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RetrievalCalibrationBucket":
        return cls(
            lower_bound=float(payload["lower_bound"]),
            upper_bound=float(payload["upper_bound"]),
            sample_count=int(payload["sample_count"]),
            mean_similarity=float(payload["mean_similarity"]),
            direction_agreement=float(payload["direction_agreement"]),
            outcome_dispersion=float(payload["outcome_dispersion"]),
            conflict_rate=float(payload["conflict_rate"]),
            calibrated_confidence=float(payload["calibrated_confidence"]),
        )


@dataclass
class RetrievalCalibrationProfile:
    metric: str
    bucket_edges: list[float]
    buckets: list[RetrievalCalibrationBucket]
    sample_count: int
    recommended_strong_match_threshold: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "bucket_edges": list(self.bucket_edges),
            "buckets": [bucket.to_payload() for bucket in self.buckets],
            "sample_count": self.sample_count,
            "recommended_strong_match_threshold": self.recommended_strong_match_threshold,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RetrievalCalibrationProfile":
        return cls(
            metric=str(payload["metric"]),
            bucket_edges=[float(value) for value in payload["bucket_edges"]],
            buckets=[RetrievalCalibrationBucket.from_payload(item) for item in payload["buckets"]],
            sample_count=int(payload["sample_count"]),
            recommended_strong_match_threshold=float(payload["recommended_strong_match_threshold"]),
        )


@dataclass(frozen=True)
class RetrievedStateMatch:
    item_index: int
    item_id: str
    symbol: str
    timestamp: datetime
    similarity_raw: float
    similarity: float
    cluster_id: str | None
    outcome: OutcomeSummary
    risk: RiskSummary


@dataclass(frozen=True)
class RetrievedClusterMatch:
    cluster_id: str
    similarity_raw: float
    similarity: float
    member_count: int
    path_stats: PathStatistics
    risk_stats: RiskStatistics
    regime_composition: dict[str, float]


@dataclass(frozen=True)
class MemoryDecisionSupport:
    top_matches: list[RetrievedStateMatch]
    cluster_matches: list[RetrievedClusterMatch]
    match_strength: float
    average_similarity: float
    weak_match: bool
    no_trade_suggested: bool
    reason_code: str
    directional_bias: int
    size_multiplier: float
    path_distribution: dict[str, float]
    risk_summary: RiskStatistics | None
    path_summary: PathStatistics | None
    calibrated_match_strength: float = 0.0
    retrieval_confidence: float = 0.0
    conflict_score: float = 0.0
    context_vector: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class EncoderQualityReport:
    item_count: int
    unit_count: int
    nearest_neighbor_similarity: float
    nearest_neighbor_directional_consistency: float
    average_outcome_dispersion: float
    unit_compactness: float
    unit_separation: float
    symbol_stability: dict[str, float]
    similarity_bucket_consistency: dict[str, float]
    regime_stability: dict[str, float] = field(default_factory=dict)
    retrieval_coverage: float = 0.0
    weak_match_rate: float = 0.0
    conflict_rate: float = 0.0
    average_calibrated_confidence: float = 0.0


@dataclass
class MemoryHealthReport:
    total_units: int
    mature_units: int
    candidate_units: int
    archived_units: int
    average_support_count: float
    average_utility: float
    stale_unit_ratio: float
    average_compactness: float
    weak_match_rate: float = 0.0
    unknown_state_rate: float = 0.0
    mature_candidate_ratio: float = 0.0
    average_retrieval_hits: float = 0.0
    topology_density: float = 0.0
    # --- SatSOM / GNG diagnostics ---
    average_saturation: float = 0.0
    average_plasticity: float = 0.0
    saturated_unit_count: int = 0
    high_plasticity_unit_count: int = 0
    conflict_heavy_unit_count: int = 0
    average_conflict_rate: float = 0.0
    graph_edge_count: int = 0
    average_edge_age: float = 0.0
    create_vs_update_ratio: float = 0.0
    quarantined_units: int = 0
