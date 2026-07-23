"""Perturbed-R² signal diagnostic (MASTER_PLAN §6.4, RESEARCH_ROADMAP §2.18).

Separates signal value from structure value: run the same backtest with
the policy's direction logits progressively replaced by calibrated noise
(R² = 1.0 → 0.0) and look at the shape of the performance curve.

- Healthy: performance degrades toward zero edge as R² → 0.
- Leakage suspect: profits survive at implausibly low R² — returns are
  entering through something other than the model's signal (lookahead).
- Structure dominated: a flat curve — the risk layer / cost structure,
  not the brain, produces the returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.backtest.engine import BacktestEngine
from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.seeds import seed_everything
from src.features.feature_pipeline import FeaturePipeline
from src.models.base import ModelOutput
from src.risk.risk_manager import RiskManager

logger = get_logger("evaluation.perturbation")

DEFAULT_R2_LEVELS: tuple[float, ...] = (1.0, 0.5, 0.1, 0.01, 0.0)

# policy_builder(config, bars, pipeline) -> a policy whose ``model`` attribute
# can be wrapped (BrainPolicy or MemoryAwarePolicy-over-BrainPolicy).
PolicyBuilder = Callable[[AppConfig, dict[str, list], FeaturePipeline], object]


class PerturbedSignalModel(torch.nn.Module):
    """Wrap a trading model, mixing its action logits with calibrated noise.

    Per row: logits are standardized, mixed as
    ``sqrt(r2) * z + sqrt(1 - r2) * noise`` (noise ~ N(0,1)), then mapped
    back to the row's original scale — so ``r2=1.0`` reproduces the base
    model bit-for-bit and ``r2=0.0`` is pure noise at the same scale. For
    decoupled-head models the confidence (P of chosen class) is recomputed
    from the perturbed logits so the whole downstream pipeline sees a
    self-consistent signal. All other outputs (risk heads, value, latents)
    pass through untouched — they are the "structure" under test.
    """

    def __init__(self, base_model: torch.nn.Module, *, signal_r2: float, seed: int) -> None:
        super().__init__()
        if not 0.0 <= signal_r2 <= 1.0:
            raise ValueError("signal_r2 must be within [0, 1]")
        self.base_model = base_model
        self.signal_r2 = float(signal_r2)
        self._generator = torch.Generator().manual_seed(int(seed))
        self.decoupled_heads = bool(getattr(base_model, "decoupled_heads", False))

    def __getattr__(self, name: str):  # delegate e.g. memory_dim, return_heads
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("base_model"), name)

    def forward(self, **kwargs) -> ModelOutput:
        output = self.base_model(**kwargs)
        if self.signal_r2 >= 1.0:
            return output
        logits = output.action_logits
        mean = logits.mean(dim=-1, keepdim=True)
        std = logits.std(dim=-1, keepdim=True).clamp_min(1.0e-8)
        z = (logits - mean) / std
        noise = torch.randn(
            logits.shape, generator=self._generator, dtype=torch.float32
        ).to(device=logits.device, dtype=logits.dtype)
        mixed_z = (self.signal_r2**0.5) * z + ((1.0 - self.signal_r2) ** 0.5) * noise
        mixed = mixed_z * std + mean
        confidence = output.confidence
        if self.decoupled_heads:
            confidence = torch.softmax(mixed, dim=-1).amax(dim=-1, keepdim=True)
        return ModelOutput(
            action_logits=mixed,
            confidence=confidence,
            target_fraction=output.target_fraction,
            fused_representation=output.fused_representation,
            branch_embeddings=output.branch_embeddings,
            branch_confidences=output.branch_confidences,
            soma_weights=output.soma_weights,
            soma_activation=output.soma_activation,
            regime_weights=output.regime_weights,
            value=output.value,
            memory_state=output.memory_state,
            temporal_embedding=output.temporal_embedding,
            risk_predictions=output.risk_predictions,
            horizon_returns=output.horizon_returns,
        )


def _wrap_policy_model(policy: object, *, signal_r2: float, seed: int) -> None:
    """Wrap the direction-signal model inside a (possibly nested) policy."""
    target = policy
    if hasattr(target, "base_policy") and hasattr(target.base_policy, "model"):
        target = target.base_policy
    if not hasattr(target, "model"):
        raise TypeError("policy has no .model to perturb (need a BrainPolicy-like policy)")
    target.model = PerturbedSignalModel(target.model, signal_r2=signal_r2, seed=seed)


@dataclass(frozen=True)
class PerturbationPoint:
    signal_r2: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class PerturbationReport:
    points: list[PerturbationPoint]
    observations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [
                {"signal_r2": p.signal_r2, "metrics": p.metrics} for p in self.points
            ],
            "observations": list(self.observations),
        }


def _observations(points: list[PerturbationPoint]) -> list[str]:
    """Advisory curve-shape observations — labeled evidence, not verdicts."""
    by_r2 = {p.signal_r2: p.metrics for p in points}
    notes: list[str] = []
    full = by_r2.get(1.0)
    if full is None or len(points) < 2:
        return notes
    full_sharpe = float(full.get("sharpe_ratio", 0.0))
    low_keys = [r for r in by_r2 if r <= 0.01]
    if low_keys and full_sharpe > 0:
        low_sharpe = max(float(by_r2[r].get("sharpe_ratio", 0.0)) for r in low_keys)
        if low_sharpe >= 0.8 * full_sharpe:
            notes.append(
                "LEAKAGE SUSPECT: near-noise signal retains >=80% of full-signal "
                "sharpe — returns are not coming from the model's signal"
            )
    sharpes = [float(p.metrics.get("sharpe_ratio", 0.0)) for p in points]
    spread = max(sharpes) - min(sharpes)
    if spread < 0.1:
        notes.append(
            "STRUCTURE DOMINATED: sharpe barely moves across the R2 curve — the "
            "risk layer / cost structure, not the signal, drives the result"
        )
    if not notes:
        notes.append("curve shape consistent with signal-driven returns")
    return notes


def run_perturbation_curve(
    config: AppConfig,
    bars_by_symbol: dict[str, list],
    policy_builder: PolicyBuilder,
    *,
    r2_levels: tuple[float, ...] = DEFAULT_R2_LEVELS,
    seed: int | None = None,
) -> PerturbationReport:
    """Backtest the same bars at each signal-R² level.

    Every level shares bars and RNG seed; only the logit mixing weight
    differs, so metric differences are attributable to signal content.
    """
    base_seed = int(config.evaluation.seed if seed is None else seed)
    points: list[PerturbationPoint] = []
    for r2 in r2_levels:
        seed_everything(base_seed)
        pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
        policy = policy_builder(config, bars_by_symbol, pipeline)
        _wrap_policy_model(policy, signal_r2=r2, seed=base_seed + 1)
        report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
            bars_by_symbol, policy
        )
        metrics = {
            key: float(report.metrics.get(key, 0.0))
            for key in ("total_return", "sharpe_ratio", "expectancy", "max_drawdown", "cvar_95", "fill_count")
        }
        points.append(PerturbationPoint(signal_r2=float(r2), metrics=metrics))
        logger.info("perturbation_point", signal_r2=r2, **metrics)
    observations = _observations(points)
    for note in observations:
        logger.info("perturbation_observation", note=note)
    return PerturbationReport(points=points, observations=observations)
