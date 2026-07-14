"""Counterfactual utility credit for memory influence (MASTER_PLAN §6.2).

Memory influence is only allowed to stay enabled if it EARNS it: paired
backtests over identical out-of-sample folds — identical bars, identical
seeds, identical policy stack — differing in exactly one thing
(``memory.influence_mode``: "off" vs "damping"). The per-fold metric
deltas are the causal credit attributable to memory. A report with
``credit_positive=False`` means memory influence must remain off.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from src.backtest.engine import BacktestEngine
from src.core.config import AppConfig
from src.core.logging import get_logger
from src.core.seeds import seed_everything
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager

logger = get_logger("evaluation.memory_credit")

# A fold whose baseline arm produced fewer fills than this is too thin to
# attribute anything to memory and is excluded (reported, not averaged).
_MIN_FILLS_PER_FOLD = 5

# policy_builder(config, fold_bars, pipeline) -> policy for one arm.
PolicyBuilder = Callable[[AppConfig, dict[str, list], FeaturePipeline], object]


@dataclass(frozen=True)
class MemoryCreditFold:
    fold_index: int
    bar_counts: dict[str, int]
    baseline_metrics: dict[str, float]
    memory_metrics: dict[str, float]
    delta_total_return: float
    delta_expectancy: float
    delta_sharpe: float
    degenerate: bool = False


@dataclass(frozen=True)
class MemoryCreditReport:
    folds: list[MemoryCreditFold]
    mean_delta_total_return: float
    mean_delta_expectancy: float
    mean_delta_sharpe: float
    positive_folds: int
    active_folds: int
    degenerate_folds: int
    sign_test_p_value: float
    credit_positive: bool
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_delta_total_return": self.mean_delta_total_return,
            "mean_delta_expectancy": self.mean_delta_expectancy,
            "mean_delta_sharpe": self.mean_delta_sharpe,
            "positive_folds": self.positive_folds,
            "active_folds": self.active_folds,
            "degenerate_folds": self.degenerate_folds,
            "sign_test_p_value": self.sign_test_p_value,
            "credit_positive": self.credit_positive,
            "folds": [
                {
                    "fold_index": f.fold_index,
                    "bar_counts": f.bar_counts,
                    "delta_total_return": f.delta_total_return,
                    "delta_expectancy": f.delta_expectancy,
                    "delta_sharpe": f.delta_sharpe,
                    "degenerate": f.degenerate,
                    "baseline_metrics": f.baseline_metrics,
                    "memory_metrics": f.memory_metrics,
                }
                for f in self.folds
            ],
        }


def sign_test_p_value(positive: int, total: int) -> float:
    """One-sided binomial tail P(X >= positive | n=total, p=0.5).

    Small-n exact computation — the number of folds is single digits.
    """
    if total <= 0:
        return 1.0
    tail = 0.0
    for k in range(positive, total + 1):
        tail += math.comb(total, k) * 0.5**total
    return min(1.0, tail)


def _slice_folds(
    bars_by_symbol: dict[str, list],
    n_folds: int,
) -> list[dict[str, list]]:
    """Split each symbol's bars into n contiguous chronological chunks."""
    folds: list[dict[str, list]] = []
    for fold_index in range(n_folds):
        fold_bars: dict[str, list] = {}
        for symbol, bars in bars_by_symbol.items():
            start = (fold_index * len(bars)) // n_folds
            end = ((fold_index + 1) * len(bars)) // n_folds
            if end - start > 0:
                fold_bars[symbol] = bars[start:end]
        if fold_bars:
            folds.append(fold_bars)
    return folds


def _run_arm(
    config: AppConfig,
    fold_bars: dict[str, list],
    policy_builder: PolicyBuilder,
    *,
    seed: int,
) -> dict[str, float]:
    """One backtest arm. Seeded identically for both arms of a pair."""
    seed_everything(seed)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    policy = policy_builder(config, fold_bars, pipeline)
    report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        fold_bars, policy
    )
    return dict(report.metrics)


def evaluate_memory_credit(
    config: AppConfig,
    bars_by_symbol: dict[str, list],
    policy_builder: PolicyBuilder,
    *,
    n_folds: int = 4,
) -> MemoryCreditReport:
    """Paired brain-only vs memory-damped backtests over contiguous folds.

    ``bars_by_symbol`` must already be out-of-sample for the checkpoint
    under test (callers enforce via src.evaluation.oos). The two arms of
    each fold share bars, seed, and full config except
    ``memory.influence_mode``.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    baseline_config = config.model_copy(deep=True)
    baseline_config.memory.influence_mode = "off"
    memory_config = config.model_copy(deep=True)
    memory_config.memory.influence_mode = "damping"

    folds: list[MemoryCreditFold] = []
    for fold_index, fold_bars in enumerate(_slice_folds(bars_by_symbol, n_folds)):
        seed = int(config.evaluation.seed) + fold_index
        baseline_metrics = _run_arm(baseline_config, fold_bars, policy_builder, seed=seed)
        memory_metrics = _run_arm(memory_config, fold_bars, policy_builder, seed=seed)
        degenerate = float(baseline_metrics.get("fill_count", 0.0)) < _MIN_FILLS_PER_FOLD
        fold = MemoryCreditFold(
            fold_index=fold_index,
            bar_counts={symbol: len(bars) for symbol, bars in fold_bars.items()},
            baseline_metrics=baseline_metrics,
            memory_metrics=memory_metrics,
            delta_total_return=float(memory_metrics.get("total_return", 0.0))
            - float(baseline_metrics.get("total_return", 0.0)),
            delta_expectancy=float(memory_metrics.get("expectancy", 0.0))
            - float(baseline_metrics.get("expectancy", 0.0)),
            delta_sharpe=float(memory_metrics.get("sharpe_ratio", 0.0))
            - float(baseline_metrics.get("sharpe_ratio", 0.0)),
            degenerate=degenerate,
        )
        folds.append(fold)
        logger.info(
            "memory_credit_fold",
            fold=fold_index,
            delta_total_return=fold.delta_total_return,
            delta_expectancy=fold.delta_expectancy,
            baseline_fills=baseline_metrics.get("fill_count", 0.0),
            memory_fills=memory_metrics.get("fill_count", 0.0),
            degenerate=degenerate,
        )

    active = [f for f in folds if not f.degenerate]
    degenerate_count = len(folds) - len(active)
    if active:
        mean_return = sum(f.delta_total_return for f in active) / len(active)
        mean_expectancy = sum(f.delta_expectancy for f in active) / len(active)
        mean_sharpe = sum(f.delta_sharpe for f in active) / len(active)
    else:
        mean_return = mean_expectancy = mean_sharpe = 0.0
    positive = sum(1 for f in active if f.delta_total_return > 0)
    p_value = sign_test_p_value(positive, len(active))
    # The gate: memory influence earns its place only when it helps in
    # aggregate AND in the majority of folds. No folds → no credit.
    credit_positive = bool(active) and mean_return > 0.0 and positive * 2 >= len(active)

    report = MemoryCreditReport(
        folds=folds,
        mean_delta_total_return=mean_return,
        mean_delta_expectancy=mean_expectancy,
        mean_delta_sharpe=mean_sharpe,
        positive_folds=positive,
        active_folds=len(active),
        degenerate_folds=degenerate_count,
        sign_test_p_value=p_value,
        credit_positive=credit_positive,
        config_snapshot={
            "n_folds": n_folds,
            "damping_min_multiplier": config.memory.damping_min_multiplier,
            "damping_min_neighbors": config.memory.damping_min_neighbors,
            "damping_full_at_return": config.memory.damping_full_at_return,
            "checkpoint": str(config.model.checkpoint_path),
        },
    )
    logger.info(
        "memory_credit_report",
        credit_positive=report.credit_positive,
        mean_delta_total_return=report.mean_delta_total_return,
        positive_folds=f"{report.positive_folds}/{report.active_folds}",
        sign_test_p_value=report.sign_test_p_value,
        degenerate_folds=report.degenerate_folds,
    )
    return report
