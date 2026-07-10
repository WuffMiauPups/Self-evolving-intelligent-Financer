"""Walk-forward cross-validation engine.

Splits historical data into strictly chronological train/val/test folds
and evaluates a policy on each test fold, so every test window is
structurally out-of-sample relative to the data that precedes it.
Between folds a fresh pipeline and policy are built, per-symbol policy
state is reset, and (when ``reset_memory_per_fold`` is set) the memory
bank is rebuilt from that fold's training bars with a forward-outcome
embargo.  Degenerate folds (zero fills — the policy never traded) are
excluded from aggregate statistics and reported separately under
``degenerate_folds``.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import torch

from src.backtest.engine import BacktestEngine
from src.backtest.performance import BacktestReport
from src.core.config import AppConfig, WalkForwardConfig
from src.core.seeds import seed_everything
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager
from src.strategies.base import MemoryCapablePolicy, TradingPolicy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldDefinition:
    """Immutable specification of a single walk-forward fold."""

    fold_index: int
    train_start: int
    train_end: int  # exclusive
    val_start: int
    val_end: int  # exclusive
    test_start: int
    test_end: int  # exclusive


@dataclass
class FoldResult:
    """Results from evaluating a single fold."""

    fold_definition: FoldDefinition
    test_metrics: dict[str, float]
    backtest_report: BacktestReport
    elapsed_seconds: float


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds."""

    folds: list[FoldResult]
    aggregate_metrics: dict[str, dict[str, float]]
    config_snapshot: dict[str, Any]
    fold_definitions: list[FoldDefinition]
    total_elapsed_seconds: float = 0.0


def generate_folds(total_bars: int, config: WalkForwardConfig) -> list[FoldDefinition]:
    """Generate non-overlapping chronological fold definitions.

    Each fold has strictly sequential train -> val -> test windows.
    The train window either has a fixed size (rolling) or expands
    from bar 0 (expanding).  Test windows never overlap.

    Returns an empty list if the data is too short for even one fold.
    """
    folds: list[FoldDefinition] = []
    fold_size = config.train_bars + config.val_bars + config.test_bars

    if total_bars < fold_size:
        return folds

    fold_index = 0
    # Start the first fold's test window right after train+val
    test_start = config.train_bars + config.val_bars

    while test_start + config.test_bars <= total_bars:
        test_end = test_start + config.test_bars
        val_end = test_start
        val_start = val_end - config.val_bars

        if config.window_type == "expanding":
            train_start = 0
        else:  # rolling
            train_start = val_start - config.train_bars

        train_end = val_start

        if train_start < 0:
            break
        if train_end - train_start < config.min_train_bars:
            break

        folds.append(FoldDefinition(
            fold_index=fold_index,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
        ))
        fold_index += 1
        test_start += config.step_bars

    return folds


def _slice_bars(
    bars_by_symbol: dict[str, list[MarketBar]],
    start: int,
    end: int,
) -> dict[str, list[MarketBar]]:
    """Slice all symbols to [start, end) bar indices."""
    return {
        symbol: bars[start:end]
        for symbol, bars in bars_by_symbol.items()
    }


# Type alias for a function that builds a policy from config + data.
PolicyBuilder = Callable[
    [AppConfig, dict[str, list[MarketBar]], FeaturePipeline],
    object,
]


class WalkForwardRunner:
    """Runs walk-forward evaluation across chronological folds.

    Usage::

        runner = WalkForwardRunner(config, bars_by_symbol)
        result = runner.run(my_policy_builder)
        print(result.aggregate_metrics)
    """

    def __init__(
        self,
        config: AppConfig,
        bars_by_symbol: dict[str, list[MarketBar]],
    ) -> None:
        self.config = config
        self.bars_by_symbol = bars_by_symbol

    def run(self, policy_builder: PolicyBuilder) -> WalkForwardResult:
        """Execute walk-forward evaluation.

        Args:
            policy_builder: A callable that takes (config, bars_by_symbol, pipeline)
                and returns a policy object with a ``propose(snapshot)`` method.
                Called once per fold with the fold's training data, so the policy
                can be trained/initialized on that data only.

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        wf_config = self.config.evaluation.walk_forward
        min_bars = min(len(bars) for bars in self.bars_by_symbol.values())
        fold_defs = generate_folds(min_bars, wf_config)

        if not fold_defs:
            logger.warning(
                "walk_forward_no_folds total_bars=%d train=%d val=%d test=%d",
                min_bars, wf_config.train_bars, wf_config.val_bars, wf_config.test_bars,
            )
            return WalkForwardResult(
                folds=[],
                aggregate_metrics={},
                config_snapshot=self.config.model_dump(mode="json"),
                fold_definitions=[],
            )

        logger.info("walk_forward_start folds=%d", len(fold_defs))
        total_start = time.monotonic()
        fold_results: list[FoldResult] = []

        for fold_def in fold_defs:
            fold_start = time.monotonic()
            logger.info(
                "walk_forward_fold fold=%d train=[%d,%d) val=[%d,%d) test=[%d,%d)",
                fold_def.fold_index,
                fold_def.train_start, fold_def.train_end,
                fold_def.val_start, fold_def.val_end,
                fold_def.test_start, fold_def.test_end,
            )

            # Seed for reproducibility
            seed_everything(self.config.evaluation.seed + fold_def.fold_index)

            # Slice data for this fold — train bars for policy building,
            # test bars for evaluation
            train_bars = _slice_bars(
                self.bars_by_symbol, fold_def.train_start, fold_def.train_end
            )
            test_bars = _slice_bars(
                self.bars_by_symbol, fold_def.test_start, fold_def.test_end
            )

            # Build fresh pipeline and policy for this fold
            pipeline = FeaturePipeline(self.config.features, device=torch.device("cpu"))
            policy = policy_builder(self.config, train_bars, pipeline)

            # Reset all mutable state to prevent cross-fold leakage
            if isinstance(policy, TradingPolicy):
                policy.reset_all_state(reason="walk_forward_fold_start")

            # Rebuild memory bank from training data only (with embargo)
            # to prevent future-outcome leakage across folds.
            if wf_config.reset_memory_per_fold and isinstance(policy, MemoryCapablePolicy):
                if policy.updater is not None and hasattr(policy, 'retriever'):
                    from src.memory.updater import build_memory_bank_from_bars
                    from src.models.state_encoder import ModelStateEncoder
                    if hasattr(policy, 'encoder') and isinstance(policy.encoder, ModelStateEncoder):
                        embargo = max(self.config.memory.forward_horizons) if self.config.memory.forward_horizons else 0
                        try:
                            fresh_bank = build_memory_bank_from_bars(
                                bars_by_symbol=train_bars,
                                pipeline=pipeline,
                                encoder=policy.encoder,
                                memory_config=self.config.memory,
                                warmup_bars=self.config.runtime.warmup_bars,
                                embargo_bars=embargo,
                            )
                            policy.bank = fresh_bank
                            logger.info(
                                "walk_forward_memory_rebuilt fold=%d items=%d",
                                fold_def.fold_index, len(fresh_bank),
                            )
                        except (ValueError, RuntimeError) as exc:
                            logger.warning(
                                "walk_forward_memory_rebuild_failed fold=%d error=%s",
                                fold_def.fold_index, str(exc),
                            )

            # Run backtest on the test slice
            risk_manager = RiskManager(self.config.risk.model_copy())
            engine = BacktestEngine(self.config, pipeline, risk_manager)
            report = engine.run(test_bars, policy)

            # Collect memory/fusion metrics if available
            test_metrics = dict(report.metrics)
            if isinstance(policy, MemoryCapablePolicy):
                test_metrics.update(policy.metrics_snapshot())

            fold_elapsed = time.monotonic() - fold_start
            fold_results.append(FoldResult(
                fold_definition=fold_def,
                test_metrics=test_metrics,
                backtest_report=report,
                elapsed_seconds=fold_elapsed,
            ))
            logger.info(
                "walk_forward_fold_done fold=%d return=%.4f sharpe=%.3f elapsed=%.1fs",
                fold_def.fold_index,
                test_metrics.get("total_return", 0.0),
                test_metrics.get("sharpe_ratio", 0.0),
                fold_elapsed,
            )

        total_elapsed = time.monotonic() - total_start
        aggregate = _aggregate_metrics(fold_results)

        logger.info(
            "walk_forward_done folds=%d mean_return=%.4f mean_sharpe=%.3f elapsed=%.1fs",
            len(fold_results),
            aggregate.get("total_return", {}).get("mean", 0.0),
            aggregate.get("sharpe_ratio", {}).get("mean", 0.0),
            total_elapsed,
        )

        return WalkForwardResult(
            folds=fold_results,
            aggregate_metrics=aggregate,
            config_snapshot=self.config.model_dump(mode="json"),
            fold_definitions=fold_defs,
            total_elapsed_seconds=total_elapsed,
        )


def _aggregate_metrics(
    fold_results: list[FoldResult],
) -> dict[str, dict[str, float]]:
    """Compute aggregate statistics across folds for each metric.

    Returns a dict of {metric_name: {mean, std, median, min, max, count}}.
    Degenerate folds (zero fills — the policy never traded, typically a
    too-short test slice) are excluded from aggregation instead of
    contributing all-zero metrics that drag every mean toward zero; the
    excluded count is reported under ``degenerate_folds``.
    """
    if not fold_results:
        return {}

    degenerate = sum(
        1 for fr in fold_results if fr.test_metrics.get("fill_count", 0.0) <= 0
    )
    active = [fr for fr in fold_results if fr.test_metrics.get("fill_count", 0.0) > 0]
    if active:
        fold_results = active

    # Collect all metric keys across folds
    all_keys: set[str] = set()
    for fr in fold_results:
        all_keys.update(fr.test_metrics.keys())

    import math
    import statistics

    aggregate: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [
            fr.test_metrics[key]
            for fr in fold_results
            if key in fr.test_metrics and math.isfinite(fr.test_metrics[key])
        ]
        if not values:
            continue
        mean_val = sum(values) / len(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        median_val = statistics.median(values)
        aggregate[key] = {
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "min": min(values),
            "max": max(values),
            "count": float(len(values)),
            "profitable_folds": float(sum(1 for v in values if v > 0)) if key == "total_return" else 0.0,
        }

    if degenerate:
        aggregate["degenerate_folds"] = {
            "mean": float(degenerate),
            "std": 0.0,
            "median": float(degenerate),
            "min": float(degenerate),
            "max": float(degenerate),
            "count": float(degenerate),
            "profitable_folds": 0.0,
        }
    return aggregate
