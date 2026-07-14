"""Experiment suite execution for the GUI.

Runs named experiments (ablation presets) as single backtests, collects
metrics, and produces a comparison table.  Designed to be invoked from a
Qt worker thread via CallableWorker — no Qt imports here.

Performance optimizations:
  - Market bars loaded once and shared across all experiments.
  - Feature pipeline ``build_batch`` results are cached per symbol when
    the feature config is identical across experiments (which it is for
    most ablations — they change model/risk/memory flags, not features).
    This avoids repeating the expensive ``torch.unfold`` vectorized
    feature computation for every experiment.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import torch

from src.backtest.engine import BacktestEngine
from src.backtest.policy_factory import (
    build_baseline_policy,
    build_brain_memory_policy,
    build_brain_policy,
    build_buy_and_hold_policy,
    build_mean_reversion_policy,
)
from src.core.config import AppConfig, FeatureConfig, load_config
from src.core.seeds import seed_everything
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import (
    HistoricalDataService,
    generate_synthetic_bars,
    parse_date_or_datetime,
)
from src.data.schemas import MarketBar
from src.evaluation.reporting import format_ranking_table, save_experiment_results
from src.features.feature_pipeline import FeaturePipeline
from src.gui_services.state import ServiceError
from src.risk.risk_manager import RiskManager
from src.strategies.memory_policy import MemoryAwarePolicy

EXPERIMENTS_DIR = Path("configs/experiments")
RESULTS_DIR = Path("experiments")

# Baseline experiments that don't use a brain model
BASELINE_EXPERIMENTS = {"buy_and_hold", "momentum_baseline", "mean_reversion"}


def list_available_experiments() -> list[str]:
    """Return sorted names of available experiment presets."""
    presets: list[str] = []
    if EXPERIMENTS_DIR.exists():
        for yaml_file in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
            presets.append(yaml_file.stem)
    # Add non-config-based baselines
    for name in sorted(BASELINE_EXPERIMENTS):
        if name not in presets:
            presets.append(name)
    return presets


def _load_bars(
    config: AppConfig,
    *,
    use_real_data: bool,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, list[MarketBar]]:
    """Load historical bars (shared across experiments).

    When using real data, tries cache-only loading first (no API calls).
    If the cache covers the requested range, bars load in ~1-2 seconds.
    Falls back to API fetch only for symbols with no cache coverage.
    """
    if progress_callback is None:
        progress_callback = lambda _: None  # noqa: E731

    if not use_real_data:
        return {sym: generate_synthetic_bars(sym, 240) for sym in config.data.symbols}

    cache = HistoricalBarCache(config.data.cache_dir)
    start = parse_date_or_datetime(
        config.data.start_date, default=datetime.now(UTC) - timedelta(days=365 * 2),
    )
    # Pin end_date to avoid cache invalidation.  When end_date is null
    # (the default), each run uses datetime.now() which is always slightly
    # newer than the cached metadata.end, causing a trailing-gap fetch +
    # full re-save for every symbol.  For experiments we don't need the
    # latest 5 minutes — pin to start of today.
    if config.data.end_date is not None:
        end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    else:
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)

    # Try cache-only load first (no API calls = instant)
    bars: dict[str, list[MarketBar]] = {}
    uncached: list[str] = []
    for symbol in config.data.symbols:
        record = cache.load(symbol, config.data.timeframe)
        if record is not None:
            filtered = cache.filter_range(record, start=start, end=end)
            if filtered:
                bars[symbol] = filtered
                continue
        uncached.append(symbol)

    if uncached:
        progress_callback(
            f"Cache hit for {len(bars)} symbols. "
            f"Fetching {len(uncached)} from API: {', '.join(uncached[:5])}..."
        )
        if credentials_available():
            client = AlpacaClient(
                paper=config.broker.paper,
                data_feed=config.broker.data_feed,
                bar_adjustment=config.data.bar_adjustment,
            )
        else:
            client = MockBrokerClient(paper=True, data_feed="mock")
        service = HistoricalDataService(
            client=client,
            cache=cache,
            timezone_normalization=config.data.timezone_normalization,
            max_missing_gap_handling=config.data.max_missing_gap_handling,
        )
        fetched, _ = service.load_bars_for_symbols(
            uncached,
            timeframe=config.data.timeframe,
            start=start,
            end=end,
            force_refresh=False,
        )
        for sym, sym_bars in fetched.items():
            if sym_bars:
                bars[sym] = sym_bars
    else:
        progress_callback(f"All {len(bars)} symbols loaded from cache (no API calls)")

    # Drop symbols with zero bars
    return {s: b for s, b in bars.items() if b}


def _feature_config_key(cfg: FeatureConfig) -> str:
    """Hash-like key for a FeatureConfig to detect identical feature setups."""
    return cfg.model_dump_json()


def _build_policy_for_experiment(
    experiment_name: str,
    config: AppConfig,
    bars: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
):
    """Build the appropriate policy for a named experiment."""
    if experiment_name == "buy_and_hold":
        return build_buy_and_hold_policy()
    if experiment_name == "momentum_baseline":
        return build_baseline_policy()
    if experiment_name == "mean_reversion":
        return build_mean_reversion_policy()
    if config.memory.enabled:
        return build_brain_memory_policy(
            config, bars_by_symbol=bars, pipeline=pipeline, enable_memory=True,
        )
    return build_brain_policy(config, bars_by_symbol=bars, pipeline=pipeline)


class _FeatureCache:
    """Caches build_batch results so experiments with identical feature
    configs skip the expensive torch.unfold vectorized feature pass.

    Most ablation experiments (model flags, risk params, memory toggles)
    don't change the feature config at all, so a single build_batch per
    symbol serves the entire suite.
    """

    def __init__(self) -> None:
        self._cache: dict[str, dict[str, object]] = {}  # config_key → {symbol → BatchedSnapshots}

    def get_or_build(
        self,
        pipeline: FeaturePipeline,
        bars_by_symbol: dict[str, list[MarketBar]],
        feature_config: FeatureConfig,
        warmup_bars: int,
    ) -> dict[str, object]:
        """Return cached batched features or build and cache them."""
        key = _feature_config_key(feature_config)
        if key in self._cache:
            return self._cache[key]

        # Cross-sectional market context (Phase 3): computed once per
        # feature config from the full multi-symbol bar set; the config
        # key includes market_context_enabled (model_dump_json), so
        # flag-on and flag-off experiments never share a cache entry.
        market_context = None
        if feature_config.market_context_enabled:
            from src.features.market_context import compute_market_context_batch

            market_context = compute_market_context_batch(
                bars_by_symbol,
                short_window=int(feature_config.windows["short"]),
                medium_window=int(feature_config.windows["medium"]),
                epsilon=feature_config.epsilon,
            )
        batched = {
            symbol: pipeline.build_batch(
                bars,
                symbol=symbol,
                temporal_window=warmup_bars,
                market_context=market_context,
            )
            for symbol, bars in bars_by_symbol.items()
        }
        self._cache[key] = batched
        return batched


class _PrecomputedFeaturePipeline:
    """FeaturePipeline facade that serves precomputed ``build_batch`` results.

    ``BacktestEngine.run`` touches the pipeline exclusively through
    ``build_batch(bars, symbol=..., temporal_window=...)``, so this facade is
    the surgical seam that lets experiments reuse the real engine loop while
    skipping the expensive torch.unfold feature pass.  A cache entry is only
    served when it matches the requested symbol AND bar count; anything else
    falls through to the wrapped pipeline.  All other attribute access is
    delegated unchanged.
    """

    def __init__(self, pipeline: FeaturePipeline, cached_batches: dict[str, object]) -> None:
        self._pipeline = pipeline
        self._cached_batches = cached_batches

    def build_batch(self, bars, *, symbol: str, temporal_window: int, **kwargs):
        cached = self._cached_batches.get(symbol)
        if cached is not None and getattr(cached, "n_bars", -1) == len(bars):
            return cached
        return self._pipeline.build_batch(
            bars, symbol=symbol, temporal_window=temporal_window, **kwargs
        )

    def __getattr__(self, name: str):
        return getattr(self._pipeline, name)


class _CachingBacktestEngine(BacktestEngine):
    """BacktestEngine that reuses pre-cached feature batches.

    Historical note: this used to carry a hand-copied fork of
    ``BacktestEngine.run`` that silently diverged from the real engine —
    unsorted symbol order, no ``fill_timing`` support (always filled at the
    same close the decision observed), no temporal cutoff for memory
    retrieval, and no decision log.  Experiment results were therefore not
    comparable with ``scripts/run_backtest.py``.  It now inherits the base
    ``run`` verbatim and injects cached features through
    ``_PrecomputedFeaturePipeline`` — the only remaining difference is that
    ``build_batch`` is served from the experiment suite's shared cache.
    """

    def __init__(self, config, feature_pipeline, risk_manager, cached_batches):
        super().__init__(
            config=config,
            feature_pipeline=_PrecomputedFeaturePipeline(feature_pipeline, cached_batches),
            risk_manager=risk_manager,
        )


def _trim_bars(
    bars: dict[str, list[MarketBar]],
    max_bars: int,
) -> dict[str, list[MarketBar]]:
    """Keep only the last ``max_bars`` per symbol to speed up backtests."""
    return {
        symbol: bar_list[-max_bars:] if len(bar_list) > max_bars else bar_list
        for symbol, bar_list in bars.items()
    }


def run_experiment_suite(
    *,
    experiment_names: list[str],
    use_real_data: bool = False,
    fast_mode: bool = False,
    walk_forward_params: dict[str, int] | None = None,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run a suite of named experiments and return comparison results.

    Args:
        fast_mode: When True, use only the last ~1 month of data (~1000 bars
            per symbol) instead of the full history.  Reduces runtime by ~8x.
        walk_forward_params: If provided, run walk-forward evaluation instead
            of a single backtest.  Keys: train_bars, test_bars, step_bars.

    Returns a dict with:
      - "experiments": list of experiment names
      - "results": {name: {metric: value, ...}, ...}
      - "ranking_table": str (human-readable comparison)
      - "completed": int (number of experiments that finished)
      - "failed": list of (name, error_message) pairs
      - "total_seconds": float
      - "per_experiment_seconds": {name: seconds, ...}
    """
    if progress_callback is None:
        progress_callback = lambda _: None  # noqa: E731

    suite_start = time.monotonic()
    base_config = load_config("configs/default.yaml")
    if use_real_data:
        base_config.training.use_real_historical_data = True

    # Load data once (shared across all experiments)
    seed_everything(seed)
    progress_callback("Loading market data (checking cache)...")
    t0 = time.monotonic()
    bars = _load_bars(base_config, use_real_data=use_real_data, progress_callback=progress_callback)
    if not bars:
        raise ServiceError("No bars available for experiments.")
    data_time = time.monotonic() - t0

    # Fast mode: trim to last ~1 month (~2000 bars at 5-min) for speed.
    # At ~15ms per batched bar, 2000 bars = ~30s per experiment.
    # 15 experiments finish in ~8 minutes instead of hours.
    if fast_mode:
        max_bars = 1_000  # ~2 weeks of 5-min bars → ~15s per experiment
        bars = _trim_bars(bars, max_bars)
        progress_callback(
            f"Fast mode: using last {max_bars} bars/symbol (~2 weeks, ~15s/experiment)"
        )

    min_bars = min(len(b) for b in bars.values())
    progress_callback(
        f"Data loaded in {data_time:.1f}s: {len(bars)} symbols, {min_bars} bars each"
    )

    # Feature cache: build_batch results are reused when feature config matches
    feature_cache = _FeatureCache()

    all_results: dict[str, dict[str, float]] = {}
    failed: list[tuple[str, str]] = []
    per_experiment_seconds: dict[str, float] = {}

    total = len(experiment_names)
    for i, name in enumerate(experiment_names):
        exp_start = time.monotonic()
        progress_callback(f"[{i+1}/{total}] Running: {name}...")
        seed_everything(seed)

        try:
            # Load config: default → experiment base (lowers risk thresholds
            # so the trained brain's signals generate trades) → experiment overlay
            experiment_base = EXPERIMENTS_DIR / "_experiment_base.yaml"
            overlay = EXPERIMENTS_DIR / f"{name}.yaml"
            config_paths = ["configs/default.yaml"]
            if experiment_base.exists():
                config_paths.append(str(experiment_base))
            if overlay.exists():
                config_paths.append(str(overlay))
            config = load_config(*config_paths)
            if use_real_data:
                config.training.use_real_historical_data = True

            if walk_forward_params is not None:
                # Walk-forward evaluation: multiple chronological folds
                from src.evaluation.walk_forward import WalkForwardRunner

                config.evaluation.walk_forward.train_bars = walk_forward_params["train_bars"]
                config.evaluation.walk_forward.test_bars = walk_forward_params["test_bars"]
                config.evaluation.walk_forward.step_bars = walk_forward_params["step_bars"]
                config.evaluation.walk_forward.val_bars = 0
                config.evaluation.walk_forward.min_train_bars = walk_forward_params["train_bars"] // 2

                runner = WalkForwardRunner(config, bars)

                def wf_policy_builder(cfg, train_bars_slice, pip):
                    return _build_policy_for_experiment(name, cfg, train_bars_slice, pip)

                progress_callback(f"[{i+1}/{total}] {name}: walk-forward ({min_bars} bars, {len(bars)} symbols)...")
                wf_result = runner.run(wf_policy_builder)

                # Use aggregate mean as the experiment's metrics
                metrics: dict[str, float] = {}
                for key, agg in wf_result.aggregate_metrics.items():
                    metrics[key] = agg.get("mean", 0.0)
                # Add fold count and std for key metrics
                metrics["wf_fold_count"] = float(len(wf_result.folds))
                for key in ["total_return", "sharpe_ratio", "max_drawdown"]:
                    if key in wf_result.aggregate_metrics:
                        metrics[f"{key}_std"] = wf_result.aggregate_metrics[key].get("std", 0.0)
                        metrics[f"{key}_worst"] = wf_result.aggregate_metrics[key].get("min", 0.0)
                        metrics[f"{key}_best"] = wf_result.aggregate_metrics[key].get("max", 0.0)

                all_results[name] = metrics
            else:
                # Single backtest mode
                progress_callback(f"[{i+1}/{total}] {name}: building features...")
                pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

                # Get or build cached features (skips torch.unfold if config matches)
                cached_batches = feature_cache.get_or_build(
                    pipeline, bars, config.features, config.runtime.warmup_bars,
                )

                progress_callback(f"[{i+1}/{total}] {name}: building policy...")
                policy = _build_policy_for_experiment(name, config, bars, pipeline)

                progress_callback(f"[{i+1}/{total}] {name}: running backtest ({min_bars} bars x {len(bars)} symbols)...")
                risk_manager = RiskManager(config.risk.model_copy())
                engine = _CachingBacktestEngine(config, pipeline, risk_manager, cached_batches)
                report = engine.run(bars, policy)

                metrics = dict(report.metrics)
                if isinstance(policy, MemoryAwarePolicy):
                    metrics.update(policy.metrics_snapshot())

                all_results[name] = metrics
            exp_time = time.monotonic() - exp_start
            per_experiment_seconds[name] = exp_time

            ret = metrics.get("total_return", 0.0)
            sharpe = metrics.get("sharpe_ratio", 0.0)
            elapsed_total = time.monotonic() - suite_start
            remaining_est = (elapsed_total / (i + 1)) * (total - i - 1)
            progress_callback(
                f"[{i+1}/{total}] {name}: "
                f"return={ret:+.4f}  sharpe={sharpe:+.3f}  "
                f"({exp_time:.1f}s, ~{remaining_est:.0f}s remaining)"
            )

            # Save individual result
            save_experiment_results(metrics, RESULTS_DIR, name)

        except Exception as exc:  # noqa: BLE001
            exp_time = time.monotonic() - exp_start
            per_experiment_seconds[name] = exp_time
            failed.append((name, str(exc)))
            progress_callback(f"[{i+1}/{total}] {name}: FAILED - {exc}")

    # Generate ranking table
    ranking = ""
    if all_results:
        ranking = format_ranking_table(all_results)

    total_time = time.monotonic() - suite_start
    progress_callback(
        f"Done: {len(all_results)} completed, {len(failed)} failed "
        f"in {total_time:.1f}s"
    )

    return {
        "experiments": experiment_names,
        "results": all_results,
        "ranking_table": ranking,
        "completed": len(all_results),
        "failed": failed,
        "total_seconds": total_time,
        "per_experiment_seconds": per_experiment_seconds,
    }
