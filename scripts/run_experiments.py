"""Standardized experiment runner for ablation and walk-forward evaluation.

Usage examples:

    # Single ablation backtest
    python scripts/run_experiments.py --experiments brain_only

    # Full ablation suite
    python scripts/run_experiments.py --all

    # Walk-forward evaluation
    python scripts/run_experiments.py --experiments full_model,brain_only --walk-forward

    # Baseline comparison
    python scripts/run_experiments.py --experiments full_model,brain_only,brain_no_flip

    # Custom base config
    python scripts/run_experiments.py --all --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.policy_factory import (
    build_baseline_policy,
    build_brain_memory_policy,
    build_brain_policy,
    build_buy_and_hold_policy,
    build_mean_reversion_policy,
)
from src.core.config import AppConfig, load_config
from src.core.seeds import seed_everything
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import (
    HistoricalDataService,
    generate_synthetic_bars,
    parse_date_or_datetime,
)
from src.data.schemas import MarketBar
from src.evaluation.reporting import (
    build_comparison_table,
    format_ranking_table,
    save_comparison_csv,
    save_experiment_results,
)
from src.evaluation.walk_forward import WalkForwardResult, WalkForwardRunner
from src.features.feature_pipeline import FeaturePipeline
from src.risk.risk_manager import RiskManager

# All available experiment presets (matching configs/experiments/*.yaml)
ALL_EXPERIMENTS = [
    "full_model",
    "brain_only",
    "brain_no_flip",
    "memory_off",
    "no_regime_encoder",
    "no_temporal_cnn",
    "no_latent_memory",
    "no_confidence_gating",
    "no_online_memory",
    "strict_risk",
    "loose_risk",
]

# Baseline experiments that use non-brain policies
BASELINE_EXPERIMENTS = {"buy_and_hold", "momentum_baseline", "mean_reversion"}


def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(ROOT), timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _load_bars(config: AppConfig) -> dict[str, list[MarketBar]]:
    """Load historical bars from cache or API.

    Pins end_date to start-of-today when null to prevent cache
    invalidation on every run (datetime.now() is always slightly newer
    than the cached metadata.end, causing a trailing-gap API fetch).
    """
    if not config.training.use_real_historical_data:
        return {
            symbol: generate_synthetic_bars(symbol, 240)
            for symbol in config.data.symbols
        }
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
        cache=HistoricalBarCache(config.data.cache_dir),
        timezone_normalization=config.data.timezone_normalization,
        max_missing_gap_handling=config.data.max_missing_gap_handling,
    )
    start = parse_date_or_datetime(
        config.data.start_date,
        default=datetime.now(UTC) - timedelta(days=365 * 2),
    )
    # Pin end to start-of-today to maximize cache hits for experiments
    if config.data.end_date is not None:
        end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    else:
        end = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    bars_by_symbol, _ = service.load_bars_for_symbols(
        config.data.symbols,
        timeframe=config.data.timeframe,
        start=start,
        end=end,
        force_refresh=config.data.force_refresh,
    )
    return bars_by_symbol


def _build_policy_for_config(
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
) -> object:
    """Build the appropriate policy based on config flags."""
    if config.memory.enabled:
        return build_brain_memory_policy(
            config,
            bars_by_symbol=bars_by_symbol,
            pipeline=pipeline,
            enable_memory=True,
        )
    return build_brain_policy(
        config,
        bars_by_symbol=bars_by_symbol,
        pipeline=pipeline,
    )


def _run_single_backtest(
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    experiment_name: str,
) -> dict[str, float]:
    """Run a single backtest and return metrics."""
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

    if experiment_name == "buy_and_hold":
        policy = build_buy_and_hold_policy()
    elif experiment_name == "momentum_baseline":
        policy = build_baseline_policy()
    elif experiment_name == "mean_reversion":
        policy = build_mean_reversion_policy()
    else:
        policy = _build_policy_for_config(config, bars_by_symbol, pipeline)

    risk_manager = RiskManager(config.risk.model_copy())
    engine = BacktestEngine(config, pipeline, risk_manager)
    report = engine.run(bars_by_symbol, policy)

    metrics = dict(report.metrics)
    if hasattr(policy, "metrics_snapshot"):
        metrics.update(policy.metrics_snapshot())
    return metrics


def _run_walk_forward(
    config: AppConfig,
    bars_by_symbol: dict[str, list[MarketBar]],
    experiment_name: str,
) -> WalkForwardResult:
    """Run walk-forward evaluation."""
    if experiment_name in BASELINE_EXPERIMENTS:
        # Baselines don't need per-fold training
        def policy_builder(cfg, train_bars, pipeline):
            if experiment_name == "buy_and_hold":
                return build_buy_and_hold_policy()
            elif experiment_name == "momentum_baseline":
                return build_baseline_policy()
            elif experiment_name == "mean_reversion":
                return build_mean_reversion_policy()
    else:
        def policy_builder(cfg, train_bars, pipeline):
            return _build_policy_for_config(cfg, train_bars, pipeline)

    runner = WalkForwardRunner(config, bars_by_symbol)
    return runner.run(policy_builder)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiment suite")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated experiment names (e.g., 'full_model,brain_only,memory_off')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all preset experiments",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Include baseline strategies (buy_and_hold, momentum, mean_reversion)",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Base config file path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: from config)",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward evaluation instead of single backtest",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real historical data (default: synthetic for testing)",
    )
    args = parser.parse_args()

    # Determine which experiments to run
    if args.all:
        experiment_names = list(ALL_EXPERIMENTS)
    elif args.experiments:
        experiment_names = [e.strip() for e in args.experiments.split(",")]
    else:
        experiment_names = ["full_model"]

    if args.baselines:
        experiment_names.extend(sorted(BASELINE_EXPERIMENTS))

    # Load base config once
    base_config = load_config(args.config)
    if args.real_data:
        base_config.training.use_real_historical_data = True

    git_hash = _get_git_hash()
    output_dir = Path(args.output_dir or base_config.evaluation.output_dir)
    # `is not None` (not truthiness): --seed 0 is a valid, explicit seed.
    seed = args.seed if args.seed is not None else base_config.evaluation.seed

    # Load data once (shared across experiments)
    seed_everything(seed)
    print(f"Loading data...")
    bars_by_symbol = _load_bars(base_config)
    min_bars = min(len(bars) for bars in bars_by_symbol.values())
    print(f"  {len(bars_by_symbol)} symbols, {min_bars} bars each")

    # Run experiments
    all_results: dict[str, WalkForwardResult | dict[str, float]] = {}

    for experiment_name in experiment_names:
        print(f"\n{'='*60}")
        print(f"  Experiment: {experiment_name}")
        print(f"{'='*60}")

        # Load config: base → experiment base (risk thresholds) → experiment overlay
        seed_everything(seed)
        experiment_base = ROOT / "configs" / "experiments" / "_experiment_base.yaml"
        overlay_path = ROOT / "configs" / "experiments" / f"{experiment_name}.yaml"
        config_paths = [args.config]
        if experiment_base.exists():
            config_paths.append(str(experiment_base))
        if overlay_path.exists():
            config_paths.append(str(overlay_path))
        config = load_config(*config_paths)

        if args.real_data:
            config.training.use_real_historical_data = True

        if args.walk_forward:
            result = _run_walk_forward(config, bars_by_symbol, experiment_name)
            all_results[experiment_name] = result
            print(f"\n  Folds: {len(result.folds)}")
            for key in ["total_return", "sharpe_ratio", "max_drawdown"]:
                agg = result.aggregate_metrics.get(key, {})
                if agg:
                    print(f"  {key}: mean={agg.get('mean', 0):.4f}  std={agg.get('std', 0):.4f}")
        else:
            metrics = _run_single_backtest(config, bars_by_symbol, experiment_name)
            all_results[experiment_name] = metrics
            for key in ["total_return", "sharpe_ratio", "max_drawdown", "win_rate",
                         "profit_factor", "fill_count", "flat_fraction"]:
                print(f"  {key}: {metrics.get(key, 0.0):.4f}")

        # Save individual results
        save_experiment_results(
            all_results[experiment_name],
            output_dir,
            experiment_name,
            git_hash=git_hash,
        )

    # Generate comparison report
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON RANKING")
        print(f"{'='*60}")
        print(format_ranking_table(all_results))

        comparison = build_comparison_table(all_results)
        csv_path = output_dir / "comparison.csv"
        save_comparison_csv(comparison, csv_path)
        print(f"\n  CSV saved to: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
