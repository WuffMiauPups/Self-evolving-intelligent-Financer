from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.backtest.policy_factory import build_brain_memory_policy
from src.core.config import load_config
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import HistoricalDataService, generate_synthetic_bars, parse_date_or_datetime
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.models.factory import resolve_inference_model
from src.risk.risk_manager import RiskManager
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.strategies.brain_policy import BrainPolicy
from src.strategies.memory_policy import MemoryAwarePolicy


def _print_report(label: str, metrics: dict[str, float]) -> None:
    fill_count = int(metrics.get("fill_count", 0))
    veto_count = int(metrics.get("fusion_veto_count", 0))
    dampen_count = int(metrics.get("fusion_dampen_count", 0))
    support_count = int(metrics.get("fusion_support_count", 0))
    neutral_count = int(metrics.get("fusion_neutral_count", 0))
    total_fusion = veto_count + dampen_count + support_count + neutral_count
    blocked_count = veto_count + dampen_count
    blocked_frac = blocked_count / total_fusion if total_fusion > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  total_return        : {metrics.get('total_return', 0.0):+.4f}")
    print(f"  max_drawdown        : {metrics.get('max_drawdown', 0.0):.4f}")
    print(f"  fills               : {fill_count}")
    print(f"  win_rate            : {metrics.get('win_rate', 0.0):.4f}")
    print(f"  avg_return/trade    : {metrics.get('avg_return_per_trade', 0.0):+.6f}")
    print(f"  profit_factor       : {metrics.get('profit_factor', 0.0):.4f}")
    print(f"  expectancy          : {metrics.get('expectancy', 0.0):+.6f}")
    print(f"  turnover            : {metrics.get('turnover', 0.0):.4f}")
    print(f"  avg_exposure        : {metrics.get('avg_exposure', 0.0):.4f}")
    if total_fusion > 0:
        print(f"  --- fusion ---")
        print(f"  support             : {support_count}  ({metrics.get('memory_supported_rate', 0.0):.3f})")
        print(f"  neutral             : {neutral_count}  ({metrics.get('memory_neutral_rate', 0.0):.3f})")
        print(f"  dampen              : {dampen_count}  ({metrics.get('memory_dampened_rate', 0.0):.3f})")
        print(f"  veto                : {veto_count}  ({metrics.get('memory_veto_rate', 0.0):.3f})")
        print(f"  blocked(veto+damp)  : {blocked_count}  ({blocked_frac:.3f} of all fusion calls)")


def _make_memory_aware_policy(
    brain_policy: BrainPolicy,
    config,
    artifacts,
    bars_by_symbol: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
    *,
    enable_memory: bool,
    conservative_thresholds: bool = False,
    verbose_fusion: bool = False,
) -> MemoryAwarePolicy:
    """Thin shim around src.backtest.policy_factory.build_brain_memory_policy.

    Kept for callers that already have a constructed ``brain_policy`` and
    resolved ``artifacts`` in scope.  The factory rebuilds those itself
    from ``config``, but the inputs are the same so the constructed
    MemoryAwarePolicy is byte-identical to the pre-refactor behavior.
    """
    # ``brain_policy`` and ``artifacts`` are unused here — the factory
    # rebuilds them from the same config, so the semantics are preserved.
    return build_brain_memory_policy(
        config,
        bars_by_symbol=bars_by_symbol,
        pipeline=pipeline,
        enable_memory=enable_memory,
        conservative_thresholds=conservative_thresholds,
        verbose_fusion=verbose_fusion,
    )


def _collect_fusion_counts(policy: object, metrics: dict[str, float]) -> dict[str, float]:
    """Merge memory-policy diagnostic snapshot into metrics.

    Post-Great-Simplification the fusion layer is gone; ``vetoed_count`` /
    ``dampened_count`` / ``supported_count`` / ``neutral_count`` no longer
    exist on ``MemoryAwarePolicy``. The reporter still prints the labels
    so we just leave them at zero.
    """
    if not isinstance(policy, MemoryAwarePolicy):
        return metrics
    metrics = dict(metrics)
    metrics["fusion_veto_count"] = 0.0
    metrics["fusion_dampen_count"] = 0.0
    metrics["fusion_support_count"] = 0.0
    metrics["fusion_neutral_count"] = 0.0
    snap = getattr(policy, "metrics_snapshot", lambda: {})()
    metrics.update({k: v for k, v in snap.items() if k not in metrics})
    return metrics


def _load_bars_by_symbol(config, *, use_real_data: bool, allow_mock: bool) -> dict[str, list[MarketBar]]:
    if not use_real_data:
        return {
            symbol: generate_synthetic_bars(symbol, 240)
            for symbol in config.runtime.symbols
        }
    if credentials_available():
        client = AlpacaClient(paper=config.broker.paper, data_feed=config.broker.data_feed)
    elif allow_mock:
        client = MockBrokerClient(paper=True, data_feed="mock")
    else:
        client = None
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
    end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    bars_by_symbol, _ = service.load_bars_for_symbols(
        config.data.symbols,
        timeframe=config.data.timeframe,
        start=start,
        end=end,
        force_refresh=config.data.force_refresh,
    )
    return bars_by_symbol


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal backtest")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load for the neural model")
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable associative market memory decision support during the backtest.",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Backtest on cached/fetched real historical bars instead of synthetic bars.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow explicit mock bar generation for --real-data when Alpaca credentials are unavailable.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Config file(s), later files override earlier files. "
            "Default: configs/default.yaml. "
            "Passing --config REPLACES the default (it does not append to it)."
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all 4 comparison scenarios: baseline, fusion-only, fusion+memory, fusion+memory+conservative.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward cross-validation instead of a single contiguous backtest.",
    )
    args = parser.parse_args()
    # argparse action='append' appends to a non-None default instead of
    # replacing it, so the default stack is applied post-parse.
    if args.config is None:
        args.config = ["configs/default.yaml"]
    config = load_config(*args.config)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.use_memory:
        config.memory.enabled = True
        if config.memory.runtime_update_mode == "pending_only":
            config.memory.runtime_update_mode = "append_when_ready"
    use_real_data = args.real_data or config.training.use_real_historical_data
    bars_by_symbol = _load_bars_by_symbol(config, use_real_data=use_real_data, allow_mock=args.allow_mock)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

    if args.walk_forward:
        from src.evaluation.walk_forward import WalkForwardRunner
        from src.strategies.base import MemoryCapablePolicy

        def _wf_policy_builder(cfg, train_bars, pipe):
            sample_sym = next(iter(train_bars))
            snap = pipe.build(train_bars[sample_sym][:cfg.runtime.warmup_bars + 1], symbol=sample_sym)
            arts = resolve_inference_model(
                model_config=cfg.model,
                feature_config=cfg.features,
                feature_contract=snap.contract,
                device=torch.device("cpu"),
                default_model_kind="brain",
            )
            bp = BrainPolicy(
                model=arts.model,
                device=torch.device("cpu"),
                feature_contract=arts.feature_contract,
                normalization=arts.normalization,
                max_state_gap_sec=cfg.runtime.state_reset_gap_sec,
            )
            if cfg.memory.enabled:
                return _make_memory_aware_policy(bp, cfg, arts, train_bars, pipe, enable_memory=True)
            return bp

        runner = WalkForwardRunner(config, bars_by_symbol)
        result = runner.run(_wf_policy_builder)
        print(f"\n{'='*60}")
        print("  WALK-FORWARD RESULTS")
        print(f"{'='*60}")
        for fold_result in result.folds:
            fd = fold_result.fold_definition
            print(f"\n  Fold {fd.fold_index}: train=[{fd.train_start},{fd.train_end}) test=[{fd.test_start},{fd.test_end})")
            print(f"    return={fold_result.test_metrics.get('total_return', 0.0):+.4f}  "
                  f"sharpe={fold_result.test_metrics.get('sharpe_ratio', 0.0):.3f}  "
                  f"fills={int(fold_result.test_metrics.get('fill_count', 0))}  "
                  f"elapsed={fold_result.elapsed_seconds:.1f}s")
        if result.aggregate_metrics:
            print(f"\n  --- Aggregate ---")
            for key in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor"):
                agg = result.aggregate_metrics.get(key)
                if agg:
                    print(f"    {key:<20}: mean={agg['mean']:+.4f}  std={agg['std']:.4f}  "
                          f"min={agg['min']:+.4f}  max={agg['max']:+.4f}")
        print(f"\n  Total elapsed: {result.total_elapsed_seconds:.1f}s")
        return 0

    if args.compare:
        # ── Scenario 1: Baseline (no fusion, no memory) ──────────────────────
        baseline_engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
        baseline_report = baseline_engine.run(bars_by_symbol, BaselineMomentumPolicy())
        _print_report("1. BASELINE (no fusion, no memory)", baseline_report.metrics)

        # Build shared model artifacts once
        sample_symbol = next(iter(bars_by_symbol))
        sample_snapshot = pipeline.build(
            bars_by_symbol[sample_symbol][: config.runtime.warmup_bars + 1],
            symbol=sample_symbol,
        )
        artifacts = resolve_inference_model(
            model_config=config.model,
            feature_config=config.features,
            feature_contract=sample_snapshot.contract,
            device=torch.device("cpu"),
            default_model_kind="brain",
        )

        # ── Scenario 2: Fusion ON, Memory OFF ────────────────────────────────
        brain2 = BrainPolicy(
            model=artifacts.model,
            device=torch.device("cpu"),
            feature_contract=artifacts.feature_contract,
            normalization=artifacts.normalization,
            max_state_gap_sec=config.runtime.state_reset_gap_sec,
        )
        policy2 = _make_memory_aware_policy(
            brain2, config, artifacts, bars_by_symbol, pipeline,
            enable_memory=False,
        )
        engine2 = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
        report2 = engine2.run(bars_by_symbol, policy2)
        metrics2 = _collect_fusion_counts(policy2, report2.metrics)
        _print_report("2. FUSION ON, memory off", metrics2)

        # ── Scenario 3: Fusion ON, Memory ON ─────────────────────────────────
        brain3 = BrainPolicy(
            model=artifacts.model,
            device=torch.device("cpu"),
            feature_contract=artifacts.feature_contract,
            normalization=artifacts.normalization,
            max_state_gap_sec=config.runtime.state_reset_gap_sec,
        )
        policy3 = _make_memory_aware_policy(
            brain3, config, artifacts, bars_by_symbol, pipeline,
            enable_memory=True,
        )
        engine3 = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
        report3 = engine3.run(bars_by_symbol, policy3)
        metrics3 = _collect_fusion_counts(policy3, report3.metrics)
        _print_report("3. FUSION ON + MEMORY ON", metrics3)

        # ── Scenario 4: Fusion ON, Memory ON, Conservative thresholds + logging ──
        brain4 = BrainPolicy(
            model=artifacts.model,
            device=torch.device("cpu"),
            feature_contract=artifacts.feature_contract,
            normalization=artifacts.normalization,
            max_state_gap_sec=config.runtime.state_reset_gap_sec,
        )
        policy4 = _make_memory_aware_policy(
            brain4, config, artifacts, bars_by_symbol, pipeline,
            enable_memory=True,
            conservative_thresholds=True,
            verbose_fusion=True,
        )
        engine4 = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
        report4 = engine4.run(bars_by_symbol, policy4)
        metrics4 = _collect_fusion_counts(policy4, report4.metrics)
        _print_report("4. FUSION ON + MEMORY ON + CONSERVATIVE THRESHOLDS", metrics4)

        # ── Side-by-side summary ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  COMPARISON SUMMARY")
        print(f"{'='*60}")
        headers = ["Metric", "Baseline", "Fusion", "Fusion+Mem", "Conservative"]
        rows = [
            ("total_return",       "total_return"),
            ("max_drawdown",       "max_drawdown"),
            ("fills",              "fill_count"),
            ("profit_factor",      "profit_factor"),
            ("expectancy",         "expectancy"),
            ("avg_ret/trade",      "avg_return_per_trade"),
            ("veto_count",         "fusion_veto_count"),
            ("dampen_count",       "fusion_dampen_count"),
            ("support_count",      "fusion_support_count"),
            ("neutral_count",      "fusion_neutral_count"),
        ]
        all_metrics = [baseline_report.metrics, metrics2, metrics3, metrics4]
        col_w = 13
        print(f"  {'Metric':<20}" + "".join(f"{h:>{col_w}}" for h in headers[1:]))
        print(f"  {'-'*20}" + "-" * (col_w * 4))
        for display, key in rows:
            vals = [m.get(key, 0.0) for m in all_metrics]
            if key in ("fill_count", "fusion_veto_count", "fusion_dampen_count",
                       "fusion_support_count", "fusion_neutral_count"):
                row = "".join(f"{int(v):>{col_w}}" for v in vals)
            else:
                row = "".join(f"{v:>{col_w}.4f}" for v in vals)
            print(f"  {display:<20}{row}")

        return 0

    # ── Original single-scenario path ─────────────────────────────────────────
    baseline_engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
    baseline_report = baseline_engine.run(bars_by_symbol, BaselineMomentumPolicy())
    _print_report("baseline", baseline_report.metrics)

    sample_symbol = next(iter(bars_by_symbol))
    sample_snapshot = pipeline.build(
        bars_by_symbol[sample_symbol][: config.runtime.warmup_bars + 1],
        symbol=sample_symbol,
    )
    artifacts = resolve_inference_model(
        model_config=config.model,
        feature_config=config.features,
        feature_contract=sample_snapshot.contract,
        device=torch.device("cpu"),
        default_model_kind="brain",
    )
    brain_policy = BrainPolicy(
        model=artifacts.model,
        device=torch.device("cpu"),
        feature_contract=artifacts.feature_contract,
        normalization=artifacts.normalization,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
    )
    if config.memory.enabled:
        brain_policy = _make_memory_aware_policy(
            brain_policy, config, artifacts, bars_by_symbol, pipeline,
            enable_memory=True,
        )
    brain_engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
    brain_report = brain_engine.run(bars_by_symbol, brain_policy)
    label = f"{artifacts.model_kind}_{'checkpoint' if config.model.checkpoint_path else 'untrained'}"
    metrics = _collect_fusion_counts(brain_policy, brain_report.metrics)
    _print_report(label, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
