"""Backtest execution + result history.

Runs in-process via BacktestEngine. Designed to be invoked from a Qt worker
thread (no Qt imports here).
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal

import torch

from src.backtest.engine import BacktestEngine
from src.backtest.policy_factory import (
    build_baseline_policy,
    build_brain_memory_policy,
    build_brain_policy,
)
from src.core.config import AppConfig
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import (
    HistoricalDataService,
    generate_synthetic_bars,
    parse_date_or_datetime,
)
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.gui_services.config_service import load_app_config
from src.gui_services.state import BacktestResultRow, ServiceError
from src.risk.risk_manager import RiskManager
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.strategies.memory_policy import MemoryAwarePolicy

BACKTEST_HISTORY_DIR = Path("runs/backtests")

PolicyKind = Literal["baseline", "brain", "brain+memory"]


def _load_bars(
    config: AppConfig, *, use_real_data: bool, allow_mock: bool
) -> dict[str, list[MarketBar]]:
    if not use_real_data:
        return {sym: generate_synthetic_bars(sym, 240) for sym in config.runtime.symbols}
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
    from datetime import UTC, timedelta

    start = parse_date_or_datetime(
        config.data.start_date, default=datetime.now(UTC) - timedelta(days=365 * 2)
    )
    end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    bars, _ = service.load_bars_for_symbols(
        config.data.symbols,
        timeframe=config.data.timeframe,
        start=start,
        end=end,
        force_refresh=config.data.force_refresh,
    )
    return bars


def run_backtest(
    *,
    config_paths: list[str],
    use_real_data: bool = False,
    allow_mock: bool = True,
    label: str = "baseline",
    policy_kind: PolicyKind = "baseline",
    walk_forward: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> BacktestResultRow:
    """Run a single backtest end-to-end and persist the result row.

    ``policy_kind`` controls which strategy is run:
      - ``"baseline"``       — BaselineMomentumPolicy (no model, no memory)
      - ``"brain"``          — BrainPolicy loaded from config.model.checkpoint_path
      - ``"brain+memory"``   — MemoryAwarePolicy (brain + memory + fusion)

    The brain/brain+memory paths go through ``src/backtest/policy_factory.py``
    which is shared with the CLI ``scripts/run_backtest.py``.  Memory/fusion
    toggles applied via layered YAML overrides (``runs/gui_backtest_override.yaml``)
    flow in through ``config_paths``.
    """
    if progress_callback is None:
        progress_callback = lambda _msg: None  # noqa: E731

    progress_callback("Loading config…")
    config = load_app_config(*config_paths)

    progress_callback("Loading bars…")
    try:
        bars = _load_bars(config, use_real_data=use_real_data, allow_mock=allow_mock)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Failed to load market bars: {exc}") from exc
    if not bars:
        raise ServiceError("No bars available for backtest.")

    # Drop symbols that returned zero bars for the requested date range.
    # This happens when a ticker is delisted, renamed, or otherwise has
    # no coverage in the requested window on the Alpaca feed (e.g. SQ
    # after Block Inc's 2025 ticker change).  Letting an empty symbol
    # through would crash ``FeaturePipeline.build_batch`` ("requires a
    # non-empty bar sequence") or, worse, silently set ``min_length=0``
    # in the engine and produce a no-op backtest.  Warn loudly and
    # continue with the surviving symbols.
    empty_symbols = [sym for sym, bar_list in bars.items() if not bar_list]
    if empty_symbols:
        surviving = {s: b for s, b in bars.items() if b}
        progress_callback(
            f"Dropping {len(empty_symbols)} symbol(s) with zero bars: "
            f"{', '.join(empty_symbols)}"
        )
        if not surviving:
            raise ServiceError(
                "All configured symbols returned zero bars for the "
                f"requested date range ({config.data.start_date} → "
                f"{config.data.end_date}).  Widen the window or pick a "
                "preset with a longer history."
            )
        bars = surviving

    progress_callback("Building feature pipeline…")
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

    progress_callback(f"Building {policy_kind} policy…")
    try:
        policy = _build_policy(policy_kind, config, bars, pipeline)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Failed to build {policy_kind} policy: {exc}") from exc

    if walk_forward:
        progress_callback("Running walk-forward cross-validation…")
        from src.evaluation.walk_forward import WalkForwardRunner

        def _wf_builder(cfg, train_bars, pipe):
            return _build_policy(policy_kind, cfg, train_bars, pipe)

        runner = WalkForwardRunner(config, bars)
        try:
            wf_result = runner.run(_wf_builder)
        except Exception as exc:  # noqa: BLE001
            raise ServiceError(f"Walk-forward failed: {exc}") from exc

        agg = wf_result.aggregate_metrics
        row = BacktestResultRow(
            timestamp=datetime.now(),
            label=f"[WF {len(wf_result.folds)}F] {label}",
            total_return=float(agg.get("total_return", {}).get("mean", 0.0)),
            max_drawdown=float(agg.get("max_drawdown", {}).get("mean", 0.0)),
            win_rate=float(agg.get("win_rate", {}).get("mean", 0.0)),
            turnover=float(agg.get("turnover", {}).get("mean", 0.0)),
            fill_count=int(agg.get("fill_count", {}).get("mean", 0)),
            profit_factor=float(agg.get("profit_factor", {}).get("mean", 0.0)),
            expectancy=float(agg.get("expectancy", {}).get("mean", 0.0)),
            avg_return_per_trade=float(agg.get("avg_return_per_trade", {}).get("mean", 0.0)),
            sharpe_ratio=float(agg.get("sharpe_ratio", {}).get("mean", 0.0)),
            sortino_ratio=float(agg.get("sortino_ratio", {}).get("mean", 0.0)),
            avg_exposure=float(agg.get("avg_exposure", {}).get("mean", 0.0)),
            round_trip_win_rate=float(agg.get("round_trip_win_rate", {}).get("mean", 0.0)),
            config_paths=list(config_paths),
            notes=(
                f"walk_forward folds={len(wf_result.folds)} "
                f"policy={policy_kind} "
                f"bars={sum(len(v) for v in bars.values())} symbols={len(bars)}"
            ),
        )
        progress_callback("Persisting result…")
        _persist_row(row)
        return row

    progress_callback("Running backtest…")
    engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
    try:
        report = engine.run(bars, policy)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Backtest failed: {exc}") from exc

    metrics = dict(report.metrics)
    # Fusion metrics removed with the fusion layer. MemoryAwarePolicy is now
    # a thin passive wrapper; its metrics_snapshot() only exposes memory-bank
    # counts (memory_item_count / memory_pending_count / memory_finalized_count).
    row = BacktestResultRow(
        timestamp=datetime.now(),
        label=label,
        total_return=float(metrics.get("total_return", 0.0)),
        max_drawdown=float(metrics.get("max_drawdown", 0.0)),
        win_rate=float(metrics.get("win_rate", 0.0)),
        turnover=float(metrics.get("turnover", 0.0)),
        fill_count=int(metrics.get("fill_count", 0)),
        profit_factor=float(metrics.get("profit_factor", 0.0)),
        expectancy=float(metrics.get("expectancy", 0.0)),
        avg_return_per_trade=float(metrics.get("avg_return_per_trade", 0.0)),
        sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0)),
        sortino_ratio=float(metrics.get("sortino_ratio", 0.0)),
        avg_exposure=float(metrics.get("avg_exposure", 0.0)),
        round_trip_win_rate=float(metrics.get("round_trip_win_rate", 0.0)),
        config_paths=list(config_paths),
        notes=(
            f"policy={policy_kind} "
            f"bars={sum(len(v) for v in bars.values())} symbols={len(bars)}"
        ),
    )

    progress_callback("Persisting result…")
    _persist_row(row)
    return row


def _build_policy(
    policy_kind: PolicyKind,
    config: AppConfig,
    bars: dict[str, list[MarketBar]],
    pipeline: FeaturePipeline,
):
    """Dispatch to the right builder in src.backtest.policy_factory."""
    if policy_kind == "baseline":
        return build_baseline_policy()
    if policy_kind == "brain":
        return build_brain_policy(config, bars_by_symbol=bars, pipeline=pipeline)
    if policy_kind == "brain+memory":
        return build_brain_memory_policy(
            config,
            bars_by_symbol=bars,
            pipeline=pipeline,
            enable_memory=True,
        )
    raise ServiceError(f"Unknown policy_kind: {policy_kind}")


_FILENAME_FORBIDDEN_RX = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _safe_label_for_filename(label: str) -> str:
    """Strip characters that Windows forbids in filenames.

    The full label is preserved verbatim inside the JSON payload (see
    ``BacktestResultRow.label``); only the on-disk file name is sanitized.
    Windows rejects ``<>:"/\\|?*`` plus control bytes, while POSIX only
    rejects ``/`` and NUL — sanitizing the union keeps the file name
    portable across both.
    """
    cleaned = _FILENAME_FORBIDDEN_RX.sub("_", label).strip(" ._")
    # Collapse runs of underscores left behind by the substitution.
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "run"


def _persist_row(row: BacktestResultRow) -> None:
    BACKTEST_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_label = _safe_label_for_filename(row.label)
    fname = row.timestamp.strftime("%Y%m%d_%H%M%S") + f"_{safe_label}.json"
    out = BACKTEST_HISTORY_DIR / fname
    with open(out, "w", encoding="utf-8") as f:
        json.dump(row.to_dict(), f, indent=2)


def list_recent_results(*, limit: int = 50) -> list[BacktestResultRow]:
    if not BACKTEST_HISTORY_DIR.exists():
        return []
    files = sorted(BACKTEST_HISTORY_DIR.glob("*.json"), reverse=True)[:limit]
    rows: list[BacktestResultRow] = []
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                rows.append(BacktestResultRow.from_dict(json.load(f)))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            continue
    return rows
