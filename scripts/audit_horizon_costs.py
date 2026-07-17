"""Horizon-vs-cost audit (MASTER_PLAN §7.0.2).

Does ANY configured horizon clear round-trip costs often enough to matter,
or is the 5-min timeframe itself the ceiling?  Per symbol and per
H ∈ {1, 5, 15, 30, 60} bars: P(|H-bar close-to-close forward return| >
cost) and the median |move| conditional on clearing, from the exact bars
the training path loads.

Session-boundary pitfall: an H-bar window that crosses a session close
includes the overnight gap and inflates the move — pairs whose timestamps
are more than H*bar_period + 30 min apart are excluded and the excluded
fraction reported.

Spread caveat: historical bars carry spread=0, so the fee+slippage cost
floor UNDERSTATES true cost.  That only strengthens a negative verdict.

Interpretation aid: with round-trip cost c and typical |move| m, the
breakeven hit rate is (m + c) / (2m) — m = 9 bps at c = 6 bps needs 83%
directional accuracy; m = 30 bps needs 60%.

Deterministic given the bar cache; the covered data range is recorded in
the report.  Output: stdout table, runs/horizon_cost_audit.json, and a
markdown table block for docs/analysis/HORIZON_COST_AUDIT.md.

Usage:
    python scripts/audit_horizon_costs.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

HORIZONS = (1, 5, 15, 30, 60)
SESSION_SLACK_SEC = 1800.0  # 30 min beyond the nominal H-bar span
DEFAULT_REPORT_PATH = Path("runs/horizon_cost_audit.json")


def audit_symbol_horizon(
    closes: np.ndarray,
    epochs: np.ndarray,
    horizon: int,
    *,
    cost_bps: float,
    bar_seconds: float,
) -> dict[str, float] | None:
    """Forward |move| stats for one symbol at one horizon.

    Returns None when there are no usable pairs.  ``excluded_frac`` counts
    session-boundary pairs (timestamp span > H*bar_period + 30 min).
    """
    if closes.shape[0] <= horizon:
        return None
    base = closes[:-horizon]
    ahead = closes[horizon:]
    span = epochs[horizon:] - epochs[:-horizon]
    max_span = horizon * bar_seconds + SESSION_SLACK_SEC
    valid = (span <= max_span) & (base > 0)
    total = int(base.shape[0])
    excluded = total - int(valid.sum())
    if valid.sum() == 0:
        return None
    moves_bps = np.abs(ahead[valid] / base[valid] - 1.0) * 1e4
    clearing = moves_bps > cost_bps
    p_clear = float(clearing.mean())
    median_all = float(np.median(moves_bps))
    median_clear = float(np.median(moves_bps[clearing])) if clearing.any() else float("nan")
    breakeven = (
        (median_clear + cost_bps) / (2.0 * median_clear)
        if np.isfinite(median_clear) and median_clear > 0
        else float("nan")
    )
    return {
        "pairs": total,
        "excluded_frac": excluded / total if total else 0.0,
        "p_clear": p_clear,
        "median_move_bps": median_all,
        "median_clearing_move_bps": median_clear,
        "breakeven_hit_rate": breakeven,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizon-vs-cost audit (§7.0.2)")
    parser.add_argument(
        "--report", default=str(DEFAULT_REPORT_PATH), help="Where to write the JSON report"
    )
    args = parser.parse_args()

    config = load_config("configs/default.yaml", "configs/training.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("audit_horizon_costs")

    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, data_meta = _resolve_bars_by_symbol(config)

    cost_bps = float(config.training.transaction_cost_bps)
    bar_seconds = float(config.backtest.seconds_per_bar)
    logger.info(
        "horizon_cost_audit_start",
        cost_bps=cost_bps,
        bar_seconds=bar_seconds,
        horizons=list(HORIZONS),
        symbols=sorted(bars_by_symbol),
    )

    per_symbol: dict[str, dict[str, dict[str, float]]] = {}
    pooled_moves: dict[int, list[np.ndarray]] = {h: [] for h in HORIZONS}
    pooled_pairs: dict[int, int] = {h: 0 for h in HORIZONS}
    pooled_excluded: dict[int, int] = {h: 0 for h in HORIZONS}
    data_ranges: dict[str, tuple[str, str]] = {}

    for symbol in sorted(bars_by_symbol):
        bars = bars_by_symbol[symbol]
        if len(bars) < max(HORIZONS) + 1:
            logger.warning("symbol_skipped_too_few_bars", symbol=symbol, bars=len(bars))
            continue
        closes = np.array([bar.close for bar in bars], dtype=np.float64)
        epochs = np.array([bar.timestamp.timestamp() for bar in bars], dtype=np.float64)
        data_ranges[symbol] = (
            bars[0].timestamp.isoformat(),
            bars[-1].timestamp.isoformat(),
        )
        per_symbol[symbol] = {}
        for horizon in HORIZONS:
            stats = audit_symbol_horizon(
                closes, epochs, horizon, cost_bps=cost_bps, bar_seconds=bar_seconds
            )
            if stats is None:
                continue
            per_symbol[symbol][str(horizon)] = stats
            # Recompute the included moves once more for pooling (cheap and
            # keeps audit_symbol_horizon a pure per-symbol function).
            base, ahead = closes[:-horizon], closes[horizon:]
            span = epochs[horizon:] - epochs[:-horizon]
            valid = (span <= horizon * bar_seconds + SESSION_SLACK_SEC) & (base > 0)
            pooled_moves[horizon].append(np.abs(ahead[valid] / base[valid] - 1.0) * 1e4)
            pooled_pairs[horizon] += int(base.shape[0])
            pooled_excluded[horizon] += int(base.shape[0] - valid.sum())

    pooled: dict[str, dict[str, float]] = {}
    for horizon in HORIZONS:
        if not pooled_moves[horizon]:
            continue
        moves = np.concatenate(pooled_moves[horizon])
        clearing = moves > cost_bps
        median_clear = float(np.median(moves[clearing])) if clearing.any() else float("nan")
        pooled[str(horizon)] = {
            "pairs": pooled_pairs[horizon],
            "excluded_frac": pooled_excluded[horizon] / pooled_pairs[horizon],
            "p_clear": float(clearing.mean()),
            "median_move_bps": float(np.median(moves)),
            "median_clearing_move_bps": median_clear,
            "breakeven_hit_rate": (
                (median_clear + cost_bps) / (2.0 * median_clear)
                if np.isfinite(median_clear) and median_clear > 0
                else float("nan")
            ),
        }

    report = {
        "cost_bps": cost_bps,
        "bar_seconds": bar_seconds,
        "session_slack_sec": SESSION_SLACK_SEC,
        "horizons": list(HORIZONS),
        "data_meta": {k: v for k, v in data_meta.items() if k != "using_mock"},
        "data_ranges": data_ranges,
        "pooled": pooled,
        "per_symbol": per_symbol,
        "spread_caveat": (
            "historical bars carry spread=0, so cost_bps understates true "
            "round-trip cost; a negative verdict is conservative"
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _row(label: str, stats: dict[str, float]) -> str:
        return (
            f"| {label:<6} | {stats['pairs']:>9,} | {stats['excluded_frac']:>8.1%} "
            f"| {stats['p_clear']:>8.1%} | {stats['median_move_bps']:>10.1f} "
            f"| {stats['median_clearing_move_bps']:>12.1f} | {stats['breakeven_hit_rate']:>9.1%} |"
        )

    header = (
        "| H bars | pairs     | excluded | P(clear) | median bps | clearing bps | breakeven |\n"
        "|--------|-----------|----------|----------|------------|--------------|-----------|"
    )
    print(f"\n  HORIZON-VS-COST AUDIT  (round-trip cost {cost_bps:.1f} bps, bar {bar_seconds:.0f}s)")
    print("\n  POOLED (all symbols)")
    print(header)
    for horizon in HORIZONS:
        if str(horizon) in pooled:
            print(_row(f"{horizon}", pooled[str(horizon)]))
    for symbol in sorted(per_symbol):
        print(f"\n  {symbol}")
        print(header)
        for horizon in HORIZONS:
            if str(horizon) in per_symbol[symbol]:
                print(_row(f"{horizon}", per_symbol[symbol][str(horizon)]))
    print(f"\n  Report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
