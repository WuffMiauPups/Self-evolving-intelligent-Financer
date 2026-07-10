from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from src.risk.ledger import FillRecord


@dataclass
class BacktestReport:
    metrics: dict[str, float]
    equity_curve: list[float]
    fills: list[FillRecord]
    # Per-proposal record (symbol, timestamp, bar_index, action,
    # confidence) — lets the benchmark grade directional accuracy against
    # realized forward returns without re-running the policy.
    decision_log: list[dict[str, object]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.decision_log is None:
            self.decision_log = []


def compute_backtest_metrics(
    *,
    initial_equity: float,
    equity_curve: list[float],
    fills: list[FillRecord],
    exposures: list[float],
    bars_per_year: int = 19656,
    seconds_per_bar: float = 300.0,
) -> dict[str, float]:
    """Compute comprehensive backtest metrics.

    Args:
        initial_equity: Starting portfolio value.
        equity_curve: Per-bar equity values.
        fills: All executed fills.
        exposures: Per-bar exposure ratios.
        bars_per_year: Annualization factor (default 252 days * 78 five-min bars).
    """
    if not equity_curve:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "avg_exposure": 0.0,
        }

    # --- Basic return ---
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_equity) - 1.0

    # --- Max drawdown ---
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, (value / peak) - 1.0 if peak else 0.0)

    # --- Per-bar returns for ratio calculations ---
    bar_returns: list[float] = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        bar_returns.append((equity_curve[i] / prev) - 1.0 if prev > 0 else 0.0)

    # --- Sharpe ratio (annualized) ---
    sharpe_ratio = 0.0
    if bar_returns:
        mean_ret = sum(bar_returns) / len(bar_returns)
        variance = sum((r - mean_ret) ** 2 for r in bar_returns) / max(len(bar_returns) - 1, 1)
        std_ret = math.sqrt(variance)
        if std_ret > 1e-12:
            sharpe_ratio = (mean_ret / std_ret) * math.sqrt(bars_per_year)

    # --- Sortino ratio (annualized, downside deviation) ---
    sortino_ratio = 0.0
    if bar_returns:
        mean_ret = sum(bar_returns) / len(bar_returns)
        downside_sq = [r ** 2 for r in bar_returns if r < 0]
        # Downside deviation divides by ALL observations, not just the
        # negative ones — dividing by the loss count overstated the
        # deviation (and understated Sortino) in proportion to how rarely
        # the strategy lost.
        downside_dev = math.sqrt(sum(downside_sq) / len(bar_returns)) if downside_sq else 0.0
        if downside_dev > 1e-12:
            sortino_ratio = (mean_ret / downside_dev) * math.sqrt(bars_per_year)

    # --- CVaR 95 (expected shortfall of per-bar returns) ---
    cvar_95 = 0.0
    if bar_returns:
        tail_count = max(1, int(math.ceil(len(bar_returns) * 0.05)))
        worst = sorted(bar_returns)[:tail_count]
        cvar_95 = sum(worst) / len(worst)

    # --- CAGR ---
    n_bars = len(equity_curve)
    years = n_bars / bars_per_year if bars_per_year > 0 else 1.0
    cagr = 0.0
    if years > 0 and final_equity > 0 and initial_equity > 0:
        cagr = (final_equity / initial_equity) ** (1.0 / max(years, 1e-6)) - 1.0

    # --- Calmar ratio ---
    calmar_ratio = 0.0
    if abs(max_drawdown) > 1e-12:
        calmar_ratio = cagr / abs(max_drawdown)

    # --- Fill-based metrics ---
    realized = [fill.realized_pnl for fill in fills if abs(fill.realized_pnl) > 1e-9]
    wins = sum(1 for pnl in realized if pnl > 0)
    losses = len(realized) - wins
    win_rate = wins / len(realized) if realized else 0.0

    gross_profit = sum(pnl for pnl in realized if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in realized if pnl < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        # No losing trades: report a large finite sentinel instead of 0.0,
        # which ranked all-winning strategies WORST in comparisons.
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    avg_return_per_trade = sum(realized) / len(realized) if realized else 0.0
    avg_win = gross_profit / wins if wins > 0 else 0.0
    avg_loss = gross_loss / losses if losses > 0 else 0.0
    expectancy = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) if realized else 0.0

    # --- Turnover ---
    turnover = sum(abs(fill.quantity * fill.price) for fill in fills) / initial_equity if initial_equity else 0.0

    # --- Exposure ---
    avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0

    # --- Long / Short split ---
    long_fills = [f for f in fills if f.side == "buy"]
    short_fills = [f for f in fills if f.side == "sell"]
    long_count = len(long_fills)
    short_count = len(short_fills)
    long_realized = [f.realized_pnl for f in long_fills if abs(f.realized_pnl) > 1e-9]
    short_realized = [f.realized_pnl for f in short_fills if abs(f.realized_pnl) > 1e-9]
    long_win_rate = sum(1 for p in long_realized if p > 0) / len(long_realized) if long_realized else 0.0
    short_win_rate = sum(1 for p in short_realized if p > 0) / len(short_realized) if short_realized else 0.0

    # --- Round trip tracking ---
    position_tracker: dict[str, list[FillRecord]] = defaultdict(list)
    round_trip_count = 0
    round_trip_wins = 0
    total_holding_bars = 0.0
    prev_net_by_symbol: dict[str, float] = defaultdict(float)
    for fill in fills:
        tracker = position_tracker[fill.symbol]
        tracker.append(fill)
        net_qty = sum(
            f.quantity if f.side == "buy" else -f.quantity for f in tracker
        )
        # A round trip closes when the net position returns to zero OR
        # flips sign (long -> short in one fill previously never closed
        # the long trip, undercounting trips and inflating holding times).
        prev_net = prev_net_by_symbol[fill.symbol]
        prev_net_by_symbol[fill.symbol] = net_qty
        sign_flip = prev_net != 0.0 and net_qty != 0.0 and (prev_net > 0) != (net_qty > 0)
        if abs(net_qty) < 1e-9 or sign_flip:
            round_trip_count += 1
            # Sum realized PnL across all fills that made up this round trip.
            # A winning round trip has net positive realized PnL (post-fee).
            trip_pnl = sum(f.realized_pnl for f in tracker)
            if trip_pnl > 0:
                round_trip_wins += 1
            if len(tracker) >= 2:
                entry_time = tracker[0].timestamp
                exit_time = tracker[-1].timestamp
                holding_sec = (exit_time - entry_time).total_seconds()
                # Convert to bars (approximate)
                holding_bars = holding_sec / seconds_per_bar
                total_holding_bars += holding_bars
            position_tracker[fill.symbol] = []
    avg_holding_bars = total_holding_bars / round_trip_count if round_trip_count > 0 else 0.0
    round_trip_win_rate = round_trip_wins / round_trip_count if round_trip_count > 0 else 0.0

    # --- Cost tracking ---
    total_fees = sum(fill.fee for fill in fills)

    # --- Flat fraction (bars with zero exposure) ---
    flat_bars = sum(1 for e in exposures if abs(e) < 1e-9)
    flat_fraction = flat_bars / len(exposures) if exposures else 1.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "cvar_95": cvar_95,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_return_per_trade": avg_return_per_trade,
        "turnover": turnover,
        "avg_exposure": avg_exposure,
        "fill_count": float(len(fills)),
        "trade_count": float(len(fills)),
        "round_trip_count": float(round_trip_count),
        "round_trip_win_rate": round_trip_win_rate,
        "long_count": float(long_count),
        "short_count": float(short_count),
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "avg_holding_bars": avg_holding_bars,
        "total_fees": total_fees,
        "flat_fraction": flat_fraction,
    }
