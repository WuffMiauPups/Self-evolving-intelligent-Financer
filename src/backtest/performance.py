from __future__ import annotations

from dataclasses import dataclass

from src.risk.ledger import FillRecord


@dataclass
class BacktestReport:
    metrics: dict[str, float]
    equity_curve: list[float]
    fills: list[FillRecord]


def compute_backtest_metrics(
    *,
    initial_equity: float,
    equity_curve: list[float],
    fills: list[FillRecord],
    exposures: list[float],
) -> dict[str, float]:
    if not equity_curve:
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "avg_exposure": 0.0,
        }
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_drawdown = min(max_drawdown, (value / peak) - 1.0 if peak else 0.0)
    realized = [fill.realized_pnl for fill in fills if abs(fill.realized_pnl) > 1.0e-9]
    wins = sum(1 for pnl in realized if pnl > 0)
    win_rate = wins / len(realized) if realized else 0.0
    turnover = sum(abs(fill.quantity * fill.price) for fill in fills) / initial_equity if initial_equity else 0.0
    avg_exposure = sum(exposures) / len(exposures) if exposures else 0.0
    avg_return_per_trade = sum(realized) / len(realized) if realized else 0.0
    gross_profit = sum(pnl for pnl in realized if pnl > 0)
    gross_loss = abs(sum(pnl for pnl in realized if pnl < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
    expectancy = (win_rate * (gross_profit / wins) - (1.0 - win_rate) * (gross_loss / max(len(realized) - wins, 1))) if realized else 0.0
    return {
        "total_return": (equity_curve[-1] / initial_equity) - 1.0,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "turnover": turnover,
        "avg_exposure": avg_exposure,
        "fill_count": float(len(fills)),
        "avg_return_per_trade": avg_return_per_trade,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }

