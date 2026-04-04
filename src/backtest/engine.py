from __future__ import annotations

from dataclasses import dataclass

from src.backtest.performance import BacktestReport, compute_backtest_metrics
from src.backtest.simulator import BacktestSimulator
from src.core.config import AppConfig
from src.execution.paper_executor import PaperExecutor
from src.execution.schemas import OrderIntent
from src.features.feature_pipeline import FeaturePipeline
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager


@dataclass
class BacktestEngine:
    config: AppConfig
    feature_pipeline: FeaturePipeline
    risk_manager: RiskManager

    def run(self, bars_by_symbol: dict[str, list[object]], policy: object) -> BacktestReport:
        if hasattr(policy, "reset_all_state"):
            policy.reset_all_state(reason="backtest_episode_start")
        self.risk_manager.reset_state()
        sequence_by_key: dict[tuple[str, str, str], int] = {}
        portfolio = PortfolioState(
            cash=self.config.backtest.initial_cash,
            equity=self.config.backtest.initial_cash,
        )
        ledger = Ledger()
        simulator = BacktestSimulator(
            executor=PaperExecutor(
                portfolio=portfolio,
                ledger=ledger,
                fee_bps=self.config.backtest.fee_bps,
                slippage_bps=self.config.backtest.slippage_bps,
            ),
            fee_bps=self.config.backtest.fee_bps,
            slippage_bps=self.config.backtest.slippage_bps,
        )
        min_length = min(len(bars) for bars in bars_by_symbol.values())
        equity_curve: list[float] = []
        exposures: list[float] = []
        for index in range(self.config.runtime.warmup_bars, min_length):
            latest_prices = {symbol: bars[index].close for symbol, bars in bars_by_symbol.items()}
            timestamp = next(iter(bars_by_symbol.values()))[index].timestamp
            portfolio.mark_to_market(latest_prices, timestamp)
            for symbol, bars in bars_by_symbol.items():
                if hasattr(policy, "prepare_symbol_history"):
                    policy.prepare_symbol_history(symbol, bars[: index + 1])
                snapshot = self.feature_pipeline.build(bars[: index + 1], symbol=symbol)
                proposal, _ = policy.propose(snapshot)
                if hasattr(policy, "record_snapshot"):
                    policy.record_snapshot(symbol)
                decision = self.risk_manager.evaluate(
                    proposal,
                    portfolio,
                    evaluation_time=snapshot.timestamp,
                )
                if not decision.approved:
                    continue
                bucket = snapshot.timestamp.strftime("%Y%m%d%H%M")
                order_key = ("backtest", symbol, decision.side or "buy", bucket)
                sequence_by_key[order_key] = sequence_by_key.get(order_key, 0) + 1
                client_order_id = (
                    f"seif-backtest-{symbol.upper()}-{(decision.side or 'buy')[0]}-"
                    f"{bucket}-{sequence_by_key[order_key]:04d}"
                )
                intent = OrderIntent(
                    symbol=symbol,
                    side=decision.side or "buy",
                    quantity=decision.quantity,
                    client_order_id=client_order_id,
                    reference_price=snapshot.last_price,
                    timestamp=snapshot.timestamp,
                    paper=True,
                    risk_approved=True,
                    session_id="backtest",
                    sequence=sequence_by_key[order_key],
                )
                simulator.execute(intent)
                position = portfolio.get_position(symbol)
                self.risk_manager.register_fill(
                    symbol,
                    position.qty if position else 0.0,
                    timestamp=snapshot.timestamp,
                )
            portfolio.mark_to_market(latest_prices, timestamp)
            equity_curve.append(portfolio.equity)
            total_exposure = sum(abs(position.qty * position.market_price) for position in portfolio.positions.values())
            exposures.append(total_exposure / max(portfolio.equity, 1.0))
        metrics = compute_backtest_metrics(
            initial_equity=self.config.backtest.initial_cash,
            equity_curve=equity_curve,
            fills=ledger.fills,
            exposures=exposures,
        )
        if hasattr(policy, "metrics_snapshot"):
            metrics.update(policy.metrics_snapshot())
        return BacktestReport(
            metrics=metrics,
            equity_curve=equity_curve,
            fills=ledger.fills,
        )
