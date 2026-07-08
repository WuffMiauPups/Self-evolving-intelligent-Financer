from __future__ import annotations

from dataclasses import dataclass

import torch

from src.backtest.performance import BacktestReport, compute_backtest_metrics
from src.backtest.simulator import BacktestSimulator
from src.core.config import AppConfig
from src.data.schemas import bars_to_tensor
from src.execution.paper_executor import PaperExecutor
from src.execution.schemas import OrderIntent
from src.features.feature_pipeline import FeaturePipeline
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.strategies.base import BatchTradingPolicy, MemoryCapablePolicy, TradingPolicy


@dataclass
class BacktestEngine:
    config: AppConfig
    feature_pipeline: FeaturePipeline
    risk_manager: RiskManager

    def run(self, bars_by_symbol: dict[str, list[object]], policy: object) -> BacktestReport:
        if isinstance(policy, TradingPolicy):
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
        # ------------------------------------------------------------------
        # Vectorized rolling-feature precompute.
        #
        # The backtest used to call ``pipeline.build_from_tensor`` once
        # per bar per symbol.  That path precomputed the OHLCV+spread
        # tensor once (good), but still re-ran all six feature families
        # scalar-style on a window-sized slice every iteration.  For
        # ``N_bars × warmup_bars`` worth of rolling reductions that was
        # ``O(N × W)`` of Python+torch scalar ops per symbol.
        #
        # We now call ``pipeline.build_batch`` once per symbol.  The
        # rolling helpers in ``src/features/rolling.py`` compute every
        # feature across the full bar history in one vectorized pass
        # via ``torch.unfold``, and the per-bar ``snapshot_at(i)``
        # extracts a single-bar ``FeatureSnapshot`` in O(F) — no
        # feature recomputation.  The resulting snapshots are
        # numerically equivalent to the scalar path (modulo float-
        # reduction order; the max observed diff is ~1e-3 on
        # ``regime_price_z_*`` features, well within FP32 noise).
        # ------------------------------------------------------------------
        warmup_bars = self.config.runtime.warmup_bars
        batched_by_symbol = {
            symbol: self.feature_pipeline.build_batch(
                bars, symbol=symbol, temporal_window=warmup_bars
            )
            for symbol, bars in bars_by_symbol.items()
        }
        # Deterministic execution order: sorted symbols. This matches the
        # sequential path's iteration order (dict preserves insertion
        # order, but sorting is explicit and resilient to caller order).
        symbols = sorted(bars_by_symbol.keys())
        use_batch = isinstance(policy, BatchTradingPolicy)
        for index in range(warmup_bars, min_length):
            latest_prices = {symbol: bars_by_symbol[symbol][index].close for symbol in symbols}
            timestamp = bars_by_symbol[symbols[0]][index].timestamp
            portfolio.mark_to_market(latest_prices, timestamp)
            if isinstance(policy, MemoryCapablePolicy):
                # Set temporal cutoff to current bar timestamp so memory
                # retrieval only considers items strictly before this point.
                # This prevents future-outcome data leakage in backtests.
                # MemoryAwarePolicy.propose_batch reads self._temporal_cutoff
                # internally, so setting it once here covers all symbols.
                policy.set_temporal_cutoff(timestamp)
                for symbol in symbols:
                    window_start = max(0, index + 1 - warmup_bars)
                    window_end = index + 1
                    # ``prepare_symbol_history`` still wants the structured
                    # bar list (it inspects MarketBar fields).  Slice cost
                    # is O(window) but only paid when memory is enabled.
                    policy.prepare_symbol_history(
                        symbol, bars_by_symbol[symbol][window_start:window_end]
                    )
            # ------------------------------------------------------------------
            # Batched inference: collect snapshots for all symbols at this
            # bar, then run ONE forward pass.  The per-symbol sequential
            # path used to pay one Python-to-torch round-trip per symbol;
            # ``propose_batch`` packs them into a single encoder+brain
            # forward.  For N symbols that's an Nx reduction in Python
            # overhead and a smaller wins on GPU batch-dim packing.
            # ------------------------------------------------------------------
            snapshots = [batched_by_symbol[symbol].snapshot_at(index) for symbol in symbols]
            if use_batch:
                results = policy.propose_batch(snapshots)
            else:
                # Fallback for policies that don't implement BatchTradingPolicy
                # (e.g. BaselineMomentumPolicy).  Preserves correctness.
                results = [policy.propose(snapshot) for snapshot in snapshots]
            for sym_idx, symbol in enumerate(symbols):
                snapshot = snapshots[sym_idx]
                proposal, _ = results[sym_idx]
                if isinstance(policy, MemoryCapablePolicy):
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
            bars_per_year=self.config.backtest.bars_per_year,
            seconds_per_bar=self.config.backtest.seconds_per_bar,
        )
        if isinstance(policy, MemoryCapablePolicy):
            metrics.update(policy.metrics_snapshot())
        return BacktestReport(
            metrics=metrics,
            equity_curve=equity_curve,
            fills=ledger.fills,
        )
