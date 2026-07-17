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
from src.risk.constraints import action_annotation
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
        """Run a bar-driven backtest.

        Fill timing follows ``config.backtest.fill_timing``:
        ``next_open`` (default) fills a decision made on bar t at bar
        t+1's open — a decision cannot execute at the very close it just
        observed.  ``same_close`` preserves the legacy optimistic fill
        for comparisons.
        """
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
        # Cross-sectional market context (Phase 3): computed ONCE from the
        # full multi-symbol bar set and broadcast into every symbol's
        # batched snapshots.  None while the flag is off (default) — the
        # pipeline then emits no market_* features at all.
        market_context = None
        if self.config.features.market_context_enabled:
            from src.features.market_context import compute_market_context_batch

            market_context = compute_market_context_batch(
                bars_by_symbol,
                short_window=int(self.config.features.windows["short"]),
                medium_window=int(self.config.features.windows["medium"]),
                epsilon=self.config.features.epsilon,
            )
        batched_by_symbol = {
            symbol: self.feature_pipeline.build_batch(
                bars,
                symbol=symbol,
                temporal_window=warmup_bars,
                market_context=market_context,
            )
            for symbol, bars in bars_by_symbol.items()
        }
        # Deterministic execution order: sorted symbols. This matches the
        # sequential path's iteration order (dict preserves insertion
        # order, but sorting is explicit and resilient to caller order).
        symbols = sorted(bars_by_symbol.keys())
        use_batch = isinstance(policy, BatchTradingPolicy)
        # The bar loop must not start before every feature family is
        # warm: warmup_bars (48 default) < first_valid_index (long-window
        # features) fed partially zero-filled snapshots to the policy.
        first_valid = max(
            (
                getattr(batched, "first_valid_index", 0)
                for batched in batched_by_symbol.values()
            ),
            default=0,
        )
        start_index = max(warmup_bars, first_valid)
        next_open_fills = self.config.backtest.fill_timing == "next_open"
        # With next-open fills the final bar has no next open to fill at:
        # stop one bar early rather than fabricating a same-close fill.
        end_index = min_length - 1 if next_open_fills else min_length
        decision_log: list[dict[str, object]] = []
        misaligned_bars = 0
        for index in range(start_index, end_index):
            latest_prices = {symbol: bars_by_symbol[symbol][index].close for symbol in symbols}
            timestamp = bars_by_symbol[symbols[0]][index].timestamp
            misaligned_bars += sum(
                1
                for symbol in symbols[1:]
                if bars_by_symbol[symbol][index].timestamp != timestamp
            )
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
                decision_log.append(
                    {
                        "symbol": symbol,
                        "timestamp": snapshot.timestamp,
                        "bar_index": index,
                        "action": int(proposal.action),
                        "confidence": float(proposal.confidence),
                        # EV-gate inputs (§7.0.4): recorded so predicted
                        # MFE/|MAE| can be rank-correlated against realized
                        # excursions offline.  Additive keys.
                        "expected_gain_bps": float(getattr(proposal, "expected_gain_bps", 0.0)),
                        "expected_loss_bps": float(getattr(proposal, "expected_loss_bps", 0.0)),
                    }
                )
                decision = self.risk_manager.evaluate(
                    proposal,
                    portfolio,
                    evaluation_time=snapshot.timestamp,
                )
                if isinstance(policy, MemoryCapablePolicy):
                    # Memory records the EXECUTED (risk-clamped) action.
                    policy.record_snapshot(
                        symbol, action_annotation=action_annotation(proposal, decision)
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
                if next_open_fills:
                    fill_bar = bars_by_symbol[symbol][index + 1]
                    reference_price = float(fill_bar.open)
                    # Cross half the quoted spread per leg where spread
                    # data exists (live-recorded bars; historical bars
                    # carry spread=0, leaving them unchanged).
                    spread = float(getattr(fill_bar, "spread", 0.0) or 0.0)
                    if spread > 0.0:
                        half = spread / 2.0
                        reference_price += half if (decision.side or "buy") == "buy" else -half
                    fill_timestamp = fill_bar.timestamp
                else:
                    reference_price = snapshot.last_price
                    fill_timestamp = snapshot.timestamp
                intent = OrderIntent(
                    symbol=symbol,
                    side=decision.side or "buy",
                    quantity=decision.quantity,
                    client_order_id=client_order_id,
                    reference_price=reference_price,
                    timestamp=fill_timestamp,
                    paper=True,
                    risk_approved=True,
                    session_id="backtest",
                    sequence=sequence_by_key[order_key],
                    strategy_id=proposal.strategy_id,
                )
                result = simulator.execute(intent)
                if not result.accepted:
                    # A rejected fill (e.g. insufficient cash) must not
                    # register: it previously locked the symbol into a
                    # phantom exit cooldown.
                    continue
                position = portfolio.get_position(symbol)
                self.risk_manager.register_fill(
                    symbol,
                    position.qty if position else 0.0,
                    timestamp=snapshot.timestamp,
                )
                # Outcome-weighted allocation (§6.3): credit the fill's
                # realized NET PnL (PaperExecutor records it fee-adjusted)
                # back to the strategy that proposed the trade.
                if hasattr(policy, "record_fill_outcome") and ledger.fills:
                    fill = ledger.fills[-1]
                    policy.record_fill_outcome(fill.strategy_id, fill.realized_pnl)
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
        if misaligned_bars:
            metrics["timestamp_misaligned_bars"] = float(misaligned_bars)
        if isinstance(policy, MemoryCapablePolicy):
            metrics.update(policy.metrics_snapshot())
        return BacktestReport(
            metrics=metrics,
            equity_curve=equity_curve,
            fills=ledger.fills,
            decision_log=decision_log,
        )
