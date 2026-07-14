"""Durable outcome dataset tests (plan §4.5.1).

The self-evolution loop learns from realized outcomes — these pin that
fills survive restarts, per-trade PnL is net of fees, and every
decision (approved, rejected, resized) lands in the decision stream
with the EXECUTED action.
"""
from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.execution.tick_processor import _write_decision_event
from src.monitoring.decision_log import DecisionEvent, DecisionLogWriter
from src.risk.ledger import FillRecord, Ledger

T0 = datetime(2026, 7, 6, 14, 30, tzinfo=UTC)


def _fill(i: int, pnl: float = 10.0) -> FillRecord:
    return FillRecord(
        timestamp=T0,
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        price=100.0 + i,
        fee=0.5,
        realized_pnl=pnl,
        client_order_id=f"cid-{i}",
    )


def test_ledger_fills_survive_restart(tmp_path) -> None:
    path = tmp_path / "fills.jsonl"
    ledger = Ledger(persist_path=path)
    ledger.record(_fill(1))
    ledger.record(_fill(2, pnl=-3.25))

    revived = Ledger.load(path)
    assert len(revived.fills) == 2
    assert revived.fills[0].price == pytest.approx(101.0)
    assert revived.fills[1].realized_pnl == pytest.approx(-3.25)
    # Appends exactly once per fill.
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_ledger_load_skips_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "fills.jsonl"
    path.write_text(_fill(1).to_json_line() + "\n{torn line\n", encoding="utf-8")
    revived = Ledger.load(path)
    assert len(revived.fills) == 1


def test_ledger_persist_failure_never_raises(tmp_path) -> None:
    ledger = Ledger(persist_path=tmp_path)  # a directory: open() will fail
    ledger.record(_fill(1))  # must not raise
    assert len(ledger.fills) == 1


def test_paper_executor_reports_net_of_fee_pnl() -> None:
    from src.execution.paper_executor import PaperExecutor
    from src.execution.schemas import OrderIntent
    from src.risk.portfolio_state import PortfolioState

    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    ledger = Ledger()
    executor = PaperExecutor(portfolio=portfolio, ledger=ledger, fee_bps=10.0, slippage_bps=0.0)

    def _intent(side: str, seq: int) -> OrderIntent:
        side_token = "b" if side == "buy" else "s"
        return OrderIntent(
            symbol="AAPL",
            side=side,
            quantity=10.0,
            client_order_id=f"seif-test-AAPL-{side_token}-202607061430-{seq:04d}",
            reference_price=100.0 if side == "buy" else 110.0,
            timestamp=T0,
            paper=True,
            risk_approved=True,
            session_id="test",
            sequence=seq,
        )

    executor.execute(_intent("buy", 1))
    executor.execute(_intent("sell", 2))
    close_fill = ledger.fills[-1]
    gross = (110.0 - 100.0) * 10.0
    exit_fee = abs(110.0 * 10.0) * (10.0 / 10_000.0)
    assert close_fill.fee == pytest.approx(exit_fee)
    assert close_fill.realized_pnl == pytest.approx(gross - exit_fee)
    assert portfolio.realized_pnl < gross  # net of fees, not gross


def _proposal(action: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=T0,
        symbol="AAPL",
        request_id="req-1",
        action=action,
        confidence=0.8,
        target_fraction=0.05,
        strategy_id="brain",
    )


def test_decision_event_records_rejection(tmp_path) -> None:
    writer = DecisionLogWriter(tmp_path / "decisions.jsonl")
    ctx = SimpleNamespace(decision_writer=writer)
    decision = SimpleNamespace(approved=False, reasons=["daily_loss_limit"], side=None, quantity=0.0)
    _write_decision_event(ctx, proposal=_proposal(), decision=decision, result=None)
    events = DecisionLogWriter.read_all(tmp_path / "decisions.jsonl")
    assert len(events) == 1
    assert events[0]["approved"] is False
    assert events[0]["reasons"] == ["daily_loss_limit"]
    assert events[0]["executed_side"] is None


def test_decision_event_records_executed_resized_action(tmp_path) -> None:
    writer = DecisionLogWriter(tmp_path / "decisions.jsonl")
    ctx = SimpleNamespace(decision_writer=writer)
    decision = SimpleNamespace(
        approved=True, reasons=["resized_to_cap"], side="buy", quantity=50.0
    )
    result = SimpleNamespace(status="filled", client_order_id="cid-9")
    _write_decision_event(ctx, proposal=_proposal(), decision=decision, result=result)
    event = DecisionLogWriter.read_all(tmp_path / "decisions.jsonl")[0]
    assert event["approved"] is True
    assert event["executed_side"] == "buy"
    assert event["executed_qty"] == pytest.approx(50.0)
    assert event["resized"] is True
    assert event["order_status"] == "filled"


def test_decision_event_noop_without_writer() -> None:
    ctx = SimpleNamespace(decision_writer=None)
    decision = SimpleNamespace(approved=True, reasons=[], side="buy", quantity=1.0)
    _write_decision_event(ctx, proposal=_proposal(), decision=decision, result=None)  # no raise


def test_decision_event_round_trips_via_dataclass(tmp_path) -> None:
    writer = DecisionLogWriter(tmp_path / "d.jsonl")
    writer.write(
        DecisionEvent(
            ts=T0,
            symbol="MSFT",
            request_id="r",
            proposed_action=-1,
            confidence=0.6,
            target_fraction=-0.05,
            approved=True,
            executed_side="sell",
            executed_qty=3.0,
        )
    )
    event = DecisionLogWriter.read_all(tmp_path / "d.jsonl")[0]
    assert event["symbol"] == "MSFT"
    assert event["proposed_action"] == -1


# ── §4.5.2: executed-action credit assignment ──────────────────────────


def test_action_annotation_formats_executed_action() -> None:
    from src.risk.constraints import RiskDecision, action_annotation

    proposal = _proposal(action=1)
    resized = RiskDecision(approved=True, reasons=["resized_to_cap"], quantity=50.0, side="buy")
    annotation = action_annotation(proposal, resized)
    assert annotation == "action=1 approved=1 side=buy qty=50.0000 resized=1 strategy=brain"

    rejected = RiskDecision(approved=False, reasons=["daily_loss_limit"])
    annotation = action_annotation(proposal, rejected)
    assert annotation == "action=1 approved=0 side=- qty=0.0000 resized=0 strategy=brain"


def test_record_snapshot_threads_annotation_to_updater() -> None:
    from src.strategies.memory_policy import MemoryAwarePolicy

    observed: list[tuple[object, str | None]] = []

    class _Updater:
        def observe(self, encoded_state, *, action_annotation=None):
            observed.append((encoded_state, action_annotation))

    policy = MemoryAwarePolicy.__new__(MemoryAwarePolicy)
    policy.updater = _Updater()
    policy.last_encoded_by_symbol = {"AAPL": object()}
    policy.record_snapshot("AAPL", action_annotation="action=1 approved=1 side=buy qty=5.0000 resized=0")
    assert observed[0][1] == "action=1 approved=1 side=buy qty=5.0000 resized=0"


def test_intervention_rate_computed_from_counters() -> None:
    from src.execution.tick_processor import _intervention_rate
    from src.monitoring.metrics_logger import MetricsLogger

    metrics = MetricsLogger()
    ctx = SimpleNamespace(metrics=metrics)
    assert _intervention_rate(ctx) == 0.0
    for _ in range(4):
        metrics.increment("proposals")
    metrics.increment("risk_interventions")
    assert _intervention_rate(ctx) == pytest.approx(0.25)
