"""Self-monitor L0 (docs/analysis/SELF_MONITOR_DESIGN.md).

Covers the pure state machine (S1-S5, L1 shrink-only contract, replay
rules), the outcome-join edge cases from the 2026-07-22 review (side-flip,
carried-position seed, client_order_id live join), and the live
tick_processor wiring end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.monitoring.self_monitor import HealthReport, SelfMonitor, SelfMonitorConfig

T0 = "2026-07-22T14:00:00+00:00"


def _monitor(**overrides) -> SelfMonitor:
    return SelfMonitor(SelfMonitorConfig(enabled=True, **overrides))


def _resolve_trades(monitor: SelfMonitor, outcomes: list[tuple[float, bool]]) -> None:
    """Open+close one round trip per (entry_conf, won)."""
    for i, (conf, won) in enumerate(outcomes):
        sym = f"S{i}"
        ts = f"2026-07-22T14:{i % 60:02d}:00+00:00"
        monitor.record_fill(symbol=sym, side="buy", quantity=1.0, realized_pnl=0.0,
                            timestamp=ts, entry_confidence=conf)
        monitor.record_fill(symbol=sym, side="sell", quantity=1.0,
                            realized_pnl=1.0 if won else -1.0, timestamp=ts)


def test_disabled_monitor_records_nothing() -> None:
    monitor = SelfMonitor(SelfMonitorConfig(enabled=False))
    monitor.record_decision(confidence=0.9, action=1, timestamp=T0)
    monitor.record_fill(symbol="A", side="buy", quantity=1.0, realized_pnl=0.0,
                        timestamp=T0, entry_confidence=0.9)
    assert monitor.decision_count == 0
    assert len(monitor.outcomes) == 0


def test_s1_flags_low_win_rate() -> None:
    monitor = _monitor(baseline_win_rate=0.40, window=50)
    # 5 wins / 45 losses = 10% vs 40% baseline -> large negative z
    _resolve_trades(monitor, [(0.6, i < 5) for i in range(50)])
    report = monitor.health(now=T0)
    assert report.win_rate == pytest.approx(0.10)
    assert any(f.startswith("S1_winrate_low") for f in report.flags)


def test_s1_silent_at_baseline() -> None:
    monitor = _monitor(baseline_win_rate=0.40, window=50)
    _resolve_trades(monitor, [(0.6, i % 5 < 2) for i in range(50)])  # 40%
    report = monitor.health(now=T0)
    assert not any(f.startswith("S1") for f in report.flags)


def test_s2_flags_overclaiming_bucket() -> None:
    monitor = _monitor(min_bucket=15, calibration_gap=0.25)
    # Claims 0.85, wins 20% -> gap 0.65
    _resolve_trades(monitor, [(0.85, i < 4) for i in range(20)])
    report = monitor.health(now=T0)
    assert any(f.startswith("S2_miscalibrated") for f in report.flags)
    bucket = report.calibration["0.8"]
    assert bucket["n"] == 20
    assert bucket["realized"] == pytest.approx(0.2)


def test_s3_flags_pinned_confidence() -> None:
    monitor = _monitor()
    for i in range(60):
        monitor.record_decision(confidence=0.49 + 0.001 * (i % 5), action=1,
                                timestamp=f"2026-07-22T14:{i % 60:02d}:00+00:00")
    report = monitor.health(now=T0)
    assert any(f.startswith("S3_confidence_pinned") for f in report.flags)


def test_s3_silent_on_wide_distribution() -> None:
    monitor = _monitor()
    for i in range(60):
        monitor.record_decision(confidence=0.4 + 0.4 * (i % 10) / 10, action=1,
                                timestamp=f"2026-07-22T14:{i % 60:02d}:00+00:00")
    report = monitor.health(now=T0)
    assert not any(f.startswith("S3") for f in report.flags)


def test_s4_counts_close_reopen_cycles() -> None:
    monitor = _monitor(churn_per_hour=3.0)
    monitor.record_decision(confidence=0.6, action=1, timestamp="2026-07-22T14:00:00+00:00")
    monitor.record_decision(confidence=0.6, action=1, timestamp="2026-07-22T15:00:00+00:00")
    for i in range(6):  # 6 close->reopen cycles on one symbol within the hour
        ts = f"2026-07-22T14:{5 * i:02d}:00+00:00"
        monitor.record_fill(symbol="NFLX", side="buy", quantity=1.0, realized_pnl=0.0,
                            timestamp=ts, entry_confidence=0.6)
        monitor.record_fill(symbol="NFLX", side="sell", quantity=1.0, realized_pnl=0.1,
                            timestamp=ts)
    report = monitor.health(now=T0)
    assert any(f.startswith("S4_churning") for f in report.flags)


def test_l1_multiplier_is_shrink_only_and_gated() -> None:
    # Gate OFF: unhealthy monitor still returns 1.0
    monitor = _monitor(damping_enabled=False)
    _resolve_trades(monitor, [(0.85, False) for _ in range(30)])
    assert monitor.health(now=T0).damping_multiplier == 1.0
    # Gate ON: multiplier drops but NEVER below the floor, never above 1.0
    monitor = _monitor(damping_enabled=True, damping_floor=0.5)
    _resolve_trades(monitor, [(0.85, False) for _ in range(30)])
    m = monitor.health(now=T0).damping_multiplier
    assert 0.5 <= m < 1.0


def test_healthy_monitor_multiplier_is_one_even_when_enabled() -> None:
    monitor = _monitor(damping_enabled=True)
    _resolve_trades(monitor, [(0.45, i % 5 < 2) for i in range(20)])  # 40% at claimed ~0.45
    report = monitor.health(now=T0)
    assert report.flags == []
    assert report.damping_multiplier == 1.0


def test_replay_sorts_fills_and_uses_seed(tmp_path) -> None:
    """The two ledger rules (sort by timestamp, seed the carried book)."""
    import json
    decisions = tmp_path / "d.jsonl"
    fills = tmp_path / "f.jsonl"
    decisions.write_text(json.dumps({
        "ts": "2026-07-22T14:00:00+00:00", "proposed_action": 1, "confidence": 0.8,
        "client_order_id": "o1", "confidence_semantics": "p_resolved_class",
    }) + "\n", encoding="utf-8")
    # Fills OUT of order; carried long 5.0 on CARRY so its sell is a CLOSE, not a short
    rows = [
        {"timestamp": "2026-07-22T15:00:00+00:00", "symbol": "CARRY", "side": "sell",
         "quantity": 5.0, "realized_pnl": 2.0, "client_order_id": "x"},
        {"timestamp": "2026-07-22T14:30:00+00:00", "symbol": "A", "side": "buy",
         "quantity": 1.0, "realized_pnl": 0.0, "client_order_id": "o1"},
        {"timestamp": "2026-07-22T14:45:00+00:00", "symbol": "A", "side": "sell",
         "quantity": 1.0, "realized_pnl": 1.5, "client_order_id": "x2"},
    ]
    fills.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    monitor = SelfMonitor.replay_from_ledgers(
        decisions_path=decisions, fills_path=fills, day="2026-07-22",
        seed_positions={"CARRY": 5.0},
    )
    assert monitor._position["CARRY"] == pytest.approx(0.0)   # closed, never negative
    assert list(monitor.outcomes) == [(0.8, True)]            # A round trip joined to conf 0.8


def test_report_serializes_to_json_line() -> None:
    import json
    monitor = _monitor()
    line = monitor.health(now=T0).to_json_line()
    parsed = json.loads(line)
    assert parsed["ts"] == T0
    assert parsed["damping_multiplier"] == 1.0


def test_s5_flags_cost_underestimate() -> None:
    monitor = _monitor()
    for i in range(20):
        ts = f"2026-07-22T14:{i % 60:02d}:00+00:00"
        monitor.record_fill(symbol="MSTR", side="buy", quantity=1.0, realized_pnl=0.0,
                            timestamp=ts, entry_confidence=0.6,
                            fill_price=100.10, reference_price=100.00)  # 10bps worse than assumed 2
        monitor.record_fill(symbol="MSTR", side="sell", quantity=1.0, realized_pnl=-0.1,
                            timestamp=ts)
    report = monitor.health(now=T0)
    assert report.realized_slippage_bps == pytest.approx(10.0, abs=0.1)
    assert any(f.startswith("S5_cost_underestimated") for f in report.flags)


def test_s5_silent_when_slippage_matches_assumption() -> None:
    monitor = _monitor()  # assumed_slippage_bps 2.0, cost_surprise_bps 3.0
    for i in range(20):
        ts = f"2026-07-22T14:{i % 60:02d}:00+00:00"
        monitor.record_fill(symbol="MSTR", side="buy", quantity=1.0, realized_pnl=0.0,
                            timestamp=ts, entry_confidence=0.6,
                            fill_price=100.02, reference_price=100.00)  # 2bps = assumed
        monitor.record_fill(symbol="MSTR", side="sell", quantity=1.0, realized_pnl=0.1,
                            timestamp=ts)
    report = monitor.health(now=T0)
    assert not any(f.startswith("S5") for f in report.flags)


def test_live_join_entry_confidence_by_client_order_id() -> None:
    """The live path: a fill lands with an order id, not the decision's
    confidence — record_decision(client_order_id=...) lets the open join it."""
    monitor = _monitor()
    monitor.record_decision(confidence=0.82, action=1, timestamp=T0, client_order_id="oid-1")
    monitor.record_fill(symbol="AAPL", side="buy", quantity=1.0, realized_pnl=0.0,
                        timestamp=T0, client_order_id="oid-1")   # open, no entry_confidence param
    monitor.record_fill(symbol="AAPL", side="sell", quantity=1.0, realized_pnl=1.5, timestamp=T0)
    assert list(monitor.outcomes) == [(0.82, True)]


def test_side_flip_books_close_and_opens_new_side() -> None:
    """A single fill that reverses sign is a close of the old leg AND an open
    of the new one — neither round trip may be dropped (2026-07-22 review)."""
    monitor = _monitor()
    monitor.record_fill(symbol="AAPL", side="buy", quantity=1.0, realized_pnl=0.0,
                        timestamp=T0, entry_confidence=0.7)              # open long @0.7
    monitor.record_fill(symbol="AAPL", side="sell", quantity=2.0, realized_pnl=1.0,
                        timestamp=T0, entry_confidence=0.4)              # FLIP long->short, book +1
    assert monitor._position["AAPL"] == pytest.approx(-1.0)
    monitor.record_fill(symbol="AAPL", side="buy", quantity=1.0, realized_pnl=-0.5,
                        timestamp=T0)                                    # cover short, book -0.5
    assert list(monitor.outcomes) == [(0.7, True), (0.4, False)]        # both trips recorded


def test_seed_positions_prevents_phantom_short_on_carried_close() -> None:
    """A restart holding a carried long: seeding the start-of-day book means
    the first sell CLOSES it (position -> 0) instead of opening a phantom
    short, and no outcome is fabricated with a wrong confidence."""
    monitor = _monitor()
    monitor.seed_positions({"AAPL": 5.0})
    monitor.record_fill(symbol="AAPL", side="sell", quantity=5.0, realized_pnl=2.0, timestamp=T0)
    assert monitor._position["AAPL"] == pytest.approx(0.0)
    assert list(monitor.outcomes) == []


def test_tick_processor_wiring_feeds_monitor_end_to_end() -> None:
    """Pins the LIVE plumbing (not just the state machine): _write_decision_event
    feeds record_decision with the route's client_order_id, _record_fill_to_monitor
    joins the opening fill by that id, and _emit_self_report writes on cadence.
    A break in this id plumbing would silently leave S1/S2 empty in production."""
    from datetime import UTC, datetime

    from src.execution.tick_processor import (
        _emit_self_report,
        _record_fill_to_monitor,
        _write_decision_event,
    )

    monitor = _monitor()
    written: list = []
    ctx = SimpleNamespace(
        self_monitor=monitor,
        report_writer=SimpleNamespace(write=written.append),
        decision_writer=None,
        config=SimpleNamespace(
            monitoring=SimpleNamespace(self_monitor=SimpleNamespace(report_interval_decisions=1))
        ),
    )
    dt = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)
    proposal = SimpleNamespace(timestamp=dt, symbol="AAPL", action=1, confidence=0.8,
                               target_fraction=0.05, request_id="r",
                               expected_gain_bps=0.0, expected_loss_bps=0.0)
    decision = SimpleNamespace(approved=True, reasons=[], side="buy", quantity=1.0)
    result = SimpleNamespace(client_order_id="o1", status="filled")
    _write_decision_event(ctx, proposal=proposal, decision=decision, result=result)
    assert monitor.decision_count == 1

    open_fill = SimpleNamespace(symbol="AAPL", side="buy", quantity=1.0, realized_pnl=0.0,
                                timestamp=dt, client_order_id="o1")
    close_fill = SimpleNamespace(symbol="AAPL", side="sell", quantity=1.0, realized_pnl=1.5,
                                 timestamp=dt, client_order_id="o2")
    _record_fill_to_monitor(ctx, fill=open_fill)
    _record_fill_to_monitor(ctx, fill=close_fill)
    assert list(monitor.outcomes) == [(0.8, True)]   # joined by client_order_id

    _emit_self_report(ctx, now=dt)
    assert len(written) == 1                          # one report at interval=1


def test_due_for_report_cadence() -> None:
    monitor = _monitor()
    for _ in range(24):
        monitor.record_decision(confidence=0.6, action=1, timestamp=T0)
    assert not monitor.due_for_report(25)   # 24 < 25
    monitor.record_decision(confidence=0.6, action=1, timestamp=T0)
    assert monitor.due_for_report(25)       # 25th crosses the interval
    assert not monitor.due_for_report(25)   # marker advanced, not due again
