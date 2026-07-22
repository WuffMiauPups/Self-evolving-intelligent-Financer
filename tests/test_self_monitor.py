"""Self-monitor scaffold (docs/analysis/SELF_MONITOR_DESIGN.md).

Green tests pin what the scaffold already implements (S1-S4, L1 contract,
replay rules). The xfail at the bottom is the executable spec for the
weekend's remaining work (S5).
"""
from __future__ import annotations

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


@pytest.mark.xfail(reason="S5 weekend work: realized spread vs assumed cost_bps", strict=True)
def test_s5_flags_cost_underestimate() -> None:
    monitor = _monitor()
    for i in range(20):
        ts = f"2026-07-22T14:{i % 60:02d}:00+00:00"
        monitor.record_fill(symbol="MSTR", side="buy", quantity=1.0, realized_pnl=0.0,
                            timestamp=ts, entry_confidence=0.6,
                            fill_price=100.10, reference_price=100.00)  # 10bps worse than assumed
        monitor.record_fill(symbol="MSTR", side="sell", quantity=1.0, realized_pnl=-0.1,
                            timestamp=ts)
    report = monitor.health(now=T0)
    assert any(f.startswith("S5_cost_underestimated") for f in report.flags)
