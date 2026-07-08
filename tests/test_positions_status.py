"""Tests for the positions + account status writer/reader used by the
GUI's POSITIONS tab.

Coverage targets:
  * JSON round-trip for the full snapshot (account + positions).
  * Missing-field tolerance in the reader (forward-compat).
  * Atomic overwrite — two writes in a row leave a single well-formed
    file with the second payload.
  * ``PositionPeakTracker`` high-water mark + drawdown semantics:
    initial, rising, falling, re-entry resets.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.positions_status import (
    AccountSnapshot,
    PositionPeakTracker,
    PositionSnapshot,
    PositionsStatusSnapshot,
    PositionsStatusWriter,
)


def _make_account(**overrides) -> AccountSnapshot:
    base = dict(
        equity=100_500.0,
        cash=85_250.0,
        buying_power=170_500.0,
        portfolio_value=100_500.0,
        session_start_equity=100_000.0,
        session_peak_equity=100_750.0,
        session_return=0.005,
        session_drawdown=-0.0025,
        max_position_notional=5_000.0,
        max_concurrent_positions=3,
        min_trade_confidence=0.55,
        kill_switch=False,
    )
    base.update(overrides)
    return AccountSnapshot(**base)


def _make_position(symbol: str = "AAPL", **overrides) -> PositionSnapshot:
    base = dict(
        symbol=symbol,
        side="long",
        qty=0.5,
        avg_price=180.0,
        market_price=182.5,
        notional=91.25,
        unrealized_pnl=1.25,
        unrealized_pnl_pct=0.01389,
        peak_unrealized_pnl=1.30,
        drawdown=-0.05,
        drawdown_pct=-0.000556,
        entry_time=datetime(2026, 4, 8, 17, 0, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return PositionSnapshot(**base)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_positions_snapshot_json_round_trip() -> None:
    snap = PositionsStatusSnapshot(
        ts=datetime(2026, 4, 8, 17, 2, 58, tzinfo=timezone.utc),
        account=_make_account(),
        positions=[
            _make_position("AAPL"),
            _make_position("NFLX", side="long", qty=0.017),
            _make_position("MSTR", side="short", qty=-0.2),
        ],
    )
    text = snap.to_json()
    reloaded = PositionsStatusSnapshot.from_json(text)
    assert reloaded is not None
    assert reloaded.ts == snap.ts
    assert reloaded.account.equity == snap.account.equity
    assert reloaded.account.kill_switch is False
    assert len(reloaded.positions) == 3
    assert [p.symbol for p in reloaded.positions] == ["AAPL", "NFLX", "MSTR"]
    assert reloaded.positions[2].side == "short"
    assert reloaded.positions[2].qty == -0.2


def test_positions_snapshot_tolerates_missing_fields() -> None:
    # Forward-compat: a minimal payload should still parse into the
    # dataclass with sensible defaults, matching the "new fields are
    # additive" schema contract in the module docstring.
    minimal = (
        '{"ts":"2026-04-08T17:00:00+00:00",'
        '"account":{"equity":1000.0},'
        '"positions":[]}'
    )
    snap = PositionsStatusSnapshot.from_json(minimal)
    assert snap is not None
    assert snap.account.equity == 1000.0
    assert snap.account.cash == 0.0
    assert snap.account.max_concurrent_positions == 0
    assert snap.positions == []


def test_positions_snapshot_from_json_rejects_garbage() -> None:
    assert PositionsStatusSnapshot.from_json("not json") is None
    assert PositionsStatusSnapshot.from_json("[]") is None  # list, not dict
    assert PositionsStatusSnapshot.from_json('{"ts":"bogus"}') is None  # bad ts


# ---------------------------------------------------------------------------
# Writer atomic overwrite
# ---------------------------------------------------------------------------


def test_writer_atomic_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "positions_status.json"
    writer = PositionsStatusWriter(path=path)

    first = PositionsStatusSnapshot(
        ts=datetime(2026, 4, 8, 17, 0, 0, tzinfo=timezone.utc),
        account=_make_account(equity=100_000.0),
        positions=[_make_position("AAPL", unrealized_pnl=0.5)],
    )
    writer.write(first)
    assert path.exists()
    loaded_first = PositionsStatusWriter.read_latest(path)
    assert loaded_first is not None
    assert loaded_first.account.equity == 100_000.0

    # Overwrite with a second snapshot — same file, new content, no
    # stale tmp file left behind.
    second = PositionsStatusSnapshot(
        ts=datetime(2026, 4, 8, 17, 5, 0, tzinfo=timezone.utc),
        account=_make_account(equity=100_250.0),
        positions=[
            _make_position("AAPL", unrealized_pnl=2.5),
            _make_position("NFLX", unrealized_pnl=1.1),
        ],
    )
    writer.write(second)
    loaded_second = PositionsStatusWriter.read_latest(path)
    assert loaded_second is not None
    assert loaded_second.account.equity == 100_250.0
    assert len(loaded_second.positions) == 2
    assert not (path.with_suffix(path.suffix + ".tmp")).exists()


def test_reader_returns_none_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    assert PositionsStatusWriter.read_latest(missing) is None


def test_reader_returns_none_for_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("", encoding="utf-8")
    assert PositionsStatusWriter.read_latest(empty) is None


def test_reader_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    assert PositionsStatusWriter.read_latest(corrupt) is None


# ---------------------------------------------------------------------------
# PositionPeakTracker semantics
# ---------------------------------------------------------------------------


def test_peak_tracker_initial_entry_seeds_peak_at_current_pnl() -> None:
    tracker = PositionPeakTracker()
    peak, dd = tracker.update(
        symbol="AAPL", qty=1.0, unrealized_pnl=0.0, entry_epoch=1.0
    )
    assert peak == 0.0
    assert dd == 0.0


def test_peak_tracker_rises_with_pnl() -> None:
    tracker = PositionPeakTracker()
    tracker.update(symbol="AAPL", qty=1.0, unrealized_pnl=0.0, entry_epoch=1.0)
    peak, dd = tracker.update(
        symbol="AAPL", qty=1.0, unrealized_pnl=5.0, entry_epoch=1.0
    )
    assert peak == 5.0
    assert dd == 0.0


def test_peak_tracker_reports_drawdown_when_pnl_falls() -> None:
    tracker = PositionPeakTracker()
    tracker.update(symbol="AAPL", qty=1.0, unrealized_pnl=5.0, entry_epoch=1.0)
    tracker.update(symbol="AAPL", qty=1.0, unrealized_pnl=8.0, entry_epoch=1.0)
    peak, dd = tracker.update(
        symbol="AAPL", qty=1.0, unrealized_pnl=6.5, entry_epoch=1.0
    )
    assert peak == 8.0
    assert dd == -1.5


def test_peak_tracker_never_reports_positive_drawdown() -> None:
    tracker = PositionPeakTracker()
    # Even if PnL keeps rising, drawdown is clamped to zero (never > 0).
    for pnl in (0.0, 1.0, 3.0, 7.0):
        _, dd = tracker.update(
            symbol="AAPL", qty=1.0, unrealized_pnl=pnl, entry_epoch=1.0
        )
        assert dd == 0.0


def test_peak_tracker_forgets_closed_positions() -> None:
    tracker = PositionPeakTracker()
    tracker.update(symbol="AAPL", qty=1.0, unrealized_pnl=5.0, entry_epoch=1.0)
    # Close the position (qty = 0) — tracker should drop its state.
    peak, dd = tracker.update(
        symbol="AAPL", qty=0.0, unrealized_pnl=0.0, entry_epoch=1.0
    )
    assert peak == 0.0
    assert dd == 0.0
    assert "AAPL" not in tracker._state


def test_peak_tracker_resets_on_re_entry() -> None:
    tracker = PositionPeakTracker()
    # First entry: peak climbs to 10.
    tracker.update(symbol="AAPL", qty=1.0, unrealized_pnl=10.0, entry_epoch=1.0)
    # Close.
    tracker.update(symbol="AAPL", qty=0.0, unrealized_pnl=0.0, entry_epoch=1.0)
    # Re-enter with a DIFFERENT entry epoch — peak must reset.
    peak, dd = tracker.update(
        symbol="AAPL", qty=1.0, unrealized_pnl=2.0, entry_epoch=2.0
    )
    assert peak == 2.0  # Not 10 from the previous trade.
    assert dd == 0.0
