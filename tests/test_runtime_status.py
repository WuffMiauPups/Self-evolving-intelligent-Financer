"""Tests for the structured per-tick runtime status writer/reader."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.monitoring.runtime_status import (
    RuntimeStatusSnapshot,
    RuntimeStatusWriter,
)


def _snap(step: int = 1) -> RuntimeStatusSnapshot:
    return RuntimeStatusSnapshot(
        ts=datetime(2026, 4, 7, 12, 0, step, tzinfo=timezone.utc),
        step=step,
        session_id="20260407",
        symbol="AAPL",
        equity=100_000.0 + step,
        fills=step,
        safe_to_trade=True,
        last_decision={"action": 1, "confidence": 0.71, "target_fraction": 0.05, "fusion_level": "support"},
        last_risk={"approved": True, "reasons": [], "side": "buy", "quantity": 25.0},
        fusion_enabled=True,
        memory_enabled=True,
        frozen_brain=True,
    )


def test_write_and_read_latest(tmp_path: Path) -> None:
    p = tmp_path / "runtime_status.jsonl"
    writer = RuntimeStatusWriter(p)
    writer.write(_snap(1))
    writer.write(_snap(2))
    writer.write(_snap(3))
    latest = RuntimeStatusWriter.read_latest(p)
    assert latest is not None
    assert latest["step"] == 3
    assert latest["symbol"] == "AAPL"
    assert latest["last_decision"]["action"] == 1
    assert latest["last_decision"]["fusion_level"] == "support"
    assert latest["last_risk"]["approved"] is True
    assert latest["fusion_enabled"] is True
    assert latest["frozen_brain"] is True


def test_read_latest_missing_file(tmp_path: Path) -> None:
    assert RuntimeStatusWriter.read_latest(tmp_path / "no.jsonl") is None


def test_writer_swallows_io_errors(tmp_path: Path) -> None:
    # Point at a directory, not a file — open() will fail. Must not raise.
    bad = tmp_path / "subdir"
    bad.mkdir()
    writer = RuntimeStatusWriter(bad)
    writer.write(_snap(1))  # must not raise


def test_parse_line_skips_garbage() -> None:
    assert RuntimeStatusWriter.parse_line("") is None
    assert RuntimeStatusWriter.parse_line("not json") is None
    assert RuntimeStatusWriter.parse_line('{"step": 5}') == {"step": 5}


def test_runtime_service_uses_structured_snapshot(tmp_path: Path, monkeypatch) -> None:
    """RuntimeService.get_status should consume the structured snapshot
    instead of relying on log keyword scanning."""
    import src.monitoring.runtime_status as rs_mod
    import src.gui_services.runtime_service as svc_mod

    snap_path = tmp_path / "runtime_status.jsonl"
    monkeypatch.setattr(rs_mod, "DEFAULT_RUNTIME_STATUS_PATH", snap_path)
    monkeypatch.setattr(svc_mod, "DEFAULT_RUNTIME_STATUS_PATH", snap_path)

    RuntimeStatusWriter(snap_path).write(_snap(7))

    svc = svc_mod.RuntimeService(project_root=Path.cwd())
    status = svc.get_status()
    assert "step=7" in status.last_message
    assert "action=1" in status.last_decision_summary
    assert "fusion=support" in status.last_decision_summary
    assert "approved" in status.last_risk_decision_summary
    assert status.last_log_timestamp is not None
