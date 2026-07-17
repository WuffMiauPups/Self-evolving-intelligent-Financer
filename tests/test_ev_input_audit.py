"""EV-input quality audit plumbing (§7.0.4).

DecisionEvent (runtime) and the backtest decision log now carry the EV
legs exactly as the EV gate consumed them; the audit script's realized
MFE/MAE must match how targets.py builds MFE/MAE targets
(compute_outcome_summary), and old decision-log lines without the new
keys must keep loading (torn-line-style tolerance).
"""
from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.data.schemas import MarketBar
from src.memory.outcomes import compute_outcome_summary
from src.monitoring.decision_log import DecisionEvent, DecisionLogWriter

_ROOT = Path(__file__).resolve().parents[1]


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_ev_inputs", _ROOT / "scripts" / "audit_ev_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


T = datetime(2026, 7, 17, 14, 0, tzinfo=UTC)


def _bar(index: int, close: float) -> MarketBar:
    return MarketBar(
        symbol="TEST",
        timestamp=T + timedelta(minutes=5 * index),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1000.0,
    )


def test_decision_event_round_trips_ev_legs() -> None:
    event = DecisionEvent(
        ts=T,
        symbol="NVDA",
        request_id="r-1",
        proposed_action=1,
        confidence=0.51,
        target_fraction=0.01,
        approved=True,
        expected_gain_bps=42.5,
        expected_loss_bps=17.25,
    )
    payload = json.loads(event.to_json_line())
    assert payload["expected_gain_bps"] == pytest.approx(42.5)
    assert payload["expected_loss_bps"] == pytest.approx(17.25)


def test_loader_tolerates_old_lines_without_ev_legs(tmp_path: Path) -> None:
    old_line = json.dumps(
        {
            "ts": T.isoformat(),
            "symbol": "NVDA",
            "request_id": "r-old",
            "proposed_action": 1,
            "confidence": 0.5,
            "target_fraction": 0.01,
            "approved": False,
            "reasons": ["negative_ev"],
        }
    )
    log_path = tmp_path / "decisions.jsonl"
    log_path.write_text(old_line + "\n", encoding="utf-8")
    writer = DecisionLogWriter(log_path)
    writer.write(
        DecisionEvent(
            ts=T,
            symbol="NVDA",
            request_id="r-new",
            proposed_action=1,
            confidence=0.5,
            target_fraction=0.01,
            approved=True,
            expected_gain_bps=10.0,
            expected_loss_bps=5.0,
        )
    )
    events = DecisionLogWriter.read_all(log_path)
    assert len(events) == 2
    assert "expected_gain_bps" not in events[0]  # old line untouched, still loads
    assert events[1]["expected_gain_bps"] == pytest.approx(10.0)


def test_realized_excursions_match_targets_construction() -> None:
    """The audit's realized MFE/MAE must equal compute_outcome_summary's
    excursions — the exact way targets.py builds the mfe/mae targets."""
    audit = _load_audit_module()
    closes = [100.0, 100.4, 99.2, 101.1, 100.7, 99.8, 100.2, 101.5, 98.9, 100.05, 100.3, 99.95]
    bars = [_bar(i, c) for i, c in enumerate(closes)]
    horizons = [1, 3, 6]
    index = 2
    outcome = compute_outcome_summary(
        bars, index, horizons=horizons, path_label_rules={}
    )
    result = audit.realized_excursions_bps(
        np.array(closes, dtype=np.float64), index, max(horizons)
    )
    assert result is not None
    mfe_bps, mae_bps = result
    # compute_outcome_summary runs in float32; tolerance covers the cast.
    assert mfe_bps == pytest.approx(outcome.max_favorable_excursion * 1e4, rel=1e-4, abs=0.05)
    assert mae_bps == pytest.approx(abs(outcome.max_adverse_excursion) * 1e4, rel=1e-4, abs=0.05)


def test_realized_excursions_refuse_short_forward_window() -> None:
    audit = _load_audit_module()
    closes = np.array([100.0, 100.1, 100.2], dtype=np.float64)
    assert audit.realized_excursions_bps(closes, 0, 6) is None


def test_spearman_sanity() -> None:
    audit = _load_audit_module()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert audit.spearman(x, x * 10.0 + 3.0) == pytest.approx(1.0)
    assert audit.spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(audit.spearman(x, np.ones_like(x)))  # constant side: undefined
    monotone_nonlinear = np.array([1.0, 8.0, 27.0, 64.0, 125.0])
    assert audit.spearman(x, monotone_nonlinear) == pytest.approx(1.0)


def test_engine_and_runtime_records_carry_ev_legs_source_contract() -> None:
    engine_source = (_ROOT / "src" / "backtest" / "engine.py").read_text(encoding="utf-8")
    record_block = engine_source.split("decision_log.append(", 1)[1].split("}", 1)[0]
    assert '"expected_gain_bps"' in record_block
    assert '"expected_loss_bps"' in record_block

    tick_source = (_ROOT / "src" / "execution" / "tick_processor.py").read_text(encoding="utf-8")
    event_block = tick_source.split("def _write_decision_event", 1)[1].split(
        "def _write_decision_observation", 1
    )[0]
    assert "expected_gain_bps" in event_block
    assert "expected_loss_bps" in event_block
