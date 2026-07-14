"""Close-the-loop producers (MASTER_PLAN §4.5.3 / §4.5.6 / §4.5.7).

The 2026-07-14 audit found the loop's CONSUMERS shipped without their
producers: the retrain trigger read an observation stream nothing wrote,
the backward-transfer floor gated slices nothing produced, and weight
drift was never logged. These tests pin the producers.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.core.config import RiskConfig
from src.execution.order_router import OrderRouter
from src.execution.tick_processor import _apply_broker_fill
from src.monitoring.health import HealthState
from src.risk.ledger import Ledger
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.training.replay_buffer import load_observations
from src.training.retrain import compute_weight_drift

T0 = datetime(2026, 7, 14, 14, 0, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# §4.5.3: the production observation writer
# ---------------------------------------------------------------------------


def _ctx(observation_path: Path) -> SimpleNamespace:
    from src.core.logging import get_logger  # noqa: F401 - parity with live ctx

    return SimpleNamespace(
        client=None,
        portfolio=PortfolioState(cash=100_000.0, equity=100_000.0),
        ledger=Ledger(),
        risk_manager=RiskManager(RiskConfig()),
        router=OrderRouter(paper_executor=None),
        health=HealthState(started_at=T0, last_tick_at=T0),
        order_journal=None,
        processed_fill_ids=set(),
        strategy_by_client_order_id={},
        clamped_by_client_order_id={},
        policy=SimpleNamespace(),
        config=SimpleNamespace(
            runtime=SimpleNamespace(timeframe="5Min"),
            execution=SimpleNamespace(
                observation_stream_enabled=True,
                observation_stream_path=str(observation_path),
            ),
        ),
    )


class _NullLogger:
    def info(self, *args, **kwargs) -> None: ...

    def warning(self, *args, **kwargs) -> None: ...


def _order(client_order_id: str, side: str, price: float, ts: datetime) -> dict:
    return {
        "client_order_id": client_order_id,
        "symbol": "AAPL",
        "filled_qty": 10.0,
        "filled_avg_price": price,
        "side": side,
        "status": "filled",
        "updated_at": ts,
    }


def test_broker_fills_produce_decision_observations(tmp_path: Path) -> None:
    stream = tmp_path / "observations.jsonl"
    ctx = _ctx(stream)
    # The closing order was risk-resized at submit time.
    ctx.clamped_by_client_order_id["cid-close"] = True
    _apply_broker_fill(ctx, order=_order("cid-open", "buy", 100.0, T0), logger=_NullLogger())
    _apply_broker_fill(
        ctx, order=_order("cid-close", "sell", 105.0, T0 + timedelta(minutes=5)), logger=_NullLogger()
    )

    observations = load_observations(stream)
    assert [obs.decision_id for obs in observations] == ["cid-open", "cid-close"]
    opening, closing = observations
    assert opening.executed_action == 1 and opening.clamped is False
    assert closing.executed_action == -1
    assert closing.realized_pnl == pytest.approx(50.0)  # (105-100) * 10, net
    assert closing.clamped is True
    # The clamp stash is consumed, not leaked.
    assert ctx.clamped_by_client_order_id == {}


def test_observation_stream_can_be_disabled(tmp_path: Path) -> None:
    stream = tmp_path / "observations.jsonl"
    ctx = _ctx(stream)
    ctx.config.execution.observation_stream_enabled = False
    _apply_broker_fill(ctx, order=_order("cid-1", "buy", 100.0, T0), logger=_NullLogger())
    assert not stream.exists()


def test_observation_stream_feeds_the_retrain_trigger(tmp_path: Path) -> None:
    """End-to-end §4.5.3: fills written by the production writer are
    readable by the exact loader the --check-retrain-trigger CLI uses."""
    stream = tmp_path / "observations.jsonl"
    ctx = _ctx(stream)
    for index in range(6):
        side = "buy" if index % 2 == 0 else "sell"
        price = 100.0 + index
        _apply_broker_fill(
            ctx,
            order=_order(f"cid-{index}", side, price, T0 + timedelta(minutes=5 * index)),
            logger=_NullLogger(),
        )
    observations = load_observations(stream)
    assert len(observations) == 6
    from src.training.retrain import check_retrain_trigger_from_observations

    fired, reason = check_retrain_trigger_from_observations(observations)
    # Below min_observations: the trigger must refuse, loudly typed.
    assert fired is False
    assert reason.startswith("insufficient_observations")


def test_paper_path_writes_observations_source_contract() -> None:
    """The paper fill sink must emit observations too (same helper)."""
    source = (REPO_ROOT / "src" / "execution" / "tick_processor.py").read_text(encoding="utf-8")
    assert source.count("_write_decision_observation(ctx, fill=") >= 2, (
        "both the broker and paper fill sinks must write the outcome stream"
    )


# ---------------------------------------------------------------------------
# §4.5.6: per-module weight drift
# ---------------------------------------------------------------------------


def _state(scale_head: float = 1.0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(3)
    return {
        "encoder.weight": torch.randn(8, 4, generator=g),
        "encoder.bias": torch.randn(8, generator=g),
        "head.weight": torch.randn(3, 8, generator=g) * scale_head,
    }


def test_weight_drift_identity_is_one_per_module() -> None:
    state = _state()
    drift = compute_weight_drift(state, {k: v.clone() for k, v in state.items()})
    assert set(drift) == {"encoder", "head"}
    for value in drift.values():
        assert value == pytest.approx(1.0, abs=1e-6)


def test_weight_drift_localizes_the_moved_module() -> None:
    anchor = _state()
    candidate = {k: v.clone() for k, v in anchor.items()}
    candidate["head.weight"] = torch.randn(3, 8, generator=torch.Generator().manual_seed(99))
    drift = compute_weight_drift(candidate, anchor)
    assert drift["encoder"] == pytest.approx(1.0, abs=1e-6)
    assert drift["head"] < 0.9, "rewritten module must show cosine well below 1"


def test_weight_drift_skips_shape_mismatches() -> None:
    anchor = _state()
    candidate = dict(_state())
    candidate["head.weight"] = torch.randn(5, 8)  # architecture change
    drift = compute_weight_drift(candidate, anchor)
    assert "head" not in drift
    assert drift["encoder"] == pytest.approx(1.0, abs=1e-6)


def test_anchored_training_logs_drift_source_contract() -> None:
    source = (REPO_ROOT / "src" / "training" / "supervised.py").read_text(encoding="utf-8")
    assert "compute_weight_drift(model.state_dict(), anchor_state_dict)" in source
    assert "anchor_weight_drift" in source


# ---------------------------------------------------------------------------
# §4.5.7: backward-transfer slice producer
# ---------------------------------------------------------------------------


def test_slice_decision_log_buckets_align_on_window() -> None:
    from src.evaluation.benchmark import slice_decision_log

    window_start = datetime(2026, 3, 1, tzinfo=UTC)
    window_end = datetime(2026, 7, 1, tzinfo=UTC)
    log = [
        {"timestamp": window_start + timedelta(days=offset), "symbol": "AAPL"}
        for offset in (0, 10, 40, 70, 100, 121)
    ]
    buckets = slice_decision_log(
        log, window_start=window_start, window_end=window_end, n_slices=4
    )
    assert len(buckets) == 4
    counts = [len(members) for _, members in buckets]
    assert sum(counts) == len(log)
    assert counts == [2, 1, 1, 2]  # days 0,10 | 40 | 70 | 100,121 (last edge inclusive)
    # Boundary names are window-derived, so a second log over the same
    # window buckets identically.
    names = [name for name, _ in buckets]
    assert names[0].startswith("s0:2026-03-01")
    assert len(set(names)) == len(names), "slice names must be unique"


def test_backward_transfer_slices_grade_per_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluation.benchmark as benchmark_module

    # 200 synthetic bars; decisions graded against realized forward moves.
    from src.data.historical_data import generate_synthetic_bars

    bars = generate_synthetic_bars("AAPL", 200)
    bars_by_symbol = {"AAPL": bars}

    def _log_for(actions: list[int]) -> list[dict[str, object]]:
        step = max(1, len(bars) // len(actions))
        return [
            {
                "symbol": "AAPL",
                "timestamp": bars[i * step].timestamp,
                "bar_index": i * step,
                "action": actions[i],
                "confidence": 0.6,
            }
            for i in range(len(actions))
        ]

    logs = iter([_log_for([1] * 60), _log_for([-1] * 60)])
    monkeypatch.setattr(
        benchmark_module, "_brain_decision_log", lambda cfg, bars: next(logs)
    )
    monkeypatch.setattr(
        benchmark_module,
        "enforce_out_of_sample",
        lambda bars, **kwargs: (bars, False),
    )
    from src.core.config import load_config

    config = load_config("configs/default.yaml")
    config.evaluation.promotion.backward_transfer_min_decisions = 5
    slice_metrics = benchmark_module.benchmark_backward_transfer_slices(
        config,
        bars_by_symbol,
        candidate_checkpoint="cand.pt",
        incumbent_checkpoint="inc.pt",
        oos_cutoff=None,
        n_slices=3,
    )
    assert len(slice_metrics) == 3
    for candidate_ba, incumbent_ba in slice_metrics.values():
        assert 0.0 <= candidate_ba <= 1.0 and 0.0 <= incumbent_ba <= 1.0


def test_backward_transfer_skips_thin_slices(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.evaluation.benchmark as benchmark_module
    from src.data.historical_data import generate_synthetic_bars

    bars = generate_synthetic_bars("AAPL", 200)
    thin_log = [
        {"symbol": "AAPL", "timestamp": bars[5].timestamp, "bar_index": 5, "action": 1, "confidence": 0.6}
    ]
    logs = iter([thin_log, list(thin_log)])
    monkeypatch.setattr(benchmark_module, "_brain_decision_log", lambda cfg, bars: next(logs))
    monkeypatch.setattr(
        benchmark_module, "enforce_out_of_sample", lambda bars, **kwargs: (bars, False)
    )
    from src.core.config import load_config

    config = load_config("configs/default.yaml")
    slice_metrics = benchmark_module.benchmark_backward_transfer_slices(
        config,
        {"AAPL": bars},
        candidate_checkpoint="cand.pt",
        incumbent_checkpoint="inc.pt",
        oos_cutoff=None,
        n_slices=3,
    )
    assert slice_metrics == {}, "one decision per window cannot evidence forgetting"


def test_promotion_gate_receives_slice_metrics_source_contract() -> None:
    source = (REPO_ROOT / "scripts" / "run_train.py").read_text(encoding="utf-8")
    assert "slice_metrics=slice_metrics" in source
    assert "benchmark_backward_transfer_slices(" in source
