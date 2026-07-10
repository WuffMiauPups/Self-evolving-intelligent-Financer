"""Evaluation honesty tests (plan §4.4): OOS enforcement, honest fills,
metric math golden fixtures, tradeable baselines, and the deterministic
scorecard row.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
import torch

from src.backtest.engine import BacktestEngine
from src.backtest.performance import compute_backtest_metrics
from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.evaluation.oos import enforce_out_of_sample, filter_out_of_sample
from src.evaluation.walk_forward import _aggregate_metrics
from src.features.feature_pipeline import FeaturePipeline
from src.risk.ledger import FillRecord
from src.risk.risk_manager import RiskManager
from src.strategies.baselines import BuyAndHoldPolicy


T0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def _config():
    config = load_config("configs/default.yaml")
    config.data.symbols = ["AAPL"]
    config.runtime.symbols = ["AAPL"]
    return config


# ── Metric math golden fixtures ─────────────────────────────────────────


def _fill(ts_minutes: int, side: str, qty: float, price: float, pnl: float) -> FillRecord:
    return FillRecord(
        timestamp=T0 + timedelta(minutes=ts_minutes),
        symbol="AAPL",
        side=side,
        quantity=qty,
        price=price,
        fee=0.0,
        realized_pnl=pnl,
        client_order_id=f"t{ts_minutes}",
    )


def test_sortino_divides_by_all_observations() -> None:
    # returns: +1%, -2%, +1%, +1% => mean 0.25%; downside dev uses N=4:
    # sqrt(0.02^2 / 4) = 0.01 -> sortino = 0.25 * sqrt(bars_per_year)... verify exactly.
    curve = [100.0, 101.0, 98.98, 99.9698, 100.9695]
    metrics = compute_backtest_metrics(
        initial_equity=100.0,
        equity_curve=curve,
        fills=[],
        exposures=[0.0] * len(curve),
        bars_per_year=252,
    )
    bar_returns = [curve[i] / curve[i - 1] - 1.0 for i in range(1, len(curve))]
    mean_ret = sum(bar_returns) / len(bar_returns)
    downside_dev = (sum(r * r for r in bar_returns if r < 0) / len(bar_returns)) ** 0.5
    expected = (mean_ret / downside_dev) * (252 ** 0.5)
    assert metrics["sortino_ratio"] == pytest.approx(expected, rel=1e-9)


def test_cvar_95_is_mean_of_worst_tail() -> None:
    # 20 bar returns: exactly one lands in the 5% tail (the worst one).
    curve = [100.0]
    returns = [0.01] * 19 + [-0.10]
    for r in returns:
        curve.append(curve[-1] * (1.0 + r))
    metrics = compute_backtest_metrics(
        initial_equity=100.0,
        equity_curve=curve,
        fills=[],
        exposures=[0.0] * len(curve),
    )
    assert metrics["cvar_95"] == pytest.approx(-0.10, rel=1e-9)


def test_profit_factor_all_winners_is_not_worst() -> None:
    fills = [_fill(1, "sell", 1.0, 100.0, 50.0), _fill(2, "sell", 1.0, 100.0, 25.0)]
    metrics = compute_backtest_metrics(
        initial_equity=100.0,
        equity_curve=[100.0, 101.0],
        fills=fills,
        exposures=[0.0, 0.0],
    )
    assert metrics["profit_factor"] == pytest.approx(999.0)


def test_round_trip_closes_on_sign_flip() -> None:
    # long 10 -> sell 20 (flip to short 10): the long trip must close.
    fills = [
        _fill(1, "buy", 10.0, 100.0, 0.0),
        _fill(2, "sell", 20.0, 105.0, 50.0),
    ]
    metrics = compute_backtest_metrics(
        initial_equity=10_000.0,
        equity_curve=[10_000.0, 10_050.0],
        fills=fills,
        exposures=[0.1, 0.1],
    )
    assert metrics["round_trip_count"] == pytest.approx(1.0)
    assert metrics["round_trip_win_rate"] == pytest.approx(1.0)


def test_degenerate_folds_excluded_from_aggregation() -> None:
    good = SimpleNamespace(test_metrics={"total_return": 0.10, "fill_count": 5.0})
    dead = SimpleNamespace(test_metrics={"total_return": 0.0, "fill_count": 0.0})
    aggregate = _aggregate_metrics([good, dead, dead])
    assert aggregate["total_return"]["mean"] == pytest.approx(0.10)
    assert aggregate["degenerate_folds"]["count"] == pytest.approx(2.0)


# ── OOS enforcement ─────────────────────────────────────────────────────


def _bars(n: int = 40):
    return {"AAPL": generate_synthetic_bars("AAPL", n)}


def test_filter_out_of_sample_drops_covered_bars() -> None:
    bars = _bars(40)
    cutoff = bars["AAPL"][19].timestamp
    filtered, dropped = filter_out_of_sample(bars, cutoff)
    assert dropped == 20
    assert all(bar.timestamp > cutoff for bar in filtered["AAPL"])


def test_enforce_oos_hard_fails_when_fully_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    bars = _bars(40)
    beyond_all = bars["AAPL"][-1].timestamp
    monkeypatch.setattr(
        "src.evaluation.oos.checkpoint_training_end", lambda _path: beyond_all
    )
    with pytest.raises(RuntimeError, match="No out-of-sample bars"):
        enforce_out_of_sample(bars, checkpoint_path="stub.pt", allow_in_sample=False)
    # Override tags instead of failing.
    tagged_bars, tagged = enforce_out_of_sample(
        bars, checkpoint_path="stub.pt", allow_in_sample=True
    )
    assert tagged is True and tagged_bars is bars


def test_enforce_oos_without_checkpoint_is_passthrough() -> None:
    bars = _bars(10)
    out, tagged = enforce_out_of_sample(bars, checkpoint_path=None)
    assert out is bars and tagged is False


# ── Honest fills: next-bar-open timing ─────────────────────────────────


def test_next_open_fills_use_next_bars_open() -> None:
    config = _config()
    config.backtest.fill_timing = "next_open"
    config.backtest.slippage_bps = 0.0
    config.backtest.fee_bps = 0.0
    bars = generate_synthetic_bars("AAPL", 240)
    open_by_ts = {bar.timestamp: float(bar.open) for bar in bars}
    close_by_ts = {bar.timestamp: float(bar.close) for bar in bars}
    engine = BacktestEngine(
        config,
        FeaturePipeline(config.features, device=torch.device("cpu")),
        RiskManager(config.risk.model_copy()),
    )
    report = engine.run({"AAPL": bars}, BuyAndHoldPolicy())
    assert report.fills, "buy-and-hold must trade through the EV gate"
    for fill in report.fills:
        assert fill.price == pytest.approx(open_by_ts[fill.timestamp]), (
            "next_open fill must execute at the fill bar's OPEN"
        )
        assert fill.price != pytest.approx(close_by_ts[fill.timestamp]) or (
            open_by_ts[fill.timestamp] == close_by_ts[fill.timestamp]
        )


def test_same_close_mode_preserved_for_comparison() -> None:
    config = _config()
    config.backtest.fill_timing = "same_close"
    config.backtest.slippage_bps = 0.0
    config.backtest.fee_bps = 0.0
    bars = generate_synthetic_bars("AAPL", 240)
    close_by_ts = {bar.timestamp: float(bar.close) for bar in bars}
    engine = BacktestEngine(
        config,
        FeaturePipeline(config.features, device=torch.device("cpu")),
        RiskManager(config.risk.model_copy()),
    )
    report = engine.run({"AAPL": bars}, BuyAndHoldPolicy())
    assert report.fills
    for fill in report.fills:
        assert fill.price == pytest.approx(close_by_ts[fill.timestamp])


def test_decision_log_is_emitted() -> None:
    config = _config()
    bars = generate_synthetic_bars("AAPL", 240)
    engine = BacktestEngine(
        config,
        FeaturePipeline(config.features, device=torch.device("cpu")),
        RiskManager(config.risk.model_copy()),
    )
    report = engine.run({"AAPL": bars}, BuyAndHoldPolicy())
    assert report.decision_log
    first = report.decision_log[0]
    assert set(first) == {"symbol", "timestamp", "bar_index", "action", "confidence"}
    assert first["action"] == 1


# ── Scorecard: deterministic end-to-end row ────────────────────────────


def test_benchmark_appends_deterministic_row(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.evaluation.benchmark import append_scorecard_row, run_benchmark
    from src.models.factory import build_checkpoint_meta, build_model_from_contract
    from src.training.checkpoints import save_checkpoint

    config = _config()
    bars = generate_synthetic_bars("AAPL", 400)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    snapshot = pipeline.build(bars[: config.runtime.warmup_bars + 1], symbol="AAPL")
    model = build_model_from_contract(
        model_kind="brain", model_config=config.model, feature_contract=snapshot.contract
    )
    cutoff = bars[199].timestamp
    meta = build_checkpoint_meta(
        model_kind="brain",
        model_config=config.model,
        feature_config=config.features,
        feature_contract=snapshot.contract,
        extra={
            "training_split_ranges": {
                "train": {"AAPL": {"end_timestamp": cutoff.isoformat()}},
                "val": {},
            },
        },
    )
    checkpoint_path = tmp_path / "brain_test.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        meta=meta,
    )
    config.model.checkpoint_path = str(checkpoint_path)

    row1 = run_benchmark(config, {"AAPL": list(bars)})
    row2 = run_benchmark(config, {"AAPL": list(bars)})
    assert row1 == row2, "identical inputs must produce an identical scorecard row"
    assert row1.tag == "oos"
    assert row1.data_start > cutoff.date().isoformat() or row1.data_start >= cutoff.date().isoformat()

    scorecard = tmp_path / "SCORECARD.md"
    append_scorecard_row(row1, path=scorecard)
    append_scorecard_row(row2, path=scorecard)
    lines = scorecard.read_text(encoding="utf-8").strip().splitlines()
    assert lines[-1] == lines[-2]

    # A checkpoint whose training covers everything must refuse.
    meta_all = dict(meta)
    meta_all["training_split_ranges"] = {
        "train": {"AAPL": {"end_timestamp": bars[-1].timestamp.isoformat()}},
        "val": {},
    }
    covered_path = tmp_path / "covered.pt"
    save_checkpoint(
        covered_path,
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        meta=meta_all,
    )
    config.model.checkpoint_path = str(covered_path)
    with pytest.raises(RuntimeError, match="No out-of-sample bars"):
        run_benchmark(config, {"AAPL": list(bars)})
