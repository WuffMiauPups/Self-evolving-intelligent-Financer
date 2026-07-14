"""Cross-sectional market-context features (RESEARCH_ROADMAP Phase 3).

Pins the full contract of ``src/features/market_context.py`` and its
pipeline wiring:

* hand-computed feature values on tiny synthetic bars (SPY/QQQ returns
  and realized vol, universe breadth, symbol-vs-SPY relative strength);
* graceful degradation — missing SPY/QQQ or a missing context object
  emits 0.0 (never NaN, never a crash) with a ONE-TIME warning;
* ``features.market_context_enabled`` flips the feature contract, so a
  checkpoint trained with the other setting refuses cleanly with the
  existing contract/config-mismatch errors — never a shape crash — while
  a legacy checkpoint saved BEFORE the flag existed still loads when the
  flag is off;
* disabled flag == byte-identical pipeline output (no market_* names,
  supplied contexts ignored bit-for-bit, shared features unperturbed);
* the on-disk feature cache re-keys when the flag flips or the market
  context data changes (FEATURE_SCHEMA_VERSION salt + config snapshot +
  context signature).

Scalar-vs-batch parity for the market features lives in
tests/test_feature_parity.py (test_scalar_vs_batch_parity_with_market_context).
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import torch

from src.core.config import load_config
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar
from src.features import market_context
from src.features.feature_pipeline import FeaturePipeline
from src.features.market_context import (
    MARKET_FEATURE_NAMES,
    compute_market_context_batch,
    compute_market_context_features,
    compute_market_context_snapshot,
)


@pytest.fixture(autouse=True)
def _no_feature_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SETAI_DISABLE_FEATURE_CACHE", "1")


@pytest.fixture(autouse=True)
def _fresh_warning_registry():
    market_context.reset_market_context_warnings()
    yield
    market_context.reset_market_context_warnings()


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, **kwargs: object) -> None:
        self.messages.append(message)


def _bars_from_closes(closes: list[float], *, symbol: str) -> list[MarketBar]:
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    return [
        MarketBar(
            symbol=symbol,
            timestamp=start + timedelta(minutes=index),
            open=close,
            high=close * 1.001,
            low=close * 0.999,
            close=close,
            volume=1_000.0,
            spread=0.01,
        )
        for index, close in enumerate(closes)
    ]


def _make_bars(count: int, *, seed: int, symbol: str) -> list[MarketBar]:
    generator = torch.Generator().manual_seed(seed)
    steps = torch.randn(count, generator=generator) * 0.01
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    bars: list[MarketBar] = []
    price = 100.0
    for index in range(count):
        close = max(price * (1.0 + float(steps[index])), 1.0)
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=start + timedelta(minutes=index),
                open=price,
                high=max(price, close) * 1.002,
                low=min(price, close) * 0.998,
                close=close,
                volume=10_000.0,
                spread=0.01,
            )
        )
        price = close
    return bars


def _universe(n_bars: int = 130) -> dict[str, list[MarketBar]]:
    return {
        "AAPL": _make_bars(n_bars, seed=7, symbol="AAPL"),
        "SPY": _make_bars(n_bars, seed=11, symbol="SPY"),
        "QQQ": _make_bars(n_bars, seed=13, symbol="QQQ"),
        "MSFT": _make_bars(n_bars, seed=17, symbol="MSFT"),
    }


# ---------------------------------------------------------------------------
# 1. Hand-computed feature values on tiny synthetic bars
# ---------------------------------------------------------------------------


def test_snapshot_values_hand_computed() -> None:
    """SPY: 100→110→121 (two +10% steps), QQQ: 100→90→99, AAA: 100→100→130,
    BBB: 100→120→90.  short=1, medium=2, epsilon=0."""
    bars_by_symbol = {
        "SPY": _bars_from_closes([100.0, 110.0, 121.0], symbol="SPY"),
        "QQQ": _bars_from_closes([100.0, 90.0, 99.0], symbol="QQQ"),
        "AAA": _bars_from_closes([100.0, 100.0, 130.0], symbol="AAA"),
        "BBB": _bars_from_closes([100.0, 120.0, 90.0], symbol="BBB"),
    }
    snapshot = compute_market_context_snapshot(
        bars_by_symbol, short_window=1, medium_window=2, epsilon=0.0
    )
    values = snapshot.values
    assert values["market_spy_return_short"] == pytest.approx(121.0 / 110.0 - 1.0, rel=1e-6)
    assert values["market_spy_return_medium"] == pytest.approx(0.21, rel=1e-6)
    # SPY made two identical +10% log steps → realized vol is exactly 0.
    assert values["market_spy_vol_medium"] == pytest.approx(0.0, abs=1e-7)
    assert values["market_qqq_return_short"] == pytest.approx(99.0 / 90.0 - 1.0, rel=1e-6)
    assert values["market_qqq_return_medium"] == pytest.approx(-0.01, rel=1e-4)
    expected_qqq_vol = (math.log(1.1) - math.log(0.9)) / 2.0
    assert values["market_qqq_vol_medium"] == pytest.approx(expected_qqq_vol, rel=1e-5)
    # Medium-window returns: SPY +21%, AAA +30% positive; QQQ -1%, BBB -10%.
    assert values["market_breadth_medium"] == pytest.approx(0.5)
    assert snapshot.spy_available is True
    assert snapshot.spy_return_medium == pytest.approx(0.21, rel=1e-6)


def test_relative_strength_hand_computed() -> None:
    """AAA medium return (+30%) minus SPY medium return (+21%) = +9%."""
    bars_by_symbol = {
        "SPY": _bars_from_closes([100.0, 110.0, 121.0], symbol="SPY"),
        "AAA": _bars_from_closes([100.0, 100.0, 130.0], symbol="AAA"),
    }
    snapshot = compute_market_context_snapshot(
        bars_by_symbol, short_window=1, medium_window=2, epsilon=0.0
    )
    aaa_tensor = torch.tensor(
        [[c, c, c, c, 1_000.0, 0.01] for c in (100.0, 100.0, 130.0)],
        dtype=torch.float32,
    )
    features = compute_market_context_features(
        aaa_tensor, context=snapshot, medium_window=2, epsilon=0.0
    )
    assert float(features["market_rel_strength_medium"]) == pytest.approx(
        0.30 - 0.21, rel=1e-5
    )
    # Insufficient symbol history (numel <= medium) → 0.0, not -spy_return.
    short_tensor = aaa_tensor[:2]
    degraded = compute_market_context_features(
        short_tensor, context=snapshot, medium_window=2, epsilon=0.0
    )
    assert float(degraded["market_rel_strength_medium"]) == 0.0


def test_batch_columns_hand_computed() -> None:
    bars_by_symbol = {
        "SPY": _bars_from_closes([100.0, 110.0, 121.0], symbol="SPY"),
        "QQQ": _bars_from_closes([100.0, 90.0, 99.0], symbol="QQQ"),
        "AAA": _bars_from_closes([100.0, 100.0, 130.0], symbol="AAA"),
        "BBB": _bars_from_closes([100.0, 120.0, 90.0], symbol="BBB"),
    }
    batch = compute_market_context_batch(
        bars_by_symbol, short_window=1, medium_window=2, epsilon=0.0
    )
    assert len(batch.timeline) == 3
    spy_medium = batch.columns["market_spy_return_medium"]
    # Warmup rows (medium window needs 3 bars) are zero; final row is +21%.
    assert float(spy_medium[0]) == 0.0
    assert float(spy_medium[1]) == 0.0
    assert float(spy_medium[2]) == pytest.approx(0.21, rel=1e-6)
    breadth = batch.columns["market_breadth_medium"]
    assert float(breadth[0]) == 0.0
    assert float(breadth[1]) == 0.0
    assert float(breadth[2]) == pytest.approx(0.5)
    assert batch.spy_available is True
    assert bool(batch.spy_row_mask.all())


# ---------------------------------------------------------------------------
# 2. Graceful degradation: missing SPY/QQQ, missing context — 0.0 + one warning
# ---------------------------------------------------------------------------


def test_missing_spy_qqq_emit_zeros_and_warn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(market_context, "_LOGGER", recorder)
    bars_by_symbol = {"AAA": _bars_from_closes([100.0, 105.0, 110.0], symbol="AAA")}
    for _ in range(3):  # repeated ticks must not re-warn
        snapshot = compute_market_context_snapshot(
            bars_by_symbol, short_window=1, medium_window=2, epsilon=0.0
        )
    assert snapshot.spy_available is False
    for name in (
        "market_spy_return_short",
        "market_spy_return_medium",
        "market_spy_vol_medium",
        "market_qqq_return_short",
        "market_qqq_return_medium",
        "market_qqq_vol_medium",
    ):
        assert snapshot.values[name] == 0.0
    # AAA itself is up → breadth still computes over the available universe.
    assert snapshot.values["market_breadth_medium"] == pytest.approx(1.0)
    spy_warnings = [m for m in recorder.messages if "spy" in m.lower()]
    qqq_warnings = [m for m in recorder.messages if "qqq" in m.lower()]
    assert len(spy_warnings) == 1
    assert len(qqq_warnings) == 1


def test_enabled_pipeline_without_context_emits_finite_zeros_and_warns_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RecordingLogger()
    monkeypatch.setattr(market_context, "_LOGGER", recorder)
    config = load_config("configs/default.yaml")
    features = config.features.model_copy(update={"market_context_enabled": True})
    pipeline = FeaturePipeline(features, device=torch.device("cpu"))
    bars = _make_bars(80, seed=7, symbol="AAPL")

    # Scalar path: single-symbol build with no context (the walk-forward /
    # single-symbol-test shape).  Never crashes, never NaN.
    for _ in range(2):
        snapshot = pipeline.build(bars, symbol="AAPL")
    market_rows = {
        name: float(snapshot.features[row])
        for row, name in enumerate(snapshot.feature_names)
        if name.startswith("market_")
    }
    assert set(market_rows) == set(MARKET_FEATURE_NAMES)
    assert all(value == 0.0 for value in market_rows.values())
    assert torch.isfinite(snapshot.features).all()

    # Batch path: same degradation contract.
    batched = pipeline.build_batch(bars, symbol="AAPL", temporal_window=48)
    for name in MARKET_FEATURE_NAMES:
        column = batched.features[:, batched.feature_names.index(name)]
        assert torch.equal(column, torch.zeros_like(column))

    context_warnings = [m for m in recorder.messages if "market_context_unavailable" in m]
    # One warning per path (scalar + batch), each fired exactly once.
    assert len(context_warnings) == 2


def test_batch_context_without_spy_zeroes_spy_features() -> None:
    bars_by_symbol = {
        "AAA": _bars_from_closes([100.0, 105.0, 110.0, 120.0], symbol="AAA"),
        "QQQ": _bars_from_closes([100.0, 101.0, 102.0, 103.0], symbol="QQQ"),
    }
    batch = compute_market_context_batch(
        bars_by_symbol, short_window=1, medium_window=2, epsilon=0.0
    )
    assert batch.spy_available is False
    for name in ("market_spy_return_short", "market_spy_return_medium", "market_spy_vol_medium"):
        assert torch.equal(batch.columns[name], torch.zeros_like(batch.columns[name]))
    # Relative strength degrades to 0.0 without the SPY anchor.
    aaa_tensor = torch.tensor(
        [[c, c, c, c, 1_000.0, 0.01] for c in (100.0, 105.0, 110.0, 120.0)],
        dtype=torch.float32,
    )
    features = market_context.compute_market_context_features_rolling(
        aaa_tensor,
        [bar.timestamp for bar in bars_by_symbol["AAA"]],
        context=batch,
        medium_window=2,
        epsilon=0.0,
    )
    rel = features["market_rel_strength_medium"]
    assert torch.equal(rel, torch.zeros_like(rel))


def test_timestamp_alignment_late_start_symbol() -> None:
    """A symbol whose history starts later must gather SPY's rows by
    timestamp, not by array index."""
    spy_bars = _bars_from_closes([100.0, 110.0, 121.0, 133.1, 146.41], symbol="SPY")
    late_bars = [
        MarketBar(
            symbol="AAA",
            timestamp=spy_bars[3 + i].timestamp,  # starts at SPY's 4th bar
            open=100.0,
            high=100.1,
            low=99.9,
            close=100.0,
            volume=1_000.0,
            spread=0.01,
        )
        for i in range(2)
    ]
    batch = compute_market_context_batch(
        {"SPY": spy_bars, "AAA": late_bars}, short_window=1, medium_window=2, epsilon=0.0
    )
    aaa_tensor = torch.tensor(
        [[100.0, 100.1, 99.9, 100.0, 1_000.0, 0.01]] * 2, dtype=torch.float32
    )
    features = market_context.compute_market_context_features_rolling(
        aaa_tensor,
        [bar.timestamp for bar in late_bars],
        context=batch,
        medium_window=2,
        epsilon=0.0,
    )
    # AAA's first bar is SPY's 4th: SPY medium return there is 133.1/110 - 1.
    assert float(features["market_spy_return_medium"][0]) == pytest.approx(
        133.1 / 110.0 - 1.0, rel=1e-6
    )
    assert float(features["market_spy_return_medium"][1]) == pytest.approx(
        146.41 / 121.0 - 1.0, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 3. Contract bump + old-checkpoint clean refusal
# ---------------------------------------------------------------------------


def test_enabled_flag_changes_contract_and_regime_wiring() -> None:
    config = load_config("configs/default.yaml")
    assert config.features.market_context_enabled is False
    features_on = config.features.model_copy(update={"market_context_enabled": True})
    bars = _make_bars(80, seed=7, symbol="AAPL")
    contract_off = FeaturePipeline(config.features, device=torch.device("cpu")).build_contract(
        bars, symbol="AAPL"
    )
    snapshot_on = FeaturePipeline(features_on, device=torch.device("cpu")).build(
        bars, symbol="AAPL"
    )
    contract_on = snapshot_on.contract

    assert contract_on.feature_dim == contract_off.feature_dim + len(MARKET_FEATURE_NAMES)
    # Regime detection sees the market: market_* joins the regime input.
    assert contract_on.regime_dim == contract_off.regime_dim + len(MARKET_FEATURE_NAMES)
    assert set(MARKET_FEATURE_NAMES) <= set(contract_on.regime_feature_names)
    # And the regime model branch receives them via the implicit prefix map.
    assert set(MARKET_FEATURE_NAMES) <= set(contract_on.branch_feature_names["regime"])
    assert contract_on.branch_dims["regime"] == (
        contract_off.branch_dims["regime"] + len(MARKET_FEATURE_NAMES)
    )

    # The existing contract-mismatch error — not a shape crash.
    with pytest.raises(ConfigurationError, match="mismatch"):
        contract_off.assert_matches(contract_on, context="checkpoint_contract[brain.pt]")


def test_old_checkpoint_refuses_cleanly_when_flag_enabled(tmp_path: Path) -> None:
    """A checkpoint trained with the flag OFF must refuse to load into a
    flag-ON runtime with the existing ConfigurationError — never a shape
    crash inside the model."""
    from src.models.factory import (
        build_checkpoint_meta,
        build_model_from_contract,
        resolve_inference_model,
    )
    from src.training.checkpoints import save_checkpoint

    config = load_config("configs/default.yaml")
    features_on = config.features.model_copy(update={"market_context_enabled": True})
    bars = _make_bars(80, seed=7, symbol="AAPL")
    contract_off = FeaturePipeline(config.features, device=torch.device("cpu")).build_contract(
        bars, symbol="AAPL"
    )
    contract_on = FeaturePipeline(features_on, device=torch.device("cpu")).build_contract(
        bars, symbol="AAPL"
    )
    model = build_model_from_contract(
        model_kind="brain", model_config=config.model, feature_contract=contract_off
    )
    checkpoint_path = tmp_path / "brain_off.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        meta=build_checkpoint_meta(
            model_kind="brain",
            model_config=config.model,
            feature_config=config.features,
            feature_contract=contract_off,
        ),
    )
    with pytest.raises(ConfigurationError, match="mismatch"):
        resolve_inference_model(
            model_config=config.model.model_copy(
                update={"checkpoint_path": str(checkpoint_path)}
            ),
            feature_config=features_on,
            feature_contract=contract_on,
            device=torch.device("cpu"),
        )


def test_legacy_checkpoint_without_flag_key_loads_when_disabled(tmp_path: Path) -> None:
    """Checkpoints saved BEFORE market_context_enabled existed carry a
    feature_config dict without the key.  With the flag off (default)
    they must keep loading — the factory normalizes the stored dict
    through FeatureConfig so new defaulted fields never orphan shipped
    checkpoints (brain.pt)."""
    from src.models.factory import (
        build_checkpoint_meta,
        build_model_from_contract,
        resolve_inference_model,
    )
    from src.training.checkpoints import save_checkpoint

    config = load_config("configs/default.yaml")
    bars = _make_bars(80, seed=7, symbol="AAPL")
    contract = FeaturePipeline(config.features, device=torch.device("cpu")).build_contract(
        bars, symbol="AAPL"
    )
    model = build_model_from_contract(
        model_kind="brain", model_config=config.model, feature_contract=contract
    )
    meta = build_checkpoint_meta(
        model_kind="brain",
        model_config=config.model,
        feature_config=config.features,
        feature_contract=contract,
    )
    # Simulate the pre-Phase-3 artifact: the stored dict has no flag key.
    assert meta["feature_config"].pop("market_context_enabled") is False
    checkpoint_path = tmp_path / "brain_legacy.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        meta=meta,
    )
    artifacts = resolve_inference_model(
        model_config=config.model.model_copy(update={"checkpoint_path": str(checkpoint_path)}),
        feature_config=config.features,
        feature_contract=contract,
        device=torch.device("cpu"),
    )
    assert artifacts.model_kind == "brain"
    artifacts.feature_contract.assert_matches(contract, context="legacy_load")


# ---------------------------------------------------------------------------
# 4. Disabled flag == byte-identical output
# ---------------------------------------------------------------------------


def test_disabled_flag_output_is_byte_identical() -> None:
    config = load_config("configs/default.yaml")
    assert config.features.market_context_enabled is False
    features_on = config.features.model_copy(update={"market_context_enabled": True})
    bars_by_symbol = _universe(100)
    bars = bars_by_symbol["AAPL"]
    pipeline_off = FeaturePipeline(config.features, device=torch.device("cpu"))
    pipeline_on = FeaturePipeline(features_on, device=torch.device("cpu"))

    snapshot_off = pipeline_off.build(bars, symbol="AAPL")
    assert not any(name.startswith("market_") for name in snapshot_off.feature_names)

    # Supplying a context while disabled is ignored bit-for-bit.
    scalar_context = compute_market_context_snapshot(
        bars_by_symbol, short_window=5, medium_window=20, epsilon=config.features.epsilon
    )
    snapshot_off_ctx = pipeline_off.build(bars, symbol="AAPL", market_context=scalar_context)
    assert snapshot_off.feature_names == snapshot_off_ctx.feature_names
    assert torch.equal(snapshot_off.features, snapshot_off_ctx.features)
    assert torch.equal(snapshot_off.regime_input, snapshot_off_ctx.regime_input)
    assert torch.equal(snapshot_off.temporal_tensor, snapshot_off_ctx.temporal_tensor)
    for branch_name, branch_tensor in snapshot_off.branch_inputs.items():
        assert torch.equal(branch_tensor, snapshot_off_ctx.branch_inputs[branch_name])

    # Enabling ONLY adds market_* columns: every shared feature value is
    # bit-identical, so the disabled path is provably untouched by the
    # new family's code.
    snapshot_on = pipeline_on.build(bars, symbol="AAPL", market_context=scalar_context)
    shared_names = [
        name for name in snapshot_on.feature_names if not name.startswith("market_")
    ]
    assert shared_names == snapshot_off.feature_names
    for name in shared_names:
        off_value = snapshot_off.features[snapshot_off.feature_names.index(name)]
        on_value = snapshot_on.features[snapshot_on.feature_names.index(name)]
        assert torch.equal(off_value, on_value), name

    # Batch path: disabled + context supplied is ignored bit-for-bit too.
    batch_context = compute_market_context_batch(
        bars_by_symbol, short_window=5, medium_window=20, epsilon=config.features.epsilon
    )
    batched_off = pipeline_off.build_batch(bars, symbol="AAPL", temporal_window=48)
    batched_off_ctx = pipeline_off.build_batch(
        bars, symbol="AAPL", temporal_window=48, market_context=batch_context
    )
    assert batched_off.feature_names == batched_off_ctx.feature_names
    assert not any(name.startswith("market_") for name in batched_off.feature_names)
    assert torch.equal(batched_off.features, batched_off_ctx.features)
    assert torch.equal(batched_off.regime_input, batched_off_ctx.regime_input)


# ---------------------------------------------------------------------------
# 5. Cache invalidation: fingerprints change when the flag flips
# ---------------------------------------------------------------------------


def test_flag_flip_changes_feature_cache_fingerprint() -> None:
    from src.features.feature_cache import FEATURE_SCHEMA_VERSION, compute_fingerprint

    # Phase 3 bumped the schema salt (2 → 3): every pre-bump cache entry
    # misses even before the per-config keys are compared.
    assert FEATURE_SCHEMA_VERSION >= 3

    config = load_config("configs/default.yaml")
    features_on = config.features.model_copy(update={"market_context_enabled": True})
    pipeline_off = FeaturePipeline(config.features, device=torch.device("cpu"))
    pipeline_on = FeaturePipeline(features_on, device=torch.device("cpu"))
    snapshot_off = pipeline_off._feature_config_snapshot()
    snapshot_on = pipeline_on._feature_config_snapshot()
    assert snapshot_off["market_context_enabled"] is False
    assert snapshot_on["market_context_enabled"] is True

    bars_signature = {"count": 10, "close_sum": 1000.0}
    fingerprint_off = compute_fingerprint(
        bars_signature=bars_signature,
        symbol="AAPL",
        feature_config_snapshot=snapshot_off,
        temporal_window=48,
    )
    fingerprint_on = compute_fingerprint(
        bars_signature=bars_signature,
        symbol="AAPL",
        feature_config_snapshot=snapshot_on,
        temporal_window=48,
    )
    assert fingerprint_off != fingerprint_on

    # The market-context DATA also participates: same bars + config but a
    # different context (e.g. re-fetched SPY history) re-keys the entry.
    universe = _universe(80)
    context_a = compute_market_context_batch(
        universe, short_window=5, medium_window=20, epsilon=config.features.epsilon
    )
    context_b = compute_market_context_batch(
        {**universe, "SPY": _make_bars(80, seed=99, symbol="SPY")},
        short_window=5,
        medium_window=20,
        epsilon=config.features.epsilon,
    )
    fingerprint_ctx_a = compute_fingerprint(
        bars_signature=bars_signature,
        symbol="AAPL",
        feature_config_snapshot=snapshot_on,
        temporal_window=48,
        market_context_signature=context_a.signature(),
    )
    fingerprint_ctx_b = compute_fingerprint(
        bars_signature=bars_signature,
        symbol="AAPL",
        feature_config_snapshot=snapshot_on,
        temporal_window=48,
        market_context_signature=context_b.signature(),
    )
    assert fingerprint_ctx_a != fingerprint_ctx_b
    assert fingerprint_ctx_a != fingerprint_on


def test_warm_cache_entry_not_served_across_flag_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a warm flag-off cache entry must MISS when the flag is
    on (and vice versa) — the batch output shapes differ, so serving a
    stale entry would be a silent contract violation."""
    from src.features import feature_cache

    monkeypatch.delenv("SETAI_DISABLE_FEATURE_CACHE", raising=False)
    monkeypatch.setattr(feature_cache, "DEFAULT_CACHE_DIR", tmp_path)

    config = load_config("configs/default.yaml")
    features_on = config.features.model_copy(update={"market_context_enabled": True})
    bars_by_symbol = _universe(90)
    bars = bars_by_symbol["AAPL"]

    pipeline_off = FeaturePipeline(config.features, device=torch.device("cpu"))
    batched_off = pipeline_off.build_batch(bars, symbol="AAPL", temporal_window=48)
    assert not any(name.startswith("market_") for name in batched_off.feature_names)
    assert len(list(tmp_path.glob("features_*.pt"))) == 1  # warm entry written

    context = compute_market_context_batch(
        bars_by_symbol, short_window=5, medium_window=20, epsilon=config.features.epsilon
    )
    pipeline_on = FeaturePipeline(features_on, device=torch.device("cpu"))
    batched_on = pipeline_on.build_batch(
        bars, symbol="AAPL", temporal_window=48, market_context=context
    )
    # A stale hit would have returned the off-entry without market_* names.
    assert any(name.startswith("market_") for name in batched_on.feature_names)
    assert len(list(tmp_path.glob("features_*.pt"))) == 2  # distinct entries
