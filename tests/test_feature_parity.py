"""Scalar-vs-batch feature pipeline equivalence (MASTER_PLAN §4.3.1) and
live-path input sanitation (§4.3.3).

The load-bearing train/live parity invariant: for every bar index the
vectorized training/backtest path (``build_batch`` → ``snapshot_at(i)``)
must produce the same features the live path (``build`` /
``build_from_tensor``) produces on the ``temporal_window``-sized bar
slice ending at ``i``.  This had no test before §4.3.

Regression provenance — on pre-fix code this file FAILED, proving it
bites.  The small-window config below caught:

* ``rsi_14`` — the batch path zeroed ``[: _RSI_PERIOD + 1]`` (bars 0..14)
  while the scalar guard ``n > _RSI_PERIOD`` makes bar 14 the first
  valid bar → zero-vs-value mismatch at exactly ``i == 14``.  Any config
  with ``windows.long <= 14`` exposes it; the default (long=60) hid it.
* ``momentum_{W}`` — the batch path's first valid bar was ``W - 1``
  (W bars available) but the scalar path computes momentum under the
  same ``numel > W`` guard as ``return_{W}`` (W+1 bars) → mismatch at
  exactly ``i == W - 1``.  Needs a momentum window with
  ``W - 1 >= windows.long``; ``W=12`` with ``long=10`` below does it.
* ``return_{W}`` / ``log_return_{W}`` epsilon skew — the batch helpers
  guard the anchor denominator with ``+ epsilon`` while the scalar path
  divided raw: an epsilon-sized (~5e-6 at price ~100) systematic skew on
  every bar, and the raw scalar division is the §4.3.3 zero-close → inf
  hole.  Both paths now share the guarded denominator.

The DEFAULT config also failed pre-fix, on ``volatility_proxy_t``:
``snapshot_at`` reset ``close_return_t`` at the first column of each
per-bar temporal window but not ``volatility_proxy_t`` (= its abs), so
every full window (``i >= temporal_window``) leaked the prior bar's
return into column 0.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from src.core.config import load_config
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline, FeatureSnapshot

# Tolerance for scalar-vs-batch equivalence.  The two paths share the
# same formulas but reduce in different orders (``torch.unfold`` windows
# vs contiguous slices), so exact bit-equality is not achievable.  With
# realistically-conditioned bars (~1% moves) the observed fp32 drift is
# <= 2.6e-5 abs / 2.5e-5 rel, worst on ``regime_price_z_*`` (a z-score
# of price, mean ~100 vs std ~1, the most cancellation-sensitive
# family).  Near-constant prices (variance/mean^2 at fp32 epsilon) can
# exceed even this, which is reduction noise, not semantics.  Every
# defect this file pins produced zero-vs-value mismatches of 1e-2..1,
# orders of magnitude above this tolerance.
PARITY_RTOL = 1.0e-4
PARITY_ATOL = 5.0e-5


def _make_bars(count: int, *, seed: int = 7, symbol: str = "AAPL") -> list[MarketBar]:
    """Deterministic synthetic bars with ~1% moves.

    ``generate_synthetic_bars`` produces ~0.01% moves, which puts price
    variance near fp32 epsilon and drowns the parity comparison in
    cancellation noise on z-score features.  These bars keep the
    comparison about semantics (indexing / zero-fill / formulas).
    """
    generator = torch.Generator().manual_seed(seed)
    steps = torch.randn(count, generator=generator) * 0.01
    vol_noise = torch.rand(count, generator=generator)
    start = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
    bars: list[MarketBar] = []
    price = 100.0
    for index in range(count):
        close = max(price * (1.0 + float(steps[index])), 1.0)
        open_price = price
        high = max(open_price, close) * 1.002
        low = min(open_price, close) * 0.998
        bars.append(
            MarketBar(
                symbol=symbol,
                timestamp=start + timedelta(minutes=index),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=10_000.0 * (0.8 + 0.4 * float(vol_noise[index])),
                spread=0.01 + (index % 5) * 0.001,
            )
        )
        price = close
    return bars


@pytest.fixture(autouse=True)
def _no_feature_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SETAI_DISABLE_FEATURE_CACHE", "1")


def _assert_snapshots_match(
    scalar: FeatureSnapshot, batch: FeatureSnapshot, index: int
) -> None:
    assert scalar.feature_names == batch.feature_names
    for row, name in enumerate(scalar.feature_names):
        torch.testing.assert_close(
            batch.features[row],
            scalar.features[row],
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            msg=lambda m, name=name: f"feature {name!r} at bar {index}: {m}",
        )
    assert scalar.branch_feature_names == batch.branch_feature_names
    for branch_name in scalar.branch_inputs:
        torch.testing.assert_close(
            batch.branch_inputs[branch_name],
            scalar.branch_inputs[branch_name],
            rtol=PARITY_RTOL,
            atol=PARITY_ATOL,
            msg=lambda m, b=branch_name: f"branch {b!r} at bar {index}: {m}",
        )
    torch.testing.assert_close(
        batch.regime_input,
        scalar.regime_input,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        msg=lambda m: f"regime_input at bar {index}: {m}",
    )
    torch.testing.assert_close(
        batch.temporal_tensor,
        scalar.temporal_tensor,
        rtol=PARITY_RTOL,
        atol=PARITY_ATOL,
        msg=lambda m: f"temporal_tensor at bar {index}: {m}",
    )


def _run_parity(feature_config, *, temporal_window: int, n_bars: int) -> None:
    """Compare per-index scalar output against the batched container.

    Comparison starts at ``windows['long']`` — deliberately BELOW the
    corrected ``first_valid_index`` so the historical off-by-one sites
    (RSI at bar 14, momentum_{W} at bar W-1) stay pinned.  Indices below
    ``windows['long']`` are excluded because the regime family's
    clipped-prefix cumulative std (cumsum-based) accumulates fp32 noise
    beyond the parity tolerance there; the contract only guarantees
    equivalence from ``first_valid_index`` anyway.
    """
    pipeline = FeaturePipeline(feature_config, device=torch.device("cpu"))
    bars = _make_bars(n_bars)
    batched = pipeline.build_batch(bars, symbol="AAPL", temporal_window=temporal_window)
    for index in range(int(feature_config.windows["long"]), n_bars):
        window = bars[max(0, index + 1 - temporal_window): index + 1]
        scalar = pipeline.build(window, symbol="AAPL")
        _assert_snapshots_match(scalar, batched.snapshot_at(index), index)


def test_scalar_vs_batch_parity_default_config() -> None:
    config = load_config("configs/default.yaml")
    _run_parity(
        config.features,
        temporal_window=config.runtime.warmup_bars,
        n_bars=130,
    )


def test_scalar_vs_batch_parity_small_window_config() -> None:
    """Small windows drop ``first_valid_index`` below the RSI (14) and
    MACD (35) warmups and place a momentum boundary (W=12) above
    ``windows.long`` — the exact geometry that exposed the pre-fix
    off-by-ones (see module docstring)."""
    config = load_config("configs/default.yaml")
    small = config.features.model_copy(
        update={
            "windows": {"short": 3, "medium": 5, "long": 10},
            "volatility_windows": [5],
            "momentum_windows": [3, 12],
            "volume_windows": [5],
        }
    )
    _run_parity(small, temporal_window=40, n_bars=90)


def test_scalar_vs_batch_parity_with_market_context() -> None:
    """Phase 3 market-context features must satisfy the same scalar/batch
    invariant as every other family: ``build_batch(...).snapshot_at(i)``
    equals ``build`` on the ``temporal_window`` slice ending at ``i``,
    with the scalar context computed from the same-length slices of every
    symbol's bars.  Covers all eight ``market_*`` features (SPY/QQQ
    proxies, breadth over a 4-symbol universe, relative strength)."""
    from src.features.market_context import (
        compute_market_context_batch,
        compute_market_context_snapshot,
    )

    config = load_config("configs/default.yaml")
    features = config.features.model_copy(update={"market_context_enabled": True})
    short_w = int(features.windows["short"])
    medium_w = int(features.windows["medium"])
    temporal_window = config.runtime.warmup_bars
    n_bars = 130
    bars_by_symbol = {
        "AAPL": _make_bars(n_bars, seed=7, symbol="AAPL"),
        "SPY": _make_bars(n_bars, seed=11, symbol="SPY"),
        "QQQ": _make_bars(n_bars, seed=13, symbol="QQQ"),
        "MSFT": _make_bars(n_bars, seed=17, symbol="MSFT"),
    }
    pipeline = FeaturePipeline(features, device=torch.device("cpu"))
    batch_context = compute_market_context_batch(
        bars_by_symbol,
        short_window=short_w,
        medium_window=medium_w,
        epsilon=features.epsilon,
    )
    batched = pipeline.build_batch(
        bars_by_symbol["AAPL"],
        symbol="AAPL",
        temporal_window=temporal_window,
        market_context=batch_context,
    )
    saw_nonzero_market_value = False
    for index in range(int(features.windows["long"]), n_bars):
        slices = {
            symbol: bars[max(0, index + 1 - temporal_window): index + 1]
            for symbol, bars in bars_by_symbol.items()
        }
        scalar_context = compute_market_context_snapshot(
            slices,
            short_window=short_w,
            medium_window=medium_w,
            epsilon=features.epsilon,
        )
        scalar = pipeline.build(
            slices["AAPL"], symbol="AAPL", market_context=scalar_context
        )
        batch_snapshot = batched.snapshot_at(index)
        _assert_snapshots_match(scalar, batch_snapshot, index)
        market_rows = [
            row
            for row, name in enumerate(scalar.feature_names)
            if name.startswith("market_")
        ]
        assert len(market_rows) == 8
        if any(float(scalar.features[row]) != 0.0 for row in market_rows):
            saw_nonzero_market_value = True
    # Guard against a vacuous pass where both paths agree on all-zero
    # market features (e.g. a broken context wiring on both sides).
    assert saw_nonzero_market_value


def test_first_valid_index_covers_all_feature_family_warmups() -> None:
    """``first_valid_index`` used to be ``windows['long']`` alone; MACD
    (35), RSI/ATR (14) and per-window return warmups can exceed it for
    small-window configs."""
    config = load_config("configs/default.yaml")
    bars = _make_bars(90)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    batched = pipeline.build_batch(bars, symbol="AAPL", temporal_window=48)
    # Default config: windows.long (60) still dominates every warmup.
    assert batched.first_valid_index == 60

    small = config.features.model_copy(
        update={
            "windows": {"short": 3, "medium": 5, "long": 10},
            "volatility_windows": [5],
            "momentum_windows": [3, 12],
            "volume_windows": [5],
        }
    )
    small_pipeline = FeaturePipeline(small, device=torch.device("cpu"))
    small_batched = small_pipeline.build_batch(bars, symbol="AAPL", temporal_window=40)
    # MACD needs 35 bars of history (slow SMA seed + signal seed) and the
    # temporal window (40) is large enough for it to be active, so the
    # MACD warmup dominates windows.long (10).
    assert small_batched.first_valid_index == 35


# ── §4.3.3 live-path input sanitation ───────────────────────────────────


def test_zero_close_bar_produces_finite_features_on_live_path() -> None:
    """A pathological zero-close bar used to send inf through the
    ``return_1`` denominator and inf/NaN through the volatility family
    (log of an unclamped inf ratio → inf std).  The live path now
    epsilon-guards return denominators and nan_to_num's the snapshot."""
    from dataclasses import replace

    config = load_config("configs/default.yaml")
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bars = _make_bars(80)
    # Interior dead bar (zero prices/volume) + zero close in the
    # return_1 denominator position (second-to-last bar).
    bars[40] = replace(bars[40], open=0.0, high=0.0, low=0.0, close=0.0, volume=0.0)
    bars[-2] = replace(bars[-2], close=0.0)
    snapshot = pipeline.build(bars, symbol="AAPL")
    assert torch.isfinite(snapshot.features).all()
    for branch_name, branch_tensor in snapshot.branch_inputs.items():
        assert torch.isfinite(branch_tensor).all(), f"non-finite branch {branch_name!r}"
    assert torch.isfinite(snapshot.regime_input).all()
    assert torch.isfinite(snapshot.temporal_tensor).all()


def test_feature_epsilon_has_a_single_default_source() -> None:
    """§4.3.3 epsilon unification: configs/default.yaml is the canonical
    source of ``features.epsilon`` (5e-4) and every feature formula
    receives it via ``FeatureConfig`` — no file under src/features may
    hardcode a numeric epsilon default or fallback.  The one remaining
    literal is the ``FeatureConfig.epsilon`` pydantic field default,
    which the YAML value overrides on every config load; aligning that
    literal with the YAML is tracked as §4.6.5 (defaults alignment).
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    # No numeric epsilon literals in the feature code itself.
    literal_pattern = re.compile(r"(?<![\w])epsilon\s*[:=]\s*[0-9]")
    offenders: list[str] = []
    for path in sorted((repo_root / "src" / "features").glob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if literal_pattern.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, f"hardcoded feature epsilon literals: {offenders}"

    # The feature-epsilon default is defined exactly once in src/.
    default_pattern = re.compile(r"^\s*epsilon\s*:\s*float\s*=")
    definitions: list[str] = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if default_pattern.match(line):
                definitions.append(f"{path.relative_to(repo_root)}:{lineno}")
    assert len(definitions) == 1, f"expected one epsilon default, found: {definitions}"

    # And the canonical YAML value is what the loaded config carries.
    config = load_config("configs/default.yaml")
    assert config.features.epsilon == pytest.approx(5.0e-4)
