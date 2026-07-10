from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch

from src.core.config import FeatureConfig
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar, bars_to_tensor
from src.features.momentum_features import (
    compute_momentum_summaries,
    compute_momentum_summaries_rolling,
)
from src.features.multi_timeframe import (
    compute_multi_timeframe_features,
    compute_multi_timeframe_features_rolling,
)
from src.features.price_features import (
    compute_price_features,
    compute_price_features_rolling,
    compute_technical_indicators,
    compute_technical_indicators_rolling,
)
from src.features.regime_features import (
    compute_regime_features,
    compute_regime_features_rolling,
)
from src.features.volume_features import (
    compute_volume_features,
    compute_volume_features_rolling,
)
from src.features.volatility_features import (
    compute_volatility_features,
    compute_volatility_features_rolling,
)
from src.features.calendar_features import (
    compute_calendar_features,
    compute_calendar_features_rolling,
)


@dataclass(frozen=True)
class FeatureContract:
    feature_names: tuple[str, ...]
    branch_feature_names: dict[str, tuple[str, ...]]
    regime_feature_names: tuple[str, ...]
    temporal_feature_names: tuple[str, ...]
    feature_dtype: str

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    @property
    def regime_dim(self) -> int:
        return len(self.regime_feature_names)

    @property
    def temporal_dim(self) -> int:
        return len(self.temporal_feature_names)

    @property
    def branch_dims(self) -> dict[str, int]:
        return {
            branch_name: len(feature_names)
            for branch_name, feature_names in self.branch_feature_names.items()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "branch_feature_names": {
                branch_name: list(feature_names)
                for branch_name, feature_names in self.branch_feature_names.items()
            },
            "regime_feature_names": list(self.regime_feature_names),
            "temporal_feature_names": list(self.temporal_feature_names),
            "feature_dtype": self.feature_dtype,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureContract":
        return cls(
            feature_names=tuple(payload["feature_names"]),
            branch_feature_names={
                branch_name: tuple(feature_names)
                for branch_name, feature_names in payload["branch_feature_names"].items()
            },
            regime_feature_names=tuple(payload["regime_feature_names"]),
            temporal_feature_names=tuple(payload.get("temporal_feature_names", [])),
            feature_dtype=str(payload["feature_dtype"]),
        )

    def assert_matches(self, other: "FeatureContract", *, context: str) -> None:
        if self.feature_names != other.feature_names:
            raise ConfigurationError(f"{context}: feature ordering mismatch")
        if self.branch_feature_names != other.branch_feature_names:
            raise ConfigurationError(f"{context}: branch feature assignment mismatch")
        if self.regime_feature_names != other.regime_feature_names:
            raise ConfigurationError(f"{context}: regime feature ordering mismatch")
        if self.temporal_feature_names != other.temporal_feature_names:
            raise ConfigurationError(f"{context}: temporal feature ordering mismatch")
        if self.feature_dtype != other.feature_dtype:
            raise ConfigurationError(f"{context}: feature dtype mismatch")

    def validate_snapshot(self, snapshot: "FeatureSnapshot", *, context: str) -> None:
        self.assert_matches(snapshot.contract, context=context)


@dataclass
class FeatureSnapshot:
    symbol: str
    timestamp: datetime
    last_price: float
    spread_bps: float
    features: torch.Tensor
    branch_inputs: dict[str, torch.Tensor]
    regime_input: torch.Tensor
    temporal_tensor: torch.Tensor
    feature_names: list[str]
    branch_feature_names: dict[str, list[str]]
    regime_feature_names: list[str]
    temporal_feature_names: list[str]

    @property
    def contract(self) -> FeatureContract:
        return FeatureContract(
            feature_names=tuple(self.feature_names),
            branch_feature_names={
                branch_name: tuple(feature_names)
                for branch_name, feature_names in self.branch_feature_names.items()
            },
            regime_feature_names=tuple(self.regime_feature_names),
            temporal_feature_names=tuple(self.temporal_feature_names),
            feature_dtype=str(self.features.dtype),
        )


@dataclass
class BatchedFeatureSnapshots:
    """Vectorized feature table for a full bar history.

    Produced by ``FeaturePipeline.build_batch`` in a single pass via
    the rolling helpers in ``src/features/rolling.py``.  Each column
    holds the per-bar feature values for every index in
    ``[first_valid_index, len(bars))``.  ``snapshot_at`` extracts a
    per-bar ``FeatureSnapshot`` in O(F) — no feature recomputation.

    The primary win is for backtest / training / memory-build where we
    used to call ``pipeline.build_from_tensor`` once per bar.  Those
    paths now call ``pipeline.build_batch`` once per symbol and slice
    the container per bar, amortizing the per-feature reduction cost
    across the whole history via ``torch.unfold``.
    """
    symbol: str
    feature_names: list[str]
    branch_feature_names: dict[str, list[str]]
    regime_feature_names: list[str]
    temporal_feature_names: list[str]
    features: torch.Tensor  # (N_bars, F)
    branch_inputs: dict[str, torch.Tensor]  # each (N_bars, branch_dim)
    regime_input: torch.Tensor  # (N_bars, R)
    temporal_columns: torch.Tensor  # (T, N_bars)  — full-history pointwise
    temporal_window: int  # number of bars per per-bar temporal slice
    last_prices: torch.Tensor  # (N_bars,)
    spread_values: torch.Tensor  # (N_bars,)
    timestamps: list[datetime]
    first_valid_index: int  # zero-fill prefix length

    def __post_init__(self) -> None:
        # Precompute name->row-index lookup for the temporal feature
        # table.  ``snapshot_at`` is called once per bar per symbol on
        # hot paths (backtest / memory build) and previously called
        # ``self.temporal_feature_names.index(name)`` three times per
        # invocation — a linear scan of a ~9-item list.  Caching the
        # dict drops the per-bar cost from 50-200µs to <1µs.
        # ``temporal_feature_names`` is assigned once at construction
        # and never mutated thereafter, so this cache is safe.
        self._temporal_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.temporal_feature_names)
        }

    @property
    def n_bars(self) -> int:
        return self.features.shape[0]

    def snapshot_at(self, index: int) -> FeatureSnapshot:
        """Return a per-bar ``FeatureSnapshot`` at the given bar index.

        The snapshot's ``temporal_tensor`` matches what the scalar
        pipeline would return for a ``warmup_bars``-long window ending
        at this bar — we slice the precomputed temporal columns and
        then apply the same "first-bar reset" the scalar path applies
        to ``close_return`` / ``log_return`` / ``volume_change`` /
        ``volatility_proxy``.  Equivalence with the scalar path is
        within fp32 reduction-order noise (observed <= ~2.6e-5, pinned
        at rtol=1e-4 / atol=5e-5 by tests/test_feature_parity.py).
        """
        if not 0 <= index < self.n_bars:
            raise IndexError(f"snapshot_at index out of range: {index} (n={self.n_bars})")
        w = self.temporal_window
        temporal_start = max(0, index + 1 - w)
        # Clone so our per-window reset does not mutate the shared
        # precomputed columns (subsequent ``snapshot_at`` calls must
        # see the untouched full-history temporal_columns).
        temporal_tensor = self.temporal_columns[:, temporal_start: index + 1].clone()
        # Scalar-path semantics: at the first row of every per-bar
        # window, close_return / log_return / volume_change are
        # forced to 0 because that row "has no prior bar" within the
        # window.  The batched columns hold the *real* prior-bar
        # values from the full history, so we need to override them
        # here to match.  Indices come from the temporal_feature_names
        # ordering produced by ``_build_temporal_tensor``.
        # ``volatility_proxy_t`` is ``close_return.abs()`` and inherits
        # the same first-row reset in the scalar path (§4.3.1 fix —
        # omitting it leaked the prior bar's return into column 0 of
        # every full window).
        if temporal_tensor.shape[1] > 0:
            for reset_name in (
                "close_return_t",
                "log_return_t",
                "volume_change_t",
                "volatility_proxy_t",
            ):
                row_idx = self._temporal_name_to_idx.get(reset_name)
                if row_idx is None:
                    continue
                temporal_tensor[row_idx, 0] = 0.0
        last_price = float(self.last_prices[index].item())
        spread = float(self.spread_values[index].item())
        spread_bps = (spread / last_price) * 10_000.0 if last_price > 0 else 0.0
        return FeatureSnapshot(
            symbol=self.symbol,
            timestamp=self.timestamps[index],
            last_price=last_price,
            spread_bps=spread_bps,
            features=self.features[index],
            branch_inputs={k: v[index] for k, v in self.branch_inputs.items()},
            regime_input=self.regime_input[index],
            temporal_tensor=temporal_tensor,
            feature_names=self.feature_names,
            branch_feature_names=self.branch_feature_names,
            regime_feature_names=self.regime_feature_names,
            temporal_feature_names=self.temporal_feature_names,
        )


class FeaturePipeline:
    def __init__(
        self,
        config: FeatureConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype
        # --- Per-pipeline name-resolution cache -----------------------------
        # Feature names are deterministic given the FeatureConfig: the set of
        # feature names produced by ``build()`` is identical across every
        # call on the same pipeline instance. The old code recomputed the
        # full branch-prefix match + regime-prefix filter per sample, which
        # profiled as 0.76s out of 6.7s total feature-pipeline time on a
        # 3k-sample build. We now memoize by the tuple of sorted feature
        # names (first call pays the cost, all subsequent calls reuse it).
        # The cache is keyed on the tuple itself so any future feature-set
        # change would miss and rebuild automatically.
        self._name_cache: dict[
            tuple[str, ...],
            tuple[dict[str, list[str]], list[str]],
        ] = {}

    def build(self, bars: Sequence[MarketBar], *, symbol: str) -> FeatureSnapshot:
        if not bars:
            raise ValueError("At least one bar is required to build features")
        tensor = bars_to_tensor(bars, device=self.device, dtype=self.dtype)
        return self.build_from_tensor(tensor, symbol=symbol, last_bar=bars[-1])

    def build_from_tensor(
        self,
        bars_tensor: torch.Tensor,
        *,
        symbol: str,
        last_bar: MarketBar | None = None,
        timestamp: datetime | None = None,
        last_price: float | None = None,
        spread: float | None = None,
    ) -> FeatureSnapshot:
        """Build a feature snapshot from a precomputed ``(T, 6)`` tensor.

        This is the runtime hot path: the per-tick BarBuffer hands its
        preallocated tensor view directly to this method, skipping the
        per-bar Python list comprehension that ``bars_to_tensor`` would
        otherwise have to do on every call.

        ``last_bar`` is the most-recent ``MarketBar`` and provides the
        symbol/timestamp/price metadata that the snapshot needs.  When
        callers do not have a ``MarketBar`` (e.g. backtest precompute
        paths) they may instead pass ``timestamp``, ``last_price``, and
        ``spread`` explicitly.
        """
        if bars_tensor.dim() != 2 or bars_tensor.shape[0] == 0:
            raise ValueError(
                f"build_from_tensor requires a non-empty (T, 6) tensor, got shape "
                f"{tuple(bars_tensor.shape)}"
            )
        if last_bar is not None:
            resolved_timestamp = last_bar.timestamp
            resolved_close = float(last_bar.close)
            resolved_spread = float(last_bar.spread)
        else:
            if timestamp is None or last_price is None:
                raise ValueError(
                    "build_from_tensor requires either last_bar or (timestamp, last_price)"
                )
            resolved_timestamp = timestamp
            resolved_close = float(last_price)
            resolved_spread = float(spread if spread is not None else 0.0)
        spread_bps = (
            (resolved_spread / resolved_close) * 10_000.0 if resolved_close > 0 else 0.0
        )

        values: dict[str, torch.Tensor] = {}
        values.update(compute_price_features(bars_tensor, self.config.momentum_windows, self.config.epsilon))
        values.update(compute_volatility_features(bars_tensor, self.config.volatility_windows, self.config.epsilon))
        values.update(compute_volume_features(bars_tensor, self.config.volume_windows, self.config.epsilon))
        values.update(compute_momentum_summaries(bars_tensor, self.config.momentum_windows, self.config.epsilon))
        values.update(compute_multi_timeframe_features(bars_tensor, self.config.windows, self.config.epsilon))
        values.update(
            compute_regime_features(
                bars_tensor,
                short_window=self.config.windows["short"],
                medium_window=self.config.windows["medium"],
                long_window=self.config.windows["long"],
                epsilon=self.config.epsilon,
            )
        )
        values.update(compute_technical_indicators(bars_tensor, self.config.epsilon))
        values.update(compute_calendar_features(resolved_timestamp, device=bars_tensor.device))

        feature_names = sorted(values.keys())
        cache_key = tuple(feature_names)
        cached = self._name_cache.get(cache_key)
        if cached is None:
            branch_feature_names = self._resolve_branch_feature_names(feature_names)
            regime_names = [name for name in feature_names if name.startswith("regime_")]
            self._name_cache[cache_key] = (branch_feature_names, regime_names)
        else:
            branch_feature_names, regime_names = cached
        stacked = torch.stack([values[name] for name in feature_names], dim=0)
        # Live-path input sanitation (§4.3.3): a pathological bar (e.g.
        # zero close) can still push a non-finite value through a
        # feature formula (log of a zero-close ratio inside a rolling
        # std, ...).  The model must never see NaN/inf, so replace them
        # with 0 at the snapshot boundary.  This is a no-op on finite
        # features, which keeps scalar/batch parity intact
        # (tests/test_feature_parity.py).
        stacked = torch.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
        branch_inputs = self._assign_branch_inputs(feature_names, stacked, branch_feature_names)
        regime_input = torch.stack([values[name] for name in regime_names], dim=0)
        regime_input = torch.nan_to_num(regime_input, nan=0.0, posinf=0.0, neginf=0.0)
        temporal_names, temporal_tensor = self._build_temporal_tensor(bars_tensor)
        temporal_tensor = torch.nan_to_num(temporal_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return FeatureSnapshot(
            symbol=symbol,
            timestamp=resolved_timestamp,
            last_price=resolved_close,
            spread_bps=spread_bps,
            features=stacked,
            branch_inputs=branch_inputs,
            regime_input=regime_input,
            temporal_tensor=temporal_tensor,
            feature_names=feature_names,
            branch_feature_names=branch_feature_names,
            regime_feature_names=regime_names,
            temporal_feature_names=temporal_names,
        )

    def build_contract(self, bars: Sequence[MarketBar], *, symbol: str) -> FeatureContract:
        return self.build(bars, symbol=symbol).contract

    def build_batch(
        self,
        bars: Sequence[MarketBar],
        *,
        symbol: str,
        temporal_window: int,
    ) -> BatchedFeatureSnapshots:
        """Vectorized rolling feature pipeline for a full bar history.

        Produces a ``BatchedFeatureSnapshots`` container holding every
        per-bar feature tensor for the full ``bars`` sequence in one
        vectorized pass.  Callers (backtest, data_module, memory build)
        iterate via ``container.snapshot_at(i)`` to get an individual
        ``FeatureSnapshot`` without paying for per-bar feature
        computation.

        Invariant: for bar indices ``i >= first_valid_index`` (derived
        from ALL feature-family warmups — see
        ``_derive_first_valid_index``) the returned snapshot matches
        what ``FeaturePipeline.build`` would return if called with
        ``bars[max(0, i + 1 - temporal_window): i + 1]``, within fp32
        reduction-order noise (pinned at rtol=1e-4 / atol=5e-5 by
        tests/test_feature_parity.py; the live path additionally
        sanitizes non-finite values, a no-op on well-formed bars).
        Earlier bars are zero-filled — callers must never read from
        them (use ``first_valid_index`` to guard).

        ``temporal_window`` matches the per-bar temporal tensor size
        used by the scalar path (typically ``config.runtime.warmup_bars``).
        It controls how many historical bars the model's temporal CNN
        sees at each bar, NOT which bars get features computed.

        On-disk cache: ``build_batch`` is deterministic given
        ``(bars, feature config, temporal_window)``.  We compute a
        fingerprint and short-circuit on cache hits.  Disable with
        ``SETAI_DISABLE_FEATURE_CACHE=1``.  Changing the feature
        config auto-invalidates the cache (different fingerprint).
        """
        from src.features.feature_cache import (
            cache_is_disabled,
            compute_fingerprint,
            load_cached,
            save_cached,
        )

        if not bars:
            raise ValueError("build_batch requires a non-empty bar sequence")

        fingerprint: str | None = None
        if not cache_is_disabled():
            bars_signature = self._compute_bars_signature(bars)
            fingerprint = compute_fingerprint(
                bars_signature=bars_signature,
                symbol=symbol,
                feature_config_snapshot=self._feature_config_snapshot(),
                temporal_window=temporal_window,
            )
            cached = load_cached(fingerprint)
            if cached is not None:
                return cached

        result = self._build_batch_impl(bars, symbol=symbol, temporal_window=temporal_window)

        if fingerprint is not None:
            save_cached(fingerprint, result)

        return result

    def _compute_bars_signature(self, bars: Sequence[MarketBar]) -> dict[str, Any]:
        """Signature for cache keying: endpoints + a cheap content hash.

        Endpoint fields alone (count / first / last) collide when only
        interior bars change — e.g. a re-fetch with corrected adjusted
        prices for the same date range would silently keep serving the
        stale cached features.  The close/volume sums plus a handful of
        sampled interior closes make any single-bar edit change the
        fingerprint without hashing the full history.
        """
        bars_list = list(bars)
        if not bars_list:
            return {"count": 0}
        first, last = bars_list[0], bars_list[-1]
        n = len(bars_list)
        sample_indices = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        return {
            "count": n,
            "first_ts": first.timestamp.isoformat(),
            "last_ts": last.timestamp.isoformat(),
            "first_close": float(first.close),
            "last_close": float(last.close),
            # math.fsum: exact float summation, so any value edit shifts
            # the sum deterministically (no fp-rounding near-misses).
            "close_sum": float(math.fsum(bar.close for bar in bars_list)),
            "volume_sum": float(math.fsum(bar.volume for bar in bars_list)),
            "sampled_closes": [float(bars_list[i].close) for i in sample_indices],
        }

    def _feature_config_snapshot(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "windows": dict(cfg.windows) if hasattr(cfg, "windows") else {},
            "volatility_windows": list(cfg.volatility_windows) if hasattr(cfg, "volatility_windows") else [],
            "momentum_windows": list(cfg.momentum_windows) if hasattr(cfg, "momentum_windows") else [],
            "volume_windows": list(cfg.volume_windows) if hasattr(cfg, "volume_windows") else [],
            "epsilon": float(cfg.epsilon) if hasattr(cfg, "epsilon") else 0.0,
            "branch_feature_prefixes": {
                k: list(v) for k, v in getattr(cfg, "branch_feature_prefixes", {}).items()
            },
        }

    def _build_batch_impl(
        self,
        bars: Sequence[MarketBar],
        *,
        symbol: str,
        temporal_window: int,
    ) -> BatchedFeatureSnapshots:
        """Uncached build_batch body. See :meth:`build_batch` for contract."""
        if not bars:
            raise ValueError("build_batch requires a non-empty bar sequence")
        bars_tensor = bars_to_tensor(bars, device=self.device, dtype=self.dtype)
        n = bars_tensor.shape[0]
        epsilon = self.config.epsilon
        # --- Window semantics to match per-bar scalar path ---
        # The scalar pipeline is called with a ``temporal_window``-sized
        # bar slice at each bar.  Inside that slice, each feature
        # family's per-window guard uses one of two strategies:
        #
        #   - price / volatility / volume / momentum / multi_timeframe:
        #     "zero if config_window > available" — if a configured
        #     window exceeds the slice size, that feature is always 0
        #     regardless of bar index.
        #
        #   - regime_features:
        #     "clip to min(config_window, available)" — the reduction
        #     still fires, but on the smaller available window.
        #
        # We mirror both behaviours here:
        #   - For non-regime families we run the rolling helpers with
        #     the ORIGINAL config windows and then, as a post-step,
        #     zero out any feature whose config_window > temporal_window
        #     (so the batched output has the same zero columns the
        #     scalar path would have produced).
        #   - For regime we pre-clip the window to temporal_window so
        #     the rolling helpers compute the same reduction the
        #     scalar path would have computed on the clipped window.
        regime_short_w = max(1, min(int(self.config.windows["short"]), temporal_window))
        regime_medium_w = max(1, min(int(self.config.windows["medium"]), temporal_window))
        regime_long_w = max(1, min(int(self.config.windows["long"]), temporal_window))

        # --- Vectorized per-family rolling features -------------------------
        values: dict[str, torch.Tensor] = {}
        values.update(
            compute_price_features_rolling(bars_tensor, self.config.momentum_windows, epsilon)
        )
        values.update(
            compute_volatility_features_rolling(bars_tensor, self.config.volatility_windows, epsilon)
        )
        values.update(
            compute_volume_features_rolling(bars_tensor, self.config.volume_windows, epsilon)
        )
        values.update(
            compute_momentum_summaries_rolling(bars_tensor, self.config.momentum_windows, epsilon)
        )
        values.update(
            compute_multi_timeframe_features_rolling(bars_tensor, self.config.windows, epsilon)
        )
        values.update(
            compute_regime_features_rolling(
                bars_tensor,
                short_window=regime_short_w,
                medium_window=regime_medium_w,
                long_window=regime_long_w,
                epsilon=epsilon,
            )
        )
        values.update(compute_calendar_features_rolling([bar.timestamp for bar in bars], device=bars_tensor.device))
        values.update(
            compute_technical_indicators_rolling(bars_tensor, epsilon)
        )

        # --- Post-step: zero out non-regime features whose config_window
        # exceeds the per-bar temporal_window ---
        # The scalar path at each bar has ``close.numel() == temporal_window``
        # and returns zero for any requested window larger than that.
        # Match that here.  Regime features were already clipped, so
        # they're not affected by this loop.
        def _zero_if_oversized(prefix: str, cfg_windows: list[int]) -> None:
            for w in cfg_windows:
                if w > temporal_window:
                    key = f"{prefix}{w}"
                    if key in values:
                        values[key] = torch.zeros(
                            n, dtype=bars_tensor.dtype, device=bars_tensor.device
                        )

        # volatility_{W}: "if log_returns.numel() >= W" → volatility
        # scalar uses log_returns (length = numel - 1), so the
        # oversized check is ``W > temporal_window - 1``.
        for w in self.config.volatility_windows:
            if w > temporal_window - 1:
                key = f"volatility_{w}"
                if key in values:
                    values[key] = torch.zeros(
                        n, dtype=bars_tensor.dtype, device=bars_tensor.device
                    )
        # price features: ``if close.numel() > window`` means
        # ``temporal_window > W`` ⇔ ``W < temporal_window``.  So
        # oversized is ``W >= temporal_window``.
        for w in self.config.momentum_windows:
            if w >= temporal_window:
                for prefix in ("return_", "log_return_", "momentum_"):
                    key = f"{prefix}{w}"
                    if key in values:
                        values[key] = torch.zeros(
                            n, dtype=bars_tensor.dtype, device=bars_tensor.device
                        )
        # volume_{W}, volume_change_{W}, volume_z_{W}: ``>= W`` → oversized is ``W > temporal_window``.
        _zero_if_oversized("volume_z_", self.config.volume_windows)
        _zero_if_oversized("volume_change_", self.config.volume_windows)
        # momentum_slope_{W}: same ``>= W``.
        _zero_if_oversized("momentum_slope_", self.config.momentum_windows)
        # multi-timeframe features: ``>= W``.
        for label, w in self.config.windows.items():
            if w > temporal_window:
                for prefix in ("regime_close_to_ma_", "regime_price_z_"):
                    key = f"{prefix}{label}"
                    if key in values:
                        values[key] = torch.zeros(
                            n, dtype=bars_tensor.dtype, device=bars_tensor.device
                        )
        # Technical indicators: scalar path guards on ``n > period``,
        # so zero when the per-bar temporal_window is not large enough.
        from src.features.price_features import (
            _RSI_PERIOD,
            _ATR_PERIOD,
            _MACD_MIN_BARS,
        )
        if temporal_window <= _RSI_PERIOD:
            values["rsi_14"] = torch.zeros(
                n, dtype=bars_tensor.dtype, device=bars_tensor.device
            )
        if temporal_window <= _ATR_PERIOD:
            values["atr_14"] = torch.zeros(
                n, dtype=bars_tensor.dtype, device=bars_tensor.device
            )
        if temporal_window <= _MACD_MIN_BARS:
            for key in ("macd_line", "macd_signal", "macd_histogram"):
                values[key] = torch.zeros(
                    n, dtype=bars_tensor.dtype, device=bars_tensor.device
                )

        feature_names = sorted(values.keys())
        cache_key = tuple(feature_names)
        cached = self._name_cache.get(cache_key)
        if cached is None:
            branch_feature_names = self._resolve_branch_feature_names(feature_names)
            regime_names = [name for name in feature_names if name.startswith("regime_")]
            self._name_cache[cache_key] = (branch_feature_names, regime_names)
        else:
            branch_feature_names, regime_names = cached

        # Stack features into (F, N) then transpose to (N, F) for row-major
        # per-bar access.  ``torch.stack`` along dim=1 gives us (N, F) directly.
        features_table = torch.stack(
            [values[name] for name in feature_names], dim=1
        )  # (N, F)

        branch_inputs_table: dict[str, torch.Tensor] = {}
        for branch_name, matched in branch_feature_names.items():
            if not matched:
                branch_inputs_table[branch_name] = torch.zeros(
                    n, 1, dtype=features_table.dtype, device=features_table.device
                )
                continue
            branch_inputs_table[branch_name] = torch.stack(
                [values[name] for name in matched], dim=1
            )  # (N, branch_dim)

        if regime_names:
            regime_input_table = torch.stack(
                [values[name] for name in regime_names], dim=1
            )  # (N, R)
        else:
            regime_input_table = torch.zeros(
                n, 0, dtype=features_table.dtype, device=features_table.device
            )

        # --- Pointwise temporal columns on the full history ----------------
        temporal_names, temporal_columns = self._build_temporal_columns(bars_tensor)

        first_valid_index = self._derive_first_valid_index(temporal_window)
        return BatchedFeatureSnapshots(
            symbol=symbol,
            feature_names=feature_names,
            branch_feature_names=branch_feature_names,
            regime_feature_names=regime_names,
            temporal_feature_names=temporal_names,
            features=features_table,
            branch_inputs=branch_inputs_table,
            regime_input=regime_input_table,
            temporal_columns=temporal_columns,
            temporal_window=temporal_window,
            last_prices=bars_tensor[:, 3].clone(),
            spread_values=bars_tensor[:, 5].clone(),
            timestamps=[bar.timestamp for bar in bars],
            first_valid_index=first_valid_index,
        )

    def _derive_first_valid_index(self, temporal_window: int) -> int:
        """First bar index at which every feature family is fully warmed.

        Previously this was ``windows['long']`` alone, but family
        warmups can exceed it for small-window configs: MACD needs 35
        bars (slow-SMA seed + signal seed), RSI/ATR need 14, and each
        ``return_{W}`` needs ``W + 1`` bars.  Per family, the first
        valid index mirrors the scalar path's insufficient-history
        guard; windows large enough to be zeroed everywhere by the
        ``temporal_window`` post-step are trivially consistent at every
        index and therefore excluded.  ``windows['long']`` is kept as a
        floor so existing consumers of ``first_valid_index`` never see
        it shrink.
        """
        from src.features.price_features import (
            _ATR_PERIOD,
            _MACD_MIN_BARS,
            _RSI_PERIOD,
        )

        candidates = [max(int(self.config.windows.get("long", 0)), 0)]
        # return_/log_return_/momentum_{W}: scalar guard ``numel > W``.
        candidates.extend(w for w in self.config.momentum_windows if w < temporal_window)
        # momentum_slope_{W}: scalar guard ``numel >= W``.
        candidates.extend(w - 1 for w in self.config.momentum_windows if w <= temporal_window)
        # volatility_{W}: scalar guard ``numel - 1 >= W`` (log returns).
        candidates.extend(w for w in self.config.volatility_windows if w <= temporal_window - 1)
        # volume_z_/volume_change_{W}: scalar guard ``numel >= W``.
        candidates.extend(w - 1 for w in self.config.volume_windows if w <= temporal_window)
        # regime_close_to_ma_/regime_price_z_{label}: scalar guard ``numel >= W``.
        candidates.extend(w - 1 for w in self.config.windows.values() if w <= temporal_window)
        # Technical indicators: scalar guard ``numel > period``.
        if temporal_window > _RSI_PERIOD:
            candidates.append(_RSI_PERIOD)
        if temporal_window > _ATR_PERIOD:
            candidates.append(_ATR_PERIOD)
        if temporal_window > _MACD_MIN_BARS:
            candidates.append(_MACD_MIN_BARS)
        return max(candidates)

    def _build_temporal_columns(
        self, bars_tensor: torch.Tensor
    ) -> tuple[list[str], torch.Tensor]:
        """Same pointwise temporal-feature math as ``_build_temporal_tensor``
        but returns the full ``(9, N)`` column tensor.  ``build_batch``
        slices columns per bar in ``snapshot_at``; the scalar path
        continues to call ``_build_temporal_tensor`` on per-bar window
        slices (no change).  The math is identical, so a slice of the
        batched columns equals what the scalar path would compute on
        the same window."""
        names, tensor = self._build_temporal_tensor(bars_tensor)
        return names, tensor

    def infer_branch_dimensions(self, feature_names: list[str]) -> dict[str, int]:
        return {
            branch_name: len(self._match_features(feature_names, prefixes))
            for branch_name, prefixes in self.config.branch_feature_prefixes.items()
        }

    def _assign_branch_inputs(
        self,
        feature_names: list[str],
        feature_values: torch.Tensor,
        branch_feature_names: dict[str, list[str]] | None = None,
    ) -> dict[str, torch.Tensor]:
        values_by_name = {name: feature_values[index] for index, name in enumerate(feature_names)}
        resolved_branch_feature_names = branch_feature_names or self._resolve_branch_feature_names(feature_names)
        branch_inputs: dict[str, torch.Tensor] = {}
        for branch_name, matched in resolved_branch_feature_names.items():
            if not matched:
                branch_inputs[branch_name] = torch.zeros(1, dtype=feature_values.dtype, device=feature_values.device)
                continue
            branch_inputs[branch_name] = torch.stack([values_by_name[name] for name in matched], dim=0)
        return branch_inputs

    def _resolve_branch_feature_names(self, feature_names: list[str]) -> dict[str, list[str]]:
        return {
            branch_name: self._match_features(feature_names, prefixes)
            for branch_name, prefixes in self.config.branch_feature_prefixes.items()
        }

    def _build_temporal_tensor(self, bars_tensor: torch.Tensor) -> tuple[list[str], torch.Tensor]:
        if bars_tensor.dim() != 2 or bars_tensor.shape[-1] < 6:
            raise ValueError(f"Expected bars tensor shape (time, 6+), got {tuple(bars_tensor.shape)}")
        open_, high, low, close, volume, spread = [bars_tensor[:, index] for index in range(6)]
        epsilon = self.config.epsilon

        prev_close = torch.roll(close, shifts=1, dims=0)
        prev_close[0] = close[0]
        prev_volume = torch.roll(volume, shifts=1, dims=0)
        prev_volume[0] = volume[0]

        close_return = (close - prev_close) / (prev_close.abs() + epsilon)
        close_return[0] = 0.0
        log_return = torch.log((close.abs() + epsilon) / (prev_close.abs() + epsilon))
        log_return[0] = 0.0
        range_ratio = (high - low) / (close.abs() + epsilon)
        body_ratio = (close - open_) / (open_.abs() + epsilon)
        upper_wick = (high - torch.maximum(open_, close)) / (close.abs() + epsilon)
        lower_wick = (torch.minimum(open_, close) - low) / (close.abs() + epsilon)
        volume_change = (volume - prev_volume) / (prev_volume.abs() + epsilon)
        volume_change[0] = 0.0
        spread_bps = (spread / (close.abs() + epsilon)) * 10_000.0
        volatility_proxy = close_return.abs()

        temporal_names = [
            "close_return_t",
            "log_return_t",
            "range_ratio_t",
            "body_ratio_t",
            "upper_wick_t",
            "lower_wick_t",
            "volume_change_t",
            "spread_bps_t",
            "volatility_proxy_t",
        ]
        temporal_tensor = torch.stack(
            [
                close_return,
                log_return,
                range_ratio,
                body_ratio,
                upper_wick,
                lower_wick,
                volume_change,
                spread_bps,
                volatility_proxy,
            ],
            dim=0,
        )
        return temporal_names, temporal_tensor

    @staticmethod
    def _match_features(feature_names: list[str], prefixes: list[str]) -> list[str]:
        return [name for name in feature_names if any(name.startswith(prefix) for prefix in prefixes)]
