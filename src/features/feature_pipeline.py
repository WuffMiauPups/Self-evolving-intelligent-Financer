from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import torch

from src.core.config import FeatureConfig
from src.core.exceptions import ConfigurationError
from src.data.schemas import MarketBar, bars_to_tensor
from src.features.momentum_features import compute_momentum_summaries
from src.features.multi_timeframe import compute_multi_timeframe_features
from src.features.price_features import compute_price_features
from src.features.regime_features import compute_regime_features
from src.features.volume_features import compute_volume_features
from src.features.volatility_features import compute_volatility_features


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

    def build(self, bars: Sequence[MarketBar], *, symbol: str) -> FeatureSnapshot:
        if not bars:
            raise ValueError("At least one bar is required to build features")
        tensor = bars_to_tensor(bars, device=self.device, dtype=self.dtype)
        values: dict[str, torch.Tensor] = {}
        values.update(compute_price_features(tensor, self.config.momentum_windows, self.config.epsilon))
        values.update(compute_volatility_features(tensor, self.config.volatility_windows, self.config.epsilon))
        values.update(compute_volume_features(tensor, self.config.volume_windows, self.config.epsilon))
        values.update(compute_momentum_summaries(tensor, self.config.momentum_windows, self.config.epsilon))
        values.update(compute_multi_timeframe_features(tensor, self.config.windows, self.config.epsilon))
        values.update(
            compute_regime_features(
                tensor,
                short_window=self.config.windows["short"],
                medium_window=self.config.windows["medium"],
                long_window=self.config.windows["long"],
                epsilon=self.config.epsilon,
            )
        )

        feature_names = sorted(values.keys())
        branch_feature_names = self._resolve_branch_feature_names(feature_names)
        regime_names = [name for name in feature_names if name.startswith("regime_")]
        stacked = torch.stack([values[name] for name in feature_names], dim=0)
        branch_inputs = self._assign_branch_inputs(feature_names, stacked, branch_feature_names)
        regime_input = torch.stack([values[name] for name in regime_names], dim=0)
        temporal_names, temporal_tensor = self._build_temporal_tensor(tensor)
        return FeatureSnapshot(
            symbol=symbol,
            timestamp=bars[-1].timestamp,
            last_price=bars[-1].close,
            spread_bps=((bars[-1].spread / bars[-1].close) * 10_000.0) if bars[-1].close > 0 else 0.0,
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
