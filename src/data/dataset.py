from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeatureContract, FeaturePipeline, FeatureSnapshot

if TYPE_CHECKING:
    from src.training.sample_columns import ColumnarSamples


@dataclass
class TrainingSample:
    features: torch.Tensor
    branch_inputs: dict[str, torch.Tensor]
    regime_input: torch.Tensor
    temporal_tensor: torch.Tensor
    label: torch.Tensor
    symbol: str = ""
    timestamp: datetime | None = None
    direction_name: str = ""
    path_label: torch.Tensor | None = None
    path_name: str = ""
    regression_targets: dict[str, float] = field(default_factory=dict)
    future_returns: dict[int, float] = field(default_factory=dict)


class BarWindowDataset(Dataset[TrainingSample]):
    def __init__(
        self,
        bars_by_symbol: dict[str, list[MarketBar]],
        pipeline: FeaturePipeline,
        *,
        warmup_bars: int,
        horizon: int,
        label_epsilon: float,
    ) -> None:
        self.samples: list[TrainingSample] = []
        self.feature_contract: FeatureContract | None = None
        for symbol, bars in bars_by_symbol.items():
            for index in range(warmup_bars, len(bars) - horizon):
                snapshot: FeatureSnapshot = pipeline.build(bars[max(0, index + 1 - warmup_bars): index + 1], symbol=symbol)
                if self.feature_contract is None:
                    self.feature_contract = snapshot.contract
                else:
                    self.feature_contract.validate_snapshot(
                        snapshot,
                        context=f"training_dataset[{symbol}]",
                    )
                current_close = bars[index].close
                future_close = bars[index + horizon].close
                future_return = (future_close / current_close) - 1.0
                if future_return > label_epsilon:
                    label = 2
                    direction_name = "long"
                elif future_return < -label_epsilon:
                    label = 0
                    direction_name = "short"
                else:
                    label = 1
                    direction_name = "flat"
                self.samples.append(
                    TrainingSample(
                        features=snapshot.features,
                        branch_inputs=dict(snapshot.branch_inputs),
                        regime_input=snapshot.regime_input,
                        temporal_tensor=snapshot.temporal_tensor,
                        label=torch.tensor(label, dtype=torch.long),
                        symbol=symbol,
                        timestamp=snapshot.timestamp,
                        direction_name=direction_name,
                        future_returns={horizon: float(future_return)},
                    )
                )
        if self.feature_contract is None:
            raise ValueError("No training samples were generated; increase history length or reduce warmup")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrainingSample:
        return self.samples[index]


def collate_training_samples(batch: list[TrainingSample]) -> dict[str, Any]:
    branch_names = batch[0].branch_inputs.keys()
    max_time = max(sample.temporal_tensor.shape[-1] for sample in batch)
    temporal_tensor = torch.stack(
        [
            F.pad(sample.temporal_tensor, (max_time - sample.temporal_tensor.shape[-1], 0))
            for sample in batch
        ],
        dim=0,
    )
    payload: dict[str, Any] = {
        "features": torch.stack([sample.features for sample in batch], dim=0),
        "branch_inputs": {
            name: torch.stack([sample.branch_inputs[name] for sample in batch], dim=0)
            for name in branch_names
        },
        "regime_input": torch.stack([sample.regime_input for sample in batch], dim=0),
        "temporal_tensor": temporal_tensor,
        "label": torch.stack([sample.label for sample in batch], dim=0),
        "symbol": [sample.symbol for sample in batch],
        "timestamp": [sample.timestamp for sample in batch],
        "direction_name": [sample.direction_name for sample in batch],
        "path_name": [sample.path_name for sample in batch],
        "future_returns": [dict(sample.future_returns) for sample in batch],
    }
    if any(sample.path_label is not None for sample in batch):
        payload["path_label"] = torch.stack(
            [
                sample.path_label if sample.path_label is not None else torch.tensor(0, dtype=torch.long)
                for sample in batch
            ],
            dim=0,
        )
    if any(sample.regression_targets for sample in batch):
        target_names = sorted(
            {
                target_name
                for sample in batch
                for target_name in sample.regression_targets
            }
        )
        payload["regression_targets"] = {
            target_name: torch.tensor(
                [float(sample.regression_targets.get(target_name, 0.0)) for sample in batch],
                dtype=torch.float32,
            )
            for target_name in target_names
        }
    return payload


def count_nonfinite_columns(columns: "ColumnarSamples") -> dict[str, int]:
    """Per-field count of non-finite (NaN/±inf) values in the tensor columns.

    The collate path sanitizes these to 0.0 via ``nan_to_num``; this makes
    the sanitation observable instead of silent (§4.7) — a large count means
    the feature pipeline is emitting garbage that training would otherwise
    quietly learn as zeros.
    """
    counts: dict[str, int] = {}

    def _add(name: str, tensor: torch.Tensor) -> None:
        if tensor.is_floating_point():
            bad = int((~torch.isfinite(tensor)).sum().item())
            if bad:
                counts[name] = bad

    _add("features", columns.features)
    for name, tensor in columns.branch_inputs.items():
        _add(f"branch_inputs.{name}", tensor)
    _add("regime_input", columns.regime_input)
    _add("temporal_tensor", columns.temporal_tensor)
    for name, tensor in columns.regression_targets.items():
        _add(f"regression_targets.{name}", tensor)
    return counts


def make_columnar_collate(columns: "ColumnarSamples"):
    """Create a collate function that slices columnar tensors directly.

    Bypasses per-sample ``TrainingSample`` creation and ``torch.stack``.
    Uses advanced indexing on pre-stacked tensors: O(1) per field instead
    of O(batch_size).

    The returned callable accepts a list of integer indices (as produced by
    an ``IndexDataset``) and returns a dict with the same structure as
    ``collate_training_samples``.
    """
    # Counted once at loader construction (not per batch — the columns are
    # immutable), so surfacing costs nothing on the hot path.
    nonfinite_counts = count_nonfinite_columns(columns)
    if nonfinite_counts:
        logging.getLogger(__name__).warning(
            "collate_nonfinite_inputs sanitized_to_zero=%s", nonfinite_counts
        )

    def collate(indices: list[int]) -> dict[str, Any]:
        idx = torch.tensor(indices, dtype=torch.long,
                           device=columns.features.device)

        def _safe(t: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(t)

        payload: dict[str, Any] = {
            "features": _safe(columns.features[idx]),
            "branch_inputs": {
                name: _safe(tensor[idx]) for name, tensor in columns.branch_inputs.items()
            },
            "regime_input": _safe(columns.regime_input[idx]),
            "temporal_tensor": _safe(columns.temporal_tensor[idx]),
            "label": columns.label[idx],
        }

        # Metadata (lists -- cheap list comprehension)
        idx_list = idx.tolist()
        payload["symbol"] = [columns.symbols[i] for i in idx_list]
        payload["timestamp"] = [columns.timestamps[i] for i in idx_list]
        payload["direction_name"] = [columns.direction_names[i] for i in idx_list]
        payload["path_name"] = [columns.path_names[i] for i in idx_list]

        # Path label -- optional tensor column
        if columns.path_label is not None:
            payload["path_label"] = columns.path_label[idx]

        # Regression targets -- already stored as {name: Tensor[N]}
        if columns.regression_targets:
            payload["regression_targets"] = {
                name: torch.nan_to_num(tensor[idx])
                for name, tensor in columns.regression_targets.items()
            }

        # Future returns -- match old collate format: list[dict[int, float]]
        if columns.future_returns:
            horizons = list(columns.future_returns.keys())
            payload["future_returns"] = [
                {h: float(columns.future_returns[h][i].item()) for h in horizons}
                for i in idx_list
            ]
        else:
            payload["future_returns"] = [{} for _ in idx_list]

        return payload

    return collate
