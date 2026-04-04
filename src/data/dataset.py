from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeatureContract, FeaturePipeline, FeatureSnapshot


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
                snapshot: FeatureSnapshot = pipeline.build(bars[: index + 1], symbol=symbol)
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


class HistoricalSequenceDataset(Dataset[TrainingSample]):
    def __init__(self, samples: list[TrainingSample], *, feature_contract: FeatureContract, split: str) -> None:
        self.samples = list(samples)
        self.feature_contract = feature_contract
        self.split = split

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
