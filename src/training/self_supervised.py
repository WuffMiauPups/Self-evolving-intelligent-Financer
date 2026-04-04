from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import TrainingSample
from src.training.losses import feature_prediction_loss
from src.utils.torch_utils import move_batch_to_device


@dataclass
class SelfSupervisedSample:
    current: TrainingSample
    target_features: torch.Tensor


class NextFeatureDataset(Dataset[SelfSupervisedSample]):
    def __init__(self, samples: list[TrainingSample]) -> None:
        self.items = [
            SelfSupervisedSample(current=samples[index], target_features=samples[index + 1].features)
            for index in range(len(samples) - 1)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> SelfSupervisedSample:
        return self.items[index]


def collate_self_supervised(batch: list[SelfSupervisedSample]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    branch_names = batch[0].current.branch_inputs.keys()
    return {
        "features": torch.stack([item.current.features for item in batch]),
        "branch_inputs": {
            name: torch.stack([item.current.branch_inputs[name] for item in batch])
            for name in branch_names
        },
        "regime_input": torch.stack([item.current.regime_input for item in batch]),
        "target_features": torch.stack([item.target_features for item in batch]),
    }


class NextFeatureHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


def run_self_supervised_epoch(
    *,
    model: nn.Module,
    projection_head: NextFeatureHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    projection_head.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            branch_inputs=batch["branch_inputs"],
            regime_input=batch["regime_input"],
            full_features=batch["features"],
        )
        prediction = projection_head(output.fused_representation)
        loss = feature_prediction_loss(prediction, batch["target_features"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)

