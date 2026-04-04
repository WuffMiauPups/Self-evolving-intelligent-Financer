from __future__ import annotations

from dataclasses import dataclass

import torch

from src.training.replay_buffer import ReplayBuffer


@dataclass
class RLBatch:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


def compute_bootstrap_targets(
    rewards: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    return rewards + gamma * next_values * (1.0 - dones)


class RLTrainerScaffold:
    def __init__(self, replay_buffer: ReplayBuffer, gamma: float = 0.99) -> None:
        self.replay_buffer = replay_buffer
        self.gamma = gamma

    def build_batch(self, batch_size: int, *, device: torch.device | None = None) -> RLBatch:
        batch = self.replay_buffer.sample(batch_size, device=device)
        return RLBatch(
            observation=batch["observation"],
            action=batch["action"],
            reward=batch["reward"],
            next_observation=batch["next_observation"],
            done=batch["done"],
        )

