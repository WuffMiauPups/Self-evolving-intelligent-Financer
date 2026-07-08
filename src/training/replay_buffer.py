from __future__ import annotations

# STATUS: RESERVED — not wired into any active code path.
# This module is scaffolding for future capability. It is tested but not
# invoked by the training pipeline, runtime, or CLI entry points.

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Experience:
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._items: deque[Experience] = deque(maxlen=capacity)

    def push(self, experience: Experience) -> None:
        self._items.append(experience)

    def __len__(self) -> int:
        return len(self._items)

    def sample(self, batch_size: int, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
        if batch_size > len(self._items):
            raise ValueError("Not enough items in replay buffer")
        indices = torch.randperm(len(self._items))[:batch_size].tolist()
        items = [self._items[index] for index in indices]
        batch = {
            "observation": torch.stack([item.observation for item in items]),
            "action": torch.stack([item.action for item in items]),
            "reward": torch.stack([item.reward for item in items]),
            "next_observation": torch.stack([item.next_observation for item in items]),
            "done": torch.stack([item.done for item in items]),
        }
        if device is not None:
            batch = {key: value.to(device) for key, value in batch.items()}
        return batch

