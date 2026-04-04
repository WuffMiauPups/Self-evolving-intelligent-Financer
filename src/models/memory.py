from __future__ import annotations

import torch
from torch import nn


class LatentMemory(nn.Module):
    def __init__(self, input_dim: int, memory_dim: int) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.cell = nn.GRUCell(input_size=input_dim, hidden_size=memory_dim)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.memory_dim, device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor, memory_state: torch.Tensor | None = None) -> torch.Tensor:
        if memory_state is None:
            memory_state = self.initial_state(
                inputs.shape[0],
                device=inputs.device,
                dtype=inputs.dtype,
            )
        return self.cell(inputs, memory_state)
