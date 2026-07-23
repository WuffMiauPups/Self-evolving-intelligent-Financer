from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalEncoderOutput:
    embedding: torch.Tensor


class TemporalResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, dropout: float, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(output_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv1d(input_channels, output_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = self.dropout(hidden)
        return self.activation(hidden + residual)


class MarketTemporalCNN(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        channels: list[int],
        kernel_sizes: list[int],
        temporal_blocks: int,
        dropout: float,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if not channels:
            raise ValueError("channels must not be empty")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must not be empty")
        self.input_channels = input_channels
        self.channels = list(channels)
        self.kernel_sizes = list(kernel_sizes)
        self.temporal_blocks = max(temporal_blocks, 1)
        self.dilations = list(dilations) if dilations else [1] * self.temporal_blocks

        blocks: list[nn.Module] = []
        in_channels = input_channels
        for block_index in range(self.temporal_blocks):
            out_channels = self.channels[min(block_index, len(self.channels) - 1)]
            kernel_size = self.kernel_sizes[block_index % len(self.kernel_sizes)]
            dilation = self.dilations[block_index % len(self.dilations)]
            blocks.append(
                TemporalResidualBlock(
                    input_channels=in_channels,
                    output_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dilation=dilation,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = in_channels
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim),
            nn.GELU(),
        )

    def forward(self, temporal_tensor: torch.Tensor) -> TemporalEncoderOutput:
        if temporal_tensor.dim() == 2:
            temporal_tensor = temporal_tensor.unsqueeze(0)
        if temporal_tensor.dim() != 3:
            raise ValueError(
                f"Expected temporal_tensor shape (batch, channels, time), got {tuple(temporal_tensor.shape)}"
            )
        if temporal_tensor.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected temporal input channels {self.input_channels}, got {temporal_tensor.shape[1]}"
            )
        activations = self.blocks(temporal_tensor)
        pooled = self.pool(activations)
        embedding = self.projection(pooled)
        return TemporalEncoderOutput(embedding=embedding)
