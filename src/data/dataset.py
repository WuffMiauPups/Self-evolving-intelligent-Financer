from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch

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
