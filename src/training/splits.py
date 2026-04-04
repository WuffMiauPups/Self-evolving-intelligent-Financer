from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SymbolSplit:
    symbol: str
    split: SplitName
    start_index: int
    end_index: int
    start_timestamp: datetime | None
    end_timestamp: datetime | None
    sample_count: int


def build_symbol_splits(
    *,
    symbol: str,
    eligible_indices: list[int],
    timestamps: list[datetime],
    train_split: float,
    val_split: float,
    test_split: float,
) -> list[SymbolSplit]:
    del test_split
    sample_count = len(eligible_indices)
    if sample_count == 0:
        return [
            SymbolSplit(symbol=symbol, split="train", start_index=0, end_index=0, start_timestamp=None, end_timestamp=None, sample_count=0),
            SymbolSplit(symbol=symbol, split="val", start_index=0, end_index=0, start_timestamp=None, end_timestamp=None, sample_count=0),
            SymbolSplit(symbol=symbol, split="test", start_index=0, end_index=0, start_timestamp=None, end_timestamp=None, sample_count=0),
        ]
    train_count = int(sample_count * train_split)
    val_count = int(sample_count * val_split)
    train_count = min(max(train_count, 1), sample_count)
    remaining = sample_count - train_count
    if remaining <= 1:
        val_count = max(0, remaining)
    else:
        val_count = min(max(val_count, 1), remaining - 1)
    test_count = max(sample_count - train_count - val_count, 0)
    boundaries = {
        "train": (0, train_count),
        "val": (train_count, train_count + val_count),
        "test": (train_count + val_count, train_count + val_count + test_count),
    }
    splits: list[SymbolSplit] = []
    for split_name, (left, right) in boundaries.items():
        left_index = eligible_indices[left] if left < sample_count else 0
        right_index = eligible_indices[right - 1] if right > left and right - 1 < sample_count else left_index
        split_timestamps = timestamps[left:right]
        splits.append(
            SymbolSplit(
                symbol=symbol,
                split=split_name,  # type: ignore[arg-type]
                start_index=left_index,
                end_index=right_index,
                start_timestamp=split_timestamps[0] if split_timestamps else None,
                end_timestamp=split_timestamps[-1] if split_timestamps else None,
                sample_count=max(right - left, 0),
            )
        )
    return splits
