from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import torch

from src.data.schemas import MarketBar


class BarBuffer:
    """Rolling per-symbol bar buffer with a parallel tensor backing.

    The runtime hot path used to call ``snapshot()`` once per symbol per
    tick to get a Python list of ``MarketBar`` instances, then handed
    that list to ``bars_to_tensor`` inside the feature pipeline.  At
    50 symbols × 200 bars per buffer that is ~10K Python attribute
    lookups per tick just to materialize the OHLCV tensor — work that
    has to repeat every tick because the bar list is rebuilt from
    scratch each call.

    The buffer now keeps a preallocated ``(maxlen, 6)`` float32 tensor
    in lock-step with the deque.  ``snapshot_tensor()`` returns a view
    over the live rows in O(1), and the feature pipeline can consume
    that view directly via ``FeaturePipeline.build_from_tensor`` without
    paying for any per-bar Python work.  ``snapshot()`` still returns
    a list of ``MarketBar`` for callers (tests, build_memory CLI) that
    care about the structured records.
    """

    _COLS = 6  # [open, high, low, close, volume, spread]

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("BarBuffer maxlen must be positive")
        self.maxlen = maxlen
        self._bars: deque[MarketBar] = deque(maxlen=maxlen)
        # Preallocated tensor backing.  We keep the most recent bar at
        # row index ``len(self._bars) - 1`` and shift left when the
        # buffer is full.  Shifting is O(maxlen) but only happens at
        # max capacity; appends below capacity are O(1).
        self._tensor: torch.Tensor = torch.zeros(
            (maxlen, self._COLS), dtype=torch.float32
        )
        self._count: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def append(self, bar: MarketBar) -> None:
        if self._count >= self.maxlen:
            # Shift left in place to drop the oldest row.  ``clone`` is
            # required because torch tensor slice assignment with
            # overlapping memory is undefined.
            self._tensor[:-1] = self._tensor[1:].clone()
            target_row = self.maxlen - 1
        else:
            target_row = self._count
            self._count += 1
        self._tensor[target_row, 0] = bar.open
        self._tensor[target_row, 1] = bar.high
        self._tensor[target_row, 2] = bar.low
        self._tensor[target_row, 3] = bar.close
        self._tensor[target_row, 4] = bar.volume
        self._tensor[target_row, 5] = bar.spread
        self._bars.append(bar)

    def extend(self, bars: Iterable[MarketBar]) -> None:
        for bar in bars:
            self.append(bar)

    def clear(self) -> None:
        self._bars.clear()
        self._count = 0
        self._tensor.zero_()

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------
    def snapshot(self) -> list[MarketBar]:
        """Return a fresh list of all bars currently in the buffer.

        Kept for backward compatibility with callers that need the
        full ``MarketBar`` records (build_memory, tests, snapshots that
        leave the runtime path).  Runtime hot paths should use
        ``snapshot_tensor`` + ``last_bar`` instead.
        """
        return list(self._bars)

    def snapshot_tensor(self) -> torch.Tensor:
        """Return a ``(count, 6)`` view of the live rows in chronological order.

        Returns a slice into the preallocated tensor — no allocation,
        no Python loop, no per-bar attribute lookup.  The returned
        tensor shares storage with the buffer; callers must not mutate
        it.  Subsequent ``append`` calls update the underlying tensor
        in place, so callers should not retain the view across writes
        unless they ``clone()`` first.
        """
        if self._count == 0:
            return self._tensor[:0]
        return self._tensor[: self._count]

    def last_bar(self) -> MarketBar:
        """Return the most recently appended bar (for symbol/timestamp metadata)."""
        return self._bars[-1]

    def __len__(self) -> int:
        return self._count
