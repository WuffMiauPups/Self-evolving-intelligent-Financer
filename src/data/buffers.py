from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from src.data.schemas import MarketBar


class BarBuffer:
    def __init__(self, maxlen: int) -> None:
        self._bars: deque[MarketBar] = deque(maxlen=maxlen)

    def append(self, bar: MarketBar) -> None:
        self._bars.append(bar)

    def extend(self, bars: Iterable[MarketBar]) -> None:
        self._bars.extend(bars)

    def clear(self) -> None:
        self._bars.clear()

    def snapshot(self) -> list[MarketBar]:
        return list(self._bars)

    def __len__(self) -> int:
        return len(self._bars)

