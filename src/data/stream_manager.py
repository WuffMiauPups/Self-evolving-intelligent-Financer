from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.data.alpaca_client import BrokerDataClient
from src.data.schemas import MarketBar


@dataclass
class PollingStreamManager:
    client: BrokerDataClient
    symbols: list[str]
    timeframe: str
    poll_interval_sec: float
    last_errors: dict[str, str] = field(default_factory=dict)

    async def stream(self) -> AsyncIterator[dict[str, MarketBar]]:
        while True:
            self.last_errors = {}
            bars: dict[str, MarketBar] = {}
            for symbol in self.symbols:
                try:
                    bars[symbol] = self.client.get_latest_bar(symbol=symbol, timeframe=self.timeframe)
                except Exception as exc:
                    self.last_errors[symbol] = str(exc)
            if not bars and self.last_errors:
                raise RuntimeError(f"No market data available: {self.last_errors}")
            yield bars
            await asyncio.sleep(self.poll_interval_sec)
