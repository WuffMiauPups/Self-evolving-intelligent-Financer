from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from src.data.alpaca_client import BrokerDataClient
from src.data.schemas import MarketBar

logger = logging.getLogger(__name__)


@dataclass
class PollingStreamManager:
    client: BrokerDataClient
    symbols: list[str]
    timeframe: str
    poll_interval_sec: float
    last_errors: dict[str, str] = field(default_factory=dict)
    consecutive_empty_polls: int = 0
    # Number of consecutive empty polls (every symbol failing) before we
    # *yield an empty dict* and let the trade loop re-evaluate health on the
    # next tick. The trade loop already gates everything on
    # ``health.safe_to_trade`` and per-symbol session/data guards, so an
    # empty poll is a routine transient condition (broker timeout, brief
    # outage, weekend gap), not grounds for permanently killing the loop.
    empty_poll_warn_every: int = 10
    # After this many consecutive all-symbol failures, yield a sentinel
    # so the trade loop can halt.  0 = disabled (never trip).
    circuit_breaker_threshold: int = 60

    async def stream(self) -> AsyncIterator[dict[str, MarketBar]]:
        while True:
            self.last_errors = {}
            bars: dict[str, MarketBar] = {}
            # Prefer the batch endpoint (``get_latest_bars``) when the
            # underlying client exposes one.  Alpaca's SDK fetches the
            # latest bar + quote for every symbol in a single request
            # pair, so the whole universe costs two HTTPS round-trips
            # instead of ``2 × N_symbols``.  At 50 symbols this collapses
            # per-tick broker-I/O wall time from several seconds to
            # roughly the network latency of one call.
            batch_fn = getattr(self.client, "get_latest_bars", None)
            batch_raised = False
            if callable(batch_fn):
                try:
                    bars, errors = batch_fn(list(self.symbols), self.timeframe)
                    self.last_errors = dict(errors)
                except Exception as exc:  # noqa: BLE001
                    # Catastrophic batch failure — fall through to the
                    # per-symbol loop so a broken batch implementation
                    # never strands the runtime without data.
                    logger.warning("stream_batch_poll_failed error=%s", exc)
                    batch_raised = True
                    bars = {}
            if (batch_raised or not callable(batch_fn)) and not bars:
                # No batch path, or the batch path raised: run the
                # legacy per-symbol loop.  Clear any ``_batch`` marker
                # error so downstream consumers only see real per-symbol
                # failures.
                self.last_errors = {}
                for symbol in self.symbols:
                    try:
                        bars[symbol] = self.client.get_latest_bar(
                            symbol=symbol, timeframe=self.timeframe
                        )
                    except Exception as exc:
                        self.last_errors[symbol] = str(exc)
            if not bars and self.last_errors:
                # All symbols failed in a single poll.  This used to raise
                # RuntimeError, which permanently killed the paper runtime
                # on the first transient broker glitch.  Instead, log the
                # condition (rate-limited) and yield an empty dict so the
                # trade loop can re-evaluate guards on the next tick.  The
                # trade loop's per-tick stream_error metrics already capture
                # individual symbol failures via ``self.last_errors``.
                self.consecutive_empty_polls += 1
                if (
                    self.consecutive_empty_polls == 1
                    or self.consecutive_empty_polls % self.empty_poll_warn_every == 0
                ):
                    logger.warning(
                        "stream_all_symbols_failed consecutive=%d errors=%s",
                        self.consecutive_empty_polls,
                        self.last_errors,
                    )
                if (
                    self.circuit_breaker_threshold > 0
                    and self.consecutive_empty_polls >= self.circuit_breaker_threshold
                ):
                    logger.error(
                        "stream_circuit_breaker_tripped consecutive=%d threshold=%d",
                        self.consecutive_empty_polls,
                        self.circuit_breaker_threshold,
                    )
                    return
            else:
                self.consecutive_empty_polls = 0
            yield bars
            await asyncio.sleep(self.poll_interval_sec)
