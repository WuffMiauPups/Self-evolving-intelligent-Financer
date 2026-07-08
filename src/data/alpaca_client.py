from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from src.core.exceptions import DataUnavailableError, ExecutionError
from src.core.time_utils import utc_now
from src.data.schemas import MarketBar


def credentials_available() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_API_SECRET"))


def _timeframe_from_string(timeframe: str) -> Any:
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        amount = int(normalized[:-3])
        return TimeFrame(amount, TimeFrameUnit.Minute)
    if normalized.endswith("hour"):
        amount = int(normalized[:-4])
        return TimeFrame(amount, TimeFrameUnit.Hour)
    if normalized.endswith("day"):
        amount = int(normalized[:-3])
        return TimeFrame(amount, TimeFrameUnit.Day)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _historical_lookback_delta(
    timeframe: str,
    limit: int,
    *,
    bars_per_trading_day: int = 390,
    min_lookback_days_minute: int = 3,
    min_lookback_days_hourly: int = 7,
    min_lookback_multiplier_daily: int = 3,
    min_lookback_days_daily: int = 30,
) -> timedelta:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        trading_days = max(min_lookback_days_minute, (limit // bars_per_trading_day) + 2)
        return timedelta(days=trading_days)
    if normalized.endswith("hour"):
        lookback_days = max(5, (limit // min_lookback_days_hourly) + 3)
        return timedelta(days=lookback_days)
    if normalized.endswith("day"):
        return timedelta(days=max(limit * min_lookback_multiplier_daily, min_lookback_days_daily))
    return timedelta(days=5)


def _normalize_order_status(status: object) -> str:
    return str(status).split(".")[-1].lower()


def _is_terminal_order_status(status: str) -> bool:
    return status in {"filled", "canceled", "cancelled", "expired", "rejected"}


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _safe_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return utc_now()


class BrokerDataClient(Protocol):
    paper: bool
    data_feed: str

    def get_account(self) -> dict[str, Any]:
        ...

    def get_clock(self) -> dict[str, Any] | None:
        ...

    def get_asset(self, symbol: str) -> dict[str, Any] | None:
        ...

    def get_open_orders(self) -> list[dict[str, Any]]:
        ...

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        ...

    def get_positions(self) -> list[dict[str, Any]]:
        ...

    def cancel_order(self, order_id: str) -> bool:
        ...

    def get_historical_bars(self, symbol: str, timeframe: str, limit: int) -> list[MarketBar]:
        ...

    def get_historical_bars_range(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        ...

    def get_latest_bar(self, symbol: str, timeframe: str) -> MarketBar:
        ...

    def get_latest_bars(
        self, symbols: list[str], timeframe: str
    ) -> tuple[dict[str, MarketBar], dict[str, str]]:
        """Batch variant of ``get_latest_bar``.

        Returns a ``(bars, errors)`` tuple where ``bars`` maps each
        symbol that succeeded to its latest ``MarketBar`` and ``errors``
        maps each symbol that failed to a short error string.  Clients
        may implement this as a single batched API call (Alpaca
        supports ``symbol_or_symbols=[...]`` natively) or fall back to
        a per-symbol loop.
        """
        ...

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        ...


class AlpacaClient:
    def __init__(self, *, paper: bool = True, data_feed: str = "iex") -> None:
        if not credentials_available():
            raise DataUnavailableError("Alpaca credentials are not configured in the environment")
        self.paper = paper
        self.data_feed = data_feed
        self._trading_client = None
        self._data_client = None

    @property
    def trading_client(self) -> Any:
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient

            self._trading_client = TradingClient(
                os.environ["ALPACA_API_KEY"],
                os.environ["ALPACA_API_SECRET"],
                paper=self.paper,
            )
        return self._trading_client

    @property
    def data_client(self) -> Any:
        if self._data_client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._data_client = StockHistoricalDataClient(
                os.environ["ALPACA_API_KEY"],
                os.environ["ALPACA_API_SECRET"],
            )
        return self._data_client

    def get_account(self) -> dict[str, Any]:
        account = self.trading_client.get_account()
        return {
            "status": str(getattr(account, "status", "")),
            "buying_power": _safe_float(getattr(account, "buying_power", 0.0)),
            "cash": _safe_float(getattr(account, "cash", 0.0)),
            "equity": _safe_float(getattr(account, "equity", 0.0)),
            "last_equity": _safe_float(getattr(account, "last_equity", 0.0)),
            "portfolio_value": _safe_float(getattr(account, "portfolio_value", 0.0)),
            "trading_blocked": bool(getattr(account, "trading_blocked", False)),
            "account_blocked": bool(getattr(account, "account_blocked", False)),
            "trade_suspended_by_user": bool(getattr(account, "trade_suspended_by_user", False)),
            "shorting_enabled": bool(getattr(account, "shorting_enabled", True)),
        }

    def get_clock(self) -> dict[str, Any] | None:
        clock = self.trading_client.get_clock()
        return {
            "is_open": bool(getattr(clock, "is_open", False)),
            "timestamp": _safe_datetime(getattr(clock, "timestamp", utc_now())),
            "next_open": _safe_datetime(getattr(clock, "next_open", utc_now())),
            "next_close": _safe_datetime(getattr(clock, "next_close", utc_now())),
        }

    def get_asset(self, symbol: str) -> dict[str, Any] | None:
        asset = self.trading_client.get_asset(symbol)
        if asset is None:
            return None
        return {
            "symbol": str(getattr(asset, "symbol", symbol)),
            "status": str(getattr(asset, "status", "")),
            "tradable": bool(getattr(asset, "tradable", False)),
            "shortable": bool(getattr(asset, "shortable", False)),
            "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
            "fractionable": bool(getattr(asset, "fractionable", False)),
            "asset_class": str(getattr(asset, "asset_class", "")),
            "exchange": str(getattr(asset, "exchange", "")),
        }

    def get_open_orders(self) -> list[dict[str, Any]]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        orders = self.trading_client.get_orders(
            filter=GetOrdersRequest(status=QueryOrderStatus.OPEN)
        )
        return [self._normalize_order(order) for order in orders]

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        # Alpaca's dedicated endpoint covers closed orders too, so an
        # order that filled between polls (and thus vanished from the
        # open-orders list) is still resolvable with its fill price.
        try:
            order = self.trading_client.get_order_by_client_id(client_order_id)
            if order is not None:
                return self._normalize_order(order)
        except Exception:
            pass
        try:
            orders = self.get_open_orders()
        except Exception:
            orders = []
        for order in orders:
            if order["client_order_id"] == client_order_id:
                return order
        return None

    def get_positions(self) -> list[dict[str, Any]]:
        positions = self.trading_client.get_all_positions()
        return [
            {
                "symbol": str(getattr(position, "symbol", "")),
                "qty": _safe_float(getattr(position, "qty", 0.0)),
                "avg_entry_price": _safe_float(getattr(position, "avg_entry_price", 0.0)),
                "current_price": _safe_float(getattr(position, "current_price", 0.0)),
                "market_value": _safe_float(getattr(position, "market_value", 0.0)),
                "side": str(getattr(position, "side", "")),
            }
            for position in positions
        ]

    def cancel_order(self, order_id: str) -> bool:
        self.trading_client.cancel_order_by_id(order_id)
        return True

    def get_historical_bars(self, symbol: str, timeframe: str, limit: int) -> list[MarketBar]:
        from alpaca.data.requests import StockBarsRequest

        end = datetime.now(UTC)
        start = end - _historical_lookback_delta(timeframe, limit)
        return self.get_historical_bars_range(symbol=symbol, timeframe=timeframe, start=start, end=end)[-limit:]

    def get_historical_bars_range(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        from alpaca.data.requests import StockBarsRequest

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=_timeframe_from_string(timeframe),
            start=start,
            end=end,
            feed=self.data_feed,
        )
        bars = self.data_client.get_stock_bars(request)
        series = bars.data.get(symbol, [])
        return [
            MarketBar(
                symbol=symbol,
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=float(bar.volume),
                spread=0.0,
            )
            for bar in series
        ]

    def get_latest_bar(self, symbol: str, timeframe: str) -> MarketBar:
        del timeframe
        from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest

        latest_bar = self.data_client.get_stock_latest_bar(
            StockLatestBarRequest(symbol_or_symbols=[symbol], feed=self.data_feed)
        )[symbol]
        latest_quote = self.data_client.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=[symbol], feed=self.data_feed)
        )[symbol]
        spread = max(float(latest_quote.ask_price) - float(latest_quote.bid_price), 0.0)
        return MarketBar(
            symbol=symbol,
            timestamp=latest_bar.timestamp,
            open=float(latest_bar.open),
            high=float(latest_bar.high),
            low=float(latest_bar.low),
            close=float(latest_bar.close),
            volume=float(latest_bar.volume),
            spread=spread,
        )

    def get_latest_bars(
        self, symbols: list[str], timeframe: str
    ) -> tuple[dict[str, MarketBar], dict[str, str]]:
        """Fetch the latest bar for many symbols in two SDK calls total.

        Alpaca's ``StockLatestBarRequest`` accepts ``symbol_or_symbols``
        as a list natively, so the whole universe is one HTTPS round-
        trip for the bars and one more for the quotes.  This turns what
        used to be ``N × (latency + serialization)`` (at 50 symbols,
        roughly 2.5–5s per tick against the IEX feed) into 2 round-
        trips total — a 25–50x improvement on broker-I/O wall time.

        Any symbol that fails (missing in the response, parse error,
        network hiccup) goes into the ``errors`` dict; successful ones
        go into the ``bars`` dict.  Individual symbol failures do NOT
        abort the whole batch.
        """
        del timeframe
        bars_out: dict[str, MarketBar] = {}
        errors_out: dict[str, str] = {}
        if not symbols:
            return bars_out, errors_out
        from alpaca.data.requests import StockLatestBarRequest, StockLatestQuoteRequest

        try:
            bar_response = self.data_client.get_stock_latest_bar(
                StockLatestBarRequest(symbol_or_symbols=list(symbols), feed=self.data_feed)
            )
        except Exception as exc:  # noqa: BLE001
            # Whole-batch failure: record the error for every symbol and
            # let the trade loop re-evaluate guards on the next tick.
            err = f"get_stock_latest_bar_failed: {exc}"
            return bars_out, {symbol: err for symbol in symbols}
        try:
            quote_response = self.data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=list(symbols), feed=self.data_feed)
            )
        except Exception as exc:  # noqa: BLE001
            # Quote failure alone should not hide the bars — we fall
            # through and default the spread to 0 for each symbol.
            quote_response = {}
            errors_out["_quote_batch"] = f"get_stock_latest_quote_failed: {exc}"
        for symbol in symbols:
            bar = bar_response.get(symbol)
            if bar is None:
                errors_out[symbol] = "no_latest_bar"
                continue
            quote = quote_response.get(symbol) if quote_response else None
            if quote is not None:
                try:
                    spread = max(float(quote.ask_price) - float(quote.bid_price), 0.0)
                except Exception:  # noqa: BLE001
                    spread = 0.0
            else:
                spread = 0.0
            try:
                bars_out[symbol] = MarketBar(
                    symbol=symbol,
                    timestamp=bar.timestamp,
                    open=float(bar.open),
                    high=float(bar.high),
                    low=float(bar.low),
                    close=float(bar.close),
                    volume=float(bar.volume),
                    spread=spread,
                )
            except Exception as exc:  # noqa: BLE001
                errors_out[symbol] = f"parse_failed: {exc}"
        return bars_out, errors_out

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        order = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide(side.lower()),
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        response = self.trading_client.submit_order(order)
        return self._normalize_order(response)

    @staticmethod
    def _normalize_order(order: Any) -> dict[str, Any]:
        status = _normalize_order_status(getattr(order, "status", "unknown"))
        qty = _safe_float(getattr(order, "qty", 0.0))
        filled_qty = _safe_float(getattr(order, "filled_qty", 0.0))
        remaining_qty = max(qty - filled_qty, 0.0)
        return {
            "id": str(getattr(order, "id", "")) or None,
            "client_order_id": str(getattr(order, "client_order_id", "")),
            "symbol": str(getattr(order, "symbol", "")),
            "qty": qty,
            "filled_qty": filled_qty,
            "remaining_qty": remaining_qty,
            "filled_avg_price": _safe_float(getattr(order, "filled_avg_price", 0.0)),
            "side": str(getattr(order, "side", "")),
            "status": status,
            "submitted_at": _safe_datetime(getattr(order, "submitted_at", utc_now())),
            "updated_at": _safe_datetime(getattr(order, "updated_at", utc_now())),
            "terminal": _is_terminal_order_status(status),
        }


@dataclass
class MockBrokerClient:
    seed_price: float = 100.0
    step: int = 0
    paper: bool = True
    data_feed: str = "mock"
    account_state: dict[str, Any] = field(
        default_factory=lambda: {
            "status": "ACTIVE",
            "buying_power": 100000.0,
            "cash": 100000.0,
            "equity": 100000.0,
            "last_equity": 100000.0,
            "portfolio_value": 100000.0,
            "trading_blocked": False,
            "account_blocked": False,
            "trade_suspended_by_user": False,
            "shorting_enabled": True,
        }
    )
    clock_state: dict[str, Any] = field(
        default_factory=lambda: {
            "is_open": True,
            "timestamp": utc_now(),
            "next_open": utc_now(),
            "next_close": utc_now() + timedelta(hours=6),
        }
    )
    assets: dict[str, dict[str, Any]] = field(default_factory=dict)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    submitted_orders: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_account(self) -> dict[str, Any]:
        return dict(self.account_state)

    def get_clock(self) -> dict[str, Any] | None:
        clock = dict(self.clock_state)
        clock["timestamp"] = utc_now()
        return clock

    def get_asset(self, symbol: str) -> dict[str, Any] | None:
        if symbol in self.assets:
            return dict(self.assets[symbol])
        return {
            "symbol": symbol,
            "status": "active",
            "tradable": True,
            "shortable": True,
            "easy_to_borrow": True,
            "fractionable": True,
            "asset_class": "us_equity",
            "exchange": "MOCK",
        }

    def get_open_orders(self) -> list[dict[str, Any]]:
        return [dict(order) for order in self.open_orders.values()]

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        if client_order_id in self.open_orders:
            return dict(self.open_orders[client_order_id])
        if client_order_id in self.submitted_orders:
            return dict(self.submitted_orders[client_order_id])
        return None

    def get_positions(self) -> list[dict[str, Any]]:
        return [dict(position) for position in self.positions.values()]

    def cancel_order(self, order_id: str) -> bool:
        for client_order_id, order in list(self.open_orders.items()):
            if order.get("id") == order_id:
                order["status"] = "canceled"
                order["terminal"] = True
                self.submitted_orders[client_order_id] = dict(order)
                del self.open_orders[client_order_id]
                return True
        return False

    def get_historical_bars(self, symbol: str, timeframe: str, limit: int) -> list[MarketBar]:
        del timeframe
        return self._generate_bars(symbol, limit)

    def get_historical_bars_range(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        return self._generate_bars_for_range(symbol, timeframe=timeframe, start=start, end=end)

    def get_latest_bar(self, symbol: str, timeframe: str) -> MarketBar:
        del timeframe
        self.step += 1
        return self._generate_bars(symbol, 1)[-1]

    def get_latest_bars(
        self, symbols: list[str], timeframe: str
    ) -> tuple[dict[str, MarketBar], dict[str, str]]:
        del timeframe
        bars_out: dict[str, MarketBar] = {}
        errors_out: dict[str, str] = {}
        for symbol in symbols:
            self.step += 1
            bars_out[symbol] = self._generate_bars(symbol, 1)[-1]
        return bars_out, errors_out

    def submit_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: str,
        client_order_id: str,
    ) -> dict[str, Any]:
        if qty <= 0:
            raise ExecutionError("Mock order qty must be positive")
        fill_price = max(self.seed_price + self.step * 0.1, 1.0)
        order = {
            "id": f"mock-{client_order_id}",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "qty": qty,
            "filled_qty": qty,
            "remaining_qty": 0.0,
            "filled_avg_price": fill_price,
            "side": side,
            "status": "filled",
            "submitted_at": utc_now(),
            "updated_at": utc_now(),
            "terminal": True,
        }
        self.submitted_orders[client_order_id] = dict(order)
        self._apply_mock_fill(symbol=symbol, qty=qty, side=side, fill_price=fill_price)
        return order

    def _apply_mock_fill(self, *, symbol: str, qty: float, side: str, fill_price: float) -> None:
        """Track the fill in the mock position book and account.

        Keeps mock rehearsals on the same lifecycle as the broker path:
        submits mutate positions/cash, so per-tick hydration and account
        sync exercise the real reconciliation flow instead of wiping the
        local book every tick.
        """
        signed = qty if side.lower() == "buy" else -qty
        existing = self.positions.get(symbol)
        old_qty = float(existing["qty"]) if existing else 0.0
        old_avg = float(existing["avg_entry_price"]) if existing else 0.0
        new_qty = old_qty + signed
        if abs(new_qty) < 1.0e-9:
            self.positions.pop(symbol, None)
        else:
            if existing is None or old_qty == 0 or old_qty * signed < 0:
                avg_price = fill_price if existing is None or old_qty * new_qty <= 0 else old_avg
            else:
                avg_price = ((abs(old_qty) * old_avg) + (abs(signed) * fill_price)) / (abs(old_qty) + abs(signed))
            self.positions[symbol] = {
                "symbol": symbol,
                "qty": new_qty,
                "avg_entry_price": avg_price,
                "current_price": fill_price,
                "market_value": new_qty * fill_price,
                "side": "long" if new_qty > 0 else "short",
            }
        self.account_state["cash"] = float(self.account_state.get("cash", 0.0)) - signed * fill_price
        notional = sum(float(p["qty"]) * float(p["current_price"]) for p in self.positions.values())
        self.account_state["equity"] = float(self.account_state["cash"]) + notional
        self.account_state["portfolio_value"] = self.account_state["equity"]

    def _generate_bars(self, symbol: str, count: int) -> list[MarketBar]:
        bars: list[MarketBar] = []
        current = datetime.now(UTC) - timedelta(minutes=max(count - 1, 0))
        price = self.seed_price + self.step * 0.1
        for idx in range(count):
            drift = 0.05 * ((self.step + idx) % 7 - 3)
            close = max(price + drift, 1.0)
            open_price = max(close - 0.1, 1.0)
            high = close + 0.15
            low = max(close - 0.2, 0.1)
            volume = 1_000 + ((self.step + idx) % 10) * 50
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=current + timedelta(minutes=idx),
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    spread=0.01,
                )
            )
            price = close
        self.step += count
        return bars

    def _generate_bars_for_range(
        self,
        symbol: str,
        *,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        normalized = timeframe.strip().lower()
        if normalized.endswith("min"):
            step_delta = timedelta(minutes=max(int(normalized[:-3]), 1))
        elif normalized.endswith("hour"):
            step_delta = timedelta(hours=max(int(normalized[:-4]), 1))
        elif normalized.endswith("day"):
            step_delta = timedelta(days=max(int(normalized[:-3]), 1))
        else:
            step_delta = timedelta(minutes=1)
        bars: list[MarketBar] = []
        current = start.astimezone(UTC) if start.tzinfo is not None else start.replace(tzinfo=UTC)
        end_utc = end.astimezone(UTC) if end.tzinfo is not None else end.replace(tzinfo=UTC)
        step_index = 0
        price = self.seed_price + self.step * 0.1
        while current <= end_utc:
            drift = 0.05 * ((self.step + step_index) % 7 - 3)
            close = max(price + drift, 1.0)
            open_price = max(close - 0.1, 1.0)
            high = close + 0.15
            low = max(close - 0.2, 0.1)
            volume = 1_000 + ((self.step + step_index) % 10) * 50
            bars.append(
                MarketBar(
                    symbol=symbol,
                    timestamp=current,
                    open=open_price,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    spread=0.01,
                )
            )
            price = close
            current += step_delta
            step_index += 1
        self.step += step_index
        return bars
