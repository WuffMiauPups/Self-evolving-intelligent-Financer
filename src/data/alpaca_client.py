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


def _historical_lookback_delta(timeframe: str, limit: int) -> timedelta:
    normalized = timeframe.strip().lower()
    if normalized.endswith("min"):
        trading_days = max(3, (limit // 390) + 2)
        return timedelta(days=trading_days)
    if normalized.endswith("hour"):
        lookback_days = max(5, (limit // 7) + 3)
        return timedelta(days=lookback_days)
    if normalized.endswith("day"):
        return timedelta(days=max(limit * 3, 30))
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
        order = {
            "id": f"mock-{client_order_id}",
            "client_order_id": client_order_id,
            "symbol": symbol,
            "qty": qty,
            "filled_qty": qty,
            "remaining_qty": 0.0,
            "side": side,
            "status": "filled",
            "submitted_at": utc_now(),
            "updated_at": utc_now(),
            "terminal": True,
        }
        self.submitted_orders[client_order_id] = dict(order)
        return order

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
