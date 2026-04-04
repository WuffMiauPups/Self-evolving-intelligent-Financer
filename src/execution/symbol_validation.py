from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.time_utils import utc_now
from src.data.alpaca_client import BrokerDataClient
from src.execution.reason_codes import ReasonCode


def _normalize_asset_field(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw.split(".")[-1].lower()


@dataclass(frozen=True)
class SymbolEligibilityResult:
    symbol: str
    eligible: bool
    reason_code: str
    checked_at: datetime
    details: dict[str, Any] = field(default_factory=dict)
    quarantined: bool = False
    release_mode: str | None = None


class SymbolValidationService:
    def __init__(self, client: BrokerDataClient, timeframe: str, *, using_mock: bool) -> None:
        self.client = client
        self.timeframe = timeframe
        self.using_mock = using_mock
        self._cache: dict[str, SymbolEligibilityResult] = {}

    def validate(self, symbol: str) -> SymbolEligibilityResult:
        if symbol in self._cache:
            return self._cache[symbol]

        checked_at = utc_now()
        if self.using_mock:
            result = SymbolEligibilityResult(
                symbol=symbol,
                eligible=True,
                reason_code=ReasonCode.MOCK_MODE_ACTIVE,
                checked_at=checked_at,
                details={"provider": "mock"},
            )
            self._cache[symbol] = result
            return result

        try:
            asset = self.client.get_asset(symbol)
        except Exception as exc:
            result = SymbolEligibilityResult(
                symbol=symbol,
                eligible=False,
                reason_code=ReasonCode.SYMBOL_UNKNOWN,
                checked_at=checked_at,
                details={"error": str(exc)},
                quarantined=True,
                release_mode="until_restart",
            )
            self._cache[symbol] = result
            return result

        if asset is None:
            result = SymbolEligibilityResult(
                symbol=symbol,
                eligible=False,
                reason_code=ReasonCode.SYMBOL_UNKNOWN,
                checked_at=checked_at,
                quarantined=True,
                release_mode="until_restart",
            )
            self._cache[symbol] = result
            return result

        status = _normalize_asset_field(asset.get("status", ""))
        asset_class = _normalize_asset_field(asset.get("asset_class", ""))
        details = dict(asset)
        if status != "active":
            reason_code = ReasonCode.SYMBOL_INACTIVE
        elif not bool(asset.get("tradable", False)):
            reason_code = ReasonCode.SYMBOL_NOT_TRADABLE
        elif asset_class not in {"us_equity", "stock"}:
            reason_code = ReasonCode.SYMBOL_ASSET_CLASS_UNSUPPORTED
        elif not bool(asset.get("shortable", False)):
            reason_code = ReasonCode.SYMBOL_NOT_SHORTABLE
        elif not bool(asset.get("fractionable", False)):
            reason_code = ReasonCode.SYMBOL_NOT_FRACTIONABLE
        else:
            try:
                bars = self.client.get_historical_bars(symbol, self.timeframe, 1)
            except Exception as exc:
                bars = []
                details["data_error"] = str(exc)
            reason_code = ReasonCode.STARTUP_OK if bars else ReasonCode.SYMBOL_NO_MARKET_DATA
            details["bar_count"] = len(bars)

        eligible = reason_code == ReasonCode.STARTUP_OK
        result = SymbolEligibilityResult(
            symbol=symbol,
            eligible=eligible,
            reason_code=reason_code,
            checked_at=checked_at,
            details=details,
            quarantined=not eligible,
            release_mode=None if eligible else "until_restart",
        )
        self._cache[symbol] = result
        return result

    def validate_all(self, symbols: list[str]) -> dict[str, SymbolEligibilityResult]:
        return {symbol: self.validate(symbol) for symbol in symbols}
