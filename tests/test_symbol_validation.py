from __future__ import annotations

import pytest

from src.data.alpaca_client import MockBrokerClient
from src.execution.reason_codes import ReasonCode
from src.execution.symbol_validation import SymbolEligibilityResult, SymbolValidationService
from src.execution.runtime_bootstrap import resolve_runtime_symbols


def test_ineligible_symbols_are_skipped_by_default() -> None:
    client = MockBrokerClient(
        assets={
            "AAPL": {
                "symbol": "AAPL",
                "status": "active",
                "tradable": True,
                "shortable": True,
                "easy_to_borrow": True,
                "fractionable": True,
                "asset_class": "us_equity",
                "exchange": "MOCK",
            },
            "BAD": {
                "symbol": "BAD",
                "status": "inactive",
                "tradable": False,
                "shortable": False,
                "easy_to_borrow": False,
                "fractionable": False,
                "asset_class": "us_equity",
                "exchange": "MOCK",
            },
        }
    )
    service = SymbolValidationService(client, "1Min", using_mock=False)
    results = service.validate_all(["AAPL", "BAD"])
    eligible, ineligible = resolve_runtime_symbols(results, block_on_symbol_ineligible=False)
    assert eligible == ["AAPL"]
    assert "BAD" in ineligible
    assert ineligible["BAD"].reason_code == ReasonCode.SYMBOL_INACTIVE


def test_runtime_blocks_when_no_eligible_symbols_remain() -> None:
    results = {
        "BAD": SymbolEligibilityResult(
            symbol="BAD",
            eligible=False,
            reason_code=ReasonCode.SYMBOL_NOT_TRADABLE,
            checked_at=MockBrokerClient().get_clock()["timestamp"],
            quarantined=True,
            release_mode="until_restart",
        )
    }
    with pytest.raises(RuntimeError):
        resolve_runtime_symbols(results, block_on_symbol_ineligible=False)


def test_symbol_validation_accepts_alpaca_enum_style_asset_fields() -> None:
    client = MockBrokerClient(
        assets={
            "AAPL": {
                "symbol": "AAPL",
                "status": "AssetStatus.ACTIVE",
                "tradable": True,
                "shortable": True,
                "easy_to_borrow": True,
                "fractionable": True,
                "asset_class": "AssetClass.US_EQUITY",
                "exchange": "AssetExchange.NASDAQ",
            }
        }
    )
    service = SymbolValidationService(client, "1Min", using_mock=False)
    result = service.validate("AAPL")
    assert result.eligible is True
    assert result.reason_code == ReasonCode.STARTUP_OK
