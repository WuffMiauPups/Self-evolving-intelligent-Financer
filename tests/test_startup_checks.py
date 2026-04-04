from __future__ import annotations

from src.core.config import load_config
from src.data.alpaca_client import MockBrokerClient
from src.execution.reason_codes import ReasonCode
from src.execution.startup_checks import run_startup_checks


def test_startup_fails_without_explicit_live_confirmation() -> None:
    config = load_config("configs/default.yaml")
    config.broker.paper = False
    config.broker.live_enabled = True
    config.broker.confirm_live = False
    config.broker.base_url = "https://api.alpaca.markets"
    report = run_startup_checks(
        config=config,
        client=MockBrokerClient(paper=False),
        using_mock=False,
        confirm_live_cli=False,
        probe_symbol="AAPL",
    )
    assert report.safe_to_trade is False
    assert any(check.reason_code == ReasonCode.LIVE_CONFIRMATION_REQUIRED for check in report.checks)


def test_startup_fails_on_live_paper_mode_mismatch() -> None:
    config = load_config("configs/default.yaml")
    config.broker.paper = True
    config.broker.live_enabled = False
    config.broker.base_url = "https://api.alpaca.markets"
    report = run_startup_checks(
        config=config,
        client=MockBrokerClient(paper=True),
        using_mock=False,
        confirm_live_cli=False,
        probe_symbol="AAPL",
    )
    assert report.safe_to_trade is False
    assert any(check.reason_code == ReasonCode.ACCOUNT_MODE_MISMATCH for check in report.checks)


def test_startup_blocks_on_unhealthy_account_state() -> None:
    config = load_config("configs/default.yaml")
    client = MockBrokerClient()
    client.account_state["trading_blocked"] = True
    report = run_startup_checks(
        config=config,
        client=client,
        using_mock=False,
        confirm_live_cli=False,
        probe_symbol="AAPL",
    )
    assert report.safe_to_trade is False
    assert any(check.reason_code == ReasonCode.ACCOUNT_TRADING_BLOCKED for check in report.checks)


def test_startup_reports_mock_mode_explicitly() -> None:
    config = load_config("configs/default.yaml")
    report = run_startup_checks(
        config=config,
        client=MockBrokerClient(),
        using_mock=True,
        confirm_live_cli=False,
        probe_symbol="AAPL",
    )
    assert any(check.reason_code == ReasonCode.MOCK_MODE_ACTIVE for check in report.checks)


def test_startup_accepts_alpaca_enum_style_active_status() -> None:
    config = load_config("configs/default.yaml")
    client = MockBrokerClient()
    client.account_state["status"] = "AccountStatus.ACTIVE"
    report = run_startup_checks(
        config=config,
        client=client,
        using_mock=False,
        confirm_live_cli=False,
        probe_symbol="AAPL",
    )
    status_check = next(check for check in report.checks if check.name == "account_status")
    assert status_check.passed is True
    assert status_check.details["normalized_status"] == "active"
