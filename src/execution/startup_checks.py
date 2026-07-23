from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.config import AppConfig
from src.data.alpaca_client import BrokerDataClient, credentials_available
from src.execution.reason_codes import ReasonCode


def _normalize_account_status(status: object) -> str:
    raw = str(status or "").strip()
    if not raw:
        return ""
    return raw.split(".")[-1].lower()


@dataclass(frozen=True)
class StartupCheckResult:
    name: str
    passed: bool
    reason_code: str
    severity: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class StartupReport:
    safe_to_trade: bool
    mode: str
    using_mock: bool
    checks: list[StartupCheckResult]
    account: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "safe_to_trade": self.safe_to_trade,
            "mode": self.mode,
            "using_mock": self.using_mock,
            "check_count": len(self.checks),
            "failures": [check.reason_code for check in self.checks if not check.passed],
        }


def run_startup_checks(
    *,
    config: AppConfig,
    client: BrokerDataClient,
    using_mock: bool,
    confirm_live_cli: bool,
    probe_symbol: str | None,
) -> StartupReport:
    checks: list[StartupCheckResult] = []
    mode = "paper" if config.broker.paper else "live"
    account_snapshot: dict[str, Any] | None = None
    clock_snapshot: dict[str, Any] | None = None

    if config.broker.live_enabled:
        live_confirmed = bool(config.broker.confirm_live and confirm_live_cli)
        checks.append(
            StartupCheckResult(
                name="live_confirmation",
                passed=live_confirmed,
                reason_code=ReasonCode.STARTUP_OK if live_confirmed else ReasonCode.LIVE_CONFIRMATION_REQUIRED,
                severity="fatal" if not live_confirmed else "info",
                details={
                    "config_confirm_live": config.broker.confirm_live,
                    "cli_confirm_live": confirm_live_cli,
                },
            )
        )
    else:
        checks.append(
            StartupCheckResult(
                name="live_disabled",
                passed=True,
                reason_code=ReasonCode.LIVE_MODE_DISABLED,
                severity="info",
                details={"paper": config.broker.paper},
            )
        )

    if using_mock:
        checks.append(
            StartupCheckResult(
                name="credentials_or_mock",
                passed=config.broker.paper,
                reason_code=ReasonCode.MOCK_MODE_ACTIVE if config.broker.paper else ReasonCode.MISSING_CREDENTIALS,
                severity="info" if config.broker.paper else "fatal",
                details={"mock_mode": True},
            )
        )
    else:
        checks.append(
            StartupCheckResult(
                name="credentials_present",
                passed=credentials_available(),
                reason_code=ReasonCode.STARTUP_OK if credentials_available() else ReasonCode.MISSING_CREDENTIALS,
                severity="fatal" if not credentials_available() else "info",
                details={"mock_mode": False},
            )
        )

    expected_base_url = (config.broker.base_url or "").lower()
    if config.broker.require_account_match and expected_base_url:
        paper_fragment_present = "paper" in expected_base_url
        matches = paper_fragment_present if config.broker.paper else not paper_fragment_present
        checks.append(
            StartupCheckResult(
                name="endpoint_mode_match",
                passed=matches,
                reason_code=ReasonCode.STARTUP_OK if matches else ReasonCode.ACCOUNT_MODE_MISMATCH,
                severity="fatal" if not matches else "info",
                details={"base_url": config.broker.base_url, "paper": config.broker.paper},
            )
        )

    try:
        account_snapshot = client.get_account()
        checks.append(
            StartupCheckResult(
                name="broker_connectivity",
                passed=True,
                reason_code=ReasonCode.STARTUP_OK,
                severity="info",
                details={"provider": config.broker.provider, "mode": mode},
            )
        )
    except Exception as exc:
        checks.append(
            StartupCheckResult(
                name="broker_connectivity",
                passed=False,
                reason_code=ReasonCode.BROKER_CONNECTIVITY_FAILED,
                severity="fatal",
                details={"error": str(exc)},
            )
        )
        return StartupReport(
            safe_to_trade=False,
            mode=mode,
            using_mock=using_mock,
            checks=checks,
            account=None,
        )

    raw_account_status = account_snapshot.get("status", "")
    account_status = _normalize_account_status(raw_account_status)
    account_blocked = any(
        bool(account_snapshot.get(flag, False))
        for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
    )
    checks.append(
        StartupCheckResult(
            name="account_status",
            passed=account_status == "active" and not account_blocked,
            reason_code=ReasonCode.STARTUP_OK
            if account_status == "active" and not account_blocked
            else (ReasonCode.ACCOUNT_TRADING_BLOCKED if account_blocked else ReasonCode.ACCOUNT_STATUS_INVALID),
            severity="fatal" if account_status != "active" or account_blocked else "info",
            details={
                "status": raw_account_status,
                "normalized_status": account_status,
                "trading_blocked": account_snapshot.get("trading_blocked"),
                "account_blocked": account_snapshot.get("account_blocked"),
                "trade_suspended_by_user": account_snapshot.get("trade_suspended_by_user"),
            },
        )
    )

    buying_power = account_snapshot.get("buying_power")
    checks.append(
        StartupCheckResult(
            name="buying_power",
            passed=buying_power is not None and float(buying_power) >= 0.0,
            reason_code=ReasonCode.STARTUP_OK
            if buying_power is not None and float(buying_power) >= 0.0
            else ReasonCode.ACCOUNT_BUYING_POWER_UNAVAILABLE,
            severity="fatal" if buying_power is None or float(buying_power) < 0.0 else "info",
            details={"buying_power": buying_power},
        )
    )

    try:
        clock_snapshot = client.get_clock()
        checks.append(
            StartupCheckResult(
                name="market_clock",
                passed=True,
                reason_code=ReasonCode.STARTUP_OK,
                severity="info",
                details={"clock_available": clock_snapshot is not None},
            )
        )
    except Exception as exc:
        checks.append(
            StartupCheckResult(
                name="market_clock",
                passed=False,
                reason_code=ReasonCode.BROKER_CONNECTIVITY_FAILED,
                severity="block",
                details={"error": str(exc)},
            )
        )

    if probe_symbol is not None:
        try:
            bars = client.get_historical_bars(probe_symbol, config.runtime.timeframe, 1)
            checks.append(
                StartupCheckResult(
                    name="data_feed_probe",
                    passed=bool(bars),
                    reason_code=ReasonCode.STARTUP_OK if bars else ReasonCode.DATA_FEED_UNAVAILABLE,
                    severity="fatal" if not bars else "info",
                    details={"symbol": probe_symbol, "bar_count": len(bars)},
                )
            )
        except Exception as exc:
            checks.append(
                StartupCheckResult(
                    name="data_feed_probe",
                    passed=False,
                    reason_code=ReasonCode.DATA_FEED_UNAVAILABLE,
                    severity="fatal",
                    details={"symbol": probe_symbol, "error": str(exc)},
                )
            )

    safe_to_trade = all(check.passed or check.severity == "info" for check in checks)
    return StartupReport(
        safe_to_trade=safe_to_trade,
        mode=mode,
        using_mock=using_mock,
        checks=checks,
        account=account_snapshot,
    )
