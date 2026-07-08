"""Structured positions + account snapshot writer.

The trade loop writes a single JSON file at ``runs/positions_status.json``
every tick.  Unlike ``runtime_status.jsonl`` (append-only), this file is
**overwritten atomically** on every write — consumers always read the
latest snapshot and we never accumulate unbounded history on disk.

The GUI's ``PositionsTab`` reads this file every ~5 seconds to display
open positions, buying power, account drawdown, and per-position
drawdown without having to query the broker directly or hold a second
copy of the portfolio state in the GUI process.

Schema (one JSON object, top-level fields all required):

    {
        "ts":                 ISO-8601,
        "account": {
            "equity":              float,
            "cash":                float,
            "buying_power":        float,
            "portfolio_value":     float,
            "session_start_equity":float,
            "session_peak_equity": float,
            "session_return":      float,   # fraction vs session start
            "session_drawdown":    float,   # fraction below peak (<=0)
            "max_position_notional":     float,
            "max_concurrent_positions":  int,
            "min_trade_confidence":      float,
            "kill_switch":               bool,
        },
        "positions": [
            {
                "symbol":              str,
                "side":                "long" | "short",
                "qty":                 float,
                "avg_price":           float,
                "market_price":        float,
                "notional":            float,
                "unrealized_pnl":      float,
                "unrealized_pnl_pct":  float,   # vs entry notional
                "peak_unrealized_pnl": float,   # high-water mark since entry
                "drawdown":            float,   # current_pnl - peak_pnl (<=0)
                "drawdown_pct":        float,   # vs entry notional
                "entry_time":          ISO-8601 | null,
            },
            ...
        ],
    }

Design rules:
- Atomic overwrite (tmp file + rename) so the GUI can read safely.
- Best-effort: write failures NEVER raise into the trade loop.
- No Qt / no GUI / no service layer imports here.
- Stable schema: new fields are additive; readers tolerate missing keys.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_POSITIONS_STATUS_PATH = Path("runs/positions_status.json")


@dataclass
class AccountSnapshot:
    """Account-level state published to the Positions tab.

    All monetary fields are in the broker's quote currency (USD for
    Alpaca).  Drawdown / return are stored as fractions, not percentages,
    so the GUI can format them however it wants.
    """

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    session_start_equity: float
    session_peak_equity: float
    session_return: float
    session_drawdown: float
    max_position_notional: float
    max_concurrent_positions: int
    min_trade_confidence: float
    kill_switch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "equity": float(self.equity),
            "cash": float(self.cash),
            "buying_power": float(self.buying_power),
            "portfolio_value": float(self.portfolio_value),
            "session_start_equity": float(self.session_start_equity),
            "session_peak_equity": float(self.session_peak_equity),
            "session_return": float(self.session_return),
            "session_drawdown": float(self.session_drawdown),
            "max_position_notional": float(self.max_position_notional),
            "max_concurrent_positions": int(self.max_concurrent_positions),
            "min_trade_confidence": float(self.min_trade_confidence),
            "kill_switch": bool(self.kill_switch),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AccountSnapshot":
        return cls(
            equity=float(data.get("equity", 0.0)),
            cash=float(data.get("cash", 0.0)),
            buying_power=float(data.get("buying_power", 0.0)),
            portfolio_value=float(data.get("portfolio_value", 0.0)),
            session_start_equity=float(data.get("session_start_equity", 0.0)),
            session_peak_equity=float(data.get("session_peak_equity", 0.0)),
            session_return=float(data.get("session_return", 0.0)),
            session_drawdown=float(data.get("session_drawdown", 0.0)),
            max_position_notional=float(data.get("max_position_notional", 0.0)),
            max_concurrent_positions=int(data.get("max_concurrent_positions", 0)),
            min_trade_confidence=float(data.get("min_trade_confidence", 0.0)),
            kill_switch=bool(data.get("kill_switch", False)),
        )


@dataclass
class PositionSnapshot:
    """Per-symbol state for the Positions tab.

    The ``peak_unrealized_pnl`` + ``drawdown`` pair is computed by the
    writer (not the broker) — it tracks the high-water mark of unrealized
    PnL since the position was opened and the current drop from that
    peak.  The tracker resets when the position closes (qty crosses zero)
    or flips direction (long -> short).
    """

    symbol: str
    side: str
    qty: float
    avg_price: float
    market_price: float
    notional: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    peak_unrealized_pnl: float
    drawdown: float
    drawdown_pct: float
    entry_time: datetime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "qty": float(self.qty),
            "avg_price": float(self.avg_price),
            "market_price": float(self.market_price),
            "notional": float(self.notional),
            "unrealized_pnl": float(self.unrealized_pnl),
            "unrealized_pnl_pct": float(self.unrealized_pnl_pct),
            "peak_unrealized_pnl": float(self.peak_unrealized_pnl),
            "drawdown": float(self.drawdown),
            "drawdown_pct": float(self.drawdown_pct),
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PositionSnapshot":
        entry_raw = data.get("entry_time")
        entry_time: datetime | None = None
        if entry_raw:
            try:
                entry_time = datetime.fromisoformat(str(entry_raw))
            except ValueError:
                entry_time = None
        return cls(
            symbol=str(data.get("symbol", "")),
            side=str(data.get("side", "flat")),
            qty=float(data.get("qty", 0.0)),
            avg_price=float(data.get("avg_price", 0.0)),
            market_price=float(data.get("market_price", 0.0)),
            notional=float(data.get("notional", 0.0)),
            unrealized_pnl=float(data.get("unrealized_pnl", 0.0)),
            unrealized_pnl_pct=float(data.get("unrealized_pnl_pct", 0.0)),
            peak_unrealized_pnl=float(data.get("peak_unrealized_pnl", 0.0)),
            drawdown=float(data.get("drawdown", 0.0)),
            drawdown_pct=float(data.get("drawdown_pct", 0.0)),
            entry_time=entry_time,
        )


@dataclass
class PositionsStatusSnapshot:
    ts: datetime
    account: AccountSnapshot
    positions: list[PositionSnapshot] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ts": self.ts.isoformat(),
                "account": self.account.to_dict(),
                "positions": [p.to_dict() for p in self.positions],
            },
            separators=(",", ":"),
            default=str,
        )

    @classmethod
    def from_json(cls, text: str) -> "PositionsStatusSnapshot | None":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        try:
            ts = datetime.fromisoformat(str(data["ts"]))
        except (KeyError, ValueError):
            return None
        account_raw = data.get("account") or {}
        positions_raw = data.get("positions") or []
        if not isinstance(account_raw, dict) or not isinstance(positions_raw, list):
            return None
        account = AccountSnapshot.from_dict(account_raw)
        positions = [PositionSnapshot.from_dict(p) for p in positions_raw if isinstance(p, dict)]
        return cls(ts=ts, account=account, positions=positions)


class PositionsStatusWriter:
    """Atomic-overwrite writer.  Failures never raise into the trade loop.

    Uses a sibling ``.tmp`` file + os.replace so the reader always sees
    either the old snapshot or the new one, never a partially-written
    mix.  On Windows, os.replace is atomic for same-volume moves.
    """

    def __init__(self, path: Path | str = DEFAULT_POSITIONS_STATUS_PATH) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def write(self, snap: PositionsStatusSnapshot) -> None:
        try:
            payload = snap.to_json()
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fp:
                fp.write(payload)
            os.replace(tmp, self.path)
        except Exception:
            # Observability writer must never block trading.
            pass

    @classmethod
    def read_latest(
        cls, path: Path | str = DEFAULT_POSITIONS_STATUS_PATH
    ) -> PositionsStatusSnapshot | None:
        p = Path(path)
        if not p.exists():
            return None
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return None
        if not text.strip():
            return None
        return PositionsStatusSnapshot.from_json(text)


class PositionPeakTracker:
    """Per-symbol high-water-mark tracker for unrealized PnL.

    Used by the trade loop to compute per-position drawdown without
    forcing the portfolio dataclass to carry a runtime-only field.  The
    tracker is a pure in-memory dict keyed by symbol; it forgets entries
    when a position closes or flips direction so a re-entry starts with
    a fresh peak.
    """

    def __init__(self) -> None:
        self._state: dict[str, dict[str, float]] = {}

    def update(
        self,
        *,
        symbol: str,
        qty: float,
        unrealized_pnl: float,
        entry_epoch: float,
    ) -> tuple[float, float]:
        """Record the current ``unrealized_pnl`` and return
        ``(peak_pnl, drawdown)``.  ``drawdown`` is ``current - peak``,
        clamped to ``<= 0``.

        ``entry_epoch`` is used to detect re-entries: when it changes,
        the tracker resets.  Callers pass
        ``position.entry_time.timestamp()`` or ``0.0`` for flat.
        """
        if abs(qty) < 1.0e-9:
            # Flat position — clear any stale tracker.
            self._state.pop(symbol, None)
            return 0.0, 0.0
        entry_key = float(entry_epoch)
        state = self._state.get(symbol)
        if state is None or state["entry_epoch"] != entry_key:
            # New or re-entered position — seed the peak at the current PnL.
            state = {"entry_epoch": entry_key, "peak": float(unrealized_pnl)}
            self._state[symbol] = state
        elif unrealized_pnl > state["peak"]:
            state["peak"] = float(unrealized_pnl)
        peak = state["peak"]
        drawdown = min(0.0, float(unrealized_pnl) - peak)
        return peak, drawdown

    def forget(self, symbol: str) -> None:
        self._state.pop(symbol, None)
