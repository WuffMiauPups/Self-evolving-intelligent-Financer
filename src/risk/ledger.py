from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FillRecord:
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    fee: float
    realized_pnl: float  # net of fees
    client_order_id: str

    def to_json_line(self) -> str:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict) -> FillRecord:
        return cls(
            timestamp=datetime.fromisoformat(str(payload["timestamp"])),
            symbol=str(payload["symbol"]),
            side=str(payload["side"]),
            quantity=float(payload["quantity"]),
            price=float(payload["price"]),
            fee=float(payload.get("fee", 0.0)),
            realized_pnl=float(payload.get("realized_pnl", 0.0)),
            client_order_id=str(payload.get("client_order_id", "")),
        )


@dataclass
class Ledger:
    """Fill history — the raw material of the self-evolution loop.

    With ``persist_path`` set, every fill is appended to a JSONL file as
    it is recorded, so realized outcomes survive restarts and can feed
    retraining/replay.  Persistence is best-effort: a disk error logs
    and never raises into the trading path.
    """

    fills: list[FillRecord] = field(default_factory=list)
    persist_path: str | Path | None = None

    def record(self, fill: FillRecord) -> None:
        self.fills.append(fill)
        if self.persist_path is None:
            return
        try:
            path = Path(self.persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fp:
                fp.write(fill.to_json_line() + "\n")
        except Exception as exc:  # noqa: BLE001 - persistence must not kill trading
            logger.warning("ledger_persist_failed path=%s error=%s", self.persist_path, exc)

    @classmethod
    def load(cls, path: str | Path, *, persist_path: str | Path | None = None) -> Ledger:
        """Rehydrate fills from a JSONL file (torn/corrupt lines skipped)."""
        ledger = cls(persist_path=persist_path if persist_path is not None else path)
        file_path = Path(path)
        if not file_path.exists():
            return ledger
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ledger.fills.append(FillRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return ledger

