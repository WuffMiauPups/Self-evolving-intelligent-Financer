"""Per-decision outcome event stream (MASTER_PLAN §4.5.1).

One JSONL line per proposal the policy makes at runtime: what the model
wanted, what the risk layer decided (including clamps), and what was
actually submitted.  Together with the persisted Ledger (realized
fills) this is the durable dataset the self-evolution loop learns from
— crucially it records the EXECUTED action, not just intent.

Best-effort append-only writer in the style of RuntimeStatusWriter:
failures never raise into the trade loop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_DECISION_LOG_PATH = Path("runs/decisions.jsonl")


@dataclass(frozen=True)
class DecisionEvent:
    ts: datetime
    symbol: str
    request_id: str
    proposed_action: int
    confidence: float
    target_fraction: float
    approved: bool
    reasons: list[str] = field(default_factory=list)
    executed_side: str | None = None
    executed_qty: float = 0.0
    resized: bool = False
    order_status: str | None = None
    client_order_id: str | None = None
    # EV-gate inputs (§7.0.4): the risk head's predicted MFE / |MAE| in bps
    # exactly as the EV gate consumed them.  Additive fields — old lines
    # simply lack the keys and ``read_all`` (raw dicts) stays tolerant.
    expected_gain_bps: float = 0.0
    expected_loss_bps: float = 0.0

    def to_json_line(self) -> str:
        payload = {
            "ts": self.ts.isoformat(),
            "symbol": self.symbol,
            "request_id": self.request_id,
            "proposed_action": int(self.proposed_action),
            "confidence": float(self.confidence),
            "target_fraction": float(self.target_fraction),
            "approved": bool(self.approved),
            "reasons": list(self.reasons),
            "executed_side": self.executed_side,
            "executed_qty": float(self.executed_qty),
            "resized": bool(self.resized),
            "order_status": self.order_status,
            "client_order_id": self.client_order_id,
            "expected_gain_bps": float(self.expected_gain_bps),
            "expected_loss_bps": float(self.expected_loss_bps),
        }
        return json.dumps(payload, separators=(",", ":"), default=str)


class DecisionLogWriter:
    """Append-only JSONL writer; failures never reach the trade loop."""

    def __init__(self, path: Path | str = DEFAULT_DECISION_LOG_PATH) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def write(self, event: DecisionEvent) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(event.to_json_line() + "\n")
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def read_all(path: Path | str = DEFAULT_DECISION_LOG_PATH) -> list[dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events
