"""Rebuild logs/.open_watch_seed = the START-OF-DAY position book.

watch_opens.py replays the day's fills on top of this seed; without it a
sell that trims/closes a carried position is misclassified as a fresh
short (nine false alerts on 2026-07-21). Run BEFORE the trading day's
first fill: at that point broker positions minus today's net fills equals
broker positions, so the seed is simply the current broker book.

Safe to run intraday too: it subtracts the day's net fills first.

Usage:
    python scripts/make_watch_seed.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FILLS = Path("runs/fills.jsonl")
SEED = Path("logs/.open_watch_seed")


def main() -> int:
    from src.core.config import load_config

    load_config("configs/default.yaml")  # loads .env for broker credentials

    from src.data.alpaca_client import AlpacaClient

    client = AlpacaClient(paper=True)
    broker = {p.symbol: float(p.qty) for p in client.trading_client.get_all_positions()}

    net_today: dict[str, float] = defaultdict(float)
    today = datetime.now(UTC).date().isoformat()
    if FILLS.exists():
        for line in FILLS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("timestamp", "")).startswith(today):
                sign = 1.0 if row["side"] == "buy" else -1.0
                net_today[row["symbol"]] += float(row["quantity"]) * sign

    seed = {}
    for symbol in set(broker) | set(net_today):
        qty = broker.get(symbol, 0.0) - net_today.get(symbol, 0.0)
        if abs(qty) > 1e-9:
            seed[symbol] = qty

    SEED.parent.mkdir(parents=True, exist_ok=True)
    SEED.write_text(json.dumps(seed), encoding="utf-8")
    print(f"seed written ({len(seed)} carried positions):")
    for symbol in sorted(seed):
        print(f"  {symbol:<6} {seed[symbol]:+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
