"""OOS confidence-inversion test (follow-up to the live 2026-07-21 finding).

Live paper trades showed Spearman(confidence, realized pnl) = -0.428
(n=207) -- but that join used the confidence of the CLOSING decision
(fills carry the closing order's client_order_id). This script asks the
question on the OOS backtest window with a much larger n and separates:

  ENTRY confidence -- what arm 1 (Platt + threshold) would threshold, and
  CLOSE confidence -- what the live join actually measured.

The join is exact, not fuzzy: the engine logs decisions with bar_index and
next_open fills land at bar_index + 1, so a fill's decision row is
(symbol, fill_bar - 1) matched on action sign. Measurement only.

Usage:
    python scripts/test_confidence_inversion.py --real-data --out runs/confidence_inversion_oos.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scipy.stats import spearmanr

from src.core.config import load_config
from src.core.logging import configure_logging, get_logger

EPS = 1e-9


def stats_block(pairs: list[tuple[float, float]]) -> dict[str, object]:
    """rho/p, quartile win/mean-pnl table, winner-vs-loser mean confidence."""
    if len(pairs) < 12:
        return {"n": len(pairs), "note": "too few"}
    pairs = sorted(pairs, key=lambda x: x[0])
    conf = [p[0] for p in pairs]
    pnl = [p[1] for p in pairs]
    rho, pval = spearmanr(conf, pnl)
    q = len(pairs) // 4
    quartiles = []
    for i in range(4):
        chunk = pairs[i * q : (i + 1) * q] if i < 3 else pairs[3 * q :]
        cp = [x[1] for x in chunk]
        quartiles.append(
            {
                "conf_range": [round(chunk[0][0], 4), round(chunk[-1][0], 4)],
                "n": len(chunk),
                "win_rate": round(sum(1 for x in cp if x > 0) / len(cp), 4),
                "mean_pnl": round(sum(cp) / len(cp), 4),
                "total_pnl": round(sum(cp), 2),
            }
        )
    winners = [c for c, p in pairs if p > 0]
    losers = [c for c, p in pairs if p < 0]
    return {
        "n": len(pairs),
        "spearman_rho": round(float(rho), 4),
        "p_value": float(f"{pval:.3g}"),
        "quartiles": quartiles,
        "mean_conf_winners": round(sum(winners) / len(winners), 5) if winners else None,
        "mean_conf_losers": round(sum(losers) / len(losers), 5) if losers else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OOS confidence-inversion test")
    parser.add_argument("--real-data", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config = load_config("configs/default.yaml")
    configure_logging(
        config.logging.level,
        config.logging.json_logs,
        file_enabled=config.logging.file_enabled,
        file_path=config.logging.file_path,
    )
    logger = get_logger("confidence_inversion")
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if not config.model.checkpoint_path or not args.real_data:
        logger.warning("test_requires_checkpoint_and_real_data")
        return 2
    config.device.preferred = "cpu"

    import torch

    from src.backtest.engine import BacktestEngine
    from src.backtest.policy_factory import build_brain_policy
    from src.evaluation.oos import enforce_out_of_sample
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.training.data_loading import _resolve_bars_by_symbol

    bars_by_symbol, _ = _resolve_bars_by_symbol(config)
    bars_by_symbol, oos_meta = enforce_out_of_sample(
        bars_by_symbol, checkpoint_path=config.model.checkpoint_path
    )
    config = config.model_copy(deep=True)
    config.backtest.fill_timing = "next_open"
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    policy = build_brain_policy(config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    report = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars_by_symbol, policy
    )

    # --- Exact fill -> decision join ------------------------------------
    ts_index = {
        symbol: {bar.timestamp: i for i, bar in enumerate(bars)}
        for symbol, bars in bars_by_symbol.items()
    }
    decisions = defaultdict(list)
    for row in report.decision_log:
        decisions[(row["symbol"], int(row["bar_index"]))].append(row)

    def decision_conf(symbol, fill_ts, want_sign):
        """Confidence of the decision behind a fill (bar j-1, matched on sign).

        want_sign=+1/-1 requires that exact action; want_sign=None accepts the
        first row whose action is not the entry direction (close decisions can
        be flat, opposite, or a guard exit)."""
        j = ts_index[symbol].get(fill_ts)
        if j is None or j == 0:
            return None
        for row in decisions.get((symbol, j - 1), ()):
            if want_sign is not None and int(row["action"]) == want_sign:
                return float(row["confidence"])
            if want_sign is None:
                return float(row["confidence"])
        return None

    # --- Round-trip construction (replay; backtest book starts flat) ----
    position: dict[str, float] = defaultdict(float)
    open_rt: dict[str, dict] = {}
    round_trips: list[dict] = []
    unmatched_entries = 0
    for fill in report.fills:
        symbol = fill.symbol
        qty = float(fill.quantity) * (1.0 if fill.side == "buy" else -1.0)
        before = position[symbol]
        after = before + qty
        position[symbol] = after
        rt = open_rt.get(symbol)
        if rt is not None:
            rt["pnl"] += float(fill.realized_pnl)
            crossed = (before > EPS and after < -EPS) or (before < -EPS and after > EPS)
            if abs(after) <= EPS or crossed:
                rt["close_conf"] = decision_conf(symbol, fill.timestamp, None)
                round_trips.append(open_rt.pop(symbol))
                rt = None
        if rt is None and abs(after) > EPS and symbol not in open_rt:
            side = 1 if after > EPS else -1
            entry_conf = decision_conf(symbol, fill.timestamp, side)
            if entry_conf is None:
                unmatched_entries += 1
            open_rt[symbol] = {
                "symbol": symbol,
                "side": side,
                "entry_conf": entry_conf,
                "close_conf": None,
                "pnl": 0.0,
                "entry_ts": str(fill.timestamp),
            }

    entry_pairs = [(rt["entry_conf"], rt["pnl"]) for rt in round_trips if rt["entry_conf"] is not None]
    close_pairs = [(rt["close_conf"], rt["pnl"]) for rt in round_trips if rt["close_conf"] is not None]
    by_side = {
        name: [(rt["entry_conf"], rt["pnl"]) for rt in round_trips
               if rt["side"] == sign and rt["entry_conf"] is not None]
        for name, sign in (("long", 1), ("short", -1))
    }
    by_symbol = defaultdict(list)
    for rt in round_trips:
        if rt["entry_conf"] is not None:
            by_symbol[rt["symbol"]].append((rt["entry_conf"], rt["pnl"]))

    result = {
        "checkpoint": str(config.model.checkpoint_path),
        "oos_meta": {k: str(v) for k, v in (oos_meta or {}).items()},
        "round_trips": len(round_trips),
        "unclosed_at_end": len(open_rt),
        "unmatched_entries": unmatched_entries,
        "entry_confidence": stats_block(entry_pairs),
        "entry_confidence_by_side": {k: stats_block(v) for k, v in by_side.items()},
        "close_confidence": stats_block(close_pairs),
        "per_symbol_entry_rho": {
            symbol: {
                "n": len(pairs),
                "rho": round(float(spearmanr([p[0] for p in pairs], [p[1] for p in pairs])[0]), 4),
            }
            for symbol, pairs in sorted(by_symbol.items())
            if len(pairs) >= 12
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
