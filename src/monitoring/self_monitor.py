"""Self-monitor: the bot's live feedback on its own trading (L0 scaffold).

Design: docs/analysis/SELF_MONITOR_DESIGN.md. Contract in one line:
measure live performance against the model's own stated expectations,
write the verdict down, and respond only by becoming MORE conservative
(shrink-only; never grow, flip, or override a veto).

Signal status (L0):
  S1 rolling win-rate surprise      IMPLEMENTED
  S2 live confidence calibration    IMPLEMENTED
  S3 confidence-distribution drift  IMPLEMENTED
  S4 churn (close->reopen cycles)   IMPLEMENTED
  S5 realized vs assumed cost       IMPLEMENTED (needs reference_price at the fill site)
  L1 damping multiplier             IMPLEMENTED, config-gated OFF
  live/backtest wiring              WIRED (tick_processor + backtest engine, gated off)

No I/O in the metric path; ReportWriter is separate and best-effort. The
config type lives in src/core/config/monitoring.py (AppConfig.monitoring.
self_monitor) and is re-exported here for the tests and the replay path.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from src.core.config.monitoring import SelfMonitorConfig

DEFAULT_REPORT_PATH = Path("runs/self_report.jsonl")

__all__ = ["SelfMonitor", "SelfMonitorConfig", "HealthReport", "ReportWriter"]


@dataclass(frozen=True)
class HealthReport:
    ts: str
    decisions: int
    resolved: int
    win_rate: float | None
    win_rate_z: float | None          # S1: binomial surprise vs baseline
    calibration: dict[str, dict]      # S2: bucket -> {n, claimed, realized}
    conf_p10: float | None            # S3: live confidence distribution
    conf_p90: float | None
    churn_cycles_per_hour: float      # S4
    realized_slippage_bps: float | None  # S5: median realized slippage
    flags: list[str]
    damping_multiplier: float         # 1.0 unless damping_enabled and unhealthy

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"), default=str)


class SelfMonitor:
    """Pure state machine. Feed decisions + fills; ask for health().

    Wiring (weekend): tick_processor calls record_decision where it writes
    the DecisionEvent and record_fill where the fill lands; the backtest
    engine makes the SAME two calls (shared path, 10.4).
    """

    def __init__(self, config: SelfMonitorConfig | None = None) -> None:
        self.config = config or SelfMonitorConfig()
        self.decision_count = 0
        self.confidences: deque[float] = deque(maxlen=500)
        # (confidence_at_entry, won) for resolved round trips
        self.outcomes: deque[tuple[float, bool]] = deque(maxlen=self.config.window)
        # per-symbol running position + last entry confidence + close times
        self._position: dict[str, float] = defaultdict(float)
        self._entry_conf: dict[str, float] = {}
        self._closes: deque[str] = deque(maxlen=200)      # close timestamps
        self._reopen_cycles: deque[str] = deque(maxlen=200)
        self._last_close: dict[str, str] = {}
        self._first_ts: str | None = None
        self._last_ts: str | None = None
        # client_order_id -> entry confidence, for the live outcome join
        # (a fill lands asynchronously, so it carries the order id, not the
        # decision's confidence). Backtest passes entry_confidence directly.
        self._conf_by_order: dict[str, float] = {}
        self._slippage_bps: deque[float] = deque(maxlen=200)  # S5
        self._last_report_at = 0                              # decisions at last emit

    # ---- feed ------------------------------------------------------------

    def record_decision(
        self,
        *,
        confidence: float,
        action: int,
        timestamp: str,
        client_order_id: str | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        self.decision_count += 1
        self._first_ts = self._first_ts or timestamp
        self._last_ts = timestamp
        if action != 0:
            self.confidences.append(float(confidence))
            if client_order_id:
                self._conf_by_order[str(client_order_id)] = float(confidence)

    def record_fill(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        realized_pnl: float,
        timestamp: str,
        entry_confidence: float | None = None,
        client_order_id: str | None = None,
        # S5: realized fill vs the pre-slippage reference (backtest: fill
        # bar open vs crossed price; live: quote mid vs fill — reference is
        # not yet plumbed at the live fill site, so S5 stays dormant there).
        fill_price: float | None = None,
        reference_price: float | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        # S5 — realized slippage in bps, whenever a reference is available.
        if (
            fill_price is not None
            and reference_price is not None
            and abs(float(reference_price)) > 1e-9
        ):
            self._slippage_bps.append(
                abs(float(fill_price) - float(reference_price)) / abs(float(reference_price)) * 1e4
            )
        signed = float(quantity) * (1.0 if side == "buy" else -1.0)
        before = self._position[symbol]
        after = before + signed
        self._position[symbol] = after
        # A single fill can OPEN (flat -> position), CLOSE (position -> flat),
        # or FLIP through zero (long -> short in one shot: booked by the shared
        # close-out math, portfolio_state.py). A flip is a close of the old
        # leg AND an open of the new one; handling it as neither drops both
        # round trips from S1/S2 (2026-07-22 review).
        opened = abs(before) <= 1e-9 and abs(after) > 1e-9
        closed = abs(before) > 1e-9 and abs(after) <= 1e-9
        flipped = (before > 1e-9 and after < -1e-9) or (before < -1e-9 and after > 1e-9)

        if closed or flipped:
            # Book the old leg's outcome with the confidence banked at its open.
            self._closes.append(timestamp)
            self._last_close[symbol] = timestamp
            conf = self._entry_conf.pop(symbol, None)
            if conf is not None and abs(realized_pnl) > 0.0:
                self.outcomes.append((conf, realized_pnl > 0.0))
        if opened or flipped:
            # Store the new position's entry confidence: explicit param
            # (backtest/replay) wins, else join the order id to a recorded
            # decision (live). pop() the join so the map cannot grow unbounded.
            conf = entry_confidence
            if conf is None and client_order_id:
                conf = self._conf_by_order.pop(str(client_order_id), None)
            if conf is not None:
                self._entry_conf[symbol] = float(conf)
            # S4 churn: a clean flat -> reopen after a prior close is a cycle;
            # a flip is a continuous reversal, not a close-then-reopen.
            if opened and symbol in self._last_close:
                self._reopen_cycles.append(timestamp)

    def seed_positions(self, positions: dict[str, float]) -> None:
        """Seed the start-of-day book so a carried position's first close is
        not misread as a fresh open. The live path calls this with the
        reconciled broker book at bootstrap; the replay path passes it via
        ``replay_from_ledgers(seed_positions=...)``. Pure state, no I/O."""
        for symbol, qty in positions.items():
            if abs(float(qty)) > 1e-9:
                self._position[symbol] = float(qty)

    # ---- judge -----------------------------------------------------------

    def health(self, *, now: str) -> HealthReport:
        cfg = self.config
        flags: list[str] = []

        # S1 — rolling win-rate surprise (binomial z vs baseline).
        win_rate = z = None
        if len(self.outcomes) >= max(10, cfg.window // 2):
            n = len(self.outcomes)
            wins = sum(1 for _, won in self.outcomes if won)
            win_rate = wins / n
            p = cfg.baseline_win_rate
            z = (win_rate - p) / math.sqrt(p * (1.0 - p) / n)
            if z <= -cfg.surprise_z:
                flags.append(f"S1_winrate_low z={z:.2f} ({win_rate:.0%} vs {p:.0%})")

        # S2 — live calibration: does claimed P(win) match realized?
        buckets: dict[str, dict] = {}
        by_bucket: dict[str, list[bool]] = defaultdict(list)
        claimed: dict[str, list[float]] = defaultdict(list)
        for conf, won in self.outcomes:
            key = f"{0.1 * int(conf * 10):.1f}"
            by_bucket[key].append(won)
            claimed[key].append(conf)
        for key, wins_list in sorted(by_bucket.items()):
            n = len(wins_list)
            realized = sum(wins_list) / n
            mean_claimed = sum(claimed[key]) / n
            buckets[key] = {"n": n, "claimed": round(mean_claimed, 3), "realized": round(realized, 3)}
            if n >= cfg.min_bucket and abs(mean_claimed - realized) > cfg.calibration_gap:
                flags.append(f"S2_miscalibrated bucket={key} claims {mean_claimed:.2f} wins {realized:.0%} (n={n})")

        # S3 — confidence-distribution drift (pinning / collapse).
        p10 = p90 = None
        if len(self.confidences) >= 50:
            ordered = sorted(self.confidences)
            p10 = ordered[len(ordered) // 10]
            p90 = ordered[(len(ordered) * 9) // 10]
            if p90 - p10 < 0.05:
                flags.append(f"S3_confidence_pinned p10={p10:.3f} p90={p90:.3f}")

        # S4 — churn cycles per hour.
        churn = 0.0
        hours = _hours_between(self._first_ts, self._last_ts)
        if hours > 0.25:
            churn = len(self._reopen_cycles) / hours
            if churn > cfg.churn_per_hour:
                flags.append(f"S4_churning {churn:.1f} cycles/hour")

        # S5 — realized slippage vs the EV gate's assumed cost.
        slippage_p50 = None
        if len(self._slippage_bps) >= cfg.min_cost_samples:
            ordered_slip = sorted(self._slippage_bps)
            slippage_p50 = ordered_slip[len(ordered_slip) // 2]
            if slippage_p50 - cfg.assumed_slippage_bps > cfg.cost_surprise_bps:
                flags.append(
                    f"S5_cost_underestimated median {slippage_p50:.1f}bps "
                    f"vs assumed {cfg.assumed_slippage_bps:.1f}bps"
                )

        # L1 — shrink-only ladder. Multiplier in [floor, 1.0]; one notch
        # (25% of the span) per active flag. NEVER below floor, NEVER >1.
        multiplier = 1.0
        if cfg.damping_enabled and flags:
            span = 1.0 - cfg.damping_floor
            multiplier = max(cfg.damping_floor, 1.0 - 0.25 * span * len(flags))

        return HealthReport(
            ts=now,
            decisions=self.decision_count,
            resolved=len(self.outcomes),
            win_rate=None if win_rate is None else round(win_rate, 4),
            win_rate_z=None if z is None else round(z, 2),
            calibration=buckets,
            conf_p10=None if p10 is None else round(p10, 4),
            conf_p90=None if p90 is None else round(p90, 4),
            churn_cycles_per_hour=round(churn, 2),
            realized_slippage_bps=None if slippage_p50 is None else round(slippage_p50, 2),
            flags=flags,
            damping_multiplier=multiplier,
        )

    def due_for_report(self, interval: int) -> bool:
        """True (and advances the marker) when `interval` decisions have
        accrued since the last emission — the report cadence gate."""
        if self.decision_count - self._last_report_at >= max(1, interval):
            self._last_report_at = self.decision_count
            return True
        return False

    # ---- offline replay (validate the monitor on banked ledgers) ---------

    @classmethod
    def replay_from_ledgers(
        cls,
        *,
        decisions_path: Path | str = "runs/decisions.jsonl",
        fills_path: Path | str = "runs/fills.jsonl",
        day: str,
        config: SelfMonitorConfig | None = None,
        seed_positions: dict[str, float] | None = None,
    ) -> "SelfMonitor":
        """Rebuild monitor state from one day's ledgers.

        fills.jsonl is NOT time-ordered and positions carry overnight —
        sort by timestamp and pass the start-of-day book (the two rules
        every ledger consumer must follow; see CONFIDENCE_AUDIT.md 7).
        Entry confidence is joined from the decision's client_order_id.
        """
        cfg = config or SelfMonitorConfig(enabled=True)
        monitor = cls(cfg)
        if seed_positions:
            monitor.seed_positions(seed_positions)

        conf_by_order: dict[str, float] = {}
        for row in _read_jsonl(decisions_path):
            if not str(row.get("ts", "")).startswith(day):
                continue
            monitor.record_decision(
                confidence=float(row.get("confidence", 0.0)),
                action=int(row.get("proposed_action", 0)),
                timestamp=str(row.get("ts", "")),
            )
            order_id = row.get("client_order_id")
            if order_id:
                conf_by_order[str(order_id)] = float(row.get("confidence", 0.0))

        fills = [r for r in _read_jsonl(fills_path) if str(r.get("timestamp", "")).startswith(day)]
        fills.sort(key=lambda r: r["timestamp"])
        for row in fills:
            monitor.record_fill(
                symbol=str(row["symbol"]),
                side=str(row["side"]),
                quantity=float(row["quantity"]),
                realized_pnl=float(row.get("realized_pnl", 0.0)),
                timestamp=str(row["timestamp"]),
                entry_confidence=conf_by_order.get(str(row.get("client_order_id", ""))),
            )
        return monitor


class ReportWriter:
    """Append-only JSONL writer; failures never reach the trade loop."""

    def __init__(self, path: Path | str = DEFAULT_REPORT_PATH) -> None:
        self.path = Path(path)

    def write(self, report: HealthReport) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(report.to_json_line() + "\n")
        except Exception:  # noqa: BLE001 - observability must never block trading
            pass


def _hours_between(start: str | None, end: str | None) -> float:
    if not start or not end:
        return 0.0
    from datetime import datetime

    try:
        return max(
            (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds() / 3600.0,
            0.0,
        )
    except ValueError:
        return 0.0


def _read_jsonl(path: Path | str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
