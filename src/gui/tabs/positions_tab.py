"""Positions tab — live view of open positions + account state.

Reads ``runs/positions_status.json`` every 5 seconds via
``positions_service.get_positions_snapshot`` and renders:

  * An account summary strip (equity / cash / buying power / session
    return / session drawdown + the three risk caps).
  * A table of open positions showing side, qty, avg price, mark price,
    notional, unrealized PnL (absolute + %), **peak PnL + per-position
    drawdown**, and age.

The runtime is the sole source of truth — the GUI never queries Alpaca
directly.  That keeps the GUI decoupled from broker credentials and
means a stopped runtime simply shows stale data instead of crashing.
"""
from __future__ import annotations

from datetime import datetime, timezone

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import (
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_TEXT,
    MetricCard,
)
from src.gui_services.positions_service import get_positions_snapshot
from src.monitoring.positions_status import PositionsStatusSnapshot


class PositionsTab(QWidget):
    """Live positions + account-state view.

    Poll cadence is 5 s (per user request) — positions don't change
    faster than this in practice and a 5 s reader keeps the GUI thread
    load near zero even with a 40-symbol universe.
    """

    POLL_MS = 5000

    # Table column layout.  Keep this in sync with ``_render_positions``
    # below — changing one without the other breaks column headers.
    _COLUMNS = [
        ("Symbol", 70),
        ("Side", 55),
        ("Qty", 80),
        ("Avg", 80),
        ("Mark", 80),
        ("Notional", 90),
        ("PnL ($)", 90),
        ("PnL (%)", 75),
        ("Peak", 90),
        ("Drawdown", 90),
        ("DD (%)", 75),
        ("Age", 80),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # --- Account summary strip -------------------------------------
        account_box = QGroupBox("Account")
        grid = QGridLayout(account_box)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.card_equity = MetricCard("Equity")
        self.card_cash = MetricCard("Cash")
        self.card_buying_power = MetricCard("Buying power")
        self.card_return = MetricCard("Session return")
        self.card_drawdown = MetricCard("Session drawdown")
        self.card_peak = MetricCard("Session peak equity")
        grid.addWidget(self.card_equity, 0, 0)
        grid.addWidget(self.card_cash, 0, 1)
        grid.addWidget(self.card_buying_power, 0, 2)
        grid.addWidget(self.card_return, 0, 3)
        grid.addWidget(self.card_drawdown, 0, 4)
        grid.addWidget(self.card_peak, 0, 5)

        # Risk-cap strip underneath the account cards.  These come from
        # the runtime's risk config at spawn time; they're shown here so
        # the operator knows what ceilings the risk manager is enforcing
        # without having to check the YAML files.
        self.lbl_risk_caps = QLabel("")
        self.lbl_risk_caps.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 11px; border: 0;"
        )
        self.lbl_risk_caps.setWordWrap(True)
        grid.addWidget(self.lbl_risk_caps, 1, 0, 1, 6)

        root.addWidget(account_box)

        # --- Open positions table --------------------------------------
        positions_box = QGroupBox("Open Positions")
        pos_layout = QVBoxLayout(positions_box)
        self.lbl_positions_header = QLabel("—")
        self.lbl_positions_header.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 11px; border: 0;"
        )
        pos_layout.addWidget(self.lbl_positions_header)

        self.table = QTableWidget(0, len(self._COLUMNS))
        self.table.setHorizontalHeaderLabels([h[0] for h in self._COLUMNS])
        header = self.table.horizontalHeader()
        for idx, (_, width) in enumerate(self._COLUMNS):
            header.resizeSection(idx, width)
        header.setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        pos_layout.addWidget(self.table)

        root.addWidget(positions_box, 1)

        # --- Footer status --------------------------------------------
        self.lbl_footer = QLabel("No runtime data yet.")
        self.lbl_footer.setStyleSheet(
            f"color: {COLOR_MUTED}; font-size: 11px; border: 0;"
        )
        root.addWidget(self.lbl_footer)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        snap = get_positions_snapshot()
        if snap is None:
            self._render_empty()
            return
        self._render_account(snap)
        self._render_positions(snap)
        self._render_footer(snap)

    # ------------------------------------------------------------------
    def _render_empty(self) -> None:
        self.card_equity.set_value("—", color=COLOR_MUTED)
        self.card_cash.set_value("—", color=COLOR_MUTED)
        self.card_buying_power.set_value("—", color=COLOR_MUTED)
        self.card_return.set_value("—", color=COLOR_MUTED)
        self.card_drawdown.set_value("—", color=COLOR_MUTED)
        self.card_peak.set_value("—", color=COLOR_MUTED)
        self.lbl_risk_caps.setText("risk caps: (no runtime data)")
        self.lbl_positions_header.setText("no open positions")
        self.table.setRowCount(0)
        self.lbl_footer.setText(
            "No runtime data yet — start paper trading from the Runtime tab."
        )

    def _render_account(self, snap: PositionsStatusSnapshot) -> None:
        acct = snap.account
        self.card_equity.set_value(f"${acct.equity:,.2f}")
        self.card_cash.set_value(f"${acct.cash:,.2f}")
        self.card_buying_power.set_value(f"${acct.buying_power:,.2f}")
        # Colour the return / drawdown cards to match their sign.  Return
        # can be positive or negative; drawdown is always <=0.
        ret_color = (
            COLOR_POSITIVE
            if acct.session_return > 0
            else (COLOR_NEGATIVE if acct.session_return < 0 else COLOR_TEXT)
        )
        self.card_return.set_value(
            f"{acct.session_return * 100:+.2f}%", color=ret_color
        )
        dd_color = COLOR_NEGATIVE if acct.session_drawdown < -1.0e-9 else COLOR_TEXT
        self.card_drawdown.set_value(
            f"{acct.session_drawdown * 100:+.2f}%", color=dd_color
        )
        self.card_peak.set_value(f"${acct.session_peak_equity:,.2f}")

        caps_bits = [
            f"max position: ${acct.max_position_notional:,.0f}",
            f"max concurrent: {acct.max_concurrent_positions}",
            # min_trade_confidence removed — trade gating is EV-based now
            # (see src/risk/risk_manager.py).
        ]
        if acct.kill_switch:
            caps_bits.append("KILL SWITCH: ON")
        self.lbl_risk_caps.setText("  |  ".join(caps_bits))

    def _render_positions(self, snap: PositionsStatusSnapshot) -> None:
        positions = snap.positions
        self.lbl_positions_header.setText(
            f"{len(positions)} open position{'s' if len(positions) != 1 else ''}"
        )
        self.table.setRowCount(len(positions))
        now = datetime.now(timezone.utc)
        for row, p in enumerate(positions):
            side_color = COLOR_POSITIVE if p.side == "long" else COLOR_NEGATIVE
            pnl_color = (
                COLOR_POSITIVE
                if p.unrealized_pnl > 0
                else (COLOR_NEGATIVE if p.unrealized_pnl < 0 else COLOR_TEXT)
            )
            dd_color = COLOR_NEGATIVE if p.drawdown < -1.0e-9 else COLOR_TEXT

            age_str = "—"
            if p.entry_time is not None:
                delta = now - p.entry_time
                total_sec = int(delta.total_seconds())
                if total_sec < 0:
                    age_str = "0s"
                elif total_sec < 60:
                    age_str = f"{total_sec}s"
                elif total_sec < 3600:
                    age_str = f"{total_sec // 60}m {total_sec % 60}s"
                elif total_sec < 86400:
                    age_str = f"{total_sec // 3600}h {(total_sec % 3600) // 60}m"
                else:
                    age_str = f"{total_sec // 86400}d {(total_sec % 86400) // 3600}h"

            cells = [
                (p.symbol, COLOR_TEXT, Qt.AlignmentFlag.AlignCenter),
                (p.side.upper(), side_color, Qt.AlignmentFlag.AlignCenter),
                (f"{p.qty:.4f}", COLOR_TEXT, Qt.AlignmentFlag.AlignRight),
                (f"{p.avg_price:.4f}", COLOR_TEXT, Qt.AlignmentFlag.AlignRight),
                (f"{p.market_price:.4f}", COLOR_TEXT, Qt.AlignmentFlag.AlignRight),
                (f"${p.notional:,.2f}", COLOR_TEXT, Qt.AlignmentFlag.AlignRight),
                (f"{p.unrealized_pnl:+,.2f}", pnl_color, Qt.AlignmentFlag.AlignRight),
                (
                    f"{p.unrealized_pnl_pct * 100:+.2f}%",
                    pnl_color,
                    Qt.AlignmentFlag.AlignRight,
                ),
                (f"{p.peak_unrealized_pnl:+,.2f}", COLOR_TEXT, Qt.AlignmentFlag.AlignRight),
                (f"{p.drawdown:+,.2f}", dd_color, Qt.AlignmentFlag.AlignRight),
                (
                    f"{p.drawdown_pct * 100:+.2f}%",
                    dd_color,
                    Qt.AlignmentFlag.AlignRight,
                ),
                (age_str, COLOR_MUTED, Qt.AlignmentFlag.AlignCenter),
            ]
            for col, (text, color, align) in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setForeground(_brush(color))
                item.setTextAlignment(align)
                self.table.setItem(row, col, item)

    def _render_footer(self, snap: PositionsStatusSnapshot) -> None:
        age_sec = max(0.0, (datetime.now(timezone.utc) - snap.ts).total_seconds())
        if age_sec < 10:
            freshness = f"updated {age_sec:.0f}s ago"
        elif age_sec < 60:
            freshness = f"updated {age_sec:.0f}s ago"
        elif age_sec < 3600:
            freshness = f"updated {age_sec / 60:.0f}m ago — runtime may be stopped"
        else:
            freshness = (
                f"updated {age_sec / 3600:.1f}h ago — runtime definitely stopped"
            )
        self.lbl_footer.setText(freshness)


def _brush(color_hex: str):
    """QBrush from a hex color string.  Imported lazily so the widget
    file has zero Qt imports at module load time outside what's needed
    for UI construction."""
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(color_hex))
