"""Persistent top header bar — always-visible key metrics across all tabs.

Pulls from ``runtime_service`` (state, symbols, last-tick timestamp) and
``equity_service`` (latest equity, session return, drawdown).  Refreshes
on a QTimer at 2s cadence — both sources are cheap file reads so this
does not block the GUI thread.
"""
from __future__ import annotations

from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from src.gui.workers import CallableWorker
from src.gui.widgets import (
    COLOR_ACCENT,
    COLOR_INFO,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_POSITIVE,
    COLOR_TEXT,
    HeaderMetric,
    StatusBadge,
)
from src.gui_services.equity_service import get_live_equity_series
from src.gui_services.runtime_service import get_runtime_service
from src.gui_services.state import (
    EquitySeries,
    RuntimeState,
    RuntimeStatus,
)


def _sep() -> QFrame:
    sep = QFrame()
    sep.setFrameShape(QFrame.Shape.VLine)
    sep.setStyleSheet(
        f"color: {COLOR_PANEL_BORDER}; background: {COLOR_PANEL_BORDER}; "
        f"border: 0; max-width: 1px;"
    )
    return sep


class GlobalHeader(QWidget):
    """Persistent operator header shown above the tab widget."""

    POLL_MS = 2500

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.svc = get_runtime_service()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setStyleSheet(
            f"GlobalHeader {{ background-color: {COLOR_PANEL_BG}; "
            f"border-bottom: 2px solid {COLOR_ACCENT}; }}"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 8, 14, 8)
        outer.setSpacing(6)

        # ── Row 1: title + state badge + inline metrics ──
        row1 = QHBoxLayout()
        row1.setSpacing(12)

        self.lbl_title = QLabel("SETAI  ::  SELF-EVOLVING TRADING AI")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(13)
        self.lbl_title.setFont(title_font)
        self.lbl_title.setStyleSheet(
            f"color: {COLOR_ACCENT}; letter-spacing: 2px; border: 0;"
        )
        row1.addWidget(self.lbl_title)

        self.badge = StatusBadge("stopped")
        row1.addWidget(self.badge)

        row1.addWidget(_sep())

        self.m_equity   = HeaderMetric("Equity")
        self.m_return   = HeaderMetric("Session Return")
        self.m_dd       = HeaderMetric("Max Drawdown")
        self.m_symbols  = HeaderMetric("Symbols")
        self.m_tick     = HeaderMetric("Last Tick")
        self.m_pid      = HeaderMetric("PID")

        for w in (
            self.m_equity, self.m_return, self.m_dd,
            self.m_symbols, self.m_tick, self.m_pid,
        ):
            row1.addWidget(w)
        row1.addStretch(1)
        outer.addLayout(row1)

        # ── Row 2: the single most recent operator message ──
        self.lbl_msg = QLabel("— no runtime activity yet —")
        self.lbl_msg.setStyleSheet(
            f"color: {COLOR_MUTED}; font-family: 'Cascadia Mono', Consolas, monospace; "
            f"font-size: 11px; border: 0; background: transparent;"
        )
        self.lbl_msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        outer.addWidget(self.lbl_msg)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Kick off background data fetch — never blocks the GUI thread."""
        if getattr(self, '_refreshing', False):
            return  # skip if previous refresh still running
        self._refreshing = True
        worker = CallableWorker(self._fetch_data)
        worker.signals.finished.connect(self._on_data_fetched)
        worker.signals.failed.connect(self._on_fetch_failed)
        QThreadPool.globalInstance().start(worker)

    def _fetch_data(self) -> tuple:
        """Runs in background thread — safe to do I/O here."""
        from src.gui_services.equity_service import get_live_equity_series
        try:
            status = self.svc.get_status()
        except Exception:
            status = None
        try:
            series = get_live_equity_series()
        except Exception:
            series = None
        return (status, series)

    def _on_data_fetched(self, result: object) -> None:
        """Runs on main thread — safe to update widgets here."""
        self._refreshing = False
        if not isinstance(result, tuple) or len(result) != 2:
            return
        status, series = result
        if status is not None and series is not None:
            self._render(status, series)

    def _on_fetch_failed(self, msg: str) -> None:
        self._refreshing = False

    def _render(self, status: RuntimeStatus, series: EquitySeries) -> None:
        # State badge
        self.badge.set_state(status.state.value)

        # PID
        self.m_pid.set_value(
            str(status.pid) if status.pid else "—",
            color=COLOR_TEXT if status.pid else COLOR_MUTED,
        )

        # Symbols — the header has a fixed horizontal budget per metric so
        # we show the count plus the first few tickers followed by "+N more".
        # The full list lives in the Runtime tab's SYMBOLS card.
        if status.symbols:
            preview_count = 3
            preview = ", ".join(status.symbols[:preview_count])
            remainder = len(status.symbols) - preview_count
            if remainder > 0:
                symbols_text = f"{len(status.symbols)} :: {preview} +{remainder} more"
            else:
                symbols_text = f"{len(status.symbols)} :: {preview}"
        else:
            symbols_text = "—"
        self.m_symbols.set_value(symbols_text)

        # Last tick
        if status.last_log_timestamp:
            self.m_tick.set_value(
                status.last_log_timestamp.strftime("%H:%M:%S"),
                color=COLOR_INFO,
            )
        else:
            self.m_tick.set_value("—", color=COLOR_MUTED)

        # Equity + return + drawdown
        if series.points:
            equity = series.latest_equity
            ret = series.session_return
            dd = series.max_drawdown
            self.m_equity.set_value(f"${equity:,.2f}", color=COLOR_TEXT)
            ret_color = (
                COLOR_POSITIVE if ret > 0 else
                (COLOR_NEGATIVE if ret < 0 else COLOR_MUTED)
            )
            self.m_return.set_value(f"{ret * 100:+.2f}%", color=ret_color)
            dd_color = COLOR_NEGATIVE if dd < 0 else COLOR_MUTED
            self.m_dd.set_value(f"{dd * 100:.2f}%", color=dd_color)
        else:
            self.m_equity.set_value("—", color=COLOR_MUTED)
            self.m_return.set_value("—", color=COLOR_MUTED)
            self.m_dd.set_value("—", color=COLOR_MUTED)

        # Bottom-row message
        running = status.state == RuntimeState.RUNNING
        msg = status.last_message or status.last_decision_summary or ""
        if status.error_message:
            self.lbl_msg.setText(f"ERROR :: {status.error_message}")
            self.lbl_msg.setStyleSheet(
                f"color: {COLOR_NEGATIVE}; font-family: 'Cascadia Mono', Consolas, monospace; "
                f"font-size: 11px; border: 0; background: transparent; font-weight: 600;"
            )
        elif msg:
            prefix = "LIVE" if running else "IDLE"
            self.lbl_msg.setText(f"{prefix} :: {msg}")
            color = COLOR_TEXT if running else COLOR_MUTED
            self.lbl_msg.setStyleSheet(
                f"color: {color}; font-family: 'Cascadia Mono', Consolas, monospace; "
                f"font-size: 11px; border: 0; background: transparent;"
            )
        else:
            self.lbl_msg.setText("— no runtime activity yet —")
            self.lbl_msg.setStyleSheet(
                f"color: {COLOR_MUTED}; font-family: 'Cascadia Mono', Consolas, monospace; "
                f"font-size: 11px; border: 0; background: transparent;"
            )
