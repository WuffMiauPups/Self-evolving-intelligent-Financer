"""Risk / Execution tab — read-only view + emergency stop via runtime service."""
from __future__ import annotations

from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MetricCard, StatusBadge
from src.gui.workers import CallableWorker
from src.gui_services.risk_service import get_risk_status
from src.gui_services.runtime_service import get_runtime_service
from src.gui_services.state import RiskStatus


class RiskTab(QWidget):
    POLL_MS = 6000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.svc = get_runtime_service()
        self._fetch_pending = False
        self._build_ui()
        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("Risk authority is FINAL — GUI cannot bypass it."))
        header.addStretch(1)
        self.kill_badge = StatusBadge("off")
        header.addWidget(QLabel("Kill switch:"))
        header.addWidget(self.kill_badge)
        root.addLayout(header)

        env_box = QGroupBox("Risk Envelope (from config)")
        grid = QGridLayout(env_box)
        self.card_max_pos = MetricCard("Max position notional")
        self.card_max_loss = MetricCard("Max daily loss")
        self.card_cooldown = MetricCard("Cooldown sec")
        self.card_agg_notional = MetricCard("Max aggregate notional")
        self.card_leverage = MetricCard("Max leverage ratio")
        self.card_concurrent = MetricCard("Max concurrent positions")
        grid.addWidget(self.card_max_pos, 0, 0)
        grid.addWidget(self.card_max_loss, 0, 1)
        grid.addWidget(self.card_cooldown, 0, 2)
        grid.addWidget(self.card_agg_notional, 1, 0)
        grid.addWidget(self.card_leverage, 1, 1)
        grid.addWidget(self.card_concurrent, 1, 2)
        root.addWidget(env_box)

        dd_box = QGroupBox("Drawdown & EV Gate Costs")
        dd_grid = QGridLayout(dd_box)
        self.card_rolling_dd = MetricCard("Max rolling drawdown")
        self.card_dd_window = MetricCard("Drawdown window (bars)")
        self.card_daily_pct = MetricCard("Daily loss % of equity")
        # EV gate cost side: round-trip fee + slippage in bps. Trade gating is
        # confidence * gain_bps - (1 - confidence) * loss_bps - cost_bps > 0.
        # No fixed confidence threshold anymore.
        self.card_fee_bps = MetricCard("Fee (bps)")
        self.card_slippage_bps = MetricCard("Slippage (bps)")
        dd_grid.addWidget(self.card_rolling_dd, 0, 0)
        dd_grid.addWidget(self.card_dd_window, 0, 1)
        dd_grid.addWidget(self.card_daily_pct, 0, 2)
        dd_grid.addWidget(self.card_fee_bps, 1, 0)
        dd_grid.addWidget(self.card_slippage_bps, 1, 1)
        root.addWidget(dd_box)

        controls = QGroupBox("Emergency Controls")
        c = QHBoxLayout(controls)
        self.btn_stop_runtime = QPushButton("STOP RUNTIME (graceful)")
        self.btn_stop_runtime.setStyleSheet(
            "background-color: #a4262c; color: white; "
            "font-size: 13px; font-weight: bold; padding: 8px;"
        )
        c.addWidget(self.btn_stop_runtime)
        c.addStretch(1)
        self.btn_stop_runtime.clicked.connect(self._on_stop_runtime)
        root.addWidget(controls)

        notes = QLabel(
            "Open positions and live risk-block reasons are only visible inside the\n"
            "running runtime subprocess and its log file. See the Logs tab (filter:\n"
            "risk) for live block reasons. Position summaries are not exposed in v1."
        )
        notes.setStyleSheet("color: #888888;")
        root.addWidget(notes)
        root.addStretch(1)

    def _on_stop_runtime(self) -> None:
        if not self.svc.is_running():
            QMessageBox.information(self, "Not running", "Runtime is not running.")
            return
        confirm = QMessageBox.question(
            self,
            "Stop runtime",
            "Send graceful stop to the paper-trading runtime?",
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.svc.stop()

    def refresh(self) -> None:
        if self._fetch_pending:
            return
        self._fetch_pending = True
        worker = CallableWorker(self._fetch_data)
        worker.signals.finished.connect(self._on_data_fetched)
        worker.signals.failed.connect(self._on_fetch_failed)
        QThreadPool.globalInstance().start(worker)

    def _fetch_data(self):
        """Runs in worker thread -- no Qt calls here."""
        return get_risk_status()

    def _on_data_fetched(self, result: object) -> None:
        """Runs on main thread -- safe to update widgets here."""
        self._fetch_pending = False
        if not isinstance(result, RiskStatus):
            return
        self._render(result)

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_pending = False

    def _render(self, status: RiskStatus) -> None:
        self.kill_badge.set_state(
            "error" if status.kill_switch else "ok",
            "ENGAGED" if status.kill_switch else "OFF",
        )
        self.card_max_pos.set_value(f"${status.max_position_notional:,.0f}")
        self.card_max_loss.set_value(f"${status.max_daily_loss:,.0f}")
        self.card_cooldown.set_value(str(status.cooldown_seconds))
        self.card_agg_notional.set_value(f"${status.max_aggregate_notional:,.0f}")
        self.card_leverage.set_value(f"{status.max_leverage_ratio:.1f}x")
        self.card_concurrent.set_value(str(status.max_concurrent_positions))
        self.card_rolling_dd.set_value(f"{status.max_rolling_drawdown:.0%}")
        self.card_dd_window.set_value(str(status.rolling_drawdown_window))
        pct_label = f"{status.max_daily_loss_pct:.1%}" if status.max_daily_loss_pct is not None else "disabled"
        self.card_daily_pct.set_value(pct_label)
        self.card_fee_bps.set_value(f"{status.fee_bps:.1f}")
        self.card_slippage_bps.set_value(f"{status.slippage_bps:.1f}")
