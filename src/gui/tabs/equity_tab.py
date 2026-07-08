"""Live Equity tab — native QtCharts equity / PnL chart.

Reads structured snapshots from src/monitoring/equity_log.py via the
gui_services.equity_service layer. No browser, no localhost, no web frontend.
Refresh runs on a QTimer; the file read is small and cheap so we keep it on
the main thread (no Qt repaint blocking observed for typical log sizes —
the equity_service caps at 5000 points).
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCharts import (
    QChart,
    QChartView,
    QDateTimeAxis,
    QLineSeries,
    QValueAxis,
)
from PySide6.QtCore import QDateTime, QThreadPool, QTimer, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MetricCard
from src.gui.workers import CallableWorker
from src.gui_services.equity_service import get_live_equity_series
from src.gui_services.state import EquitySeries


class EquityTab(QWidget):
    POLL_MS = 4000

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

        cards = QHBoxLayout()
        self.card_equity = MetricCard("Current equity")
        self.card_return = MetricCard("Session return")
        self.card_dd = MetricCard("Max drawdown")
        self.card_points = MetricCard("Snapshots")
        for c in (self.card_equity, self.card_return, self.card_dd, self.card_points):
            cards.addWidget(c)
        cards.addStretch(1)
        root.addLayout(cards)

        chart_box = QGroupBox("Live Equity")
        chart_layout = QVBoxLayout(chart_box)

        self.chart = QChart()
        self.chart.setTitle("Equity over time")
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.series = QLineSeries()
        self.series.setName("Equity")
        self.chart.addSeries(self.series)

        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("HH:mm:ss")
        self.axis_x.setTitleText("Time")
        self.chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.series.attachAxis(self.axis_x)

        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Equity ($)")
        self.axis_y.setLabelFormat("%.2f")
        self.chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.series.attachAxis(self.axis_y)

        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setMinimumHeight(360)
        chart_layout.addWidget(self.chart_view)
        root.addWidget(chart_box, 1)

        footer = QHBoxLayout()
        self.lbl_source = QLabel("source: —")
        self.lbl_source.setStyleSheet("color: #aaaaaa;")
        self.btn_refresh = QPushButton("Refresh now")
        self.btn_refresh.clicked.connect(self.refresh)
        footer.addWidget(self.lbl_source, 1)
        footer.addWidget(self.btn_refresh)
        root.addLayout(footer)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        if getattr(self, '_refreshing', False):
            return
        self._refreshing = True
        worker = CallableWorker(self._fetch_equity)
        worker.signals.finished.connect(self._on_equity_fetched)
        worker.signals.failed.connect(lambda _: setattr(self, '_refreshing', False))
        QThreadPool.globalInstance().start(worker)

    def _fetch_equity(self):
        from src.gui_services.equity_service import get_live_equity_series
        return get_live_equity_series()

    def _on_equity_fetched(self, result):
        self._refreshing = False
        if result is not None:
            self._render(result)

    def _render(self, series: EquitySeries) -> None:
        self.series.clear()
        if not series.points:
            self.card_equity.set_value("—")
            self.card_return.set_value("—")
            self.card_dd.set_value("—")
            self.card_points.set_value("0")
            self.lbl_source.setText(
                f"source: {series.source} — {series.note}"
            )
            return

        # Push points
        x_min: datetime | None = None
        x_max: datetime | None = None
        y_min = float("inf")
        y_max = float("-inf")
        for p in series.points:
            qdt = QDateTime(p.timestamp)
            self.series.append(qdt.toMSecsSinceEpoch(), p.equity)
            if x_min is None or p.timestamp < x_min:
                x_min = p.timestamp
            if x_max is None or p.timestamp > x_max:
                x_max = p.timestamp
            y_min = min(y_min, p.equity)
            y_max = max(y_max, p.equity)

        if x_min is not None and x_max is not None:
            self.axis_x.setRange(QDateTime(x_min), QDateTime(x_max))
        if y_min < y_max:
            pad = max((y_max - y_min) * 0.05, 1e-6)
            self.axis_y.setRange(y_min - pad, y_max + pad)
        elif y_max != 0:
            self.axis_y.setRange(y_max * 0.99, y_max * 1.01)

        self.card_equity.set_value(f"{series.latest_equity:,.2f}")
        self.card_return.set_value(f"{series.session_return * 100:,.2f}%")
        self.card_dd.set_value(f"{series.max_drawdown * 100:,.2f}%")
        self.card_points.set_value(str(len(series.points)))
        self.lbl_source.setText(
            f"source: {series.source} — {series.note}"
        )
