"""Logs / Events tab — tail and filter the structured backend log file."""
from __future__ import annotations

from PySide6.QtCore import QThreadPool, QTimer
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.workers import CallableWorker
from src.gui_services.log_service import LogEntry, get_log_path, tail


_CATEGORY_LIST = ("runtime", "memory", "fusion", "risk", "execution", "errors")


class LogsTab(QWidget):
    POLL_MS = 3500

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._auto = True
        self._build_ui()
        self.refresh()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._auto_refresh)
        self._timer.start(self.POLL_MS)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        filter_box = QGroupBox("Filters")
        f = QHBoxLayout(filter_box)
        f.addWidget(QLabel("Show:"))
        self.filter_checks: dict[str, QCheckBox] = {}
        for cat in _CATEGORY_LIST:
            cb = QCheckBox(cat)
            cb.setChecked(True)
            cb.stateChanged.connect(self.refresh)
            self.filter_checks[cat] = cb
            f.addWidget(cb)
        f.addStretch(1)
        self.btn_refresh = QPushButton("Refresh")
        self.cb_auto = QCheckBox("Auto")
        self.cb_auto.setChecked(True)
        self.cb_auto.stateChanged.connect(self._toggle_auto)
        self.btn_refresh.clicked.connect(self.refresh)
        f.addWidget(self.btn_refresh)
        f.addWidget(self.cb_auto)
        root.addWidget(filter_box)

        self.lbl_path = QLabel(f"Log file: {get_log_path()}")
        self.lbl_path.setStyleSheet("color: #888;")
        root.addWidget(self.lbl_path)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(
            "QPlainTextEdit { background-color: #1a1a1a; color: #d0d0d0; "
            "font-family: 'Cascadia Mono', Consolas, monospace; font-size: 11px; }"
        )
        self.text.setMaximumBlockCount(5000)
        root.addWidget(self.text, 1)

    # ------------------------------------------------------------------
    def _toggle_auto(self, state: int) -> None:
        self._auto = bool(state)

    def _auto_refresh(self) -> None:
        if self._auto:
            self.refresh()

    def refresh(self) -> None:
        if getattr(self, '_refreshing', False):
            return
        self._refreshing = True
        active = {c for c, cb in self.filter_checks.items() if cb.isChecked()}
        worker = CallableWorker(tail, categories=active or None)
        worker.signals.finished.connect(self._on_logs_fetched)
        worker.signals.failed.connect(lambda _: setattr(self, '_refreshing', False))
        QThreadPool.globalInstance().start(worker)

    def _on_logs_fetched(self, result):
        self._refreshing = False
        if result is not None:
            self._render(result)

    def _render(self, entries: list[LogEntry]) -> None:
        # Re-render entirely (max 5000 lines, cheap).
        self.text.clear()
        cursor = self.text.textCursor()
        dim_fmt = QTextCharFormat()
        dim_fmt.setForeground(QColor("#666666"))
        level_colors = {
            "error": "#ff6b6b",
            "critical": "#ff6b6b",
            "warning": "#f5d76e",
            "info": "#4bd0d8",
            "debug": "#888888",
        }
        for e in entries:
            # Timestamp (dim)
            if e.timestamp:
                cursor.setCharFormat(dim_fmt)
                cursor.insertText(e.timestamp.strftime("%H:%M:%S") + " ")
            # Level badge
            level_fmt = QTextCharFormat()
            level_color = level_colors.get(e.level.lower(), "#888888")
            level_fmt.setForeground(QColor(level_color))
            cursor.setCharFormat(level_fmt)
            cursor.insertText(f"[{e.level.upper():<5s}] ")
            # Message with category coloring
            msg_fmt = QTextCharFormat()
            if e.is_error or e.category == "errors":
                msg_fmt.setForeground(QColor("#ff6b6b"))
            elif "veto" in e.message.lower() or "risk_block" in e.message.lower():
                msg_fmt.setForeground(QColor("#ffb454"))
            elif e.is_warning:
                msg_fmt.setForeground(QColor("#f5d76e"))
            elif e.category == "fusion":
                msg_fmt.setForeground(QColor("#7fc8f8"))
            elif e.category == "memory":
                msg_fmt.setForeground(QColor("#a4d65e"))
            else:
                msg_fmt.setForeground(QColor("#e0e0e0"))
            cursor.setCharFormat(msg_fmt)
            cursor.insertText(e.message + "\n")
        self.text.moveCursor(QTextCursor.MoveOperation.End)
