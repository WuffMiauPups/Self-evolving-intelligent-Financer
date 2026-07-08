"""Main window for the SETAI desktop control center."""
from __future__ import annotations

import sys

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.header import GlobalHeader
from src.gui.tabs.backtest_tab import BacktestTab
from src.gui.tabs.checkpoints_tab import CheckpointsTab
from src.gui.tabs.equity_tab import EquityTab
from src.gui.tabs.experiments_tab import ExperimentsTab
from src.gui.tabs.logs_tab import LogsTab
from src.gui.tabs.memory_tab import MemoryTab
from src.gui.tabs.positions_tab import PositionsTab
from src.gui.tabs.risk_tab import RiskTab
from src.gui.tabs.runtime_tab import RuntimeTab
from src.gui.tabs.training_tab import TrainingTab
from src.gui_services.runtime_service import get_runtime_service


_DARK_QSS = """
/* ── Base ──────────────────────────────────────────────────────── */
QMainWindow, QWidget {
    background-color: #0a0e14;
    color: #e2e8f0;
    font-size: 12px;
}
QMenuBar {
    background-color: #0a0e14;
    color: #e2e8f0;
    border-bottom: 1px solid #1f2937;
}
QMenuBar::item:selected { background-color: #1f2937; }
QMenu {
    background-color: #11161c;
    color: #e2e8f0;
    border: 1px solid #1f2937;
}
QMenu::item:selected { background-color: #1f6feb; }

/* ── Group boxes ──────────────────────────────────────────────── */
/* Note: every rule that sets font-weight MUST also set font-size in
   QSS — Qt's CSS engine does NOT cascade font properties from the
   parent QWidget rule above.  Without an explicit font-size here, on
   some Windows configurations Qt falls back to the platform default
   font (which uses pixelSize internally, so pointSize() returns -1)
   and emits "QFont::setPointSize: Point size <= 0 (-1)" warnings. */
QGroupBox {
    background-color: #11161c;
    border: 1px solid #1f2937;
    border-radius: 6px;
    margin-top: 14px;
    padding-top: 10px;
    font-size: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: #4bd0d8;
    letter-spacing: 1px;
}

/* ── Buttons ──────────────────────────────────────────────────── */
QPushButton {
    background-color: #1f2937;
    color: #e2e8f0;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover {
    background-color: #263244;
    border: 1px solid #4bd0d8;
    color: #4bd0d8;
}
QPushButton:pressed { background-color: #0d1117; }
QPushButton:disabled {
    background-color: #13171e;
    color: #4b5563;
    border: 1px solid #1f2937;
}

/* ── Inputs ───────────────────────────────────────────────────── */
QLineEdit, QComboBox, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #0d1117;
    color: #e2e8f0;
    border: 1px solid #1f2937;
    border-radius: 4px;
    padding: 5px 6px;
    selection-background-color: #1f6feb;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {
    border: 1px solid #4bd0d8;
}
QComboBox::drop-down { border: 0; width: 18px; }
QCheckBox { spacing: 6px; color: #cbd5e1; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #334155;
    background: #0d1117;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #4bd0d8;
    border: 1px solid #4bd0d8;
}

/* ── Tables ───────────────────────────────────────────────────── */
QTableWidget {
    background-color: #0d1117;
    color: #e2e8f0;
    gridline-color: #1f2937;
    alternate-background-color: #11161c;
    border: 1px solid #1f2937;
    selection-background-color: #1f6feb;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #11161c;
    color: #4bd0d8;
    padding: 6px;
    border: 0;
    border-bottom: 2px solid #1f2937;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}

/* ── Tab bar ──────────────────────────────────────────────────── */
QTabWidget::pane {
    border: 1px solid #1f2937;
    background-color: #0a0e14;
    top: -1px;
}
QTabBar::tab {
    background: #11161c;
    color: #94a3b8;
    padding: 9px 18px;
    border: 1px solid #1f2937;
    border-bottom: 2px solid transparent;
    margin-right: 2px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
}
QTabBar::tab:hover { color: #cbd5e1; }
QTabBar::tab:selected {
    background: #0a0e14;
    color: #4bd0d8;
    border-bottom: 2px solid #4bd0d8;
}

/* ── Scroll bars ──────────────────────────────────────────────── */
QScrollBar:vertical {
    background: #0a0e14;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1f2937;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #334155; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0a0e14;
    height: 10px;
    margin: 0;
}
QScrollBar::handle:horizontal {
    background: #1f2937;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background: #334155; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Status bar ───────────────────────────────────────────────── */
QStatusBar {
    background: #0a0e14;
    color: #94a3b8;
    border-top: 1px solid #1f2937;
}
"""


class SETAIControlCenter(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SETAI — Operator Control Center")
        self.resize(1360, 880)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Persistent at-a-glance header shown above every tab.
        self.header = GlobalHeader()
        layout.addWidget(self.header)

        tabs = QTabWidget()
        tabs.addTab(RuntimeTab(), "RUNTIME")
        tabs.addTab(PositionsTab(), "POSITIONS")
        tabs.addTab(EquityTab(), "EQUITY")
        tabs.addTab(BacktestTab(), "BACKTEST")
        tabs.addTab(ExperimentsTab(), "EXPERIMENTS")
        tabs.addTab(TrainingTab(), "TRAINING")
        tabs.addTab(MemoryTab(), "MEMORY / FUSION")
        tabs.addTab(RiskTab(), "RISK / EXECUTION")
        tabs.addTab(CheckpointsTab(), "MODEL / CHECKPOINTS")
        tabs.addTab(LogsTab(), "LOGS / EVENTS")
        layout.addWidget(tabs, 1)

        self.setCentralWidget(central)

        # File menu with explicit quit
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        self.statusBar().showMessage("SETAI control center ready.")

    def closeEvent(self, event) -> None:  # noqa: N802
        svc = get_runtime_service()
        if svc.is_running():
            confirm = QMessageBox.question(
                self,
                "Runtime is running",
                "The paper trading runtime is still running.\n"
                "Stop it before exiting?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel,
            )
            if confirm == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if confirm == QMessageBox.StandardButton.Yes:
                svc.stop()
        event.accept()


def main(argv: list[str] | None = None) -> int:
    app = QApplication(argv if argv is not None else sys.argv)
    app.setStyleSheet(_DARK_QSS)
    window = SETAIControlCenter()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
