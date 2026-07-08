"""Runtime tab — start/stop the paper trading subprocess and show its state."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThreadPool, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.workers import CallableWorker
from src.gui.widgets import (
    COLOR_INFO,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_POSITIVE,
    COLOR_TEXT,
    COLOR_WARN,
    MetricCard,
    RuntimeOverridesPanel,
    StatusBadge,
)
from src.gui_services.runtime_service import get_runtime_service
from src.gui_services.state import RuntimeState, RuntimeStatus, ServiceError


_CHECKPOINTS_DIR = Path("checkpoints")
_USE_CONFIG_DEFAULT = "(use config default)"


class RuntimeTab(QWidget):
    POLL_MS = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.svc = get_runtime_service()
        self._fetch_pending = False
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ─ Top control row ─
        controls = QGroupBox("Paper Trading Runtime")
        c_outer = QVBoxLayout(controls)
        c_outer.setSpacing(6)

        # Row 1: Start/Stop + status badge.
        row1 = QHBoxLayout()
        self.badge = StatusBadge("stopped")
        self.btn_start = QPushButton("Start Paper Trading")
        self.btn_stop = QPushButton("Stop Paper Trading")
        self.btn_stop.setEnabled(False)
        row1.addWidget(self.badge)
        row1.addWidget(self.btn_start)
        row1.addWidget(self.btn_stop)
        row1.addStretch(1)
        c_outer.addLayout(row1)

        # Row 2: Checkpoint picker (dropdown + refresh + browse).
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setMinimumWidth(360)
        self.checkpoint_combo.setToolTip(
            "Pick a checkpoint to load.  '(use config default)' falls back to\n"
            "model.checkpoint_path from the selected YAML stack."
        )
        self.btn_refresh_ckpt = QPushButton("Refresh")
        self.btn_refresh_ckpt.setToolTip("Rescan the checkpoints/ directory.")
        self.btn_browse_ckpt = QPushButton("Browse…")
        self.btn_browse_ckpt.setToolTip("Pick a checkpoint file outside checkpoints/.")
        row2.addWidget(self.checkpoint_combo, 1)
        row2.addWidget(self.btn_refresh_ckpt)
        row2.addWidget(self.btn_browse_ckpt)
        c_outer.addLayout(row2)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_refresh_ckpt.clicked.connect(self._reload_checkpoints)
        self.btn_browse_ckpt.clicked.connect(self._on_browse_checkpoint)

        # ─ Runtime toggles + memory caps (shared widget, also used by BacktestTab) ─
        self.overrides_panel = RuntimeOverridesPanel()
        self._initialize_overrides_panel_defaults()

        # ─ Status grid ─
        status_box = QGroupBox("Runtime State")
        grid = QGridLayout(status_box)
        self.card_pid = MetricCard("PID")
        self.card_started = MetricCard("Started at")
        self.card_last_log = MetricCard("Last log timestamp")
        self.card_symbols = MetricCard("Symbols")
        self.card_frozen = MetricCard("Frozen brain")
        self.card_mem = MetricCard("Memory")
        self.card_mem_updates = MetricCard("Memory updates")
        # EV gate + calibration cards replace the old fusion/hopfield/priors/
        # hysteresis/topology surface (all removed in Great Simplification).
        self.card_ev_gate = MetricCard("EV gate")
        self.card_ev_cost = MetricCard("Round-trip cost")
        self.card_platt = MetricCard("Platt calibration")
        grid.addWidget(self.card_pid, 0, 0)
        grid.addWidget(self.card_started, 0, 1)
        grid.addWidget(self.card_last_log, 0, 2)
        grid.addWidget(self.card_symbols, 1, 0, 1, 3)
        grid.addWidget(self.card_frozen, 2, 0)
        grid.addWidget(self.card_mem, 2, 1)
        grid.addWidget(self.card_mem_updates, 2, 2)
        grid.addWidget(self.card_ev_gate, 3, 0)
        grid.addWidget(self.card_ev_cost, 3, 1)
        grid.addWidget(self.card_platt, 3, 2)

        # ─ Recent decision/risk ─
        recent = QGroupBox("Most recent log evidence")
        r_layout = QVBoxLayout(recent)
        r_layout.setSpacing(6)
        self.lbl_decision = QLabel("Last decision: —")
        self.lbl_decision.setWordWrap(True)
        self.lbl_risk = QLabel("Last risk decision: —")
        self.lbl_risk.setWordWrap(True)
        self.lbl_message = QLabel("Last message: —")
        self.lbl_message.setWordWrap(True)
        _mono = (
            "font-family: 'Cascadia Mono', Consolas, monospace; "
            "font-size: 11px; border: 0; background: transparent;"
        )
        self.lbl_decision.setStyleSheet(f"color: {COLOR_INFO}; {_mono}")
        self.lbl_risk.setStyleSheet(f"color: {COLOR_WARN}; {_mono}")
        self.lbl_message.setStyleSheet(f"color: {COLOR_TEXT}; {_mono}")
        for w in (self.lbl_decision, self.lbl_risk, self.lbl_message):
            w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            r_layout.addWidget(w)

        # ─ Error banner ─
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet(
            f"color: {COLOR_NEGATIVE}; font-size: 12px; font-weight: bold; border: 0;"
        )
        self.lbl_error.setWordWrap(True)

        root.addWidget(controls)
        root.addWidget(self.overrides_panel)
        root.addWidget(status_box)
        root.addWidget(recent)
        root.addWidget(self.lbl_error)
        root.addStretch(1)

        # Populate checkpoint combo last so the service has time to preview
        # the config if the user rebuilds it.
        self._reload_checkpoints()

    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self.svc.is_running():
            QMessageBox.information(self, "Already running", "Runtime is already running.")
            return
        overrides = self.overrides_panel.to_overrides()
        overrides.checkpoint_path = self._selected_checkpoint_path()
        try:
            self.svc.start(overrides=overrides)
        except ServiceError as exc:
            QMessageBox.critical(self, "Failed to start", str(exc))
        self.refresh()

    def _on_stop(self) -> None:
        if not self.svc.is_running():
            self.refresh()
            return
        self.svc.stop()
        self.refresh()

    # ------------------------------------------------------------------
    # checkpoint picker helpers
    # ------------------------------------------------------------------
    def _reload_checkpoints(self) -> None:
        """Rescan the checkpoints/ directory and rebuild the picker combo.

        First entry is always '(use config default)' which maps to None.
        Real files are sorted by mtime descending so the most recent
        checkpoint is second in the list.  Skips anything in a _trash/
        subfolder or with 'trash' in the path.
        """
        # Preserve the current selection if possible.
        prev = self._selected_checkpoint_path()
        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItem(_USE_CONFIG_DEFAULT, None)

        files: list[Path] = []
        if _CHECKPOINTS_DIR.exists():
            for p in _CHECKPOINTS_DIR.glob("*.pt"):
                if not p.is_file():
                    continue
                if "trash" in p.name.lower():
                    continue
                files.append(p)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for p in files:
            self.checkpoint_combo.addItem(p.name, str(p))
        # Restore previous selection if it still exists in the list.
        if prev:
            for i in range(self.checkpoint_combo.count()):
                if self.checkpoint_combo.itemData(i) == prev:
                    self.checkpoint_combo.setCurrentIndex(i)
                    break
        self.checkpoint_combo.blockSignals(False)

    def _on_browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select checkpoint",
            str(_CHECKPOINTS_DIR if _CHECKPOINTS_DIR.exists() else Path.cwd()),
            "Checkpoints (*.pt *.pth);;All files (*)",
        )
        if not path:
            return
        # Add as a transient first entry (after the default) and select it.
        for i in range(self.checkpoint_combo.count()):
            if self.checkpoint_combo.itemData(i) == path:
                self.checkpoint_combo.setCurrentIndex(i)
                return
        self.checkpoint_combo.insertItem(1, Path(path).name + "  (browsed)", path)
        self.checkpoint_combo.setCurrentIndex(1)

    def _selected_checkpoint_path(self) -> str | None:
        data = self.checkpoint_combo.currentData()
        return str(data) if data else None

    def _initialize_overrides_panel_defaults(self) -> None:
        """Populate the overrides panel from the current YAML config stack.

        Config loading is done in a background thread to avoid blocking the
        GUI event loop on startup.
        """
        def _load():
            summary = self.svc.preview_config()
            caps = self.svc.preview_memory_caps()
            return (summary, caps)

        worker = CallableWorker(_load)
        worker.signals.finished.connect(self._on_overrides_loaded)
        # failed → silently ignore, same as the old except ServiceError: return
        QThreadPool.globalInstance().start(worker)

    def _on_overrides_loaded(self, result: object) -> None:
        """Runs on main thread — safe to update widgets here."""
        if not isinstance(result, tuple) or len(result) != 2:
            return
        summary, caps = result
        self.overrides_panel.load_defaults(summary, extra=caps)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Kick off background status fetch — never blocks the GUI thread."""
        if self._fetch_pending:
            return
        self._fetch_pending = True
        worker = CallableWorker(self.svc.get_status)
        worker.signals.finished.connect(self._on_status_fetched)
        worker.signals.failed.connect(self._on_fetch_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_status_fetched(self, status: object) -> None:
        """Runs on main thread — safe to update widgets here."""
        self._fetch_pending = False
        self._render(status)

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_pending = False
        self.lbl_error.setText(msg)

    def _render(self, status: RuntimeStatus) -> None:
        state = status.state
        self.badge.set_state(state.value)
        running = state == RuntimeState.RUNNING
        self.btn_start.setEnabled(state in (RuntimeState.STOPPED, RuntimeState.ERROR))
        self.btn_stop.setEnabled(running)
        # Disable the overrides + checkpoint picker while running — changes
        # would be misleading because they only apply on the next Start.
        self.overrides_panel.set_editable(not running)
        self.checkpoint_combo.setEnabled(not running)
        self.btn_refresh_ckpt.setEnabled(not running)
        self.btn_browse_ckpt.setEnabled(not running)

        self.card_pid.set_value(
            str(status.pid) if status.pid else "—",
            color=COLOR_TEXT if status.pid else COLOR_MUTED,
        )
        self.card_started.set_value(
            status.started_at.strftime("%H:%M:%S") if status.started_at else "—",
            color=COLOR_TEXT if status.started_at else COLOR_MUTED,
        )
        self.card_last_log.set_value(
            status.last_log_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            if status.last_log_timestamp
            else "—",
            color=COLOR_INFO if status.last_log_timestamp else COLOR_MUTED,
        )
        # Symbols: format as a multi-row grid so 40+ tickers render cleanly
        # inside the card instead of as one enormous overflowing line.  The
        # MetricCard value label is monospace with word-wrap enabled and a
        # Preferred vertical size policy, so newline-chunked rows flow
        # naturally without clipping.
        if status.symbols:
            per_row = 8
            rows = [
                ", ".join(status.symbols[i:i + per_row])
                for i in range(0, len(status.symbols), per_row)
            ]
            symbols_text = "\n".join(rows)
            symbols_color = COLOR_TEXT
        else:
            symbols_text = "—"
            symbols_color = COLOR_MUTED
        self.card_symbols.set_value(symbols_text, color=symbols_color)

        def _flag(val: bool, on_label: str = "ON", off_label: str = "off") -> tuple[str, str]:
            return (on_label, COLOR_POSITIVE) if val else (off_label, COLOR_MUTED)

        # "frozen" isn't really on/off good/bad — use info/warn colors.
        self.card_frozen.set_value(
            "YES (frozen)" if status.frozen_brain else "NO (training)",
            color=COLOR_INFO if status.frozen_brain else COLOR_WARN,
        )
        for card, flag in (
            (self.card_mem, status.memory_enabled),
            (self.card_mem_updates, status.memory_updates_enabled),
            (self.card_ev_gate, status.ev_gate_active),
        ):
            txt, col = _flag(flag)
            card.set_value(txt, color=col)

        # Round-trip cost in bps (fee + slippage). Always shown as info.
        self.card_ev_cost.set_value(
            f"{status.ev_cost_bps:.1f} bps",
            color=COLOR_INFO if status.ev_cost_bps > 0 else COLOR_MUTED,
        )

        # Platt calibration has three states: enabled + fitted (green), enabled
        # but not yet fitted (warn), disabled (muted).
        if status.platt_calibration_enabled and status.platt_calibration_fitted:
            self.card_platt.set_value("fitted", color=COLOR_POSITIVE)
        elif status.platt_calibration_enabled:
            self.card_platt.set_value("awaiting retrain", color=COLOR_WARN)
        else:
            self.card_platt.set_value("off", color=COLOR_MUTED)

        self.lbl_decision.setText(f"Last decision: {status.last_decision_summary or '—'}")
        self.lbl_risk.setText(f"Last risk decision: {status.last_risk_decision_summary or '—'}")
        self.lbl_message.setText(f"Last message: {status.last_message or '—'}")
        self.lbl_error.setText(status.error_message or "")
