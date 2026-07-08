"""Backtest tab — run a backtest in a worker thread, show recent runs.

v2 adds:
  - Policy picker (baseline / brain / brain+memory)
  - Embedded RuntimeOverridesPanel (same widget used by RuntimeTab)
  - Prettified preset display names in the dropdown
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MetricCard, RuntimeOverridesPanel
from src.gui.workers import CallableWorker
from src.gui_services.backtest_service import list_recent_results, run_backtest
from src.gui_services.config_service import list_config_presets
from src.gui_services.runtime_service import get_runtime_service
from src.gui_services.state import BacktestResultRow, ServiceError


_POLICY_KINDS = ["baseline", "brain", "brain+memory"]
_BACKTEST_OVERRIDE_YAML = Path("runs/gui_backtest_override.yaml")


def _prettify_preset_name(path: Path) -> str:
    """Turn ``backtest_fusion_off.yaml`` into ``Backtest Fusion Off``."""
    stem = path.stem
    words = stem.replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) if words else stem


class BacktestTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._building = False
        self._build_ui()
        self._reload_presets()
        self._reload_table()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        controls = QGroupBox("Run Backtest")
        c = QHBoxLayout(controls)
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(220)
        self.policy_combo = QComboBox()
        for kind in _POLICY_KINDS:
            self.policy_combo.addItem(kind, kind)
        self.policy_combo.setCurrentText("baseline")
        self.policy_combo.setToolTip(
            "baseline       — BaselineMomentumPolicy (no model)\n"
            "brain          — BrainPolicy loaded from config.model.checkpoint_path\n"
            "brain+memory   — MemoryAwarePolicy (brain + memory + fusion)"
        )
        self.real_data = QCheckBox("Real data")
        self.walk_forward = QCheckBox("Walk-forward")
        self.walk_forward.setToolTip(
            "Run walk-forward cross-validation instead of a single contiguous backtest.\n"
            "Splits data into chronological folds — trains on each fold's training window,\n"
            "tests on its test window. Prevents data leakage between folds.\n"
            "Configure fold sizes in evaluation.walk_forward section of the config."
        )
        self.btn_run = QPushButton("Run Backtest")
        self.btn_refresh = QPushButton("Refresh history")
        c.addWidget(QLabel("Preset:"))
        c.addWidget(self.preset_combo, 2)
        c.addWidget(QLabel("Policy:"))
        c.addWidget(self.policy_combo, 1)
        c.addWidget(self.real_data)
        c.addWidget(self.walk_forward)
        c.addWidget(self.btn_run)
        c.addWidget(self.btn_refresh)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_refresh.clicked.connect(self._reload_table)
        self.policy_combo.currentTextChanged.connect(self._on_policy_changed)

        # Shared override panel — exact same widget used by the Runtime tab.
        # Starts collapsed visually via a checkable groupbox (Qt feature) so
        # small-screen operators can tuck it away; operators who want to tune
        # can click the title to expand.  Default is expanded so the
        # existence of the control surface is obvious.
        self.overrides_panel = RuntimeOverridesPanel("Backtest overrides (toggles + memory caps)")
        self.overrides_panel.setCheckable(True)
        self.overrides_panel.setChecked(True)
        self._initialize_overrides_from_service()

        # Latest result cards
        latest = QGroupBox("Latest Result")
        grid = QGridLayout(latest)
        self.card_total = MetricCard("Total return")
        self.card_dd = MetricCard("Max drawdown")
        self.card_win = MetricCard("Win rate")
        self.card_to = MetricCard("Turnover")
        self.card_fills = MetricCard("Fills")
        self.card_pf = MetricCard("Profit factor")
        self.card_exp = MetricCard("Expectancy")
        self.card_sharpe = MetricCard("Sharpe")
        self.card_sortino = MetricCard("Sortino")
        self.card_exposure = MetricCard("Avg exposure")
        grid.addWidget(self.card_total, 0, 0)
        grid.addWidget(self.card_dd, 0, 1)
        grid.addWidget(self.card_win, 0, 2)
        grid.addWidget(self.card_to, 0, 3)
        grid.addWidget(self.card_fills, 1, 0)
        grid.addWidget(self.card_pf, 1, 1)
        grid.addWidget(self.card_exp, 1, 2)
        grid.addWidget(self.card_sharpe, 1, 3)
        grid.addWidget(self.card_sortino, 2, 0)
        grid.addWidget(self.card_exposure, 2, 1)

        # Status / progress
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #aaaaaa;")

        # History table
        history = QGroupBox("Recent Backtests")
        h_layout = QVBoxLayout(history)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Label", "Return", "Drawdown", "Win%", "Fills", "Profit Factor"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        h_layout.addWidget(self.table)

        root.addWidget(controls)
        root.addWidget(self.overrides_panel)
        root.addWidget(latest)
        root.addWidget(self.lbl_status)
        root.addWidget(history, 1)

        # Apply initial policy-aware enable/disable of memory controls.
        self._on_policy_changed(self.policy_combo.currentText())

    # ------------------------------------------------------------------
    def _reload_presets(self) -> None:
        self.preset_combo.clear()
        for path in list_config_presets():
            label = _prettify_preset_name(path)
            self.preset_combo.addItem(label, str(path))
        if self.preset_combo.count() == 0:
            self.preset_combo.addItem("Default", "configs/default.yaml")

    def _initialize_overrides_from_service(self) -> None:
        try:
            svc = get_runtime_service()
            summary = svc.preview_config()
            caps = svc.preview_memory_caps()
        except ServiceError:
            return
        self.overrides_panel.load_defaults(summary, extra=caps)

    def _on_policy_changed(self, policy_kind: str) -> None:
        memory_relevant = policy_kind == "brain+memory"
        self.overrides_panel.set_memory_controls_relevant(memory_relevant)

    def _selected_config_paths(self) -> list[str]:
        """Compose the YAML stack for this run.

        Always starts with configs/default.yaml; then the selected preset
        (unless the preset IS default.yaml); then the auto-generated GUI
        override file.
        """
        paths: list[str] = ["configs/default.yaml"]
        preset = self.preset_combo.currentData()
        if preset:
            preset_path = Path(preset)
            if preset_path.name != "default.yaml":
                paths.append(str(preset_path))
        # Write the override YAML file next to the runtime one.
        # NOTE: RuntimeOverridesPanel does not currently surface a
        # checkpoint_path picker, so `overrides.checkpoint_path` is always
        # None and the resulting YAML contains no `model.checkpoint_path`
        # entry.  The backtest therefore uses whatever
        # `config.model.checkpoint_path` the selected preset (or default.yaml)
        # declares.  If a per-run checkpoint picker is added to the panel in
        # the future, `to_overrides()` will start emitting `model.checkpoint_path`
        # automatically via `RuntimeOverrides.to_yaml_dict()` and the
        # downstream config merge will pick it up with no extra wiring here.
        overrides = self.overrides_panel.to_overrides()
        try:
            overrides.write_yaml(_BACKTEST_OVERRIDE_YAML)
        except OSError as e:
            # Non-fatal; the override file is optional.  The run just uses
            # whatever the preset + default.yaml say.  Surface the failure
            # so the operator knows their toggles will be ignored this run.
            self.lbl_status.setText(f"Warning: could not write overrides YAML: {e}")
        else:
            paths.append(str(_BACKTEST_OVERRIDE_YAML))
        return paths

    # ------------------------------------------------------------------
    def _on_run(self) -> None:
        if self._building:
            return
        self._building = True
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Starting backtest…")

        label_parts = [self.preset_combo.currentText().replace(".yaml", "")]
        label_parts.append(self.policy_combo.currentText())
        label = " :: ".join(p for p in label_parts if p)

        worker = CallableWorker(
            run_backtest,
            config_paths=self._selected_config_paths(),
            use_real_data=self.real_data.isChecked(),
            policy_kind=self.policy_combo.currentText(),
            walk_forward=self.walk_forward.isChecked(),
            label=label,
        )
        worker.signals.progress.connect(self.lbl_status.setText)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_finished(self, row: object) -> None:
        self._building = False
        self.btn_run.setEnabled(True)
        if isinstance(row, BacktestResultRow):
            self._render_latest(row)
        self.lbl_status.setText("Backtest complete.")
        self._reload_table()

    def _on_failed(self, msg: str) -> None:
        self._building = False
        self.btn_run.setEnabled(True)
        self.lbl_status.setText("Backtest failed.")
        QMessageBox.critical(self, "Backtest failed", msg)

    # ------------------------------------------------------------------
    def _render_latest(self, row: BacktestResultRow) -> None:
        self.card_total.set_value(f"{row.total_return:+.4f}")
        self.card_dd.set_value(f"{row.max_drawdown:.4f}")
        self.card_win.set_value(f"{row.win_rate:.2%}")
        self.card_to.set_value(f"{row.turnover:.4f}")
        self.card_fills.set_value(str(row.fill_count))
        self.card_pf.set_value(f"{row.profit_factor:.3f}")
        self.card_exp.set_value(f"{row.expectancy:+.6f}")
        # Risk-adjusted metrics — default to 0.0 when the field is missing
        # (older persisted rows) so we still render a numeric placeholder.
        self.card_sharpe.set_value(f"{getattr(row, 'sharpe_ratio', 0.0):.2f}")
        self.card_sortino.set_value(f"{getattr(row, 'sortino_ratio', 0.0):.2f}")
        self.card_exposure.set_value(f"{getattr(row, 'avg_exposure', 0.0):.1%}")

    def _reload_table(self) -> None:
        rows = list_recent_results(limit=100)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            items = [
                row.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                row.label,
                f"{row.total_return:+.4f}",
                f"{row.max_drawdown:.4f}",
                f"{row.win_rate:.2%}",
                str(row.fill_count),
                f"{row.profit_factor:.3f}",
            ]
            for c, text in enumerate(items):
                it = QTableWidgetItem(text)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, it)
