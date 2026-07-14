"""Experiments tab — run ablation suites, view comparison tables.

Provides a GUI for selecting experiment presets, running them in a
background worker thread, and viewing side-by-side metric comparisons
in a styled ranking table.

Key features:
  - "Run All" button runs every experiment in one click
  - Feature caching: experiments sharing the same feature config skip
    the expensive torch.unfold pass (~5-10x faster for typical suites)
  - Real-time progress with time estimates
  - Color-coded comparison table ranked by Sharpe ratio
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import (
    COLOR_ACCENT,
    COLOR_INFO,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    COLOR_POSITIVE,
    COLOR_TEXT,
    MetricCard,
    StatusBadge,
)
from src.gui.workers import CallableWorker
from src.gui_services.experiment_service import list_available_experiments, run_experiment_suite

# Experiments grouped by category for the selection panel
_EXPERIMENT_GROUPS = {
    "Core ablations": [
        ("full_model", "Full model (reference)"),
        ("brain_only", "Brain only (no memory)"),
        ("brain_no_flip", "Brain + FLAT allowed"),
    ],
    "Component ablations": [
        ("no_regime_encoder", "No regime encoder"),
        ("no_temporal_cnn", "No temporal CNN"),
        ("no_latent_memory", "No GRU latent memory"),
        ("no_confidence_gating", "No confidence gating"),
    ],
    "Memory": [
        ("memory_off", "Memory OFF"),
        ("no_online_memory", "No online memory updates"),
    ],
    "Risk variants": [
        ("strict_risk", "Strict risk"),
        ("loose_risk", "Loose risk"),
    ],
    "Baselines": [
        ("buy_and_hold", "Buy and hold"),
        ("momentum_baseline", "Momentum baseline"),
        ("mean_reversion", "Mean reversion"),
    ],
    "Ablations": [
        # After the Great Simplification, the only live toggles worth
        # ablating are the risk head (drives the EV gate), Platt calibration,
        # and the CNN on OHLCV. Presets referencing removed features
        # (fusion / hopfield / hysteresis / priors / etc.) were dropped.
        ("no_separate_risk_heads", "No separate risk heads"),
    ],
}

# Key metrics to show in the comparison table (display_name, key, format, higher_is_better)
_COMPARISON_METRICS = [
    ("Return", "total_return", "+.4f", True),
    ("Sharpe", "sharpe_ratio", "+.3f", True),
    ("Sortino", "sortino_ratio", "+.3f", True),
    ("MaxDD", "max_drawdown", ".4f", False),
    ("Win%", "win_rate", ".3f", True),
    ("PF", "profit_factor", ".3f", True),
    ("Expect.", "expectancy", "+.4f", True),
    ("Fills", "fill_count", ".0f", None),
    ("Flat%", "flat_fraction", ".3f", None),
    ("Fees", "total_fees", ".2f", False),
]


class ExperimentsTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._running = False
        self._checkboxes: dict[str, QCheckBox] = {}
        self._last_results: dict[str, dict[str, float]] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)

        # ── Left panel: experiment selection ─────────────────────────
        left = QVBoxLayout()

        # Selection group boxes by category
        select_area = QScrollArea()
        select_area.setWidgetResizable(True)
        select_area.setMaximumWidth(320)
        select_area.setStyleSheet(
            f"QScrollArea {{ border: 1px solid {COLOR_PANEL_BORDER}; "
            f"background: {COLOR_PANEL_BG}; }}"
        )
        select_widget = QWidget()
        select_layout = QVBoxLayout(select_widget)
        select_layout.setContentsMargins(8, 8, 8, 8)

        for group_name, experiments in _EXPERIMENT_GROUPS.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout(group)
            group_layout.setSpacing(4)
            for exp_id, exp_label in experiments:
                cb = QCheckBox(exp_label)
                cb.setToolTip(f"Experiment preset: {exp_id}")
                if exp_id in ("full_model", "brain_only"):
                    cb.setChecked(True)
                self._checkboxes[exp_id] = cb
                group_layout.addWidget(cb)
            select_layout.addWidget(group)

        select_layout.addStretch(1)
        select_area.setWidget(select_widget)

        # Quick-select buttons
        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_none = QPushButton("Clear All")
        btn_core = QPushButton("Core Only")
        btn_select_all.clicked.connect(self._select_all)
        btn_none.clicked.connect(self._select_none)
        btn_core.clicked.connect(self._select_core)
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_core)

        # Options
        options = QGroupBox("Options")
        opt_layout = QVBoxLayout(options)
        self.chk_real_data = QCheckBox("Use real historical data")
        self.chk_real_data.setToolTip(
            "Use cached/fetched real bars from Alpaca.\n"
            "Unchecked = synthetic bars (fast, for testing)."
        )
        self.chk_fast_mode = QCheckBox("Fast mode (~15s per experiment)")
        self.chk_fast_mode.setChecked(True)
        self.chk_fast_mode.setToolTip(
            "Use only the last ~2 weeks of data (1000 bars per symbol).\n"
            "Each experiment takes ~15s. Full 15-experiment suite: ~5 min.\n"
            "Uncheck to use the full history (much slower but more robust)."
        )
        self.chk_walk_forward = QCheckBox("Walk-forward evaluation")
        self.chk_walk_forward.setToolTip(
            "Instead of one backtest on the full period, split data into\n"
            "chronological folds: train → test → slide forward → repeat.\n"
            "Each fold trains a fresh model on past data only, then tests\n"
            "on unseen future data. Prevents overfitting and leakage.\n"
            "Much slower but statistically rigorous."
        )
        self.chk_walk_forward.stateChanged.connect(self._on_walk_forward_toggled)
        opt_layout.addWidget(self.chk_real_data)
        opt_layout.addWidget(self.chk_fast_mode)
        opt_layout.addWidget(self.chk_walk_forward)

        # Walk-forward parameters (hidden until checkbox is checked)
        self.wf_group = QGroupBox("Walk-Forward Settings")
        wf_layout = QGridLayout(self.wf_group)
        wf_layout.setHorizontalSpacing(8)

        self.spin_train = QSpinBox()
        self.spin_train.setRange(200, 50000)
        self.spin_train.setValue(2000)
        self.spin_train.setToolTip("Training window size in bars")

        self.spin_test = QSpinBox()
        self.spin_test.setRange(100, 10000)
        self.spin_test.setValue(500)
        self.spin_test.setToolTip("Test window size in bars")

        self.spin_step = QSpinBox()
        self.spin_step.setRange(100, 10000)
        self.spin_step.setValue(500)
        self.spin_step.setToolTip("Step size in bars (how far the window slides)")

        lbl_style = f"color: {COLOR_MUTED}; border: 0; background: transparent;"
        lbl_train = QLabel("Train bars:")
        lbl_train.setStyleSheet(lbl_style)
        lbl_test = QLabel("Test bars:")
        lbl_test.setStyleSheet(lbl_style)
        lbl_step = QLabel("Step bars:")
        lbl_step.setStyleSheet(lbl_style)
        wf_layout.addWidget(lbl_train, 0, 0)
        wf_layout.addWidget(self.spin_train, 0, 1)
        wf_layout.addWidget(lbl_test, 0, 2)
        wf_layout.addWidget(self.spin_test, 0, 3)
        wf_layout.addWidget(lbl_step, 1, 0)
        wf_layout.addWidget(self.spin_step, 1, 1)

        self.wf_group.setVisible(False)
        opt_layout.addWidget(self.wf_group)

        # ── Run buttons ──────────────────────────────────────────────
        _accent_btn_style = (
            f"QPushButton {{ background-color: #1a3a5c; color: {COLOR_ACCENT}; "
            f"border: 2px solid {COLOR_ACCENT}; font-size: 13px; font-weight: 700; "
            f"letter-spacing: 1px; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: #1f4a6e; }}"
            f"QPushButton:disabled {{ background-color: #13171e; color: #4b5563; "
            f"border-color: #1f2937; }}"
        )

        # "Run Selected" — runs only checked experiments
        self.btn_run = QPushButton("Run Selected")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setStyleSheet(_accent_btn_style)
        self.btn_run.setToolTip("Run only the experiments you checked above.")
        self.btn_run.clicked.connect(self._on_run_selected)

        # "Run All Experiments" — one click, runs everything
        self.btn_run_all = QPushButton("Run All Experiments")
        self.btn_run_all.setMinimumHeight(40)
        self.btn_run_all.setStyleSheet(
            f"QPushButton {{ background-color: #1a4a2c; color: {COLOR_POSITIVE}; "
            f"border: 2px solid {COLOR_POSITIVE}; font-size: 14px; font-weight: 700; "
            f"letter-spacing: 1px; border-radius: 6px; }}"
            f"QPushButton:hover {{ background-color: #1f5a3e; }}"
            f"QPushButton:disabled {{ background-color: #13171e; color: #4b5563; "
            f"border-color: #1f2937; }}"
        )
        self.btn_run_all.setToolTip(
            "Run ALL 27 experiments in one go.\n"
            "Features are cached across experiments with the same config,\n"
            "so this is much faster than running them individually."
        )
        self.btn_run_all.clicked.connect(self._on_run_all)

        self.badge = StatusBadge("ready")
        self.badge.set_state("stopped", "READY")
        self.lbl_progress = QLabel("")
        self.lbl_progress.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 11px;")
        self.lbl_progress.setWordWrap(True)
        self.lbl_progress.setMinimumHeight(40)

        left.addWidget(select_area, 1)
        left.addLayout(btn_row)
        left.addWidget(options)
        left.addWidget(self.btn_run)
        left.addWidget(self.btn_run_all)
        left.addWidget(self.badge)
        left.addWidget(self.lbl_progress)

        # ── Right panel: results ─────────────────────────────────────
        right = QVBoxLayout()

        # Summary cards
        summary = QGroupBox("Best Experiment Summary")
        summary_grid = QGridLayout(summary)
        self.card_best_name = MetricCard("Best by Sharpe")
        self.card_best_return = MetricCard("Return")
        self.card_best_sharpe = MetricCard("Sharpe")
        self.card_best_dd = MetricCard("Max Drawdown")
        self.card_completed = MetricCard("Completed")
        self.card_failed = MetricCard("Failed")
        self.card_time = MetricCard("Total Time")
        summary_grid.addWidget(self.card_best_name, 0, 0)
        summary_grid.addWidget(self.card_best_return, 0, 1)
        summary_grid.addWidget(self.card_best_sharpe, 0, 2)
        summary_grid.addWidget(self.card_best_dd, 0, 3)
        summary_grid.addWidget(self.card_completed, 1, 0)
        summary_grid.addWidget(self.card_failed, 1, 1)
        summary_grid.addWidget(self.card_time, 1, 2)

        # Comparison table
        results_group = QGroupBox("Experiment Comparison (ranked by Sharpe)")
        results_layout = QVBoxLayout(results_group)
        n_metrics = len(_COMPARISON_METRICS)
        self.results_table = QTableWidget(0, 1 + n_metrics)
        headers = ["Experiment"] + [m[0] for m in _COMPARISON_METRICS]
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)

        right.addWidget(summary)
        right.addWidget(results_group, 1)

        root.addLayout(left)
        root.addLayout(right, 1)

    # ── Quick-select helpers ─────────────────────────────────────────

    def _select_all(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(True)

    def _select_none(self) -> None:
        for cb in self._checkboxes.values():
            cb.setChecked(False)

    def _select_core(self) -> None:
        core = {"full_model", "brain_only", "brain_no_flip", "memory_off"}
        for name, cb in self._checkboxes.items():
            cb.setChecked(name in core)

    def _on_walk_forward_toggled(self, state: int) -> None:
        self.wf_group.setVisible(bool(state))

    # ── Run experiments ──────────────────────────────────────────────

    def _selected_experiments(self) -> list[str]:
        return [name for name, cb in self._checkboxes.items() if cb.isChecked()]

    def _all_experiments(self) -> list[str]:
        return list(self._checkboxes.keys())

    def _on_run_selected(self) -> None:
        """Run only the checked experiments."""
        selected = self._selected_experiments()
        if not selected:
            QMessageBox.warning(
                self, "No experiments selected",
                "Please select at least one experiment to run.",
            )
            return
        self._start_suite(selected)

    def _on_run_all(self) -> None:
        """Run all experiments in one click — selects all checkboxes first."""
        self._select_all()
        self._start_suite(self._all_experiments())

    def _start_suite(self, experiment_names: list[str]) -> None:
        if self._running:
            return

        self._running = True
        self.btn_run.setEnabled(False)
        self.btn_run_all.setEnabled(False)
        self.badge.set_state("running", "RUNNING")
        self.lbl_progress.setText(
            f"Starting {len(experiment_names)} experiments...\n"
            f"Features are cached — subsequent experiments with the same\n"
            f"config run much faster."
        )

        wf_params = None
        if self.chk_walk_forward.isChecked():
            wf_params = {
                "train_bars": self.spin_train.value(),
                "test_bars": self.spin_test.value(),
                "step_bars": self.spin_step.value(),
            }
        worker = CallableWorker(
            run_experiment_suite,
            experiment_names=experiment_names,
            use_real_data=self.chk_real_data.isChecked(),
            fast_mode=self.chk_fast_mode.isChecked(),
            walk_forward_params=wf_params,
            seed=42,
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_progress(self, msg: str) -> None:
        self.lbl_progress.setText(msg)

    def _on_finished(self, result: object) -> None:
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(True)
        self.badge.set_state("stopped", "DONE")

        if not isinstance(result, dict):
            self.lbl_progress.setText("Unexpected result format.")
            return

        self._last_results = result.get("results", {})
        completed = result.get("completed", 0)
        failed = result.get("failed", [])
        total_sec = result.get("total_seconds", 0.0)

        self.lbl_progress.setText(
            f"Completed {completed} experiments in {total_sec:.1f}s"
            + (f"\n{len(failed)} failed" if failed else "")
        )

        self._render_summary(self._last_results, completed, failed, total_sec)
        self._render_comparison_table(self._last_results)

    def _on_failed(self, msg: str) -> None:
        self._running = False
        self.btn_run.setEnabled(True)
        self.btn_run_all.setEnabled(True)
        self.badge.set_state("error", "ERROR")
        self.lbl_progress.setText(f"Suite failed: {msg}")
        QMessageBox.critical(self, "Experiment suite failed", msg)

    # ── Rendering ────────────────────────────────────────────────────

    def _render_summary(
        self,
        results: dict[str, dict[str, float]],
        completed: int,
        failed: list,
        total_seconds: float,
    ) -> None:
        self.card_completed.set_value(str(completed), color=COLOR_POSITIVE)
        self.card_failed.set_value(
            str(len(failed)),
            color=COLOR_NEGATIVE if failed else COLOR_POSITIVE,
        )
        self.card_time.set_value(f"{total_seconds:.1f}s", color=COLOR_INFO)

        if not results:
            return

        # Find best by Sharpe
        best_name = max(results, key=lambda n: results[n].get("sharpe_ratio", -999))
        best_metrics = results[best_name]

        self.card_best_name.set_value(best_name, color=COLOR_ACCENT)
        ret = best_metrics.get("total_return", 0.0)
        self.card_best_return.set_value(
            f"{ret:+.4f}",
            color=COLOR_POSITIVE if ret > 0 else COLOR_NEGATIVE,
        )
        sharpe = best_metrics.get("sharpe_ratio", 0.0)
        self.card_best_sharpe.set_value(
            f"{sharpe:+.3f}",
            color=COLOR_POSITIVE if sharpe > 0 else COLOR_NEGATIVE,
        )
        dd = best_metrics.get("max_drawdown", 0.0)
        self.card_best_dd.set_value(f"{dd:.4f}", color=COLOR_NEGATIVE if dd < -0.05 else COLOR_TEXT)

    def _render_comparison_table(self, results: dict[str, dict[str, float]]) -> None:
        if not results:
            self.results_table.setRowCount(0)
            return

        # Sort by Sharpe descending
        sorted_names = sorted(
            results.keys(),
            key=lambda n: results[n].get("sharpe_ratio", -999),
            reverse=True,
        )

        # Find best value for each metric (for highlighting)
        best_vals: dict[str, float] = {}
        for _, key, _, higher_is_better in _COMPARISON_METRICS:
            if higher_is_better is None:
                continue
            values = [results[n].get(key, 0.0) for n in sorted_names]
            if higher_is_better:
                best_vals[key] = max(values) if values else 0.0
            else:
                best_vals[key] = min(values) if values else 0.0

        self.results_table.setRowCount(len(sorted_names))

        for row_idx, name in enumerate(sorted_names):
            metrics = results[name]

            # Experiment name column (with rank)
            name_item = QTableWidgetItem(f"{row_idx + 1}. {name}")
            name_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            font = QFont()
            font.setBold(row_idx == 0)
            font.setPointSize(11)
            name_item.setFont(font)
            if row_idx == 0:
                name_item.setForeground(QColor(COLOR_ACCENT))
            self.results_table.setItem(row_idx, 0, name_item)

            # Metric columns
            for col_idx, (_, key, fmt, higher_is_better) in enumerate(_COMPARISON_METRICS):
                value = metrics.get(key, 0.0)
                text = f"{value:{fmt}}"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                # Color coding
                if higher_is_better is not None and key in best_vals:
                    if abs(value - best_vals[key]) < 1e-9:
                        item.setForeground(QColor(COLOR_POSITIVE))
                        font = QFont()
                        font.setBold(True)
                        font.setPointSize(11)
                        item.setFont(font)
                    elif key == "max_drawdown" and value < -0.10:
                        item.setForeground(QColor(COLOR_NEGATIVE))
                    elif key == "total_return":
                        if value > 0:
                            item.setForeground(QColor(COLOR_POSITIVE))
                        elif value < 0:
                            item.setForeground(QColor(COLOR_NEGATIVE))

                self.results_table.setItem(row_idx, 1 + col_idx, item)

        # Auto-size the first column wider
        self.results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
