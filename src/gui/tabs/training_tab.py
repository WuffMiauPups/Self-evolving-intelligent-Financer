"""Training tab — launch scripts/run_train.py from the GUI and tail its output.

Minimal surface: preset + fast/real-data + epochs + output checkpoint.
Power users still go through the CLI for detailed knob tuning; this tab
covers the common "start a training run without leaving the GUI" case.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import re

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import (
    COLOR_INFO,
    COLOR_MUTED,
    COLOR_NEGATIVE,
    COLOR_TEXT,
    MetricCard,
    StatusBadge,
)
from src.gui_services.state import RuntimeState, ServiceError
from src.gui_services.training_service import (
    TrainingStatus,
    get_training_service,
)


_PRESETS = ["(none)", "state-only", "classification-latent", "classification-only"]
_CHECKPOINTS_DIR = Path("checkpoints")


class TrainingTab(QWidget):
    POLL_MS = 2000

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.svc = get_training_service()
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)
        self.refresh()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        # ─ Training run configuration ─
        run_box = QGroupBox("Training Run")
        grid = QGridLayout(run_box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Preset:"), 0, 0)
        self.preset_combo = QComboBox()
        for p in _PRESETS:
            self.preset_combo.addItem(p, None if p == "(none)" else p)
        self.preset_combo.setToolTip(
            "Optional training preset from scripts/run_train.py PRESETS.\n"
            "Presets set sensible defaults for ablation studies."
        )
        grid.addWidget(self.preset_combo, 0, 1)

        self.chk_fast = QCheckBox("Fast experiment mode")
        self.chk_fast.setToolTip(
            "Enables fast_mode: reduced data + fewer epochs for rapid\n"
            "iteration.  Do NOT promote a fast-mode checkpoint to production."
        )
        grid.addWidget(self.chk_fast, 0, 2)

        self.chk_real_data = QCheckBox("Real data")
        self.chk_real_data.setChecked(True)
        self.chk_real_data.setToolTip(
            "Train from cached/fetched Alpaca historical bars.  Uncheck\n"
            "to use synthetic bars for smoke tests."
        )
        grid.addWidget(self.chk_real_data, 0, 3)

        grid.addWidget(QLabel("Epochs:"), 1, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 200)
        self.spin_epochs.setValue(3)
        self.spin_epochs.setToolTip(
            "Override training.epochs from the YAML config.  Fast mode\n"
            "may still cap this to a smaller value."
        )
        grid.addWidget(self.spin_epochs, 1, 1)

        self.chk_save_memory = QCheckBox("Save memory bank after training")
        self.chk_save_memory.setChecked(True)
        self.chk_save_memory.setToolTip(
            "Adds --save-memory to rebuild the associative memory bank\n"
            "from the new checkpoint immediately after training."
        )
        grid.addWidget(self.chk_save_memory, 1, 2, 1, 2)

        grid.addWidget(QLabel("Base checkpoint:"), 2, 0)
        self.base_combo = QComboBox()
        self.base_combo.setMinimumWidth(220)
        self.base_combo.setToolTip(
            "Optional: start training from an existing checkpoint.\n"
            "Picking nothing starts fresh."
        )
        grid.addWidget(self.base_combo, 2, 1, 1, 2)
        self.btn_base_refresh = QPushButton("Refresh")
        self.btn_base_refresh.clicked.connect(self._reload_checkpoints)
        grid.addWidget(self.btn_base_refresh, 2, 3)

        grid.addWidget(QLabel("Output checkpoint:"), 3, 0)
        self.output_field = QLineEdit()
        self.output_field.setPlaceholderText("checkpoints/exp_<ts>.pt  (leave empty for auto)")
        self.output_field.setToolTip(
            "Destination file for the new trained checkpoint.  Leave empty\n"
            "to auto-name as checkpoints/exp_YYYYMMDD_HHMMSS.pt."
        )
        grid.addWidget(self.output_field, 3, 1, 1, 2)
        self.btn_output_browse = QPushButton("Browse…")
        self.btn_output_browse.clicked.connect(self._on_browse_output)
        grid.addWidget(self.btn_output_browse, 3, 3)

        # Start/Stop + badge on the bottom row
        self.badge = StatusBadge("stopped")
        self.btn_start = QPushButton("Start training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)

        bottom = QHBoxLayout()
        bottom.addWidget(self.badge)
        bottom.addWidget(self.btn_start)
        bottom.addWidget(self.btn_stop)
        bottom.addStretch(1)
        grid.addLayout(bottom, 4, 0, 1, 4)

        root.addWidget(run_box)

        # ─ Advanced training options ─
        adv_box = QGroupBox("Advanced Training Options")
        adv_grid = QGridLayout(adv_box)
        adv_grid.setHorizontalSpacing(10)
        adv_grid.setVerticalSpacing(6)

        adv_grid.addWidget(QLabel("Early stopping patience (0=off):"), 0, 0)
        self.spin_early_stop = QSpinBox()
        self.spin_early_stop.setRange(0, 50)
        self.spin_early_stop.setValue(0)
        self.spin_early_stop.setToolTip(
            "Stop training if validation loss degrades for N consecutive\n"
            "epochs.  0 disables early stopping."
        )
        adv_grid.addWidget(self.spin_early_stop, 0, 1)

        adv_grid.addWidget(QLabel("Transaction cost (bps):"), 0, 2)
        self.spin_tx_cost = QDoubleSpinBox()
        self.spin_tx_cost.setRange(0.0, 50.0)
        self.spin_tx_cost.setSingleStep(0.5)
        self.spin_tx_cost.setValue(0.0)
        self.spin_tx_cost.setToolTip(
            "Estimated round-trip transaction cost in basis points.\n"
            "Widens the FLAT label band during target generation."
        )
        adv_grid.addWidget(self.spin_tx_cost, 0, 3)

        self.chk_platt = QCheckBox("Fit Platt calibration after training")
        self.chk_platt.setToolTip(
            "Fit Platt scaling on the validation set after training\n"
            "to calibrate prediction probabilities."
        )
        adv_grid.addWidget(self.chk_platt, 1, 0, 1, 2)

        root.addWidget(adv_box)

        # ─ Model architecture toggles ─
        arch_box = QGroupBox("Model Architecture")
        arch_grid = QGridLayout(arch_box)
        arch_grid.setHorizontalSpacing(10)
        arch_grid.setVerticalSpacing(6)

        self.chk_regime_encoder = QCheckBox("Regime encoder")
        self.chk_regime_encoder.setChecked(True)
        self.chk_regime_encoder.setToolTip("Enable regime-gated dendritic branches")
        arch_grid.addWidget(self.chk_regime_encoder, 0, 0)

        self.chk_temporal_cnn = QCheckBox("Temporal CNN")
        self.chk_temporal_cnn.setChecked(True)
        self.chk_temporal_cnn.setToolTip("Enable temporal convolutional encoder for price action patterns")
        arch_grid.addWidget(self.chk_temporal_cnn, 0, 1)

        self.chk_gru_memory = QCheckBox("GRU latent memory")
        self.chk_gru_memory.setChecked(True)
        self.chk_gru_memory.setToolTip("Enable GRU cell for temporal context across bars")
        arch_grid.addWidget(self.chk_gru_memory, 0, 2)

        self.chk_confidence_gating = QCheckBox("Branch confidence gating")
        self.chk_confidence_gating.setChecked(True)
        self.chk_confidence_gating.setToolTip("Enable per-branch confidence scores")
        arch_grid.addWidget(self.chk_confidence_gating, 1, 0)

        self.chk_soma_weighting = QCheckBox("Soma confidence weighting")
        self.chk_soma_weighting.setChecked(True)
        self.chk_soma_weighting.setToolTip("Weight soma gate by branch confidence (softmax + log trick)")
        arch_grid.addWidget(self.chk_soma_weighting, 1, 1)

        self.chk_decouple_conf = QCheckBox("Decouple confidence from soma")
        self.chk_decouple_conf.setChecked(False)
        self.chk_decouple_conf.setToolTip("Policy confidence independent of soma activation")
        arch_grid.addWidget(self.chk_decouple_conf, 1, 2)

        self.chk_risk_heads = QCheckBox("Separate risk heads (MFE/MAE/vol/dd)")
        self.chk_risk_heads.setToolTip(
            "Dedicate separate output heads for MFE, MAE, volatility,\n"
            "and drawdown prediction instead of sharing a single head."
        )
        arch_grid.addWidget(self.chk_risk_heads, 2, 0, 1, 2)

        root.addWidget(arch_box)

        # ─ Memory bank options (only matter when --save-memory is on) ─
        mem_box = QGroupBox("Memory Bank Options (used when saving memory)")
        mem_grid = QGridLayout(mem_box)
        mem_grid.setHorizontalSpacing(10)
        mem_grid.setVerticalSpacing(6)

        self.chk_mem_ran_merge = QCheckBox("RAN merge (online learning)")
        self.chk_mem_ran_merge.setToolTip("Blend similar items into prototypes during unit building")
        mem_grid.addWidget(self.chk_mem_ran_merge, 0, 0)

        self.chk_mem_auto_tune = QCheckBox("Auto-tune temperature (LOO)")
        self.chk_mem_auto_tune.setToolTip("Find optimal retrieval temperature via leave-one-out CV")
        mem_grid.addWidget(self.chk_mem_auto_tune, 0, 1)

        self.chk_mem_esom_labels = QCheckBox("ESOM regime labels")
        self.chk_mem_esom_labels.setToolTip("Learn regime labels from trade outcomes")
        mem_grid.addWidget(self.chk_mem_esom_labels, 0, 2)

        self.chk_mem_deep_proj = QCheckBox("Deep projections (W_Q/W_K/W_V)")
        self.chk_mem_deep_proj.setToolTip("Train learnable Hopfield projections on the memory bank")
        mem_grid.addWidget(self.chk_mem_deep_proj, 1, 0)

        self.chk_mem_value_decay = QCheckBox("Value-aware decay")
        self.chk_mem_value_decay.setToolTip("High-ROI patterns survive longer in memory")
        mem_grid.addWidget(self.chk_mem_value_decay, 1, 1)

        root.addWidget(mem_box)

        # Wire: disable memory options when "Save memory bank" is unchecked.
        self._mem_widgets = [
            self.chk_mem_ran_merge,
            self.chk_mem_auto_tune,
            self.chk_mem_esom_labels,
            self.chk_mem_deep_proj,
            self.chk_mem_value_decay,
        ]
        self.chk_save_memory.toggled.connect(self._on_save_memory_toggled)
        self._on_save_memory_toggled(self.chk_save_memory.isChecked())

        # ─ Training status cards ─
        status_box = QGroupBox("Training Status")
        status_grid = QGridLayout(status_box)
        self.card_pid = MetricCard("PID")
        self.card_started = MetricCard("Started at")
        self.card_epoch = MetricCard("Epoch progress")
        self.card_output = MetricCard("Output checkpoint")
        status_grid.addWidget(self.card_pid, 0, 0)
        status_grid.addWidget(self.card_started, 0, 1)
        status_grid.addWidget(self.card_epoch, 0, 2)
        status_grid.addWidget(self.card_output, 1, 0, 1, 3)
        root.addWidget(status_box)

        # ─ Ablation training suite ─
        ablation_box = QGroupBox("Ablation Training Suite")
        abl_layout = QVBoxLayout(ablation_box)

        abl_desc = QLabel(
            "Run a full ablation study: trains a baseline model, then re-trains with each\n"
            "component disabled one at a time. Results saved to checkpoints/ablation_<ts>/."
        )
        abl_desc.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 11px;")
        abl_layout.addWidget(abl_desc)

        # Checkboxes for which ablations to include
        abl_grid = QGridLayout()
        self.abl_checks: dict[str, QCheckBox] = {}
        ablations = [
            ("no_regime_encoder", "Regime encoder"),
            ("no_temporal_cnn", "Temporal CNN"),
            ("no_gru_memory", "GRU memory"),
            ("no_confidence_gating", "Confidence gating"),
            ("no_soma_weighting", "Soma weighting"),
            ("no_risk_heads", "Separate risk heads"),
            ("no_early_stopping", "Early stopping"),
            ("no_tx_cost", "Transaction costs"),
        ]
        for i, (key, label) in enumerate(ablations):
            chk = QCheckBox(label)
            chk.setChecked(True)
            self.abl_checks[key] = chk
            abl_grid.addWidget(chk, i // 4, i % 4)
        abl_layout.addLayout(abl_grid)

        # Controls
        abl_ctrl = QHBoxLayout()
        self.abl_badge = StatusBadge("stopped")
        self.btn_run_ablation = QPushButton("Run ablation suite")
        self.btn_run_ablation.clicked.connect(self._on_run_ablation)
        self.lbl_ablation_progress = QLabel("")
        self.lbl_ablation_progress.setStyleSheet(f"color: {COLOR_MUTED};")
        abl_ctrl.addWidget(self.abl_badge)
        abl_ctrl.addWidget(self.btn_run_ablation)
        abl_ctrl.addWidget(self.lbl_ablation_progress)
        abl_ctrl.addStretch(1)
        abl_layout.addLayout(abl_ctrl)

        root.addWidget(ablation_box)

        # Ablation internal state
        self._ablation_queue: list[tuple[str, dict]] = []
        self._ablation_dir: str = ""
        self._ablation_idx: int = 0
        self._ablation_poll: QTimer | None = None

        # ─ Recent training output (tail) ─
        log_box = QGroupBox("Recent training output (tail)")
        l_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setStyleSheet(
            f"background-color: #0d1117; color: {COLOR_TEXT}; "
            "font-family: 'Cascadia Mono', Consolas, monospace; font-size: 11px;"
        )
        l_layout.addWidget(self.log_view)
        root.addWidget(log_box, 1)

        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet(
            f"color: {COLOR_NEGATIVE}; font-size: 12px; font-weight: bold; border: 0;"
        )
        self.lbl_error.setWordWrap(True)
        root.addWidget(self.lbl_error)

        # Populate base checkpoint combo last.
        self._reload_checkpoints()

    # ------------------------------------------------------------------
    def _reload_checkpoints(self) -> None:
        prev = self.base_combo.currentData()
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        self.base_combo.addItem("(none — train from scratch)", None)
        if _CHECKPOINTS_DIR.exists():
            files = [
                p for p in _CHECKPOINTS_DIR.glob("*.pt")
                if p.is_file() and "trash" not in p.name.lower()
            ]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for p in files:
                self.base_combo.addItem(p.name, str(p))
        if prev:
            for i in range(self.base_combo.count()):
                if self.base_combo.itemData(i) == prev:
                    self.base_combo.setCurrentIndex(i)
                    break
        self.base_combo.blockSignals(False)

    def _on_browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Output checkpoint",
            str(_CHECKPOINTS_DIR if _CHECKPOINTS_DIR.exists() else Path.cwd()),
            "Checkpoints (*.pt *.pth)",
        )
        if path:
            self.output_field.setText(path)

    def _on_save_memory_toggled(self, checked: bool) -> None:
        for w in self._mem_widgets:
            w.setEnabled(checked)

    # ------------------------------------------------------------------
    def _on_start(self) -> None:
        if self.svc.is_running():
            QMessageBox.information(self, "Already running", "Training is already running.")
            return
        output = self.output_field.text().strip() or None
        base = self.base_combo.currentData()
        preset = self.preset_combo.currentData()
        try:
            self.svc.start(
                preset=preset,
                fast=self.chk_fast.isChecked(),
                use_real_data=self.chk_real_data.isChecked(),
                epochs=int(self.spin_epochs.value()),
                output_checkpoint=output,
                base_checkpoint=str(base) if base else None,
                save_memory=self.chk_save_memory.isChecked(),
                early_stopping_patience=int(self.spin_early_stop.value()),
                transaction_cost_bps=float(self.spin_tx_cost.value()),
                platt_calibration=self.chk_platt.isChecked(),
                separate_risk_heads=self.chk_risk_heads.isChecked(),
                regime_encoder_enabled=self.chk_regime_encoder.isChecked(),
                temporal_cnn_enabled=self.chk_temporal_cnn.isChecked(),
                gru_memory_enabled=self.chk_gru_memory.isChecked(),
                confidence_gating_enabled=self.chk_confidence_gating.isChecked(),
                soma_weighting_enabled=self.chk_soma_weighting.isChecked(),
                decouple_confidence=self.chk_decouple_conf.isChecked(),
                ran_merge=self.chk_mem_ran_merge.isChecked(),
                auto_tune_temperature=self.chk_mem_auto_tune.isChecked(),
                esom_regime_labeling=self.chk_mem_esom_labels.isChecked(),
                deep_projections=self.chk_mem_deep_proj.isChecked(),
                value_aware_decay=self.chk_mem_value_decay.isChecked(),
            )
        except ServiceError as exc:
            QMessageBox.critical(self, "Failed to start training", str(exc))
        self.refresh()

    def _on_stop(self) -> None:
        if not self.svc.is_running():
            self.refresh()
            return
        self.svc.stop()
        self.refresh()

    # ------------------------------------------------------------------
    # Ablation suite
    # ------------------------------------------------------------------
    def _on_run_ablation(self) -> None:
        if self.svc.is_running():
            QMessageBox.information(self, "Busy", "A training run is already active.")
            return

        # Build ablation queue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"checkpoints/ablation_{ts}"
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        runs: list[tuple[str, dict]] = [("baseline", {})]
        if self.abl_checks["no_regime_encoder"].isChecked():
            runs.append(("no_regime_encoder", {"regime_encoder_enabled": False}))
        if self.abl_checks["no_temporal_cnn"].isChecked():
            runs.append(("no_temporal_cnn", {"temporal_cnn_enabled": False}))
        if self.abl_checks["no_gru_memory"].isChecked():
            runs.append(("no_gru_memory", {"gru_memory_enabled": False}))
        if self.abl_checks["no_confidence_gating"].isChecked():
            runs.append(("no_confidence_gating", {"confidence_gating_enabled": False}))
        if self.abl_checks["no_soma_weighting"].isChecked():
            runs.append(("no_soma_weighting", {"soma_weighting_enabled": False}))
        if self.abl_checks["no_risk_heads"].isChecked():
            runs.append(("no_risk_heads", {"separate_risk_heads": False}))
        if self.abl_checks["no_early_stopping"].isChecked():
            runs.append(("no_early_stopping", {"early_stopping_patience": 0}))
        if self.abl_checks["no_tx_cost"].isChecked():
            runs.append(("no_tx_cost", {"transaction_cost_bps": 0.0}))

        self._ablation_queue = runs
        self._ablation_dir = out_dir
        self._ablation_idx = 0
        self.btn_start.setEnabled(False)
        self._run_next_ablation()

    def _run_next_ablation(self) -> None:
        if self._ablation_idx >= len(self._ablation_queue):
            self.abl_badge.set_state("stopped", "DONE")
            self.lbl_ablation_progress.setText(
                f"Ablation complete: {len(self._ablation_queue)} runs"
            )
            self.btn_run_ablation.setEnabled(True)
            self.btn_start.setEnabled(True)
            return

        name, overrides = self._ablation_queue[self._ablation_idx]
        total = len(self._ablation_queue)
        self.lbl_ablation_progress.setText(f"Run {self._ablation_idx + 1}/{total}: {name}")
        self.abl_badge.set_state("running", "RUNNING")
        self.btn_run_ablation.setEnabled(False)

        output = f"{self._ablation_dir}/{name}.pt"
        try:
            self.svc.start(
                fast=self.chk_fast.isChecked(),
                use_real_data=self.chk_real_data.isChecked(),
                epochs=int(self.spin_epochs.value()),
                output_checkpoint=output,
                save_memory=False,
                **overrides,
            )
        except ServiceError:
            # If a single run fails to start, skip to the next.
            self._ablation_idx += 1
            self._run_next_ablation()
            return

        self._ablation_poll = QTimer(self)
        self._ablation_poll.timeout.connect(self._check_ablation_done)
        self._ablation_poll.start(3000)

    def _check_ablation_done(self) -> None:
        if self.svc.is_running():
            return
        if self._ablation_poll is not None:
            self._ablation_poll.stop()
        self._ablation_idx += 1
        self._run_next_ablation()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        try:
            status = self.svc.get_status()
        except Exception as exc:  # noqa: BLE001
            self.lbl_error.setText(str(exc))
            return
        self._render(status)

    def _render(self, status: TrainingStatus) -> None:
        state = status.state
        self.badge.set_state(state.value)
        running = state == RuntimeState.RUNNING
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)

        # Disable config inputs while running.
        for w in (
            self.preset_combo,
            self.chk_fast,
            self.chk_real_data,
            self.spin_epochs,
            self.chk_save_memory,
            self.base_combo,
            self.btn_base_refresh,
            self.output_field,
            self.btn_output_browse,
            self.spin_early_stop,
            self.spin_tx_cost,
            self.chk_platt,
            self.chk_regime_encoder,
            self.chk_temporal_cnn,
            self.chk_gru_memory,
            self.chk_confidence_gating,
            self.chk_soma_weighting,
            self.chk_decouple_conf,
            self.chk_risk_heads,
            self.chk_mem_ran_merge,
            self.chk_mem_auto_tune,
            self.chk_mem_esom_labels,
            self.chk_mem_deep_proj,
            self.chk_mem_value_decay,
        ):
            w.setEnabled(not running)

        # Memory bank options only matter when save_memory is checked.
        if not running and not self.chk_save_memory.isChecked():
            for w in self._mem_widgets:
                w.setEnabled(False)

        # Ablation controls: disable during any training run.
        ablation_active = self._ablation_poll is not None and self._ablation_poll.isActive()
        self.btn_run_ablation.setEnabled(not running and not ablation_active)
        for chk in self.abl_checks.values():
            chk.setEnabled(not running and not ablation_active)
        # Disable regular start during ablation
        if ablation_active:
            self.btn_start.setEnabled(False)

        self.card_pid.set_value(
            str(status.pid) if status.pid else "—",
            color=COLOR_TEXT if status.pid else COLOR_MUTED,
        )
        self.card_started.set_value(
            status.started_at.strftime("%H:%M:%S") if status.started_at else "—",
            color=COLOR_TEXT if status.started_at else COLOR_MUTED,
        )
        self.card_epoch.set_value(
            status.epoch_progress or "—",
            color=COLOR_INFO if status.epoch_progress else COLOR_MUTED,
        )
        self.card_output.set_value(
            status.output_checkpoint or "—",
            color=COLOR_TEXT if status.output_checkpoint else COLOR_MUTED,
        )

        # Rebuild the log view from the recent lines buffer, formatted.
        self._render_training_log(status.recent_lines[-200:])

        self.lbl_error.setText(status.error_message or "")

    # Structlog dev renderer: "2026-04-10T10:06:34Z [info     ] event_name  key=val ..."
    _STRUCTLOG_RX = re.compile(
        r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z?)\s+"
        r"\[(\w+)\s*\]\s+"
        r"(\S+)\s*(.*)?$"
    )

    def _render_training_log(self, lines: list[str]) -> None:
        """Render training subprocess output with structured formatting."""
        # Build formatted text and compare to avoid unnecessary redraws
        formatted_lines = []
        for line in lines:
            m = self._STRUCTLOG_RX.match(line)
            if m:
                ts, level, event, kv = m.group(1), m.group(2), m.group(3), m.group(4) or ""
                # Extract just time portion
                time_part = ts.split("T")[1][:8] if "T" in ts else ts[:8]
                # Format key=value pairs on separate indented lines for readability
                kv_parts = re.split(r"\s+(?=\w+=)", kv.strip()) if kv.strip() else []
                if kv_parts:
                    kv_formatted = "  " + "  ".join(kv_parts)
                    formatted_lines.append(f"{time_part} [{level.upper():<5s}] {event}\n{kv_formatted}")
                else:
                    formatted_lines.append(f"{time_part} [{level.upper():<5s}] {event}")
            else:
                formatted_lines.append(line)

        new_text = "\n".join(formatted_lines)
        if self.log_view.toPlainText() != new_text:
            self.log_view.setPlainText(new_text)
            sb = self.log_view.verticalScrollBar()
            sb.setValue(sb.maximum())
