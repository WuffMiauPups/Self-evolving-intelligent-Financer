"""Memory / Fusion tab — inspect bank, trigger build/eval."""
from __future__ import annotations

from PySide6.QtCore import QTimer, QThreadPool
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gui.widgets import MetricCard, StatusBadge
from src.gui.workers import CallableWorker
from src.gui_services.memory_service import (
    build_memory,
    describe_memory_features,
    get_memory_status,
    plan_memory_build,
)
from src.gui_services.state import MemoryStatus


class MemoryTab(QWidget):
    POLL_MS = 7000
    DEFAULT_CONFIG_PATHS = ["configs/default.yaml"]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self._busy = False
        self._fetch_pending = False
        self._build_ui()
        self.refresh()
        self._refresh_plan()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(self.POLL_MS)

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.badge_mem = StatusBadge("off")
        self.badge_fusion = StatusBadge("off")
        header.addWidget(QLabel("Memory:"))
        header.addWidget(self.badge_mem)
        header.addSpacing(20)
        header.addWidget(QLabel("Fusion:"))
        header.addWidget(self.badge_fusion)
        header.addStretch(1)
        root.addLayout(header)

        bank_box = QGroupBox("Memory Bank")
        grid = QGridLayout(bank_box)
        self.card_items = MetricCard("Items")
        self.card_clusters = MetricCard("Clusters")
        self.card_units = MetricCard("Units")
        self.card_mature = MetricCard("Mature units")
        self.card_candidate = MetricCard("Candidate units")
        self.card_quarantined = MetricCard("Quarantined units")
        self.card_esom_nodes = MetricCard("ESOM nodes")
        self.card_esom_edges = MetricCard("ESOM edges")
        self.card_fp = MetricCard("Encoder fingerprint")
        self.card_contract = MetricCard("Feature contract")
        cards = [
            self.card_items, self.card_clusters, self.card_units,
            self.card_mature, self.card_candidate, self.card_quarantined,
            self.card_esom_nodes, self.card_esom_edges,
            self.card_fp, self.card_contract,
        ]
        for i, card in enumerate(cards):
            grid.addWidget(card, i // 4, i % 4)
        root.addWidget(bank_box)

        # ── Memory feature status (read-only, config-driven) ────────
        feat_box = QGroupBox("Memory Feature Status")
        feat_grid = QGridLayout(feat_box)
        feat_grid.setHorizontalSpacing(10)
        feat_grid.setVerticalSpacing(4)
        self._feature_labels: dict[str, QLabel] = {}
        _FEATURE_DISPLAY = [
            ("modern_hopfield", "Modern Hopfield"),
            ("multi_timescale", "Multi-timescale"),
            ("ran_merge", "RAN Merge"),
            ("esom_regime_labels", "ESOM Labels"),
            ("value_aware_decay", "Value Decay"),
            ("auto_tune_temp", "Auto-tune Temp"),
            ("deep_projections", "Deep Projections"),
            ("counterfactual", "Counterfactual Cal."),
        ]
        for i, (key, display_name) in enumerate(_FEATURE_DISPLAY):
            row, col = divmod(i, 4)
            name_lbl = QLabel(f"{display_name}:")
            name_lbl.setStyleSheet("color: #c0c0c0;")
            val_lbl = QLabel("--")
            val_lbl.setStyleSheet("color: #94a3b8; font-weight: bold;")
            feat_grid.addWidget(name_lbl, row, col * 2)
            feat_grid.addWidget(val_lbl, row, col * 2 + 1)
            self._feature_labels[key] = val_lbl
        root.addWidget(feat_box)

        # ── Memory build inputs ──────────────────────────────────────
        build_box = QGroupBox("Memory Build")
        bv = QVBoxLayout(build_box)

        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Checkpoint:"))
        self.checkpoint_field = QLineEdit()
        self.checkpoint_field.setPlaceholderText(
            "Required when config has memory.build_from_real_checkpoint=true"
        )
        self._plan_debounce = QTimer()
        self._plan_debounce.setSingleShot(True)
        self._plan_debounce.setInterval(500)
        self._plan_debounce.timeout.connect(self._refresh_plan)
        self.checkpoint_field.textChanged.connect(lambda: self._plan_debounce.start())
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._on_browse)
        self.btn_use_active = QPushButton("Use active checkpoint")
        self.btn_use_active.clicked.connect(self._on_use_active)
        ckpt_row.addWidget(self.checkpoint_field, 1)
        ckpt_row.addWidget(self.btn_browse)
        ckpt_row.addWidget(self.btn_use_active)
        bv.addLayout(ckpt_row)

        self.lbl_plan = QLabel("")
        self.lbl_plan.setWordWrap(True)
        self.lbl_plan.setStyleSheet("color: #c0c0c0;")
        bv.addWidget(self.lbl_plan)

        actions = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh status")
        self.btn_build = QPushButton("Build memory")
        actions.addWidget(self.btn_refresh)
        actions.addWidget(self.btn_build)
        actions.addStretch(1)
        self.btn_refresh.clicked.connect(self._on_full_refresh)
        self.btn_build.clicked.connect(self._on_build)
        bv.addLayout(actions)
        root.addWidget(build_box)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #aaaaaa;")
        root.addWidget(self.lbl_status)
        root.addStretch(1)

    # ------------------------------------------------------------------
    def _on_full_refresh(self) -> None:
        self.refresh()
        self._refresh_plan()

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
        status = get_memory_status()
        features = describe_memory_features(config_paths=self.DEFAULT_CONFIG_PATHS)
        return (status, features)

    def _on_data_fetched(self, result: object) -> None:
        """Runs on main thread -- safe to update widgets here."""
        self._fetch_pending = False
        if not isinstance(result, tuple) or len(result) != 2:
            return
        status, features = result
        self._render(status)
        self._render_feature_status(features)

    def _on_fetch_failed(self, msg: str) -> None:
        self._fetch_pending = False

    def _render(self, status: MemoryStatus) -> None:
        self.badge_mem.set_state("ok" if status.exists else "off",
                                 "READY" if status.exists else "MISSING")
        self.badge_fusion.set_state("ok" if status.fusion_enabled else "off",
                                    "ON" if status.fusion_enabled else "OFF")
        self.card_items.set_value(str(status.item_count))
        self.card_clusters.set_value(str(status.cluster_count))
        self.card_units.set_value(str(status.unit_count))
        self.card_mature.set_value(str(status.mature_unit_count))
        self.card_candidate.set_value(str(status.candidate_unit_count))
        self.card_quarantined.set_value(str(status.quarantined_unit_count))
        self.card_esom_nodes.set_value(str(status.esom_node_count))
        self.card_esom_edges.set_value(str(status.esom_edge_count))
        self.card_fp.set_value(status.encoder_fingerprint or "—")
        self.card_contract.set_value(status.feature_contract_id or "—")
        if status.error:
            self.lbl_status.setText(status.error)

    def _render_feature_status(self, features: dict) -> None:
        for key, lbl in self._feature_labels.items():
            enabled = features.get(key, False)
            if enabled:
                lbl.setText("ON")
                lbl.setStyleSheet("color: #4ade80; font-weight: bold;")
            else:
                lbl.setText("OFF")
                lbl.setStyleSheet("color: #94a3b8; font-weight: bold;")

    # ------------------------------------------------------------------
    def _refresh_plan(self) -> None:
        """Re-compute the build plan based on current inputs and update UI."""
        explicit = self.checkpoint_field.text().strip() or None
        plan = plan_memory_build(
            config_paths=self.DEFAULT_CONFIG_PATHS,
            explicit_checkpoint=explicit,
            use_active_registry=False,
        )
        if plan.requires_checkpoint:
            req = "REQUIRED (real-data + build_from_real_checkpoint)"
        else:
            req = "optional"
        if plan.resolved_checkpoint:
            line = (
                f"Build plan: real_data={plan.use_real_data} | "
                f"checkpoint={req} | "
                f"resolved={plan.resolved_checkpoint} (source: {plan.checkpoint_source})"
            )
            color = "#a4d65e"
        elif plan.requires_checkpoint:
            line = (
                f"Build plan: real_data={plan.use_real_data} | "
                f"checkpoint={req} | NO checkpoint resolved — pick one above"
            )
            color = "#ffb454"
        else:
            line = (
                f"Build plan: real_data={plan.use_real_data} | "
                f"checkpoint={req} | will use untrained encoder"
            )
            color = "#c0c0c0"
        self.lbl_plan.setText(line)
        self.lbl_plan.setStyleSheet(f"color: {color};")

        # Disable Build if not runnable, unless we're already busy
        if not self._busy:
            self.btn_build.setEnabled(plan.is_runnable)
            self.btn_build.setToolTip(plan.blocking_error or "")

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "checkpoints", "Checkpoints (*.pt *.pth)"
        )
        if path:
            self.checkpoint_field.setText(path)
            self._refresh_plan()

    def _on_use_active(self) -> None:
        plan = plan_memory_build(
            config_paths=self.DEFAULT_CONFIG_PATHS,
            explicit_checkpoint=None,
            use_active_registry=True,
        )
        if plan.resolved_checkpoint and plan.checkpoint_source == "active_registry":
            self.checkpoint_field.setText(plan.resolved_checkpoint)
            self.lbl_status.setText(
                f"Loaded active checkpoint from registry: {plan.resolved_checkpoint}"
            )
        elif plan.resolved_checkpoint and plan.checkpoint_source == "config":
            self.checkpoint_field.setText(plan.resolved_checkpoint)
            self.lbl_status.setText(
                f"Loaded checkpoint from config: {plan.resolved_checkpoint}"
            )
        else:
            QMessageBox.information(
                self,
                "No active checkpoint",
                "No active checkpoint found in the registry and the config does "
                "not declare one. Use Browse to select a file.",
            )
        self._refresh_plan()

    # ------------------------------------------------------------------
    def _on_build(self) -> None:
        if self._busy:
            return
        explicit = self.checkpoint_field.text().strip() or None
        # Final preflight before launch
        plan = plan_memory_build(
            config_paths=self.DEFAULT_CONFIG_PATHS,
            explicit_checkpoint=explicit,
            use_active_registry=False,
        )
        if not plan.is_runnable:
            QMessageBox.critical(self, "Cannot build memory", plan.blocking_error)
            self._refresh_plan()
            return

        self._set_busy(True)
        self.lbl_status.setText("Spawning memory build…")
        worker = CallableWorker(
            build_memory,
            config_paths=self.DEFAULT_CONFIG_PATHS,
            explicit_checkpoint=explicit,
            use_active_registry=False,
        )
        worker.signals.progress.connect(self.lbl_status.setText)
        worker.signals.finished.connect(lambda _r: self._on_done("Memory build complete."))
        worker.signals.failed.connect(self._on_failed)
        self.pool.start(worker)

    def _on_done(self, msg: str) -> None:
        self.lbl_status.setText(msg)
        self._set_busy(False)
        self.refresh()
        self._refresh_plan()

    def _on_failed(self, msg: str) -> None:
        self._set_busy(False)
        self.lbl_status.setText("Operation failed.")
        QMessageBox.critical(self, "Operation failed", msg)
        self._refresh_plan()

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for b in (self.btn_build, self.btn_browse, self.btn_use_active):
            b.setEnabled(not busy)
        if not busy:
            # Re-evaluate the build button gating
            self._refresh_plan()
