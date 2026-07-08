"""Model / Checkpoints tab — registry view + validate + file management.

v2 adds ad-hoc file management (snapshot backup, rename, tag, soft delete)
via ``src/gui_services/checkpoint_manager.py``.  These operations never
touch the registry schema — the registry remains the source of truth for
*promoted* checkpoints.  The file-management actions are about unpromoted
experiment files that the operator produces ad hoc.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.gui.workers import CallableWorker
from src.gui.widgets import MetricCard
from src.gui_services.checkpoint_manager import (
    CHECKPOINTS_DIR,
    list_checkpoint_files,
    move_to_trash,
    rename_checkpoint,
    snapshot_checkpoint,
    tag_checkpoint,
)
from src.gui_services.checkpoint_service import (
    get_active,
    get_active_checkpoint_path,
    list_history,
    validate_checkpoint,
)
from src.gui_services.state import CheckpointFile, CheckpointInfo, ServiceError


class CheckpointsTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._files: list[CheckpointFile] = []
        self._fetch_pending = False
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        active_box = QGroupBox("Active Checkpoint")
        grid = QGridLayout(active_box)
        self.card_version = MetricCard("Version ID")
        self.card_path = MetricCard("Path")
        self.card_fp = MetricCard("Fingerprint")
        self.card_runid = MetricCard("Run ID")
        grid.addWidget(self.card_version, 0, 0)
        grid.addWidget(self.card_path, 0, 1, 1, 2)
        grid.addWidget(self.card_fp, 1, 0)
        grid.addWidget(self.card_runid, 1, 1)
        root.addWidget(active_box)

        # ─ Registry actions (existing) ─
        actions = QGroupBox("Registry actions")
        a = QHBoxLayout(actions)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_validate = QPushButton("Validate checkpoint…")
        self.btn_promote = QPushButton("Promote checkpoint…")
        self.btn_promote.setEnabled(False)
        self.btn_promote.setToolTip(
            "Promotion goes through the existing PromotionGate. Disabled in v1: "
            "training pipeline already promotes via scripts/run_train.py."
        )
        a.addWidget(self.btn_refresh)
        a.addWidget(self.btn_validate)
        a.addWidget(self.btn_promote)
        a.addStretch(1)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_validate.clicked.connect(self._on_validate)
        root.addWidget(actions)

        # ─ File management actions (new) ─
        file_actions = QGroupBox("File management (selected row)")
        fa = QHBoxLayout(file_actions)
        self.btn_snapshot = QPushButton("Snapshot → backup")
        self.btn_snapshot.setToolTip(
            "Copy the selected checkpoint to a timestamped backup file in the\n"
            "checkpoints/ directory.  Safe to use while the runtime is active."
        )
        self.btn_rename = QPushButton("Rename…")
        self.btn_rename.setToolTip(
            "Rename the selected checkpoint file.  Refuses when the file is\n"
            "currently marked as active in the registry."
        )
        self.btn_tag = QPushButton("Tag…")
        self.btn_tag.setToolTip(
            "Attach a short label to the selected checkpoint.  Tags are\n"
            "stored in checkpoints/tags.json and displayed in the Tag\n"
            "column below."
        )
        self.btn_trash = QPushButton("Move to trash")
        self.btn_trash.setToolTip(
            "Move the selected checkpoint to checkpoints/_trash/.  Never\n"
            "hard-deletes — you can still recover the file by moving it\n"
            "back with a file manager."
        )
        self.btn_open_folder = QPushButton("Open folder")
        self.btn_open_folder.setToolTip("Open the checkpoints/ directory in the system file manager.")
        for btn in (
            self.btn_snapshot,
            self.btn_rename,
            self.btn_tag,
            self.btn_trash,
        ):
            btn.setEnabled(False)  # enabled when a row is selected
            fa.addWidget(btn)
        fa.addStretch(1)
        fa.addWidget(self.btn_open_folder)
        self.btn_snapshot.clicked.connect(self._on_snapshot)
        self.btn_rename.clicked.connect(self._on_rename)
        self.btn_tag.clicked.connect(self._on_tag)
        self.btn_trash.clicked.connect(self._on_trash)
        self.btn_open_folder.clicked.connect(self._on_open_folder)
        root.addWidget(file_actions)

        # ─ Files on disk (combined: file-system view + registry metadata) ─
        files_box = QGroupBox("Checkpoints on disk")
        f = QVBoxLayout(files_box)
        self.files_table = QTableWidget(0, 5)
        self.files_table.setHorizontalHeaderLabels(
            ["Name", "Tag", "Size", "Modified", "Active / Version"]
        )
        self.files_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.files_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.files_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.files_table.horizontalHeader().setStretchLastSection(True)
        self.files_table.itemSelectionChanged.connect(self._on_selection_changed)
        f.addWidget(self.files_table)
        root.addWidget(files_box, 1)

        # ─ Registry history (existing) ─
        history_box = QGroupBox("Registry History")
        h = QVBoxLayout(history_box)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Version", "Status", "Created", "Fingerprint", "Path"]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        h.addWidget(self.table)
        root.addWidget(history_box)

        details_box = QGroupBox("Validation Details")
        d = QVBoxLayout(details_box)
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(140)
        d.addWidget(self.details)
        root.addWidget(details_box)

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Kick off background data fetch — never blocks the GUI thread."""
        if self._fetch_pending:
            return
        self._fetch_pending = True
        worker = CallableWorker(self._fetch_refresh_data)
        worker.signals.finished.connect(self._on_refresh_fetched)
        worker.signals.failed.connect(self._on_refresh_failed)
        QThreadPool.globalInstance().start(worker)

    def _fetch_refresh_data(self) -> tuple:
        """Runs in background thread — registry + disk reads."""
        active = get_active()
        cfg_path = get_active_checkpoint_path() if active is None else None
        files = list_checkpoint_files(CHECKPOINTS_DIR)
        history = list_history()
        return (active, cfg_path, files, history)

    def _on_refresh_fetched(self, result: object) -> None:
        """Runs on main thread — safe to update widgets here."""
        self._fetch_pending = False
        if not isinstance(result, tuple) or len(result) != 4:
            return
        active, cfg_path, files, history = result
        self._render_active(active, cfg_path)
        self._render_files(files)
        self._render_history(history)

    def _on_refresh_failed(self, msg: str) -> None:
        self._fetch_pending = False

    def _render_active(self, active, cfg_path) -> None:
        if active is None:
            self.card_version.set_value("— (no active)")
            self.card_path.set_value(cfg_path or "— (config has none)")
            self.card_fp.set_value("—")
            self.card_runid.set_value("—")
        else:
            self.card_version.set_value(active.version_id)
            self.card_path.set_value(active.checkpoint_path)
            self.card_fp.set_value(active.fingerprint)
            self.card_runid.set_value(active.notes or "—")

    def _reload_files(self) -> None:
        self._render_files(list_checkpoint_files(CHECKPOINTS_DIR))

    def _render_files(self, files: list[CheckpointFile]) -> None:
        self._files = files
        self.files_table.setRowCount(len(self._files))
        for r, info in enumerate(self._files):
            active_str = (
                f"active :: {info.registry_version_id}" if info.is_active
                else (info.registry_version_id or "—")
            )
            cells = [
                info.path.name,
                info.tag or "",
                self._fmt_size(info.size_bytes),
                info.mtime.strftime("%Y-%m-%d %H:%M:%S"),
                active_str,
            ]
            for c, text in enumerate(cells):
                self.files_table.setItem(r, c, QTableWidgetItem(text))
        self._on_selection_changed()  # reset action button state

    def _reload_history(self) -> None:
        self._render_history(list_history())

    def _render_history(self, history: list[CheckpointInfo]) -> None:
        self.table.setRowCount(len(history))
        for r, info in enumerate(history):
            for c, txt in enumerate([
                info.version_id, info.status, info.created_at,
                info.fingerprint, info.checkpoint_path,
            ]):
                self.table.setItem(r, c, QTableWidgetItem(txt))

    # ------------------------------------------------------------------
    def _on_selection_changed(self) -> None:
        row = self.files_table.currentRow()
        has_selection = 0 <= row < len(self._files)
        for btn in (
            self.btn_snapshot,
            self.btn_rename,
            self.btn_tag,
            self.btn_trash,
        ):
            btn.setEnabled(has_selection)

    def _selected_file(self) -> CheckpointFile | None:
        row = self.files_table.currentRow()
        if 0 <= row < len(self._files):
            return self._files[row]
        return None

    # ------------------------------------------------------------------
    def _on_validate(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "checkpoints", "Checkpoints (*.pt *.pth)"
        )
        if not path:
            return
        self.details.setPlainText("Validating...")
        self.btn_validate.setEnabled(False)
        worker = CallableWorker(validate_checkpoint, path)
        worker.signals.finished.connect(self._on_validation_done)
        worker.signals.failed.connect(self._on_validation_failed)
        QThreadPool.globalInstance().start(worker)

    def _on_validation_done(self, summary: object) -> None:
        """Runs on main thread — safe to update widgets here."""
        self.btn_validate.setEnabled(True)
        if isinstance(summary, dict):
            lines = [f"{k}: {v}" for k, v in summary.items()]
            self.details.setPlainText("\n".join(lines))
        else:
            self.details.setPlainText(str(summary))

    def _on_validation_failed(self, msg: str) -> None:
        self.btn_validate.setEnabled(True)
        self.details.setPlainText(f"Validation failed: {msg}")

    def _on_snapshot(self) -> None:
        info = self._selected_file()
        if info is None:
            return
        try:
            dest = snapshot_checkpoint(info.path)
        except ServiceError as exc:
            QMessageBox.critical(self, "Snapshot failed", str(exc))
            return
        QMessageBox.information(
            self, "Snapshot saved",
            f"Copied to:\n{dest}",
        )
        self._reload_files()

    def _on_rename(self) -> None:
        info = self._selected_file()
        if info is None:
            return
        new_name, ok = QInputDialog.getText(
            self, "Rename checkpoint",
            f"New filename (current: {info.path.name})",
            text=info.path.stem,
        )
        if not ok or not new_name.strip():
            return
        try:
            rename_checkpoint(info.path, new_name)
        except ServiceError as exc:
            QMessageBox.critical(self, "Rename failed", str(exc))
            return
        self._reload_files()

    def _on_tag(self) -> None:
        info = self._selected_file()
        if info is None:
            return
        new_tag, ok = QInputDialog.getText(
            self, "Tag checkpoint",
            "Tag (leave empty to clear):",
            text=info.tag,
        )
        if not ok:
            return
        try:
            tag_checkpoint(info.path, new_tag)
        except ServiceError as exc:
            QMessageBox.critical(self, "Tag failed", str(exc))
            return
        self._reload_files()

    def _on_trash(self) -> None:
        info = self._selected_file()
        if info is None:
            return
        reply = QMessageBox.warning(
            self, "Move to trash",
            f"Move '{info.path.name}' to checkpoints/_trash/?\n\n"
            "The file will be moved (not permanently deleted).  You can\n"
            "recover it from the _trash folder with any file manager.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            move_to_trash(info.path)
        except ServiceError as exc:
            QMessageBox.critical(self, "Trash failed", str(exc))
            return
        self._reload_files()

    def _on_open_folder(self) -> None:
        target = CHECKPOINTS_DIR.resolve()
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    @staticmethod
    def _fmt_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
            n /= 1024  # type: ignore[assignment]
        return f"{n:.1f} TB"
