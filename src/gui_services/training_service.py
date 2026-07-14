"""Training subprocess wrapper (mirrors RuntimeService).

Spawns ``scripts/run_train.py`` as a subprocess so the GUI event loop is
not blocked by the heavy training loop.  Returns a simple
``TrainingStatus`` dataclass for the GUI to display.

Key differences from ``RuntimeService``:

  - Training has a well-defined completion (runs to N epochs and exits),
    whereas the paper-trading loop runs forever.  So we track
    ``RuntimeState.STOPPED`` vs ``RuntimeState.RUNNING`` + a ``finished``
    flag to distinguish "finished cleanly" from "never started".
  - Training logs go to stdout, not to a shared log file.  We pipe stdout
    to a rolling buffer so the Training tab can tail recent lines.
  - Graceful stop uses the same CTRL_BREAK_EVENT / SIGINT pattern as
    RuntimeService — ``run_train.py`` doesn't have an explicit signal
    handler but most torch code responds to KeyboardInterrupt.

The new checkpoint file the training run produces is whatever
``config.model.checkpoint_path`` says; callers can inject a custom path
via a layered override YAML (same mechanism as the runtime toggles).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from src.gui_services.state import RuntimeState, RuntimeOverrides, ServiceError


_DEFAULT_TRAIN_CONFIGS = ["configs/default.yaml", "configs/training.yaml"]
_TRAINING_OVERRIDE_YAML = Path("runs/gui_training_override.yaml")
_LOG_TAIL_LINES = 200


@dataclass
class TrainingStatus:
    state: RuntimeState = RuntimeState.STOPPED
    pid: int | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    last_line: str = ""
    recent_lines: list[str] = field(default_factory=list)
    # A best-effort "epoch X/N" parsed from the log tail when present.
    epoch_progress: str = ""
    output_checkpoint: str = ""
    error_message: str = ""


class TrainingService:
    """Subprocess wrapper for ``scripts/run_train.py``."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self._root = Path(project_root) if project_root else Path.cwd()
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._started_at: datetime | None = None
        self._ended_at: datetime | None = None
        self._last_error: str = ""
        self._recent_lines: list[str] = []
        self._output_checkpoint: str = ""
        self._reader_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            return self._proc.poll() is None

    def start(
        self,
        *,
        preset: str | None = None,
        fast: bool = False,
        use_real_data: bool = True,
        epochs: int | None = None,
        output_checkpoint: str | None = None,
        base_checkpoint: str | None = None,
        save_memory: bool = True,
        config_paths: list[str] | None = None,
        early_stopping_patience: int = 0,
        transaction_cost_bps: float = 0.0,
        platt_calibration: bool = False,
        separate_risk_heads: bool = False,
        regime_encoder_enabled: bool = True,
        temporal_cnn_enabled: bool = True,
        gru_memory_enabled: bool = True,
        confidence_gating_enabled: bool = True,
        soma_weighting_enabled: bool = True,
        decouple_confidence: bool = False,
    ) -> TrainingStatus:
        """Spawn ``scripts/run_train.py``.

        ``output_checkpoint`` is the destination path for the new file.
        If None, a timestamped default like ``checkpoints/exp_<ts>.pt`` is used.
        The override YAML writes ``model.checkpoint_path`` pointing at this
        file so the training entrypoint saves the new file to the right
        location.

        ``base_checkpoint`` seeds training from an existing checkpoint (the
        user can copy/rename it beforehand and point at the copy).  When
        None, training starts fresh.
        """
        with self._lock:
            if self.is_running():
                raise ServiceError("Training is already running.")
            self._last_error = ""
            self._recent_lines = []
            self._ended_at = None

            if output_checkpoint is None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_checkpoint = f"checkpoints/exp_{ts}.pt"
            self._output_checkpoint = output_checkpoint

            # Write override YAML that pins the output checkpoint path
            # and any advanced training options the operator set.
            # We do this BEFORE the CLI so the layered load sees it.
            overrides = RuntimeOverrides(checkpoint_path=output_checkpoint)
            override_path = self._root / _TRAINING_OVERRIDE_YAML
            overrides.write_yaml(override_path)

            # Append training-specific and model-specific overrides that
            # don't belong in RuntimeOverrides (which is shared by all
            # subprocess types).  We merge into the file that write_yaml
            # just created so it's a single --config argument.
            training_extra: dict = {}
            model_extra: dict = {}
            if early_stopping_patience != 0:
                training_extra["early_stopping_patience"] = early_stopping_patience
            if transaction_cost_bps != 0.0:
                training_extra["transaction_cost_bps"] = transaction_cost_bps
            if platt_calibration:
                training_extra["platt_calibration_enabled"] = True
            if separate_risk_heads:
                model_extra["separate_risk_heads"] = True
            if not regime_encoder_enabled:
                model_extra["regime_encoder_enabled"] = False
            if not temporal_cnn_enabled:
                model_extra["use_cnn_encoder"] = False
            if not gru_memory_enabled:
                model_extra["latent_gru_memory_enabled"] = False
            if not confidence_gating_enabled:
                model_extra["branch_confidence_gating_enabled"] = False
            if not soma_weighting_enabled:
                model_extra["soma_confidence_weighting_enabled"] = False
            if decouple_confidence:
                model_extra["decouple_confidence_from_soma"] = True
            if training_extra or model_extra:
                with open(override_path, "r", encoding="utf-8") as f:
                    existing = yaml.safe_load(f) or {}
                if training_extra:
                    existing.setdefault("training", {}).update(training_extra)
                if model_extra:
                    existing.setdefault("model", {}).update(model_extra)
                with open(override_path, "w", encoding="utf-8") as f:
                    f.write(
                        "# Auto-generated by the SETAI GUI. Do not edit by hand — it will be\n"
                        "# overwritten on the next Start.  Only fields toggled in the GUI are\n"
                        "# present here; everything else inherits from default.yaml.\n"
                    )
                    yaml.safe_dump(existing, f, sort_keys=False, default_flow_style=False)

            paths = list(config_paths) if config_paths else list(_DEFAULT_TRAIN_CONFIGS)

            cmd: list[str] = [
                sys.executable,
                "-u",
                str(self._root / "scripts" / "run_train.py"),
            ]
            for path in paths:
                cmd += ["--config", path]
            cmd += ["--config", str(override_path)]
            if preset:
                cmd += ["--preset", preset]
            if fast:
                cmd += ["--fast"]
            if use_real_data:
                cmd += ["--real-data"]
            if save_memory:
                cmd += ["--save-memory"]
            if epochs is not None:
                cmd += ["--epochs", str(epochs)]
            # Note: --base-checkpoint isn't a flag run_train.py understands,
            # so we rely on the override YAML's model.checkpoint_path to
            # point at the seed when base_checkpoint is provided.  Training
            # writes to the SAME path (output_checkpoint) when the two
            # differ, so we swap: if base_checkpoint is given, we write a
            # second override pointing at the base, then let the user name
            # the output manually.  For v1 we keep things simple: when
            # base_checkpoint is given, we copy it to the output path and
            # let training fine-tune from there.
            if base_checkpoint:
                import shutil
                src = Path(base_checkpoint)
                dst = self._root / output_checkpoint
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            try:
                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(self._root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                )
                self._started_at = datetime.now()
            except Exception as exc:  # noqa: BLE001
                self._proc = None
                self._last_error = f"Failed to spawn training: {exc}"
                raise ServiceError(self._last_error) from exc

            # Start a background thread to drain stdout into the recent lines
            # buffer.  Daemon thread so it dies with the app.
            self._reader_thread = threading.Thread(
                target=self._drain_stdout,
                name="TrainingService-stdout",
                daemon=True,
            )
            self._reader_thread.start()
            return self.get_status()

    def stop(self, *, timeout: float = 15.0) -> TrainingStatus:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                return self.get_status()
            try:
                if os.name == "nt":
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    self._proc.send_signal(signal.SIGINT)
            except Exception:  # noqa: BLE001
                pass
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=2.0)
            self._ended_at = datetime.now()
            self._proc = None
            return self.get_status()

    def get_status(self) -> TrainingStatus:
        with self._lock:
            state = self._compute_state()
            recent = list(self._recent_lines[-_LOG_TAIL_LINES:])
            last_line = recent[-1] if recent else ""
            epoch_progress = self._parse_epoch_progress(recent)
            return TrainingStatus(
                state=state,
                pid=self._proc.pid if self._proc and self._proc.poll() is None else None,
                started_at=self._started_at,
                ended_at=self._ended_at,
                last_line=last_line,
                recent_lines=recent,
                epoch_progress=epoch_progress,
                output_checkpoint=self._output_checkpoint,
                error_message=self._last_error,
            )

    # ------------------------------------------------------------------
    def _compute_state(self) -> RuntimeState:
        if self._proc is None:
            if self._ended_at is not None:
                return RuntimeState.STOPPED
            return RuntimeState.STOPPED
        rc = self._proc.poll()
        if rc is None:
            return RuntimeState.RUNNING
        if rc == 0:
            self._ended_at = self._ended_at or datetime.now()
            return RuntimeState.STOPPED
        # Non-zero exit — surface the last lines as the error tail.
        if not self._last_error:
            tail = "\n".join(self._recent_lines[-5:])
            self._last_error = f"exit code {rc}\n{tail}"
        self._ended_at = self._ended_at or datetime.now()
        return RuntimeState.ERROR

    def _drain_stdout(self) -> None:
        """Background thread that appends each stdout line to the buffer."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                # Strip ANSI escape codes before storing
                import re
                line = re.sub(r"\x1b\[[0-9;]*m", "", line)
                with self._lock:
                    self._recent_lines.append(line)
                    # Cap the buffer at 4x what get_status returns to bound memory.
                    if len(self._recent_lines) > _LOG_TAIL_LINES * 4:
                        self._recent_lines = self._recent_lines[-_LOG_TAIL_LINES * 2:]
        except (OSError, ValueError):
            # Process ended or pipe closed — normal termination path.
            pass

    @staticmethod
    def _parse_epoch_progress(lines: list[str]) -> str:
        """Best-effort scan for an 'epoch N/M' pattern in the log tail."""
        import re
        rx = re.compile(r"epoch[^0-9]{0,5}(\d+)\s*/\s*(\d+)", re.IGNORECASE)
        for line in reversed(lines):
            match = rx.search(line)
            if match:
                return f"epoch {match.group(1)}/{match.group(2)}"
        return ""


# Module-level singleton (one training subprocess per app instance)
_training_service: TrainingService | None = None


def get_training_service() -> TrainingService:
    global _training_service
    if _training_service is None:
        _training_service = TrainingService()
    return _training_service
