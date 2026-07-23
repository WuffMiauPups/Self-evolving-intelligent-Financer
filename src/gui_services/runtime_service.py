"""Paper-trading runtime control via subprocess.

Why a subprocess: src/execution/trade_engine.run_trade_loop uses asyncio.run()
which would block the GUI event loop indefinitely. The only safe way to
manage it from a desktop app is to spawn it as a child process and stop it
with a graceful signal.

This module is GUI-thread-safe: it does not use Qt classes; it returns plain
data and exposes a simple start/stop/status API. It is safe to instantiate a
single shared RuntimeService and call from any thread (state is protected
by a lock).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

from src.gui_services.config_service import (
    describe_config,
    describe_memory_caps,
    load_app_config,
)
from src.gui_services.state import (
    RuntimeOverrides,
    RuntimeState,
    RuntimeStatus,
    ServiceError,
)
from src.monitoring.runtime_status import (
    DEFAULT_RUNTIME_STATUS_PATH,
    RuntimeStatusWriter,
)


_DEFAULT_PAPER_CONFIGS = ["configs/default.yaml", "configs/paper_trade.yaml"]
_RUNTIME_OVERRIDE_YAML = Path("runs/gui_runtime_override.yaml")
_LOG_TIMESTAMP_RX = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
_ANSI_RX = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RX.sub("", text)


class RuntimeService:
    """Singleton-style service to start/stop/inspect the paper-trading runtime."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self._root = Path(project_root) if project_root else Path.cwd()
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._started_at: datetime | None = None
        self._last_error: str = ""
        self._config_paths: list[str] = list(_DEFAULT_PAPER_CONFIGS)
        self._cached_config_summary: dict[str, object] | None = None
        # Drained live by a daemon thread; without it the runtime child
        # blocks forever on a stderr write once the OS pipe buffer fills
        # (TrainingService already drains — RuntimeService did not).
        self._stderr_tail: deque[str] = deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # config selection
    # ------------------------------------------------------------------
    def preview_config(self) -> dict[str, object]:
        """Load+validate currently selected config and return a UI summary."""
        with self._lock:
            if self._cached_config_summary is not None:
                return dict(self._cached_config_summary)
            cfg = load_app_config(*self._config_paths)
            summary = describe_config(cfg)
            self._cached_config_summary = summary
            return dict(summary)

    def preview_memory_caps(self) -> dict[str, int]:
        """Return raw memory cap integers for the RuntimeOverridesPanel."""
        with self._lock:
            cfg = load_app_config(*self._config_paths)
            return describe_memory_caps(cfg)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def is_running(self) -> bool:
        with self._lock:
            if self._proc is None:
                return False
            return self._proc.poll() is None

    def start(
        self,
        *,
        overrides: RuntimeOverrides | None = None,
        checkpoint: str | None = None,
        use_memory: bool = False,
    ) -> RuntimeStatus:
        with self._lock:
            if self.is_running():
                raise ServiceError("Paper trading runtime is already running.")
            self._last_error = ""

            # Back-compat: if callers still pass the old kwargs, fold them
            # into the overrides dataclass so the emit path is uniform.
            if overrides is None:
                overrides = RuntimeOverrides()
            if checkpoint and not overrides.checkpoint_path:
                overrides.checkpoint_path = checkpoint
            if use_memory and overrides.memory_enabled is None:
                overrides.memory_enabled = True

            cmd = [
                sys.executable,
                "-u",  # unbuffered, so logs flush in real time
                str(self._root / "scripts" / "run_paper.py"),
            ]
            for path in self._config_paths:
                cmd += ["--config", path]

            # Emit the GUI overrides YAML and layer it on top.  This is how
            # toggles/caps/checkpoint flow into the subprocess — we exploit
            # the existing layered-YAML deep merge instead of adding new CLI
            # flags to run_paper.py.
            if not overrides.is_empty():
                override_path = self._root / _RUNTIME_OVERRIDE_YAML
                overrides.write_yaml(override_path)
                cmd += ["--config", str(override_path)]

            try:
                # New process group on Windows so we can send a graceful stop.
                creationflags = 0
                if os.name == "nt":
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                self._proc = subprocess.Popen(
                    cmd,
                    cwd=str(self._root),
                    stdout=subprocess.DEVNULL,  # logs go to file via the backend logger
                    stderr=subprocess.PIPE,
                    creationflags=creationflags,
                )
                self._started_at = datetime.now()
                self._stderr_tail.clear()
                self._stderr_thread = threading.Thread(
                    target=self._drain_stderr,
                    args=(self._proc,),
                    daemon=True,
                    name="runtime-stderr-drain",
                )
                self._stderr_thread.start()
            except Exception as exc:  # noqa: BLE001
                self._proc = None
                self._last_error = f"Failed to spawn runtime: {exc}"
                raise ServiceError(self._last_error) from exc
            return self.get_status()

    def stop(self, *, timeout: float = 10.0) -> RuntimeStatus:
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._proc = None
                self._started_at = None
                return self.get_status()
            try:
                if os.name == "nt":
                    self._proc.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    self._proc.send_signal(signal.SIGINT)
            except Exception:  # noqa: BLE001
                # Fall through to terminate.
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
            self._proc = None
            self._started_at = None
            return self.get_status()

    # ------------------------------------------------------------------
    # status snapshot
    # ------------------------------------------------------------------
    def get_status(self) -> RuntimeStatus:
        with self._lock:
            cfg_summary: dict[str, object] = {}
            try:
                cfg_summary = self.preview_config()
            except ServiceError as exc:
                self._last_error = str(exc)

            platt_enabled = bool(cfg_summary.get("platt_calibration_enabled", False))
            # The fitted calibrator lives NEXT TO the checkpoint it was
            # fitted for (src/training/platt_calibration.py), not at the
            # legacy runs/platt_calibration.json location this badge used
            # to poll.  Resolve it from the configured checkpoint path.
            platt_fitted = False
            if platt_enabled:
                checkpoint_path = str(cfg_summary.get("checkpoint_path") or "")
                if checkpoint_path:
                    from src.training.platt_calibration import (
                        platt_calibration_path_for_checkpoint,
                    )

                    platt_fitted = platt_calibration_path_for_checkpoint(checkpoint_path).exists()
            status = RuntimeStatus(
                state=self._compute_state(),
                pid=self._proc.pid if self._proc and self._proc.poll() is None else None,
                started_at=self._started_at,
                symbols=list(cfg_summary.get("symbols", [])),  # type: ignore[arg-type]
                frozen_brain=bool(cfg_summary.get("frozen_brain", True)),
                memory_enabled=bool(cfg_summary.get("memory_enabled", False)),
                memory_updates_enabled=bool(cfg_summary.get("memory_updates_allowed", False)),
                ev_gate_active=bool(cfg_summary.get("separate_risk_heads", False)),
                ev_cost_bps=float(cfg_summary.get("ev_cost_bps", 0.0)),
                platt_calibration_enabled=platt_enabled,
                platt_calibration_fitted=platt_fitted,
                error_message=self._last_error,
            )
            # Prefer structured runtime status snapshot (real per-tick truth)
            # over log-tail scanning. Fall back to log-tail only when no
            # snapshot file exists yet (e.g. very first ticks).
            if not self._fill_from_runtime_status(status):
                log_path = cfg_summary.get("log_path") or "logs/seif_runtime.log"
                self._fill_log_derived_fields(status, Path(str(log_path)))
            return status

    def _fill_from_runtime_status(self, status: RuntimeStatus) -> bool:
        snap = RuntimeStatusWriter.read_latest(DEFAULT_RUNTIME_STATUS_PATH)
        if not snap:
            return False
        # Timestamp
        ts = snap.get("ts")
        if ts:
            try:
                status.last_log_timestamp = datetime.fromisoformat(str(ts))
            except ValueError:
                pass
        # --- Live runtime truth: prefer the per-tick snapshot for the
        # operator-visible safety-flag badges. The config view captured at
        # spawn time can drift from the actual runtime state (e.g. when
        # ``allow_online_memory_updates=false`` causes the runtime to
        # silently downgrade ``runtime_update_mode`` to ``disabled``).
        # ``runtime_status.jsonl`` is written
        # per-tick by the trade engine and is the source of truth.
        if "memory_enabled" in snap:
            status.memory_enabled = bool(snap.get("memory_enabled"))
        if "frozen_brain" in snap:
            status.frozen_brain = bool(snap.get("frozen_brain"))
        # Decision summary
        decision = snap.get("last_decision")
        if isinstance(decision, dict):
            sym = snap.get("symbol", "")
            parts = [
                f"sym={sym}" if sym else "",
                f"action={decision.get('action')}",
                f"conf={decision.get('confidence'):.3f}" if isinstance(decision.get("confidence"), (int, float)) else "",
                f"frac={decision.get('target_fraction'):.3f}" if isinstance(decision.get("target_fraction"), (int, float)) else "",
            ]
            level = decision.get("fusion_level")
            if level:
                parts.append(f"fusion={level}")
            status.last_decision_summary = " ".join(p for p in parts if p)[:500]
        # Risk summary
        risk = snap.get("last_risk")
        if isinstance(risk, dict):
            approved = risk.get("approved")
            reasons = risk.get("reasons") or []
            head = "approved" if approved else "REJECTED"
            tail = ",".join(str(r) for r in reasons) if reasons else ""
            qty = risk.get("quantity")
            side = risk.get("side") or ""
            extra = f" qty={qty} side={side}" if approved and qty else ""
            status.last_risk_decision_summary = f"{head}{extra}{(' :: ' + tail) if tail else ''}"[:500]
        # Single-line operator message
        step = snap.get("step")
        equity = snap.get("equity")
        fills = snap.get("fills")
        safe = snap.get("safe_to_trade")
        bits = [
            f"step={step}",
            f"equity={equity:.2f}" if isinstance(equity, (int, float)) else "",
            f"fills={fills}",
            f"safe={safe}",
        ]
        status.last_message = " ".join(b for b in bits if b)[:500]
        return True

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        """Consume the child's stderr for its whole lifetime (daemon thread)."""
        try:
            if proc.stderr is None:
                return
            for raw in iter(proc.stderr.readline, b""):
                if not isinstance(raw, (bytes, bytearray)):
                    # Not a real byte pipe (e.g. a test double) — a
                    # non-bytes readline would never hit the b"" sentinel
                    # and this loop would spin forever.
                    return
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_tail.append(line)
        except Exception:  # noqa: BLE001
            pass

    def _compute_state(self) -> RuntimeState:
        if self._proc is None:
            return RuntimeState.STOPPED
        rc = self._proc.poll()
        if rc is None:
            return RuntimeState.RUNNING
        if rc == 0:
            return RuntimeState.STOPPED
        # Surface the drained stderr tail for the operator.
        if self._stderr_tail:
            self._last_error = self._stderr_tail[-1][:500]
        return RuntimeState.ERROR

    def _fill_log_derived_fields(self, status: RuntimeStatus, log_path: Path) -> None:
        if not log_path.exists():
            return
        try:
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                read = min(size, 64 * 1024)
                f.seek(size - read)
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            return
        lines = [_strip_ansi(ln).strip() for ln in tail.splitlines() if ln.strip()]
        if not lines:
            return
        status.last_message = lines[-1][:500]
        # Try to extract a timestamp from the most recent line.
        match = _LOG_TIMESTAMP_RX.search(lines[-1])
        if match:
            try:
                status.last_log_timestamp = datetime.fromisoformat(
                    match.group(1).replace(" ", "T")
                )
            except ValueError:
                pass
        # Scan for last decision/risk events (best-effort, structured or text)
        for line in reversed(lines):
            if not status.last_decision_summary and ("model_decision" in line or "decision_fusion" in line):
                status.last_decision_summary = _short(line)
            if not status.last_risk_decision_summary and ("risk_decision" in line or "risk_block" in line):
                status.last_risk_decision_summary = _short(line)
            if status.last_decision_summary and status.last_risk_decision_summary:
                break


def _short(line: str, max_len: int = 200) -> str:
    if len(line) <= max_len:
        return line
    return line[: max_len - 1] + "…"


# Module-level singleton (one runtime per app instance)
_runtime_service: RuntimeService | None = None


def get_runtime_service() -> RuntimeService:
    global _runtime_service
    if _runtime_service is None:
        _runtime_service = RuntimeService()
    return _runtime_service
