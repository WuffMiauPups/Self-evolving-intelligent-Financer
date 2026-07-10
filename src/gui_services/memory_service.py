"""Memory bank inspection + offline build (passive memory — no fusion layer)."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.gui_services.config_service import load_app_config
from src.gui_services.state import MemoryStatus, ServiceError


@dataclass
class MemoryBuildPlan:
    """Result of validating a memory-build invocation before launch.

    The GUI uses this to:
      - decide whether the Build button should be enabled
      - know which checkpoint will actually be passed to the subprocess
      - show a clear preflight error instead of letting the subprocess crash
    """
    config_paths: list[str]
    use_real_data: bool
    requires_checkpoint: bool
    resolved_checkpoint: str = ""
    checkpoint_source: str = ""   # "explicit" | "config" | "active_registry" | ""
    blocking_error: str = ""

    @property
    def is_runnable(self) -> bool:
        return not self.blocking_error


def plan_memory_build(
    *,
    config_paths: list[str],
    explicit_checkpoint: str | None = None,
    use_active_registry: bool = False,
) -> MemoryBuildPlan:
    """Validate and resolve everything needed for a memory build.

    Resolution precedence for the checkpoint:
        1. explicit_checkpoint (operator-supplied path)
        2. config.model.checkpoint_path (already in YAML)
        3. active checkpoint from registry (only if use_active_registry=True)

    Returns a plan object. The GUI must check `is_runnable` before calling
    `build_memory()`.
    """
    try:
        config = load_app_config(*config_paths)
    except ServiceError as exc:
        return MemoryBuildPlan(
            config_paths=list(config_paths),
            use_real_data=False,
            requires_checkpoint=False,
            blocking_error=str(exc),
        )

    use_real_data = bool(config.training.use_real_historical_data)
    requires_ckpt = bool(use_real_data and config.memory.build_from_real_checkpoint)

    plan = MemoryBuildPlan(
        config_paths=list(config_paths),
        use_real_data=use_real_data,
        requires_checkpoint=requires_ckpt,
    )

    # Resolution precedence
    candidate = ""
    source = ""
    if explicit_checkpoint:
        candidate = explicit_checkpoint
        source = "explicit"
    elif config.model.checkpoint_path:
        candidate = config.model.checkpoint_path
        source = "config"
    elif use_active_registry:
        try:
            from src.gui_services.checkpoint_service import get_active

            active = get_active()
            if active and active.checkpoint_path:
                candidate = active.checkpoint_path
                source = "active_registry"
        except Exception:  # noqa: BLE001
            candidate = ""

    if candidate:
        if not Path(candidate).exists():
            plan.blocking_error = f"Checkpoint path does not exist: {candidate}"
            return plan
        plan.resolved_checkpoint = candidate
        plan.checkpoint_source = source

    if requires_ckpt and not plan.resolved_checkpoint:
        plan.blocking_error = (
            "This config requires a checkpoint for the memory build "
            "(memory.build_from_real_checkpoint=true with real-data training). "
            "Select a checkpoint file or pick the active checkpoint."
        )
    return plan


def describe_memory_features(config_paths: list[str] | None = None) -> dict[str, bool]:
    """Return which memory-related features are enabled.

    Post-Great-Simplification, the memory subsystem is a passive scaffold
    (store states + outcomes + kNN lookup) and the only runtime-relevant
    signals are whether memory itself is on and whether updates are allowed.
    """
    config_paths = config_paths or ["configs/default.yaml"]
    try:
        config = load_app_config(*config_paths)
    except ServiceError:
        return {}
    return {
        "memory_enabled": bool(getattr(config.memory, "enabled", False)),
        "memory_updates": bool(getattr(config.runtime, "allow_online_memory_updates", False)),
    }


def get_memory_status(*, config_paths: list[str] | None = None) -> MemoryStatus:
    """Inspect the persisted memory bank without holding any GPU/CPU model state."""
    config_paths = config_paths or ["configs/default.yaml"]
    try:
        config = load_app_config(*config_paths)
    except ServiceError as exc:
        return MemoryStatus(error=str(exc))

    bank_path = config.memory.persist_path
    status = MemoryStatus(bank_path=bank_path)
    path = Path(bank_path)
    if not path.exists():
        status.error = "Memory bank file not found."
        return status
    status.exists = True

    try:
        from src.memory.storage import load_memory_bank

        bank = load_memory_bank(path, map_location="cpu")
        summary = bank.summary()
    except Exception as exc:  # noqa: BLE001
        status.error = f"Failed to load memory bank: {exc}"
        return status

    status.item_count = int(summary.get("item_count", 0))
    status.latent_dim = int(summary.get("latent_dim", 0))
    status.max_items = int(summary.get("max_items", 0))
    status.similarity_metric = str(summary.get("similarity_metric") or "")
    status.encoder_fingerprint = str(summary.get("encoder_fingerprint") or "")
    status.feature_contract_id = str(summary.get("feature_contract_id") or "")
    return status


def build_memory(
    *,
    config_paths: list[str],
    explicit_checkpoint: str | None = None,
    use_active_registry: bool = False,
    progress_callback: Callable[[str], None] | None = None,
) -> str:
    """Build memory by invoking scripts/build_memory.py as a subprocess.

    Performs preflight validation via plan_memory_build() and refuses to
    launch the subprocess if the plan is not runnable. The CLI built here
    is the only place that knows how to talk to scripts/build_memory.py —
    GUI widgets must not duplicate this logic.

    Subprocess isolation is used because:
      - the build loads the full model + feature pipeline and runs over all bars
      - the script already encapsulates the correct end-to-end pipeline
      - any failure stays out of the GUI process
    """
    if progress_callback is None:
        progress_callback = lambda _msg: None  # noqa: E731

    progress_callback("Validating memory-build plan…")
    plan = plan_memory_build(
        config_paths=config_paths,
        explicit_checkpoint=explicit_checkpoint,
        use_active_registry=use_active_registry,
    )
    if not plan.is_runnable:
        raise ServiceError(plan.blocking_error)

    cmd = [sys.executable, "-u", "scripts/build_memory.py"]
    for path in plan.config_paths:
        cmd += ["--config", path]
    if plan.resolved_checkpoint:
        cmd += ["--checkpoint", plan.resolved_checkpoint]
    if plan.use_real_data:
        cmd += ["--real-data", "--allow-mock"]

    progress_callback(
        f"Spawning build_memory.py "
        f"(checkpoint={plan.resolved_checkpoint or '—'}, "
        f"real_data={plan.use_real_data})"
    )
    # Stream subprocess output line-by-line so the GUI sees live progress
    # instead of nothing for ~20 minutes.  ``subprocess.run(capture_output=True)``
    # buffers everything until the process exits, which makes the build look
    # hung even when it's making progress.
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
        )
    except FileNotFoundError as exc:
        raise ServiceError(f"Cannot launch python: {exc}") from exc

    stdout_lines: list[str] = []
    assert proc.stdout is not None  # stdout=PIPE guarantees this
    for raw_line in proc.stdout:
        line = raw_line.rstrip("\r\n")
        stdout_lines.append(raw_line)
        if line:
            progress_callback(line)
    proc.wait()
    stdout_text = "".join(stdout_lines)

    if proc.returncode != 0:
        # Print full detail to console for debugging, give the operator a short msg.
        print("[memory build failed — full output]\n" + stdout_text)
        tail = [ln for ln in stdout_text.strip().splitlines() if ln.strip()][-3:]
        short = " | ".join(tail) or "non-zero exit"
        raise ServiceError(f"Memory build failed: {short}")
    progress_callback("Memory build complete.")
    return stdout_text.strip()


