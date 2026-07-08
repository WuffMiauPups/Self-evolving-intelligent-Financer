"""Configuration loading and preset discovery for the GUI."""
from __future__ import annotations

from pathlib import Path

from src.core.config import AppConfig, load_config
from src.gui_services.state import ServiceError


CONFIG_DIR = Path("configs")


def list_config_presets(config_dir: Path | str = CONFIG_DIR) -> list[Path]:
    """Return all *.yaml files in the config directory, sorted."""
    path = Path(config_dir)
    if not path.exists():
        return []
    return sorted(p for p in path.glob("*.yaml") if p.is_file())


def load_app_config(*paths: str | Path) -> AppConfig:
    """Load and validate an AppConfig from one or more YAML files."""
    if not paths:
        paths = (CONFIG_DIR / "default.yaml",)
    try:
        return load_config(*paths)
    except Exception as exc:  # noqa: BLE001
        raise ServiceError(f"Failed to load config: {exc}") from exc


def describe_config(config: AppConfig) -> dict[str, object]:
    """Return a small summary of the loaded config for UI display.

    Only surfaces flags that still map to live behaviour after the Great
    Simplification.  Dead features (fusion / topology growth / modern
    hopfield / regime hysteresis / market priors / etc.) are not exported.
    """
    fee_bps = float(getattr(config.risk, "fee_bps", 0.0))
    slippage_bps = float(getattr(config.risk, "slippage_bps", 0.0))
    return {
        "symbols": list(config.runtime.symbols),
        "memory_enabled": bool(config.memory.enabled),
        "frozen_brain": not bool(config.runtime.allow_online_model_updates),
        "memory_updates_allowed": bool(config.runtime.allow_online_memory_updates),
        "memory_update_mode": config.memory.runtime_update_mode,
        "checkpoint_path": config.model.checkpoint_path or "",
        "kill_switch": bool(config.risk.kill_switch),
        "log_path": getattr(config.logging, "file_path", "logs/seif_runtime.log"),
        # EV gate (Phase 5/7) — active whenever the risk head is trained.
        "separate_risk_heads": bool(getattr(config.model, "separate_risk_heads", False)),
        "ev_cost_bps": fee_bps + slippage_bps,
        # Confidence calibration (Phase A — Platt scaling).
        "platt_calibration_enabled": bool(getattr(config.training, "platt_calibration_enabled", False)),
        "order_journal_enabled": bool(getattr(config.execution, "order_journal_enabled", True)),
        "alert_webhook_url": str(getattr(config.runtime, "alert_webhook_url", "") or ""),
    }


def describe_memory_caps(config: AppConfig) -> dict[str, int]:
    """Return the raw numeric memory caps for the RuntimeOverridesPanel.

    Post-Simplification, only ``max_items`` survives as a user-tunable cap
    on the passive memory bank.
    """
    return {
        "max_items": int(config.memory.max_items),
    }
