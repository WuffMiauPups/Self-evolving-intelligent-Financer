"""Tests for RuntimeOverrides YAML emission and layered-config round-trip.

Post-Great-Simplification surface:
  - frozen_brain
  - memory_enabled
  - memory_updates_enabled
  - platt_calibration_enabled
  - separate_risk_heads (EV gate)
  - max_memory_items
  - checkpoint_path
"""
from __future__ import annotations

from pathlib import Path

from src.core.config import load_config
from src.gui_services.state import RuntimeOverrides


def test_empty_overrides_is_empty() -> None:
    ov = RuntimeOverrides()
    assert ov.is_empty()
    assert ov.to_yaml_dict() == {}


def test_only_set_fields_are_emitted() -> None:
    ov = RuntimeOverrides(memory_enabled=True, max_memory_items=1234)
    data = ov.to_yaml_dict()
    assert data == {"memory": {"enabled": True, "max_items": 1234}}


def test_frozen_brain_is_inverted_into_allow_online_updates() -> None:
    frozen = RuntimeOverrides(frozen_brain=True).to_yaml_dict()
    assert frozen["runtime"]["allow_online_model_updates"] is False

    unfrozen = RuntimeOverrides(frozen_brain=False).to_yaml_dict()
    assert unfrozen["runtime"]["allow_online_model_updates"] is True


def test_checkpoint_path_nests_under_model() -> None:
    data = RuntimeOverrides(checkpoint_path="checkpoints/exp_test.pt").to_yaml_dict()
    assert data["model"]["checkpoint_path"] == "checkpoints/exp_test.pt"


def test_memory_updates_maps_to_runtime_flag() -> None:
    data = RuntimeOverrides(memory_updates_enabled=False).to_yaml_dict()
    assert data["runtime"]["allow_online_memory_updates"] is False


def test_platt_calibration_nests_under_training() -> None:
    data = RuntimeOverrides(platt_calibration_enabled=True).to_yaml_dict()
    assert data["training"]["platt_calibration_enabled"] is True


def test_separate_risk_heads_nests_under_model() -> None:
    """Risk heads feed the EV gate; GUI toggle must land in model config."""
    data = RuntimeOverrides(separate_risk_heads=True).to_yaml_dict()
    assert data["model"]["separate_risk_heads"] is True


def test_write_yaml_round_trips_through_load_config(tmp_path: Path) -> None:
    """End-to-end: write an override YAML, layer it on top of default.yaml,
    and assert the resulting AppConfig reflects the overrides."""
    ov = RuntimeOverrides(
        frozen_brain=False,
        memory_enabled=False,
        platt_calibration_enabled=True,
        separate_risk_heads=True,
        max_memory_items=12345,
    )
    out = tmp_path / "override.yaml"
    ov.write_yaml(out)
    assert out.exists()

    cfg = load_config("configs/default.yaml", str(out))
    assert cfg.runtime.allow_online_model_updates is True
    assert cfg.memory.enabled is False
    assert cfg.memory.max_items == 12345
    assert cfg.training.platt_calibration_enabled is True
    assert cfg.model.separate_risk_heads is True


def test_empty_overrides_write_is_still_a_valid_file(tmp_path: Path) -> None:
    """Even when nothing is set, write_yaml should produce a parseable file."""
    out = tmp_path / "empty.yaml"
    RuntimeOverrides().write_yaml(out)
    assert out.exists()
    cfg = load_config("configs/default.yaml", str(out))
    assert cfg is not None


def test_write_yaml_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "override.yaml"
    RuntimeOverrides(memory_enabled=True).write_yaml(out)
    assert out.exists()
