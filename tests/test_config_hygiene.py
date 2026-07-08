"""Config-files hygiene tests (master plan workstream 1F).

The config package enforces ``extra='forbid'`` on every model except the two
documented inert stubs (DecisionFusionConfig / TopologyGrowthConfig), so a
YAML key without a model field is a hard load error instead of a silent
no-op. These tests make that contract permanent:

1. Every ``configs/**/*.yaml`` must load cleanly through ``load_config``
   layered over ``configs/default.yaml`` — any future dead/typo'd key in any
   preset fails this suite.
2. The two inert stubs must keep swallowing legacy fusion/topology sub-keys
   (old YAMLs and presets still carry them).
3. A typo'd key must raise at load, not validate silently.

Note: ``configs/sweep_state_only.yaml`` is NOT an AppConfig file — it is a
sweep-grid spec consumed by ``scripts/sweep_train.py`` (top-level ``preset``
/ ``sweep`` keys) — so it is excluded from the load_config sweep.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.config import (
    DecisionFusionConfig,
    RiskConfig,
    TopologyGrowthConfig,
    load_config,
)
from src.core.exceptions import ConfigurationError

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"

# YAML files under configs/ that are not AppConfig layers.
NOT_APP_CONFIG = {
    "sweep_state_only.yaml",  # sweep-grid spec for scripts/sweep_train.py
}

ALL_CONFIG_FILES = sorted(
    path
    for path in CONFIGS_DIR.rglob("*.yaml")
    if path.name not in NOT_APP_CONFIG
)


def _relative_id(path: Path) -> str:
    return path.relative_to(CONFIGS_DIR).as_posix()


# ---------------------------------------------------------------------------
# 1. Every config file validates under extra='forbid'
# ---------------------------------------------------------------------------


def test_config_sweep_is_not_empty() -> None:
    assert DEFAULT_CONFIG.exists()
    assert len(ALL_CONFIG_FILES) > 10  # presets + experiments must be swept


def test_default_yaml_loads_alone() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.app_name == "seif_trading_brain"


@pytest.mark.parametrize("path", ALL_CONFIG_FILES, ids=_relative_id)
def test_every_config_file_loads_layered_over_default(path: Path) -> None:
    """With extra='forbid', this catches every dead key in every preset."""
    config = load_config(DEFAULT_CONFIG, path)
    assert config is not None


def test_experiment_stack_loads_like_run_experiments() -> None:
    """Mirror scripts/run_experiments.py: default -> _experiment_base -> overlay."""
    base = CONFIGS_DIR / "experiments" / "_experiment_base.yaml"
    for overlay in sorted((CONFIGS_DIR / "experiments").glob("*.yaml")):
        if overlay == base:
            continue
        config = load_config(DEFAULT_CONFIG, base, overlay)
        assert config is not None


# ---------------------------------------------------------------------------
# 2. The two inert stubs intentionally swallow legacy sub-keys
# ---------------------------------------------------------------------------


def test_decision_fusion_stub_swallows_legacy_sub_keys() -> None:
    stub = DecisionFusionConfig(
        enabled=True,
        conflict_veto_threshold=0.70,
        conflict_dampen_threshold=0.40,
        some_future_unknown_key="ignored",
    )
    assert stub.enabled is True
    assert not hasattr(stub, "conflict_veto_threshold")


def test_topology_growth_stub_swallows_legacy_sub_keys() -> None:
    stub = TopologyGrowthConfig(
        enabled=False,
        min_novelty_score=0.60,
        max_insertions_per_refresh=5,
    )
    assert stub.enabled is False
    assert not hasattr(stub, "min_novelty_score")


def test_stub_sub_keys_swallowed_through_load_config(tmp_path: Path) -> None:
    overlay = tmp_path / "legacy_fusion.yaml"
    overlay.write_text(
        "memory:\n"
        "  decision_fusion:\n"
        "    enabled: false\n"
        "    conflict_veto_threshold: 0.99\n"
        "    dampen_misalignment_weight: 0.4\n"
        "  topology_growth:\n"
        "    enabled: false\n"
        "    split_conflict_threshold: 0.5\n",
        encoding="utf-8",
    )
    config = load_config(DEFAULT_CONFIG, overlay)
    assert config.memory.decision_fusion.enabled is False
    assert config.memory.topology_growth.enabled is False


# ---------------------------------------------------------------------------
# 3. Typo'd / dead keys raise at load
# ---------------------------------------------------------------------------


def test_typoed_risk_key_raises_at_load(tmp_path: Path) -> None:
    overlay = tmp_path / "typo.yaml"
    overlay.write_text("risk:\n  min_trade_confidnce: 0.5\n", encoding="utf-8")
    with pytest.raises(ValidationError) as excinfo:
        load_config(DEFAULT_CONFIG, overlay)
    assert "min_trade_confidnce" in str(excinfo.value)


def test_removed_key_raises_at_load(tmp_path: Path) -> None:
    """The exact class of key that used to no-op silently (review §8)."""
    overlay = tmp_path / "removed.yaml"
    overlay.write_text("risk:\n  min_trade_confidence: 0.55\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


def test_unknown_top_level_section_raises(tmp_path: Path) -> None:
    overlay = tmp_path / "topsection.yaml"
    overlay.write_text("strategeis:\n  enabled: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


def test_dead_runtime_regime_key_raises(tmp_path: Path) -> None:
    overlay = tmp_path / "regime.yaml"
    overlay.write_text("runtime:\n  regime_consecutive_threshold: 3\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


def test_dead_training_key_raises(tmp_path: Path) -> None:
    overlay = tmp_path / "training.yaml"
    overlay.write_text("training:\n  sequential_memory_training: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


# ---------------------------------------------------------------------------
# 4. RiskConfig validators (workstream 1F.2)
# ---------------------------------------------------------------------------


def test_negative_max_daily_loss_normalized_to_magnitude() -> None:
    assert RiskConfig(max_daily_loss=-350.0).max_daily_loss == 350.0


def test_zero_max_daily_loss_rejected() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(max_daily_loss=0.0)


@pytest.mark.parametrize("field", ["fee_bps", "slippage_bps", "max_spread_bps"])
def test_negative_bps_rejected(field: str) -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(**{field: -1.0})


@pytest.mark.parametrize(
    "field", ["cooldown_sec", "stale_data_sec", "duplicate_order_window_sec"]
)
def test_negative_seconds_rejected(field: str) -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(**{field: -1})


def test_zero_seconds_allowed() -> None:
    config = RiskConfig(cooldown_sec=0, stale_data_sec=0, duplicate_order_window_sec=0)
    assert config.cooldown_sec == 0


def test_aggregate_notional_must_cover_position_notional() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(max_aggregate_notional=1_000.0, max_position_notional=5_000.0)
    ok = RiskConfig(max_aggregate_notional=5_000.0, max_position_notional=5_000.0)
    assert ok.max_aggregate_notional == 5_000.0


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_max_daily_loss_pct_out_of_range_rejected(bad: float) -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(max_daily_loss_pct=bad)


def test_max_daily_loss_pct_boundary_and_none_allowed() -> None:
    assert RiskConfig(max_daily_loss_pct=1.0).max_daily_loss_pct == 1.0
    assert RiskConfig(max_daily_loss_pct=None).max_daily_loss_pct is None


def test_max_concurrent_positions_floor() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(max_concurrent_positions=0)
    assert RiskConfig(max_concurrent_positions=1).max_concurrent_positions == 1


def test_quantity_precision_must_be_positive() -> None:
    with pytest.raises(ConfigurationError):
        RiskConfig(quantity_precision=0.0)
    with pytest.raises(ConfigurationError):
        RiskConfig(quantity_precision=-0.001)
