"""Config-files hygiene tests (master plan workstream 1F).

The config package enforces ``extra='forbid'`` on every model, so a YAML key
without a model field is a hard load error instead of a silent no-op. These
tests make that contract permanent:

1. Every ``configs/**/*.yaml`` must load cleanly through ``load_config``
   layered over ``configs/default.yaml`` — any future dead/typo'd key in any
   preset fails this suite.
2. ``MemoryConfig``'s field set is pinned to the fields that actually drive
   behavior, so the pre-§6.2-prune dead-knob graveyard (Hopfield / ESOM /
   dynamic units / calibration / fusion) cannot silently regrow, and legacy
   fusion/topology blocks must now raise at load (the inert swallow-stubs
   were deleted).
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
    MemoryConfig,
    RiskConfig,
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
# 2. MemoryConfig field set is pinned — the dead-knob graveyard cannot regrow
# ---------------------------------------------------------------------------

# Every field must have a consumer in src/ or scripts/ (grep before adding).
# This is the complete allowed surface after the §6.2 dead-field prune; a
# new field here without a real consumer is a regression, not a config knob.
MEMORY_CONFIG_ALLOWED_FIELDS = {
    "enabled",
    "build_from_real_checkpoint",
    "latent_dim",
    "top_k",
    "similarity_metric",
    "max_items",
    "rebuild_on_startup",
    "persist_path",
    "forward_horizons",
    "path_label_rules",
    "runtime_update_mode",
    "build_history_bars",
    "require_encoder_match",
    "persist_pending_observations",
    "pending_store_path",
    "save_debounce_sec",
    "pending_max_per_symbol",
    "influence_mode",
    "damping_min_multiplier",
    "damping_min_neighbors",
    "damping_full_at_return",
}


def test_memory_config_field_set_is_pinned() -> None:
    assert set(MemoryConfig.model_fields) == MEMORY_CONFIG_ALLOWED_FIELDS


def test_legacy_fusion_and_topology_blocks_raise_at_load(tmp_path: Path) -> None:
    """The inert DecisionFusionConfig / TopologyGrowthConfig swallow-stubs
    were deleted with the rest of the graveyard: a config still carrying
    those blocks must fail loudly instead of silently no-opping."""
    overlay = tmp_path / "legacy_fusion.yaml"
    overlay.write_text(
        "memory:\n"
        "  decision_fusion:\n"
        "    enabled: false\n"
        "  topology_growth:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


def test_dead_memory_knob_raises_at_load(tmp_path: Path) -> None:
    """Representative of the ~100 pruned Hopfield/ESOM/calibration knobs."""
    overlay = tmp_path / "dead_memory.yaml"
    overlay.write_text("memory:\n  use_hopfield_retrieval: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(DEFAULT_CONFIG, overlay)


# ---------------------------------------------------------------------------
# 2b. training <-> memory forward_horizons stay in sync (MASTER_PLAN 4.6.2)
# ---------------------------------------------------------------------------
# default.yaml intentionally does NOT set memory.forward_horizons: AppConfig
# auto-syncs it from training.forward_horizons unless the memory side
# explicitly overrides.  These tests make that contract permanent so the
# offline memory build always labels with the horizons the brain was
# trained against.


def test_default_config_memory_horizons_match_training() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert list(config.memory.forward_horizons) == list(config.training.forward_horizons)
    assert config.memory.path_label_rules == config.training.path_label_rules


def test_training_overlay_keeps_memory_horizons_synced() -> None:
    """configs/training.yaml changes training.forward_horizons; the memory
    side must follow automatically because default.yaml no longer pins it."""
    config = load_config(DEFAULT_CONFIG, CONFIGS_DIR / "training.yaml")
    assert list(config.training.forward_horizons) == [1, 5, 15]
    assert list(config.memory.forward_horizons) == [1, 5, 15]


def test_explicit_memory_horizon_override_still_wins(tmp_path: Path) -> None:
    overlay = tmp_path / "memory_horizons.yaml"
    overlay.write_text("memory:\n  forward_horizons: [2, 4]\n", encoding="utf-8")
    config = load_config(DEFAULT_CONFIG, overlay)
    assert list(config.memory.forward_horizons) == [2, 4]


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
