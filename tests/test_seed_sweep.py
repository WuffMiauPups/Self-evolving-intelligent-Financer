"""Seed as a sweep axis (noise-floor measurement).

The training seed became config-driven on 2026-07-14 (§4.7 hygiene fix);
before that ``seed_everything(7)`` was hardcoded, so every "different"
sweep run trained identically and seed variance had never been measured.
These tests pin the plumbing that makes the noise floor measurable:
run_train's --seed flag and sweep_train's seed axis.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a scripts/*.py module by path (they are not a package)."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_applies_seed_override() -> None:
    from src.core.config import AppConfig

    sweep_train = _load_script("sweep_train")
    config = AppConfig()
    assert config.training.seed == 7
    swept = sweep_train.apply_overrides(config, {"seed": 29})
    assert swept.training.seed == 29
    # apply_overrides must not mutate the caller's config (each sweep run
    # starts from the same base).
    assert config.training.seed == 7


def test_sweep_rejects_unknown_axis() -> None:
    from src.core.config import AppConfig

    sweep_train = _load_script("sweep_train")
    with pytest.raises(ValueError, match="Unknown sweep key"):
        sweep_train.apply_overrides(AppConfig(), {"not_a_real_axis": 1})


def test_seed_axis_expands_to_one_run_per_seed() -> None:
    sweep_train = _load_script("sweep_train")
    combos = sweep_train.expand_grid({"seed": [7, 13, 29]})
    assert combos == [{"seed": 7}, {"seed": 13}, {"seed": 29}]
    # Seeds are not redundant with each other: pruning must keep all three.
    assert len(sweep_train.prune_redundant_combos(combos)) == 3


def test_run_train_exposes_seed_flag() -> None:
    source = (REPO_ROOT / "scripts" / "run_train.py").read_text(encoding="utf-8")
    assert '"--seed"' in source
    assert "config.training.seed = args.seed" in source


def test_seed_variance_sweep_config_holds_everything_but_seed_fixed() -> None:
    """The sweep config must vary ONLY the seed — any other axis would
    confound the noise-floor measurement it exists to produce."""
    import yaml

    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "sweep_seed_variance.yaml").read_text(encoding="utf-8")
    )
    assert list(payload["sweep"]) == ["seed"]
    assert len(payload["sweep"]["seed"]) >= 3, "need >= 3 points to see a spread"
