"""Sweep runs must never touch canonical artifacts (§10 rule 3).

Regression for the 2026-07-17 incident: a sweep's save_memory_after_training
wrote each run's bank to the canonical memory.persist_path
(artifacts/market_memory.pt), clobbering the live bank while paper trading
ran. The fingerprint guard contained it, but the sweep had no business
writing there — isolate_sweep_artifacts redirects every writable path into
the sweep dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sweep_train import isolate_sweep_artifacts  # noqa: E402

from src.core.config import AppConfig


def test_memory_and_checkpoint_paths_leave_canonical_locations() -> None:
    config = AppConfig()
    # Canonical starting points the incident clobbered.
    assert config.memory.persist_path == "artifacts/market_memory.pt"
    isolate_sweep_artifacts(config, "sweep_20260717_142927", 3)
    assert config.training.checkpoint_dir == "checkpoints/sweep_20260717_142927"
    assert config.memory.persist_path == (
        "checkpoints/sweep_20260717_142927/market_memory_run003.pt"
    )
    assert config.memory.pending_store_path == (
        "checkpoints/sweep_20260717_142927/pending_run003.pt"
    )
    # The canonical artifact paths must NOT appear anywhere writable.
    assert "artifacts/market_memory.pt" not in config.memory.persist_path
    assert "artifacts/" not in config.memory.pending_store_path


def test_run_index_makes_paths_unique_per_run() -> None:
    p1 = isolate_sweep_artifacts(AppConfig(), "sweepX", 1).memory.persist_path
    p2 = isolate_sweep_artifacts(AppConfig(), "sweepX", 2).memory.persist_path
    assert p1 != p2
    assert p1.endswith("market_memory_run001.pt")
    assert p2.endswith("market_memory_run002.pt")


def test_sweep_loop_calls_isolation_source_contract() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "sweep_train.py"
    ).read_text(encoding="utf-8")
    # Anchor on the real loop's apply_overrides call (the dry-run block also
    # iterates combos), up to the training call.
    loop_region = source.split("run_config = apply_overrides(base_config, combo)", 1)[1].split(
        "result = run_supervised_training", 1
    )[0]
    assert "isolate_sweep_artifacts(run_config" in loop_region
