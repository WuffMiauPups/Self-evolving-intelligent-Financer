"""Smoke tests for src.backtest.policy_factory.

These tests verify the three public builders return the right classes and
that the baseline builder has no hidden config dependencies.  The brain +
brain+memory builders require a resolved checkpoint and feature pipeline
so they are exercised via the test suite that already covers
scripts/run_backtest.py end-to-end (tests/test_backtest.py).  Here we
just verify the factory's dispatch shape.
"""
from __future__ import annotations

from src.backtest.policy_factory import (
    build_baseline_policy,
    build_brain_memory_policy,
    build_brain_policy,
)
from src.strategies.baseline_policy import BaselineMomentumPolicy


def test_build_baseline_returns_baseline_momentum_policy() -> None:
    policy = build_baseline_policy()
    assert isinstance(policy, BaselineMomentumPolicy)


def test_build_brain_policy_is_callable() -> None:
    # Signature sanity: the callable exists and has the right positional args.
    import inspect
    sig = inspect.signature(build_brain_policy)
    params = sig.parameters
    assert "config" in params
    assert "bars_by_symbol" in params
    assert "pipeline" in params


def test_build_brain_memory_policy_is_callable() -> None:
    import inspect
    sig = inspect.signature(build_brain_memory_policy)
    params = sig.parameters
    assert "config" in params
    assert "bars_by_symbol" in params
    assert "pipeline" in params
    assert "enable_memory" in params
    assert "conservative_thresholds" in params
    assert "verbose_fusion" in params
    # Defaults should preserve the old behavior.
    assert params["enable_memory"].default is True
    assert params["conservative_thresholds"].default is False
    assert params["verbose_fusion"].default is False
