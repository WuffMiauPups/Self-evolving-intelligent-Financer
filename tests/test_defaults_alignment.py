"""Python-defaults vs configs/default.yaml alignment (MASTER_PLAN §4.6.5).

YAML is canonical. Python-side defaults exist so code constructing config
models directly (tests, tools) behaves like the deployed config — the
2026-07-14 plan audit found 38 silently divergent fields (epochs 3 vs 5,
sequence_length 64 vs 48, inverted AMP/Platt flags, risk limits, ...).

The rule enforced here: every divergence must be on the documented
allow-list below, with a reason. New drift fails this test; so does a
stale allow-list entry after someone aligns a field.
"""
from __future__ import annotations

from src.core.config import AppConfig, load_config

# Deliberate divergences: bare-code default vs deployed YAML value.
# Each entry is field-path -> reason. Keep this list SHORT — anything
# without a load-bearing reason should be aligned instead.
DOCUMENTED_DIVERGENCES: dict[str, str] = {
    "broker.base_url": "None lets the client/env decide; YAML pins the paper endpoint",
    "data.symbols": "trading universe is an operational choice that lives in YAML; bare default stays minimal for unit tests",
    "runtime.symbols": "synced from data.symbols; same rationale",
    "memory.enabled": "a bare config must not switch subsystems on",
    "model.checkpoint_path": "None = no checkpoint; only the deployed config points at canonical brain.pt",
    "model.decoupled_heads": "False is LOAD-BEARING: pre-Phase-2 checkpoint metas missing the key must reconstruct legacy heads",
    "training.save_memory_after_training": "post-training side effect; bare configs must not trigger bank builds",
    "training.use_real_historical_data": "bare configs must not require market data or credentials",
}


def _flatten(model, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if hasattr(type(value), "model_fields"):
            out.update(_flatten(value, prefix=f"{path}."))
        else:
            out[path] = value
    return out


def test_python_defaults_match_default_yaml_except_documented() -> None:
    code = _flatten(AppConfig())
    yaml = _flatten(load_config("configs/default.yaml"))
    assert set(code) == set(yaml), "config field sets diverged between code and YAML"
    divergent = {key for key in code if code[key] != yaml[key]}
    undocumented = divergent - set(DOCUMENTED_DIVERGENCES)
    stale = set(DOCUMENTED_DIVERGENCES) - divergent
    assert not undocumented, (
        f"Undocumented default drift (align the Python default to default.yaml, "
        f"or add a reasoned entry to DOCUMENTED_DIVERGENCES): {sorted(undocumented)}"
    )
    assert not stale, (
        f"Stale DOCUMENTED_DIVERGENCES entries (field now aligned — remove them): {sorted(stale)}"
    )


def test_plan_named_fields_are_aligned() -> None:
    """The exact fields MASTER_PLAN §4.6.5 called out must stay aligned."""
    code = _flatten(AppConfig())
    yaml = _flatten(load_config("configs/default.yaml"))
    for field in (
        "risk.max_daily_loss",
        "training.epochs",
        "training.sequence_length",
        "training.use_amp",
        "training.platt_calibration_enabled",
        "training.early_stopping_patience",
        "features.epsilon",
    ):
        assert code[field] == yaml[field], f"{field}: code={code[field]!r} yaml={yaml[field]!r}"
