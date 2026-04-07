from src.core.config import RuntimeConfig, load_config


def test_config_loading_and_override() -> None:
    config = load_config("configs/default.yaml", "configs/paper_trade.yaml")
    assert config.broker.paper is True
    assert config.broker.live_enabled is False
    # Default paper runtime is UNBOUNDED (0 sentinel) — must run until stopped.
    assert config.runtime.max_runtime_steps == 0
    assert "AAPL" in config.runtime.symbols
    assert "AAPL" in config.data.symbols
    assert config.memory.enabled is True
    # Frozen-brain runtime policy
    assert config.runtime.allow_online_model_updates is False
    assert config.runtime.allow_online_memory_updates is True
    # paper_trade.yaml deliberately disables decision_fusion in the runtime
    # path until the runtime/offline conflict semantics are aligned.
    assert config.memory.decision_fusion.enabled is False


def test_runtime_config_default_is_unbounded() -> None:
    """Programmatic RuntimeConfig() defaults must be unbounded.

    Anything > 0 here would silently bound paper trading after a fixed number
    of ticks for any caller that constructs RuntimeConfig without YAML.
    """
    rc = RuntimeConfig()
    assert rc.max_runtime_steps == 0
    assert rc.allow_online_model_updates is False
    assert rc.allow_online_memory_updates is True
