from src.core.config import load_config


def test_config_loading_and_override() -> None:
    config = load_config("configs/default.yaml", "configs/paper_trade.yaml")
    assert config.broker.paper is True
    assert config.broker.live_enabled is False
    assert config.runtime.max_runtime_steps == 25
    assert "AAPL" in config.runtime.symbols
    assert "AAPL" in config.data.symbols
    assert config.memory.enabled is True
    assert config.training.use_real_historical_data is True
    assert config.risk.min_trade_confidence == 0.20
    assert config.risk.min_trade_fraction == 0.005
    # Frozen-brain runtime policy
    assert config.runtime.allow_online_model_updates is False
    assert config.runtime.allow_online_memory_updates is True
    # Decision fusion wired
    assert config.memory.decision_fusion.enabled is True
