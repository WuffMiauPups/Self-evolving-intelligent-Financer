from __future__ import annotations

from pydantic import BaseModel, model_validator

from src.core.exceptions import ConfigurationError


class BrokerConfig(BaseModel):
    provider: str = "alpaca"
    paper: bool = True
    live_enabled: bool = False
    confirm_live: bool = False
    require_account_match: bool = True
    base_url: str | None = None
    data_feed: str = "iex"
    polling_interval_sec: float = 2.0
    mock_data_on_missing_credentials: bool = True

    @model_validator(mode="after")
    def validate_live_mode(self) -> "BrokerConfig":
        if self.live_enabled and self.paper:
            raise ConfigurationError("paper=true and live_enabled=true cannot be enabled together")
        if not self.paper and not self.live_enabled:
            raise ConfigurationError("paper=false requires live_enabled=true")
        return self
