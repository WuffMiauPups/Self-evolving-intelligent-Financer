from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError

# The complete strategy registry the arbiter knows how to build
# (src/strategies/arbiter.py). Adding a primitive means teaching the
# arbiter factory about it AND whitelisting the name here.
KNOWN_STRATEGY_IDS = {"brain", "momentum", "mean_reversion"}


class ArbitrationConfig(BaseModel):
    """Strategy arbitration layer configuration (MASTER_PLAN §6.3).

    The founding-vision pillar: known real-life strategies run alongside
    the brain under SOFT CAPS. Every registered strategy proposes on every
    snapshot; the arbiter picks the directional candidate whose strategy
    holds the highest outcome-weighted allocation and scales its target
    fraction by the strategy's ``budget_fractions`` entry (the soft cap).
    A per-strategy realized-PnL EMA (``ema_alpha``) drives the weights;
    ``weight_floor`` keeps every strategy's weight nonzero, so a losing
    strategy is throttled, never permanently starved.

    Disabled by default and PAPER-ONLY until an arbitrated run beats the
    brain-only scorecard row — same promotion discipline as every other
    Phase 2 feature.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    strategies: list[str] = Field(
        default_factory=lambda: ["brain", "momentum", "mean_reversion"]
    )
    budget_fractions: dict[str, float] = Field(
        default_factory=lambda: {"brain": 1.0, "momentum": 0.5, "mean_reversion": 0.5}
    )
    ema_alpha: float = 0.1
    weight_floor: float = 0.05

    @model_validator(mode="after")
    def validate_arbitration_config(self) -> "ArbitrationConfig":
        unknown = sorted(set(self.strategies) - KNOWN_STRATEGY_IDS)
        if unknown:
            raise ConfigurationError(
                f"arbitration.strategies contains unknown entries {unknown}; "
                f"known: {sorted(KNOWN_STRATEGY_IDS)}"
            )
        if "brain" not in self.strategies:
            raise ConfigurationError("arbitration.strategies must include 'brain'")
        if len(set(self.strategies)) != len(self.strategies):
            raise ConfigurationError("arbitration.strategies must not contain duplicates")
        unknown_budgets = sorted(set(self.budget_fractions) - KNOWN_STRATEGY_IDS)
        if unknown_budgets:
            raise ConfigurationError(
                f"arbitration.budget_fractions contains unknown entries {unknown_budgets}; "
                f"known: {sorted(KNOWN_STRATEGY_IDS)}"
            )
        missing = [name for name in self.strategies if name not in self.budget_fractions]
        if missing:
            raise ConfigurationError(
                f"arbitration.budget_fractions missing entries for {missing}"
            )
        for name, fraction in self.budget_fractions.items():
            if not 0.0 < fraction <= 1.0:
                raise ConfigurationError(
                    f"arbitration.budget_fractions[{name!r}] must be within (0, 1]"
                )
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ConfigurationError("arbitration.ema_alpha must be within (0, 1]")
        if not 0.0 < self.weight_floor <= 0.5:
            raise ConfigurationError("arbitration.weight_floor must be within (0, 0.5]")
        return self
