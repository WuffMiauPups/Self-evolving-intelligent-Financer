"""Trading policy protocols — formal interface contracts.

Replaces the scattered ``hasattr(policy, ...)`` duck-typing with explicit,
runtime-checkable protocols.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.features.feature_pipeline import FeatureSnapshot
    from src.memory.updater import AssociativeMemoryUpdater
    from src.risk.constraints import TradeProposal


@runtime_checkable
class TradingPolicy(Protocol):
    """Base trading policy: propose + state management."""

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]: ...
    def reset_symbol_state(self, symbol: str, *, reason: str) -> str: ...
    def reset_all_state(self, *, reason: str) -> str: ...


@runtime_checkable
class BatchTradingPolicy(TradingPolicy, Protocol):
    """Policy that supports batched multi-symbol inference."""

    def propose_batch(
        self, snapshots: list[FeatureSnapshot],
    ) -> list[tuple[TradeProposal, dict[str, object]]]: ...


@runtime_checkable
class MemoryCapablePolicy(BatchTradingPolicy, Protocol):
    """Policy with associative memory retrieval and online learning."""

    updater: AssociativeMemoryUpdater | None

    def prepare_symbol_history(self, symbol: str, bars: list) -> list: ...
    def record_snapshot(self, symbol: str) -> None: ...
    def metrics_snapshot(self) -> dict[str, float]: ...
    def set_temporal_cutoff(self, cutoff: datetime | None) -> None: ...
