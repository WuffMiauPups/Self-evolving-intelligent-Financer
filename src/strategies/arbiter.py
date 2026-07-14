"""Strategy arbitration layer (MASTER_PLAN §6.3).

The founding-vision pillar: known real-life strategies run alongside the
brain under SOFT CAPS. Every registered strategy proposes on every
snapshot; the arbiter picks the directional candidate whose strategy
currently holds the highest outcome-weighted allocation, scales its
target fraction by the strategy's budget fraction (the soft cap), and
stamps the proposal with the winning ``strategy_id`` so the realized fill
PnL can be credited back via ``record_fill_outcome``.

Soft-cap philosophy: allocation weights are a normalized per-strategy
realized-PnL EMA with a nonzero floor — a losing strategy is throttled
toward the floor, never permanently starved. It keeps proposing and wins
arbitration again the moment its outcomes recover (or every rival goes
flat). Nothing here can grow a proposal: budget fractions are capped at
1.0 by ArbitrationConfig, so arbitration only ever shrinks size.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from src.core.config.arbitration import ArbitrationConfig
from src.features.feature_pipeline import FeatureSnapshot
from src.risk.constraints import TradeProposal
from src.strategies.base import BatchTradingPolicy, TradingPolicy
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.strategies.baselines import MeanReversionPolicy

logger = logging.getLogger(__name__)

_BRAIN_ID = "brain"

# Memory surface delegated to the brain member via ``__getattr__`` — but
# only when the brain actually has it, so ``isinstance(arbiter,
# MemoryCapablePolicy)`` stays truthful (True for a MemoryAwarePolicy
# brain, False for a bare BrainPolicy) and memory keeps learning under
# arbitration.
_BRAIN_DELEGATED_ATTRS = frozenset(
    {
        "prepare_symbol_history",
        "record_snapshot",
        "metrics_snapshot",
        "set_temporal_cutoff",
        "updater",
    }
)


def build_primitive(strategy_id: str) -> object:
    """Instantiate a bounded rule primitive by registry id.

    Primitives only implement ``propose`` — no batching, no memory.
    ArbitrationConfig whitelists the ids, so an unknown id here is a
    programming error, not an operator mistake.
    """
    if strategy_id == "momentum":
        return BaselineMomentumPolicy()
    if strategy_id == "mean_reversion":
        return MeanReversionPolicy()
    raise ValueError(f"Unknown primitive strategy id: {strategy_id!r}")


def build_arbiter(brain: object, config: ArbitrationConfig) -> "StrategyArbiter":
    """Compose a StrategyArbiter from an already-built brain policy.

    ``brain`` is the BrainPolicy — or the MemoryAwarePolicy wrapping it
    when memory is enabled — so arbitration always sits OUTSIDE memory
    wrapping. Primitives follow ``config.strategies`` order (registry
    order breaks weight ties).
    """
    primitives = [
        (strategy_id, build_primitive(strategy_id))
        for strategy_id in config.strategies
        if strategy_id != _BRAIN_ID
    ]
    return StrategyArbiter(
        brain=brain,
        primitives=primitives,
        budget_fractions=dict(config.budget_fractions),
        ema_alpha=config.ema_alpha,
        weight_floor=config.weight_floor,
    )


@dataclass
class StrategyArbiter:
    """Registry of (strategy_id, policy) pairs with outcome-weighted picks.

    Satisfies TradingPolicy + BatchTradingPolicy; the brain's memory
    surface (``record_snapshot``, ``updater``, ...) is delegated via
    ``__getattr__`` so MemoryCapablePolicy isinstance checks pass exactly
    when the brain member is memory-capable.
    """

    brain: object  # BrainPolicy or MemoryAwarePolicy
    primitives: list[tuple[str, object]] = field(default_factory=list)
    budget_fractions: dict[str, float] = field(default_factory=dict)
    ema_alpha: float = 0.1
    weight_floor: float = 0.05
    # Per-strategy realized-net-PnL EMA. All zeros at start → equal weights.
    pnl_ema: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for strategy_id in self.strategy_ids():
            self.pnl_ema.setdefault(strategy_id, 0.0)

    def __getattr__(self, name: str) -> object:
        # Only consulted when normal lookup fails; forwards the memory
        # surface iff the brain has it (see _BRAIN_DELEGATED_ATTRS note).
        if name in _BRAIN_DELEGATED_ATTRS:
            brain = self.__dict__.get("brain")
            if brain is not None and hasattr(brain, name):
                return getattr(brain, name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ---- Registry / allocation -------------------------------------------

    def strategy_ids(self) -> list[str]:
        """Registry order: brain first, then primitives in config order."""
        return [_BRAIN_ID] + [strategy_id for strategy_id, _ in self.primitives]

    def weights(self) -> dict[str, float]:
        """Normalized positive allocation scores: max(ema, 0) + floor.

        Deterministic and softmax-free. The floor keeps every strategy's
        weight nonzero — a loser is throttled, never permanently starved.
        """
        scores = {
            strategy_id: max(self.pnl_ema.get(strategy_id, 0.0), 0.0) + self.weight_floor
            for strategy_id in self.strategy_ids()
        }
        total = sum(scores.values())
        return {strategy_id: score / total for strategy_id, score in scores.items()}

    def record_fill_outcome(self, strategy_id: str, realized_net: float) -> None:
        """Fold a fill's realized net PnL into the strategy's EMA."""
        if strategy_id not in self.pnl_ema:
            logger.debug("arbiter_unknown_strategy_outcome strategy=%s", strategy_id)
            return
        previous = self.pnl_ema[strategy_id]
        self.pnl_ema[strategy_id] = (
            (1.0 - self.ema_alpha) * previous + self.ema_alpha * float(realized_net)
        )
        logger.info(
            "arbiter_weight_update strategy=%s realized_net=%.6f ema=%.6f weights=%s",
            strategy_id,
            float(realized_net),
            self.pnl_ema[strategy_id],
            {sid: round(weight, 4) for sid, weight in self.weights().items()},
        )

    # ---- TradingPolicy / BatchTradingPolicy protocol ----------------------

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        for member in self._members():
            if isinstance(member, TradingPolicy):
                member.reset_symbol_state(symbol, reason=reason)
        return reason

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        for member in self._members():
            if isinstance(member, TradingPolicy):
                member.reset_all_state(reason=reason)
        return reason

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]:
        brain_proposal, brain_diagnostics = self.brain.propose(snapshot)
        return self._arbitrate(snapshot, brain_proposal, brain_diagnostics)

    def propose_batch(
        self, snapshots: list[FeatureSnapshot]
    ) -> list[tuple[TradeProposal, dict[str, object]]]:
        if not snapshots:
            return []
        # Only the brain batches (one forward for all symbols); primitives
        # are cheap Python and propose per snapshot inside _arbitrate.
        if isinstance(self.brain, BatchTradingPolicy):
            brain_results = self.brain.propose_batch(snapshots)
        else:
            brain_results = [self.brain.propose(snapshot) for snapshot in snapshots]
        return [
            self._arbitrate(snapshot, proposal, diagnostics)
            for snapshot, (proposal, diagnostics) in zip(snapshots, brain_results)
        ]

    # ---- Internal helpers --------------------------------------------------

    def _members(self) -> list[object]:
        return [self.brain] + [member for _, member in self.primitives]

    def _arbitrate(
        self,
        snapshot: FeatureSnapshot,
        brain_proposal: TradeProposal,
        brain_diagnostics: dict[str, object],
    ) -> tuple[TradeProposal, dict[str, object]]:
        """Pick the highest-weight directional candidate; soft-cap its size.

        Candidates are collected in registry order, so ties on weight fall
        to the earliest-registered strategy (the brain first). All-flat →
        the brain's proposal stands unchanged (action 0, fraction 0).
        """
        candidates: list[tuple[str, TradeProposal]] = []
        candidate_actions: dict[str, int] = {_BRAIN_ID: int(brain_proposal.action)}
        if brain_proposal.action != 0:
            candidates.append((_BRAIN_ID, brain_proposal))
        for strategy_id, primitive in self.primitives:
            proposal, _ = primitive.propose(snapshot)
            candidate_actions[strategy_id] = int(proposal.action)
            if proposal.action != 0:
                candidates.append((strategy_id, proposal))

        weights = self.weights()
        if candidates:
            # max() keeps the FIRST maximal element → registry-order ties.
            chosen_id, chosen = max(candidates, key=lambda entry: weights[entry[0]])
            budget = float(self.budget_fractions.get(chosen_id, 1.0))
            chosen = replace(
                chosen,
                target_fraction=chosen.target_fraction * budget,
                strategy_id=chosen_id,
            )
        else:
            chosen_id = _BRAIN_ID
            budget = float(self.budget_fractions.get(_BRAIN_ID, 1.0))
            chosen = replace(brain_proposal, strategy_id=_BRAIN_ID)

        # Diagnostics stay anchored on the brain's dict so downstream
        # consumers (branch_confidences / soma_weights logging in
        # tick_processor) keep working even when a primitive wins.
        diagnostics = dict(brain_diagnostics)
        diagnostics["arbitration"] = {
            "chosen": chosen_id,
            "weights": weights,
            "candidate_actions": candidate_actions,
            "budget_fraction": budget,
        }
        return chosen, diagnostics
