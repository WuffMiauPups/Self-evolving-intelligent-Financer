"""Thin memory-aware policy wrapper.

Memory's influence is bounded by design (MASTER_PLAN §6.2):
- Encoded states are stored in the memory bank via the updater.
- Retrieval always honors the temporal cutoff and never returns the
  query's own stored state.
- With ``influence_mode="damping"``, kNN outcome statistics apply a size
  multiplier in [damping_min_multiplier, 1.0] to the proposal's target
  fraction: memory can only SHRINK a trade when similar past states lost
  money in the proposal's direction. It can never grow a trade, flip or
  veto an action, or touch confidence. Default mode is "off"
  (diagnostics only) until the counterfactual credit gate is positive.

Platt calibration (if a fitted calibrator is supplied) is applied to the base
policy's confidence before returning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime

from src.data.schemas import MarketBar
from src.memory.memory_bank import MemoryBank
from src.memory.schemas import MemoryItem
from src.memory.updater import AssociativeMemoryUpdater
from src.models.state_encoder import EncodedState, ModelStateEncoder
from src.risk.constraints import TradeProposal
from src.strategies.base import BatchTradingPolicy, TradingPolicy

logger = logging.getLogger(__name__)


@dataclass
class MemoryAwarePolicy:
    """Wrap a base policy; store encoded states; optional Platt calibration."""

    base_policy: object
    encoder: ModelStateEncoder
    bank: MemoryBank
    updater: AssociativeMemoryUpdater | None = None
    platt_calibrator: object | None = None  # PlattCalibrator
    knn_top_k: int = 5
    # Bounded influence (§6.2). Values mirror MemoryConfig.
    influence_mode: str = "off"
    damping_min_multiplier: float = 0.5
    damping_min_neighbors: int = 5
    damping_full_at_return: float = 0.002

    last_encoded_by_symbol: dict[str, EncodedState] = field(default_factory=dict)
    _temporal_cutoff: datetime | None = field(default=None, repr=False)

    def set_temporal_cutoff(self, cutoff: datetime | None) -> None:
        self._temporal_cutoff = cutoff

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        if isinstance(self.base_policy, TradingPolicy):
            self.base_policy.reset_symbol_state(symbol, reason=reason)
        self.last_encoded_by_symbol.pop(symbol, None)
        return reason

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        if isinstance(self.base_policy, TradingPolicy):
            self.base_policy.reset_all_state(reason=reason)
        self.last_encoded_by_symbol.clear()
        if self.updater is not None:
            self.updater.reset()
        return reason

    def prepare_symbol_history(self, symbol: str, bars: list[MarketBar]) -> list[object]:
        if self.updater is None:
            return []
        return self.updater.finalize_ready(symbol, bars)

    def record_snapshot(self, symbol: str, *, action_annotation: str | None = None) -> None:
        """Queue the last encoded state for outcome finalization.

        ``action_annotation`` records the EXECUTED (risk-clamped) action,
        not the model's raw intent — outcome-driven learning must credit
        what actually happened (MASTER_PLAN §4.5.2, the shield
        principle).  Callers pass a compact string like
        ``"action=1 approved=1 side=buy qty=12.5 resized=0"``.
        """
        if self.updater is None:
            return
        encoded_state = self.last_encoded_by_symbol.get(symbol)
        if encoded_state is None:
            return
        self.updater.observe(encoded_state, action_annotation=action_annotation)

    def propose(self, snapshot: object) -> tuple[TradeProposal, dict[str, object]]:
        proposal, diagnostics = self.base_policy.propose(snapshot)
        encoded_state = self.encoder.encode(snapshot)
        self.last_encoded_by_symbol[snapshot.symbol] = encoded_state
        proposal, diagnostics = self._finish_proposal(proposal, diagnostics, encoded_state)
        return proposal, diagnostics

    def propose_batch(self, snapshots: list) -> list[tuple[TradeProposal, dict[str, object]]]:
        if not snapshots:
            return []
        if not isinstance(self.base_policy, BatchTradingPolicy):
            return [self.propose(snapshot) for snapshot in snapshots]

        base_results = self.base_policy.propose_batch(snapshots)
        encoded_states = self.encoder.encode_batch(snapshots)

        results: list[tuple[TradeProposal, dict[str, object]]] = []
        for snapshot, (proposal, diagnostics), encoded_state in zip(
            snapshots, base_results, encoded_states
        ):
            self.last_encoded_by_symbol[snapshot.symbol] = encoded_state
            proposal, diagnostics = self._finish_proposal(proposal, diagnostics, encoded_state)
            results.append((proposal, diagnostics))
        return results

    def _finish_proposal(
        self,
        proposal: TradeProposal,
        diagnostics: dict,
        encoded_state: EncodedState,
    ) -> tuple[TradeProposal, dict[str, object]]:
        """Shared post-processing: one retrieval feeds diagnostics + damping."""
        neighbors = self._safe_retrieve(encoded_state)
        diagnostics = self._attach_memory_diag(diagnostics, neighbors)
        proposal, damping_diag = self._apply_damping(proposal, neighbors)
        diagnostics["memory_damping"] = damping_diag
        if self._platt_ready():
            proposal = replace(
                proposal,
                confidence=float(self.platt_calibrator.calibrate(proposal.confidence)),
            )
        return proposal, diagnostics

    def _apply_damping(
        self,
        proposal: TradeProposal,
        neighbors: list[MemoryItem],
    ) -> tuple[TradeProposal, dict[str, object]]:
        """Damping-only size prior (§6.2): multiplier ∈ [min_multiplier, 1.0].

        The action-aligned mean of neighbor outcomes decides the multiplier:
        similar past states that LOST money in the proposal's direction
        shrink the trade linearly, saturating at the floor when the mean
        adverse return reaches ``damping_full_at_return``. A favorable mean
        is clamped to 1.0 — memory never grows a position. Flat proposals
        and thin evidence (< min_neighbors outcomes) pass through untouched.
        """
        diag: dict[str, object] = {"applied": False, "multiplier": 1.0}
        if self.influence_mode != "damping" or proposal.action == 0:
            return proposal, diag
        aligned_returns = [
            float(proposal.action) * float(item.outcome.primary_return)
            for item in neighbors
            if item.outcome is not None
        ]
        diag["aligned_outcome_count"] = len(aligned_returns)
        if len(aligned_returns) < self.damping_min_neighbors:
            diag["reason"] = "insufficient_neighbors"
            return proposal, diag
        mean_aligned = sum(aligned_returns) / len(aligned_returns)
        diag["aligned_mean_return"] = mean_aligned
        if mean_aligned >= 0.0:
            return proposal, diag
        span = 1.0 - self.damping_min_multiplier
        multiplier = 1.0 - (abs(mean_aligned) / self.damping_full_at_return) * span
        multiplier = min(1.0, max(self.damping_min_multiplier, multiplier))
        diag.update({"applied": True, "multiplier": multiplier})
        return (
            replace(proposal, target_fraction=proposal.target_fraction * multiplier),
            diag,
        )

    def metrics_snapshot(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "memory_item_count": float(len(self.bank)),
        }
        if self.updater is not None:
            summary = self.updater.summary()
            metrics["memory_pending_count"] = float(summary.get("pending_count", 0))
            metrics["memory_finalized_count"] = float(summary.get("finalized_count", 0))
        return metrics

    # ---- Internal helpers ------------------------------------------------

    def _platt_ready(self) -> bool:
        calibrator = self.platt_calibrator
        return calibrator is not None and getattr(calibrator, "is_fitted", False)

    def _retrieve_neighbors(self, encoded_state: EncodedState) -> list:
        """Temporal-cutoff-respecting, self-excluding kNN retrieval.

        Every retrieval goes through here so the backtest engine's
        ``set_temporal_cutoff`` is actually honored: items at or after the
        cutoff (i.e. the future, when replaying history over a bank built
        from the full range) never surface, and the query's own stored
        (symbol, timestamp) copy is never its own neighbor.
        """
        return self.bank.query_top_k_simple(
            encoded_state.latent,
            k=self.knn_top_k,
            exclude_after=self._temporal_cutoff,
            exclude_item=(encoded_state.symbol, encoded_state.timestamp),
        )

    def _safe_retrieve(self, encoded_state: EncodedState) -> list[MemoryItem]:
        if len(self.bank) < 1:
            return []
        try:
            return self._retrieve_neighbors(encoded_state)
        except Exception as exc:  # noqa: BLE001 - retrieval must not block trading
            logger.warning("memory_retrieval_failed symbol=%s error=%s", encoded_state.symbol, exc)
            return []

    def _attach_memory_diag(self, diagnostics: dict, matches: list[MemoryItem]) -> dict:
        diagnostics = dict(diagnostics)
        diagnostics["memory_knn"] = {
            "neighbor_count": len(matches),
            "neighbor_symbols": [m.symbol for m in matches],
            "neighbor_outcomes": [
                float(m.outcome.primary_return)
                for m in matches
                if m.outcome is not None
            ],
        }
        return diagnostics
