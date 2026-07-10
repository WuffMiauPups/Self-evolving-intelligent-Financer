"""Thin memory-aware policy wrapper.

After the Great Simplification, memory is PASSIVE:
- Encoded states are stored in the memory bank via the updater.
- The bank's ``query_top_k_simple`` kNN lookup is exposed in diagnostics only.
- Memory does NOT gate, veto, or adjust trade proposals. Fusion / market-priors /
  regime-hysteresis layers have been removed.

Platt calibration (if a fitted calibrator is supplied) is applied to the base
policy's confidence before returning.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime

from src.data.schemas import MarketBar
from src.memory.memory_bank import MemoryBank
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
        diagnostics = self._attach_memory_diag(diagnostics, encoded_state)
        if self._platt_ready():
            proposal = replace(proposal, confidence=float(self.platt_calibrator.calibrate(proposal.confidence)))
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
            diagnostics = self._attach_memory_diag(diagnostics, encoded_state)
            if self._platt_ready():
                proposal = replace(
                    proposal,
                    confidence=float(self.platt_calibrator.calibrate(proposal.confidence)),
                )
            results.append((proposal, diagnostics))
        return results

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

    def _attach_memory_diag(
        self, diagnostics: dict, encoded_state: EncodedState
    ) -> dict:
        diagnostics = dict(diagnostics)
        if len(self.bank) >= 1:
            try:
                matches = self.bank.query_top_k_simple(
                    encoded_state.latent, k=self.knn_top_k
                )
                diagnostics["memory_knn"] = {
                    "neighbor_count": len(matches),
                    "neighbor_symbols": [m.symbol for m in matches],
                    "neighbor_outcomes": [
                        float(m.outcome.primary_return)
                        for m in matches
                        if m.outcome is not None
                    ],
                }
            except Exception as exc:  # noqa: BLE001 - best effort
                diagnostics["memory_knn_error"] = str(exc)
        else:
            diagnostics["memory_knn"] = {"neighbor_count": 0}
        return diagnostics
