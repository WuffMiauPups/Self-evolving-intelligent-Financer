from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from src.core.config import DecisionFusionConfig

logger = logging.getLogger(__name__)
from src.data.schemas import MarketBar
from src.memory.retrieval import AssociativeRetriever
from src.memory.schemas import MemoryDecisionSupport
from src.memory.updater import AssociativeMemoryUpdater
from src.models.state_encoder import EncodedState, ModelStateEncoder
from src.risk.constraints import TradeProposal
from src.strategies.decision_fusion import (
    DAMPEN,
    NEUTRAL,
    SUPPORT,
    VETO,
    FusionResult,
    compute_fusion,
)


@dataclass
class MemoryAwarePolicy:
    base_policy: object
    encoder: ModelStateEncoder
    retriever: AssociativeRetriever
    updater: AssociativeMemoryUpdater | None = None
    require_strong_match_for_trade: bool = True
    fusion_config: DecisionFusionConfig = field(default_factory=DecisionFusionConfig)
    retrieval_count: int = 0
    weak_match_count: int = 0
    no_trade_count: int = 0
    vetoed_count: int = 0
    dampened_count: int = 0
    supported_count: int = 0
    neutral_count: int = 0
    fusion_unknown_state_count: int = 0
    fusion_topology_warning_count: int = 0
    similarity_total: float = 0.0
    confidence_total: float = 0.0
    last_encoded_by_symbol: dict[str, EncodedState] = field(default_factory=dict)
    last_support_by_symbol: dict[str, MemoryDecisionSupport] = field(default_factory=dict)
    last_fusion_by_symbol: dict[str, FusionResult] = field(default_factory=dict)
    _logged_path_mode: bool = field(default=False, repr=False)

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        if hasattr(self.base_policy, "reset_symbol_state"):
            self.base_policy.reset_symbol_state(symbol, reason=reason)
        self.last_encoded_by_symbol.pop(symbol, None)
        self.last_support_by_symbol.pop(symbol, None)
        self.last_fusion_by_symbol.pop(symbol, None)
        return reason

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        if hasattr(self.base_policy, "reset_all_state"):
            self.base_policy.reset_all_state(reason=reason)
        self.last_encoded_by_symbol.clear()
        self.last_support_by_symbol.clear()
        self.last_fusion_by_symbol.clear()
        if self.updater is not None:
            self.updater.reset()
        self.retrieval_count = 0
        self.weak_match_count = 0
        self.no_trade_count = 0
        self.vetoed_count = 0
        self.dampened_count = 0
        self.supported_count = 0
        self.neutral_count = 0
        self.fusion_unknown_state_count = 0
        self.fusion_topology_warning_count = 0
        self.similarity_total = 0.0
        self.confidence_total = 0.0
        return reason

    def prepare_symbol_history(self, symbol: str, bars: list[MarketBar]) -> list[object]:
        if self.updater is None:
            return []
        return self.updater.finalize_ready(symbol, bars)

    def record_snapshot(self, symbol: str) -> None:
        if self.updater is None:
            return
        encoded_state = self.last_encoded_by_symbol.get(symbol)
        support = self.last_support_by_symbol.get(symbol)
        if encoded_state is None:
            return
        action_annotation = None if support is None else support.reason_code
        self.updater.observe(encoded_state, action_annotation=action_annotation)

    def propose(self, snapshot: object) -> tuple[TradeProposal, dict[str, object]]:
        proposal, diagnostics = self.base_policy.propose(snapshot)
        encoded_state = self.encoder.encode(snapshot)
        support = self.retriever.retrieve(encoded_state)
        self.last_encoded_by_symbol[snapshot.symbol] = encoded_state
        self.last_support_by_symbol[snapshot.symbol] = support
        self.retrieval_count += 1
        self.similarity_total += support.average_similarity
        self.confidence_total += support.retrieval_confidence
        if support.weak_match:
            self.weak_match_count += 1
        if support.no_trade_suggested:
            self.no_trade_count += 1

        # --- Decision fusion (if enabled) ---
        fusion: FusionResult | None = None
        if not self._logged_path_mode:
            if self.fusion_config.enabled:
                logger.info("memory_policy_mode mode=fusion")
            else:
                logger.info("memory_policy_mode mode=legacy fusion_disabled=true")
            self._logged_path_mode = True
        if self.fusion_config.enabled:
            fusion = compute_fusion(
                proposal=proposal,
                support=support,
                config=self.fusion_config,
                memory_units=self.retriever.bank.memory_units or None,
                esom_map=self.retriever.bank.esom_map,
            )
            self.last_fusion_by_symbol[snapshot.symbol] = fusion
            # Track fusion-level counters
            if fusion.unknown_state_flag:
                self.fusion_unknown_state_count += 1
            if fusion.topology_warning_flag:
                self.fusion_topology_warning_count += 1
            if fusion.support_level == SUPPORT:
                self.supported_count += 1
            elif fusion.support_level == NEUTRAL:
                self.neutral_count += 1

        adjusted, advisory = self._apply_support(proposal, support, fusion)
        # SAFETY CHECK: flat base action must never become a trade
        if proposal.action == 0 and adjusted.action != 0:
            raise RuntimeError(
                f"SAFETY VIOLATION: memory/fusion promoted flat base action to trade "
                f"(adjusted_action={adjusted.action}) — this must never happen"
            )
        if advisory["memory_vetoed"]:
            self.vetoed_count += 1
        if advisory["memory_dampened"]:
            self.dampened_count += 1
        diagnostics = dict(diagnostics)
        diagnostics["memory_support"] = {
            "reason_code": support.reason_code,
            "weak_match": support.weak_match,
            "no_trade_suggested": support.no_trade_suggested,
            "match_strength": support.match_strength,
            "calibrated_match_strength": support.calibrated_match_strength,
            "retrieval_confidence": support.retrieval_confidence,
            "average_similarity": support.average_similarity,
            "conflict_score": support.conflict_score,
            "context_norm": 0.0 if support.context_vector is None else float(support.context_vector.norm().item()),
            "directional_bias": support.directional_bias,
            "size_multiplier": support.size_multiplier,
            "top_match_count": len(support.top_matches),
            "cluster_match_count": len(support.cluster_matches),
            "path_distribution": support.path_distribution,
            "base_action": proposal.action,
            "adjusted_action": adjusted.action,
            "base_target_fraction": proposal.target_fraction,
            "adjusted_target_fraction": adjusted.target_fraction,
            "advisory_effect": advisory["advisory_effect"],
            "memory_vetoed": advisory["memory_vetoed"],
            "memory_dampened": advisory["memory_dampened"],
            "memory_alignment": advisory["memory_alignment"],
            "metadata": support.metadata,
        }
        if fusion is not None:
            diagnostics["decision_fusion"] = {
                "support_level": fusion.support_level,
                "reason_codes": fusion.reason_codes,
                "size_multiplier": fusion.size_multiplier,
                "confidence_adjustment": fusion.confidence_adjustment,
                "unknown_state_flag": fusion.unknown_state_flag,
                "topology_warning_flag": fusion.topology_warning_flag,
                "metadata": fusion.metadata,
            }
        return adjusted, diagnostics

    def metrics_snapshot(self) -> dict[str, float]:
        average_similarity = self.similarity_total / max(self.retrieval_count, 1)
        average_confidence = self.confidence_total / max(self.retrieval_count, 1)
        n = max(self.retrieval_count, 1)
        metrics: dict[str, float] = {
            "memory_retrieval_count": float(self.retrieval_count),
            "memory_average_similarity": float(average_similarity),
            "memory_average_confidence": float(average_confidence),
            "memory_weak_match_rate": float(self.weak_match_count / n),
            "memory_no_trade_rate": float(self.no_trade_count / n),
            "memory_veto_rate": float(self.vetoed_count / n),
            "memory_dampened_rate": float(self.dampened_count / n),
            "memory_supported_rate": float(self.supported_count / n),
            "memory_neutral_rate": float(self.neutral_count / n),
            "fusion_unknown_state_count": float(self.fusion_unknown_state_count),
            "fusion_topology_warning_count": float(self.fusion_topology_warning_count),
            "memory_cluster_count": float(len(self.retriever.bank.prototypes)),
            "memory_item_count": float(len(self.retriever.bank)),
            "memory_unit_count": float(len(self.retriever.bank.memory_units)),
            "memory_mature_unit_count": float(
                sum(1 for unit in self.retriever.bank.memory_units if unit.status == "mature")
            ),
            "memory_candidate_unit_count": float(
                sum(1 for unit in self.retriever.bank.memory_units if unit.status == "candidate")
            ),
            "memory_esom_node_count": float(0 if self.retriever.bank.esom_map is None else len(self.retriever.bank.esom_map.nodes)),
            "memory_quarantined_unit_count": float(
                sum(1 for unit in self.retriever.bank.memory_units if unit.status == "quarantined")
            ),
        }
        if self.updater is not None:
            summary = self.updater.summary()
            metrics.update(
                {
                    "memory_pending_count": float(summary["pending_count"]),
                    "memory_finalized_count": float(summary["finalized_count"]),
                }
            )
        return metrics

    def _apply_support(
        self,
        proposal: TradeProposal,
        support: MemoryDecisionSupport,
        fusion: FusionResult | None = None,
    ) -> tuple[TradeProposal, dict[str, object]]:
        # SAFETY INVARIANT: memory/fusion must never promote a flat base action to a trade.
        # This is checked both at entry and exit of this method.
        base_action = proposal.action
        base_target_fraction = proposal.target_fraction
        alignment = "flat"
        if base_action != 0:
            alignment = "aligned" if support.directional_bias in {0, base_action} else "conflicting"
        metadata = dict(proposal.metadata)
        metadata.update(
            {
                "memory_reason_code": support.reason_code,
                "memory_weak_match": float(support.weak_match),
                "memory_no_trade_suggested": float(support.no_trade_suggested),
                "memory_match_strength": support.match_strength,
                "memory_calibrated_match_strength": support.calibrated_match_strength,
                "memory_retrieval_confidence": support.retrieval_confidence,
                "memory_average_similarity": support.average_similarity,
                "memory_conflict_score": support.conflict_score,
                "memory_context_norm": 0.0 if support.context_vector is None else float(support.context_vector.norm().item()),
                "memory_size_multiplier": support.size_multiplier,
                "memory_directional_bias": float(support.directional_bias),
                "memory_base_action": float(base_action),
                "memory_base_target_fraction": float(base_target_fraction),
                "memory_alignment": alignment,
            }
        )
        if fusion is not None:
            metadata["fusion_support_level"] = fusion.support_level
            metadata["fusion_reason_codes"] = fusion.reason_codes

        if proposal.action == 0:
            metadata.update(
                {
                    "memory_adjusted_action": 0.0,
                    "memory_adjusted_target_fraction": 0.0,
                    "memory_vetoed": 0.0,
                    "memory_dampened": 0.0,
                    "memory_advisory_effect": "flat_base_policy",
                }
            )
            return replace(proposal, metadata=metadata), {
                "memory_vetoed": False,
                "memory_dampened": False,
                "advisory_effect": "flat_base_policy",
                "memory_alignment": alignment,
            }

        # -----------------------------------------------------------
        # Fusion-aware path (when fusion is enabled and available)
        # -----------------------------------------------------------
        if fusion is not None:
            if fusion.support_level == VETO and self.require_strong_match_for_trade:
                metadata.update(
                    {
                        "memory_adjusted_action": 0.0,
                        "memory_adjusted_target_fraction": 0.0,
                        "memory_vetoed": 1.0,
                        "memory_dampened": 0.0,
                        "memory_advisory_effect": "veto",
                    }
                )
                return replace(
                    proposal,
                    action=0,
                    target_fraction=0.0,
                    confidence=min(proposal.confidence, max(support.retrieval_confidence, 0.0) * fusion.confidence_adjustment),
                    metadata=metadata,
                ), {
                    "memory_vetoed": True,
                    "memory_dampened": False,
                    "advisory_effect": "veto",
                    "memory_alignment": alignment,
                }

            # Apply fusion multipliers
            adjusted_fraction = proposal.target_fraction * fusion.size_multiplier
            adjusted_confidence = min(
                proposal.confidence,
                max(support.retrieval_confidence, 0.0) * fusion.confidence_adjustment,
            )
            adjusted_action = proposal.action if abs(adjusted_fraction) > 1.0e-6 else 0
            adjusted_fraction = adjusted_fraction if adjusted_action != 0 else 0.0
            dampened = abs(adjusted_fraction) + 1.0e-9 < abs(base_target_fraction)

            if fusion.support_level == DAMPEN:
                advisory_effect = "dampened"
            elif fusion.support_level == SUPPORT and not dampened:
                advisory_effect = "supportive"
            elif dampened:
                advisory_effect = "dampened"
            else:
                advisory_effect = "neutral"

            metadata.update(
                {
                    "memory_adjusted_action": float(adjusted_action),
                    "memory_adjusted_target_fraction": float(adjusted_fraction),
                    "memory_vetoed": 0.0,
                    "memory_dampened": float(dampened),
                    "memory_advisory_effect": advisory_effect,
                }
            )
            return replace(
                proposal,
                action=adjusted_action,
                target_fraction=adjusted_fraction,
                confidence=adjusted_confidence,
                metadata=metadata,
            ), {
                "memory_vetoed": False,
                "memory_dampened": dampened,
                "advisory_effect": advisory_effect,
                "memory_alignment": alignment,
            }

        # -----------------------------------------------------------
        # Legacy path (fusion disabled — original behaviour preserved)
        # -----------------------------------------------------------
        if support.no_trade_suggested and self.require_strong_match_for_trade:
            metadata.update(
                {
                    "memory_adjusted_action": 0.0,
                    "memory_adjusted_target_fraction": 0.0,
                    "memory_vetoed": 1.0,
                    "memory_dampened": 0.0,
                    "memory_advisory_effect": "veto",
                }
            )
            return replace(
                proposal,
                action=0,
                target_fraction=0.0,
                confidence=min(proposal.confidence, max(support.retrieval_confidence, 0.0)),
                metadata=metadata,
            ), {
                "memory_vetoed": True,
                "memory_dampened": False,
                "advisory_effect": "veto",
                "memory_alignment": alignment,
            }
        aligned = support.directional_bias == 0 or support.directional_bias == proposal.action
        multiplier = support.size_multiplier if aligned else support.size_multiplier * 0.5
        adjusted_fraction = proposal.target_fraction * multiplier
        adjusted_action = proposal.action if abs(adjusted_fraction) > 1.0e-6 else 0
        adjusted_fraction = adjusted_fraction if adjusted_action != 0 else 0.0
        adjusted_confidence = min(proposal.confidence, max(support.retrieval_confidence, 0.0))
        dampened = abs(adjusted_fraction) + 1.0e-9 < abs(base_target_fraction)
        advisory_effect = "dampened" if dampened else "supportive"
        metadata.update(
            {
                "memory_adjusted_action": float(adjusted_action),
                "memory_adjusted_target_fraction": float(adjusted_fraction),
                "memory_vetoed": 0.0,
                "memory_dampened": float(dampened),
                "memory_advisory_effect": advisory_effect,
            }
        )
        return replace(
            proposal,
            action=adjusted_action,
            target_fraction=adjusted_fraction,
            confidence=adjusted_confidence,
            metadata=metadata,
        ), {
            "memory_vetoed": False,
            "memory_dampened": dampened,
            "advisory_effect": advisory_effect,
            "memory_alignment": alignment,
        }
