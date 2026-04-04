from __future__ import annotations

from dataclasses import dataclass

import torch

from src.core.config import MemoryConfig
from src.memory.calibration import calibrate_match_strength, dominant_regime_key
from src.memory.hopfield_retrieval import hopfield_retrieve
from src.memory.memory_bank import MemoryBank
from src.memory.path_stats import aggregate_path_statistics, aggregate_risk_statistics
from src.memory.schemas import MemoryDecisionSupport, RetrievedClusterMatch, RetrievedStateMatch
from src.models.state_encoder import EncodedState


@dataclass(frozen=True)
class RetrievalReason:
    EMPTY_BANK: str = "empty_memory_bank"
    INSUFFICIENT_SIMILARITY: str = "insufficient_similarity"
    INSUFFICIENT_SUPPORT: str = "insufficient_support"
    CONFLICTING_OUTCOMES: str = "conflicting_outcomes"
    CALIBRATED_WEAK_MATCH: str = "calibrated_weak_match"
    STRONG_MATCH: str = "strong_match"


class AssociativeRetriever:
    def __init__(self, bank: MemoryBank, config: MemoryConfig) -> None:
        self.bank = bank
        self.config = config

    def retrieve(self, encoded_state: EncodedState) -> MemoryDecisionSupport:
        if len(self.bank) == 0:
            return MemoryDecisionSupport(
                top_matches=[],
                cluster_matches=[],
                match_strength=0.0,
                average_similarity=0.0,
                weak_match=True,
                no_trade_suggested=self.config.allow_unknown_state,
                reason_code=RetrievalReason.EMPTY_BANK,
                directional_bias=0,
                size_multiplier=0.0,
                path_distribution={},
                risk_summary=None,
                path_summary=None,
                calibrated_match_strength=0.0,
                retrieval_confidence=0.0,
                conflict_score=0.0,
                metadata={"symbol": encoded_state.symbol},
            )

        top_matches = self.bank.query_top_k(
            encoded_state.latent,
            top_k=self.config.top_k,
            temperature=self.config.attention_temperature,
        )
        top_clusters = self.bank.query_top_clusters(
            encoded_state.latent,
            top_k=max(1, min(self.config.top_k, 3)),
            temperature=self.config.attention_temperature,
        )
        hopfield_result = None
        esom_neighbor_ids: list[str] = []
        if self.config.use_hopfield_retrieval and self.bank.memory_units:
            unit_keys = torch.stack([unit.prototype for unit in self.bank.memory_units], dim=0)
            conflict_values = torch.tensor(
                [
                    unit.path_stats.horizon_return_mean[max(unit.path_stats.horizon_return_mean)]
                    for unit in self.bank.memory_units
                ],
                dtype=torch.float32,
            )
            hopfield_result = hopfield_retrieve(
                query=encoded_state.latent,
                keys=unit_keys,
                values=unit_keys,
                top_k=max(1, min(self.config.top_k, len(self.bank.memory_units))),
                metric=self.config.similarity_metric,
                temperature=self.config.attention_temperature,
                min_similarity=self.config.min_similarity,
                conflict_values=conflict_values,
            )
            for unit_index in hopfield_result.top_indices:
                self.bank.memory_units[unit_index].retrieval_hits += 1
            if self.config.use_esom_map and self.bank.esom_map is not None and hopfield_result.top_indices:
                top_unit_id = self.bank.memory_units[hopfield_result.top_indices[0]].unit_id
                esom_neighbor_ids = self.bank.esom_map.neighbor_ids_for(top_unit_id)
        strong_matches = [match for match in top_matches if match.similarity >= self.config.min_similarity]
        valid_clusters = [
            match for match in top_clusters
            if match.similarity >= self.config.min_similarity and match.member_count >= self.config.min_cluster_size
        ]
        average_similarity = sum(match.similarity for match in top_matches) / max(len(top_matches), 1)
        best_similarity = top_matches[0].similarity if top_matches else 0.0
        calibration_profile, calibration_scope = self._select_calibration_profile(encoded_state)
        calibrated_match_strength, retrieval_confidence, calibration_conflict = calibrate_match_strength(
            calibration_profile if self.config.calibration_enabled and self.config.match_calibration_enabled else None,
            best_similarity,
        )
        if hopfield_result is not None:
            retrieval_confidence = max(retrieval_confidence, hopfield_result.average_similarity * (1.0 - hopfield_result.entropy / max(self.config.top_k, 1)))
            calibration_conflict = min(max(calibration_conflict + hopfield_result.conflict_score, 0.0), 1.0)
        if not strong_matches:
            return self._weak_support(
                top_matches=top_matches,
                top_clusters=top_clusters,
                average_similarity=average_similarity,
                match_strength=best_similarity,
                calibrated_match_strength=calibrated_match_strength,
                retrieval_confidence=retrieval_confidence,
                conflict_score=calibration_conflict,
                reason_code=RetrievalReason.INSUFFICIENT_SIMILARITY,
                context_vector=None if hopfield_result is None else hopfield_result.context_vector,
                metadata={
                    "hopfield_entropy": None if hopfield_result is None else hopfield_result.entropy,
                    "hopfield_top_indices": [] if hopfield_result is None else hopfield_result.top_indices,
                    "esom_neighbor_ids": esom_neighbor_ids,
                    "calibration_scope": calibration_scope,
                },
            )

        if len(strong_matches) < self.config.min_effective_matches:
            return self._weak_support(
                top_matches=top_matches,
                top_clusters=top_clusters,
                average_similarity=average_similarity,
                match_strength=best_similarity,
                calibrated_match_strength=calibrated_match_strength,
                retrieval_confidence=retrieval_confidence,
                conflict_score=calibration_conflict,
                reason_code=RetrievalReason.INSUFFICIENT_SUPPORT,
                context_vector=None if hopfield_result is None else hopfield_result.context_vector,
                metadata={
                    "hopfield_entropy": None if hopfield_result is None else hopfield_result.entropy,
                    "hopfield_top_indices": [] if hopfield_result is None else hopfield_result.top_indices,
                    "esom_neighbor_ids": esom_neighbor_ids,
                    "calibration_scope": calibration_scope,
                },
            )

        weights = torch.tensor([match.similarity for match in strong_matches], dtype=torch.float32)
        primary_returns = torch.tensor([match.outcome.primary_return for match in strong_matches], dtype=torch.float32)
        weight_sum = float(weights.sum().item()) or 1.0
        weighted_return = float((weights * primary_returns).sum().item() / weight_sum)
        coherence = float((weights * torch.sign(primary_returns)).sum().abs().item() / weight_sum)
        outcome_conflict = 1.0 - coherence
        conflict_score = min(
            max(calibration_conflict + (outcome_conflict * self.config.match_conflict_penalty), 0.0),
            1.0,
        )
        if coherence < 0.35:
            return self._weak_support(
                top_matches=top_matches,
                top_clusters=top_clusters,
                average_similarity=average_similarity,
                match_strength=best_similarity,
                calibrated_match_strength=calibrated_match_strength,
                retrieval_confidence=retrieval_confidence,
                conflict_score=conflict_score,
                reason_code=RetrievalReason.CONFLICTING_OUTCOMES,
                context_vector=None if hopfield_result is None else hopfield_result.context_vector,
                metadata={
                    "hopfield_entropy": None if hopfield_result is None else hopfield_result.entropy,
                    "hopfield_top_indices": [] if hopfield_result is None else hopfield_result.top_indices,
                    "esom_neighbor_ids": esom_neighbor_ids,
                    "calibration_scope": calibration_scope,
                },
            )

        if (
            self.config.require_calibrated_strong_match_for_trade_support
            and calibration_profile is not None
            and calibrated_match_strength < max(
                calibration_profile.recommended_strong_match_threshold,
                self.config.min_calibrated_match,
            )
        ):
            return self._weak_support(
                top_matches=top_matches,
                top_clusters=top_clusters,
                average_similarity=average_similarity,
                match_strength=best_similarity,
                calibrated_match_strength=calibrated_match_strength,
                retrieval_confidence=retrieval_confidence,
                conflict_score=conflict_score,
                reason_code=RetrievalReason.CALIBRATED_WEAK_MATCH,
                context_vector=None if hopfield_result is None else hopfield_result.context_vector,
                metadata={
                    "hopfield_entropy": None if hopfield_result is None else hopfield_result.entropy,
                    "hopfield_top_indices": [] if hopfield_result is None else hopfield_result.top_indices,
                    "esom_neighbor_ids": esom_neighbor_ids,
                    "calibration_scope": calibration_scope,
                },
            )

        flat_threshold = float(self.config.path_label_rules.get("flat_return_threshold", 0.00075))
        directional_bias = 0
        if weighted_return >= flat_threshold:
            directional_bias = 1
        elif weighted_return <= -flat_threshold:
            directional_bias = -1
        selected_outcomes = [match.outcome for match in strong_matches]
        path_summary = aggregate_path_statistics(selected_outcomes)
        risk_summary = aggregate_risk_statistics(selected_outcomes)
        size_multiplier = min(
            max((calibrated_match_strength + average_similarity) * 0.5 * max(1.0 - conflict_score, 0.1), 0.0),
            1.0,
        )
        return MemoryDecisionSupport(
            top_matches=top_matches,
            cluster_matches=valid_clusters if valid_clusters else top_clusters,
            match_strength=best_similarity,
            average_similarity=average_similarity,
            weak_match=False,
            no_trade_suggested=False,
            reason_code=RetrievalReason.STRONG_MATCH,
            directional_bias=directional_bias,
            size_multiplier=size_multiplier,
            path_distribution=path_summary.path_label_frequencies,
            risk_summary=risk_summary,
            path_summary=path_summary,
            calibrated_match_strength=calibrated_match_strength,
            retrieval_confidence=retrieval_confidence,
            conflict_score=conflict_score,
            context_vector=None if hopfield_result is None else hopfield_result.context_vector,
            metadata={
                "strong_match_count": len(strong_matches),
                "valid_cluster_count": len(valid_clusters),
                "weighted_primary_return": weighted_return,
                "coherence": coherence,
                "hopfield_entropy": None if hopfield_result is None else hopfield_result.entropy,
                "hopfield_top_indices": [] if hopfield_result is None else hopfield_result.top_indices,
                "esom_neighbor_ids": esom_neighbor_ids,
                "calibration_scope": calibration_scope,
            },
        )

    def _select_calibration_profile(self, encoded_state: EncodedState):
        if not (self.config.calibration_enabled and self.config.match_calibration_enabled):
            return None, "disabled"
        regime_key = dominant_regime_key(encoded_state.regime_weights)
        if self.config.symbol_specific_profiles:
            symbol_profile = self.bank.calibration_profiles_by_symbol.get(encoded_state.symbol)
            if symbol_profile is not None:
                return symbol_profile, f"symbol:{encoded_state.symbol}"
        if self.config.regime_specific_profiles and regime_key is not None:
            regime_profile = self.bank.calibration_profiles_by_regime.get(regime_key)
            if regime_profile is not None:
                return regime_profile, f"regime:{regime_key}"
        return self.bank.calibration_profile, "global"

    def _weak_support(
        self,
        *,
        top_matches: list[RetrievedStateMatch],
        top_clusters: list[RetrievedClusterMatch],
        average_similarity: float,
        match_strength: float,
        calibrated_match_strength: float,
        retrieval_confidence: float,
        conflict_score: float,
        reason_code: str,
        context_vector: torch.Tensor | None,
        metadata: dict[str, object] | None = None,
    ) -> MemoryDecisionSupport:
        size_multiplier = (
            0.0
            if self.config.require_strong_match_for_memory_support
            else min(calibrated_match_strength, 0.25)
        )
        return MemoryDecisionSupport(
            top_matches=top_matches,
            cluster_matches=top_clusters,
            match_strength=match_strength,
            average_similarity=average_similarity,
            weak_match=True,
            no_trade_suggested=self.config.allow_unknown_state,
            reason_code=reason_code,
            directional_bias=0,
            size_multiplier=size_multiplier,
            path_distribution={},
            risk_summary=None,
            path_summary=None,
            calibrated_match_strength=calibrated_match_strength,
            retrieval_confidence=retrieval_confidence,
            conflict_score=conflict_score,
            context_vector=context_vector,
            metadata={
                "strong_match_required": self.config.require_strong_match_for_memory_support,
                **(metadata or {}),
            },
        )
