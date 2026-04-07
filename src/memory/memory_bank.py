from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

import torch

from src.core.config import MemoryConfig
from src.features.feature_pipeline import FeatureContract
from src.memory.clustering import build_prototypes
from src.memory.esom_map import ESOMMap
from src.memory.schemas import (
    MemoryHealthReport,
    MemoryItem,
    MemoryUnit,
    PrototypeCluster,
    RetrievedClusterMatch,
    RetrievedStateMatch,
    RetrievalCalibrationProfile,
    UnitStatus,
)
from src.memory.similarity import pairwise_similarity

logger = logging.getLogger(__name__)


def feature_contract_fingerprint(contract: FeatureContract) -> str:
    payload = json.dumps(contract.to_dict(), sort_keys=True)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def compute_encoder_fingerprint(checkpoint_meta: dict[str, Any]) -> str:
    """Compute a fingerprint that uniquely identifies the encoder weights.

    This combines the checkpoint version, model config, and any training
    epoch info to detect when memory items were encoded by a different model.
    """
    relevant_keys = {}
    for key in ("checkpoint_version", "model_kind", "model_config", "epoch", "training_loss"):
        if key in checkpoint_meta:
            relevant_keys[key] = checkpoint_meta[key]
    payload = json.dumps(relevant_keys, sort_keys=True, default=str)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemoryBank:
    feature_contract: FeatureContract
    latent_dim: int
    similarity_metric: str
    max_items: int
    feature_contract_id: str | None = None
    encoder_fingerprint: str | None = None
    require_encoder_match: bool = False
    items: list[MemoryItem] = field(default_factory=list)
    prototypes: list[PrototypeCluster] = field(default_factory=list)
    memory_units: list[MemoryUnit] = field(default_factory=list)
    archived_units: list[MemoryUnit] = field(default_factory=list)
    calibration_profile: RetrievalCalibrationProfile | None = None
    calibration_profiles_by_symbol: dict[str, RetrievalCalibrationProfile] = field(default_factory=dict)
    calibration_profiles_by_regime: dict[str, RetrievalCalibrationProfile] = field(default_factory=dict)
    esom_map: ESOMMap | None = None
    latent_matrix: torch.Tensor = field(default_factory=lambda: torch.empty((0, 0), dtype=torch.float32))
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.feature_contract_id is None:
            self.feature_contract_id = feature_contract_fingerprint(self.feature_contract)
        if not self.items:
            self.latent_matrix = torch.empty((0, self.latent_dim), dtype=torch.float32)
            return
        self._rebuild_latent_matrix()

    def __len__(self) -> int:
        return len(self.items)

    def append(self, item: MemoryItem) -> None:
        self._validate_item(item)
        if len(self.items) >= self.max_items:
            self.items.pop(0)
        self.items.append(item)
        self._rebuild_latent_matrix()

    def extend(self, items: list[MemoryItem]) -> None:
        """Append many items and rebuild the latent matrix *once* at the end.

        The naive ``for item in items: self.append(item)`` is O(N²) because
        ``append`` triggers ``_rebuild_latent_matrix`` per call, and each
        rebuild stacks every item from scratch. This version validates each
        incoming item, appends them in place, enforces ``max_items``, and
        then rebuilds the latent matrix exactly once. On an offline build
        that produces thousands of items in a few big chunks, this collapses
        the per-append cost from O(N) to O(1) amortized. Post-condition is
        identical to the per-item ``append`` loop.
        """
        if not items:
            return
        for item in items:
            self._validate_item(item)
            self.items.append(item)
        overflow = len(self.items) - self.max_items
        if overflow > 0:
            del self.items[:overflow]
        self._rebuild_latent_matrix()

    def rebuild_prototypes(
        self,
        *,
        min_similarity: float,
        min_cluster_size: int,
        temperature: float,
    ) -> list[PrototypeCluster]:
        if not self.items:
            self.prototypes = []
            return self.prototypes
        self.prototypes = build_prototypes(
            self.items,
            self.latent_matrix,
            similarity_metric=self.similarity_metric,
            min_similarity=min_similarity,
            min_cluster_size=min_cluster_size,
            temperature=temperature,
        )
        for item in self.items:
            item.cluster_id = None
        for prototype in self.prototypes:
            for member_index in prototype.member_indices:
                self.items[member_index].cluster_id = prototype.cluster_id
        return self.prototypes

    def refresh_organization(self, config: MemoryConfig, *, dry_run: bool = False) -> dict[str, Any]:
        from src.memory.calibration import build_retrieval_calibration_profiles
        from src.memory.dynamic_units import build_dynamic_units
        from src.memory.esom_map import build_esom_map
        from src.memory.maintenance import maintain_memory_units
        from src.memory.metrics import compute_memory_health

        report: dict[str, Any] = {}
        resolved_units = list(self.memory_units)
        archived_units = list(self.archived_units)
        created_units = 0
        updated_units = 0
        if config.dynamic_growth_enabled:
            resolved_units, dynamic_report = build_dynamic_units(self.items, self.latent_matrix, config)
            created_units = dynamic_report.created_units
            updated_units = dynamic_report.updated_units
            report.update(
                {
                    "created_units": dynamic_report.created_units,
                    "updated_units": dynamic_report.updated_units,
                    "promoted_units": dynamic_report.promoted_units,
                    "novelty_threshold": dynamic_report.novelty_threshold,
                    "candidate_units": dynamic_report.candidate_units,
                    "mature_units": dynamic_report.mature_units,
                    "quarantined_units": dynamic_report.quarantined_units,
                    "conflict_rejections": dynamic_report.conflict_rejections,
                    "saturation_attenuated_updates": dynamic_report.saturation_attenuated_updates,
                }
            )
            resolved_units, maintenance_report = maintain_memory_units(
                resolved_units,
                self.items,
                config,
                latest_step=max(len(self.items) - 1, 0),
                dry_run=dry_run,
            )
            archived_units = archived_units + maintenance_report.archived_units
            report.update(
                {
                    "merged_units": maintenance_report.merged_units,
                    "split_units": maintenance_report.split_units,
                    "pruned_units": maintenance_report.pruned_units,
                    "archived_units": len(maintenance_report.archived_units),
                }
            )
        # --- Topology growth (conservative, optional) ---
        if config.topology_growth.enabled and not dry_run:
            from src.memory.topology_growth import evaluate_topology_growth

            resolved_units, growth_report = evaluate_topology_growth(
                resolved_units,
                self.items,
                config,
                esom_map=self.esom_map,
                current_step=max(len(self.items) - 1, 0),
            )
            report["topology_growth"] = growth_report.to_dict()
        calibration_profile = self.calibration_profile
        calibration_profiles_by_symbol = dict(self.calibration_profiles_by_symbol)
        calibration_profiles_by_regime = dict(self.calibration_profiles_by_regime)
        if config.calibration_enabled and config.match_calibration_enabled:
            calibration_profile, calibration_profiles_by_symbol, calibration_profiles_by_regime = build_retrieval_calibration_profiles(
                self.items,
                self.latent_matrix,
                config,
            )
            report["calibration_buckets"] = 0 if calibration_profile is None else len(calibration_profile.buckets)
            report["symbol_calibration_profiles"] = len(calibration_profiles_by_symbol)
            report["regime_calibration_profiles"] = len(calibration_profiles_by_regime)
        esom_map = self.esom_map
        topology_density = 0.0
        graph_edge_count = 0
        average_edge_age = 0.0
        current_step = max(len(self.items) - 1, 0)
        if config.use_esom_map:
            esom_map = build_esom_map(resolved_units, config, current_step=current_step)
            if esom_map is not None and esom_map.nodes:
                topology_density = sum(node.local_density for node in esom_map.nodes) / len(esom_map.nodes)
                graph_edge_count = len(esom_map.edges)
                average_edge_age = sum(e.age for e in esom_map.edges) / max(len(esom_map.edges), 1)
            report["esom_nodes"] = 0 if esom_map is None else len(esom_map.nodes)
            report["esom_edges"] = graph_edge_count
        health = compute_memory_health(
            resolved_units,
            archived_count=len(archived_units),
            latest_step=current_step,
            topology_density=topology_density,
            graph_edge_count=graph_edge_count,
            average_edge_age=average_edge_age,
            created_units=created_units,
            updated_units=updated_units,
        )
        report["health"] = health
        if not dry_run:
            self.memory_units = resolved_units
            self.archived_units = archived_units
            self.calibration_profile = calibration_profile
            self.calibration_profiles_by_symbol = calibration_profiles_by_symbol
            self.calibration_profiles_by_regime = calibration_profiles_by_regime
            self.esom_map = esom_map
            self.meta["organization_report"] = {
                key: (value if key != "health" else value.__dict__)
                for key, value in report.items()
            }
            self.meta["attention_temperature"] = config.attention_temperature
        return report

    def query_top_k(
        self,
        query_latent: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
    ) -> list[RetrievedStateMatch]:
        if not self.items:
            return []
        raw_scores, normalized_scores = pairwise_similarity(
            query_latent,
            self.latent_matrix,
            metric=self.similarity_metric,
            temperature=temperature,
        )
        resolved_top_k = min(top_k, len(self.items))
        top_values, top_indices = torch.topk(normalized_scores, k=resolved_top_k)
        matches: list[RetrievedStateMatch] = []
        for rank, item_index in enumerate(top_indices.tolist()):
            item = self.items[item_index]
            matches.append(
                RetrievedStateMatch(
                    item_index=item_index,
                    item_id=item.item_id,
                    symbol=item.symbol,
                    timestamp=item.timestamp,
                    similarity_raw=float(raw_scores[item_index].item()),
                    similarity=float(top_values[rank].item()),
                    cluster_id=item.cluster_id,
                    outcome=item.outcome,
                    risk=item.risk,
                )
            )
        return matches

    def query_top_clusters(
        self,
        query_latent: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
    ) -> list[RetrievedClusterMatch]:
        if self.memory_units:
            candidates = [u for u in self.memory_units if u.status != UnitStatus.QUARANTINED.value]
            if not candidates:
                return []
            centroids = torch.stack([unit.prototype for unit in candidates], dim=0)
        elif self.prototypes:
            candidates = self.prototypes
            centroids = torch.stack([prototype.centroid for prototype in candidates], dim=0)
        else:
            return []
        raw_scores, normalized_scores = pairwise_similarity(
            query_latent,
            centroids,
            metric=self.similarity_metric,
            temperature=temperature,
        )
        resolved_top_k = min(top_k, len(candidates))
        top_values, top_indices = torch.topk(normalized_scores, k=resolved_top_k)
        matches: list[RetrievedClusterMatch] = []
        for rank, cluster_index in enumerate(top_indices.tolist()):
            candidate = candidates[cluster_index]
            if isinstance(candidate, MemoryUnit):
                matches.append(
                    RetrievedClusterMatch(
                        cluster_id=candidate.unit_id,
                        similarity_raw=float(raw_scores[cluster_index].item()),
                        similarity=float(top_values[rank].item()),
                        member_count=candidate.support_count,
                        path_stats=candidate.path_stats,
                        risk_stats=candidate.risk_stats,
                        regime_composition=candidate.regime_composition,
                    )
                )
            else:
                matches.append(
                    RetrievedClusterMatch(
                        cluster_id=candidate.cluster_id,
                        similarity_raw=float(raw_scores[cluster_index].item()),
                        similarity=float(top_values[rank].item()),
                        member_count=candidate.member_count,
                        path_stats=candidate.path_stats,
                        risk_stats=candidate.risk_stats,
                        regime_composition=candidate.regime_composition,
                    )
                )
        return matches

    def summary(self) -> dict[str, Any]:
        return {
            "item_count": len(self.items),
            "latent_dim": self.latent_dim,
            "cluster_count": len(self.prototypes),
            "unit_count": len(self.memory_units),
            "mature_unit_count": sum(1 for unit in self.memory_units if unit.status == "mature"),
            "candidate_unit_count": sum(1 for unit in self.memory_units if unit.status == "candidate"),
            "quarantined_unit_count": sum(1 for unit in self.memory_units if unit.status == "quarantined"),
            "archived_unit_count": len(self.archived_units),
            "calibration_bucket_count": 0 if self.calibration_profile is None else len(self.calibration_profile.buckets),
            "symbol_calibration_profile_count": len(self.calibration_profiles_by_symbol),
            "regime_calibration_profile_count": len(self.calibration_profiles_by_regime),
            "esom_node_count": 0 if self.esom_map is None else len(self.esom_map.nodes),
            "esom_edge_count": 0 if self.esom_map is None else len(self.esom_map.edges),
            "feature_contract_id": self.feature_contract_id,
            "encoder_fingerprint": self.encoder_fingerprint,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            "feature_contract": self.feature_contract.to_dict(),
            "latent_dim": self.latent_dim,
            "similarity_metric": self.similarity_metric,
            "max_items": self.max_items,
            "feature_contract_id": self.feature_contract_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "require_encoder_match": self.require_encoder_match,
            "items": [item.to_payload() for item in self.items],
            "prototypes": [prototype.to_payload() for prototype in self.prototypes],
            "memory_units": [unit.to_payload() for unit in self.memory_units],
            "archived_units": [unit.to_payload() for unit in self.archived_units],
            "calibration_profile": None if self.calibration_profile is None else self.calibration_profile.to_payload(),
            "calibration_profiles_by_symbol": {
                symbol: profile.to_payload()
                for symbol, profile in self.calibration_profiles_by_symbol.items()
            },
            "calibration_profiles_by_regime": {
                regime: profile.to_payload()
                for regime, profile in self.calibration_profiles_by_regime.items()
            },
            "esom_map": None if self.esom_map is None else self.esom_map.to_payload(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryBank":
        bank = cls(
            feature_contract=FeatureContract.from_dict(payload["feature_contract"]),
            latent_dim=int(payload["latent_dim"]),
            similarity_metric=str(payload["similarity_metric"]),
            max_items=int(payload["max_items"]),
            feature_contract_id=str(payload["feature_contract_id"]),
            encoder_fingerprint=payload.get("encoder_fingerprint"),
            require_encoder_match=bool(payload.get("require_encoder_match", False)),
            items=[MemoryItem.from_payload(item_payload) for item_payload in payload.get("items", [])],
            prototypes=[
                PrototypeCluster.from_payload(cluster_payload) for cluster_payload in payload.get("prototypes", [])
            ],
            memory_units=[MemoryUnit.from_payload(unit_payload) for unit_payload in payload.get("memory_units", [])],
            archived_units=[MemoryUnit.from_payload(unit_payload) for unit_payload in payload.get("archived_units", [])],
            calibration_profile=None
            if payload.get("calibration_profile") is None
            else RetrievalCalibrationProfile.from_payload(payload["calibration_profile"]),
            calibration_profiles_by_symbol={
                str(symbol): RetrievalCalibrationProfile.from_payload(profile_payload)
                for symbol, profile_payload in payload.get("calibration_profiles_by_symbol", {}).items()
            },
            calibration_profiles_by_regime={
                str(regime): RetrievalCalibrationProfile.from_payload(profile_payload)
                for regime, profile_payload in payload.get("calibration_profiles_by_regime", {}).items()
            },
            esom_map=None if payload.get("esom_map") is None else ESOMMap.from_payload(payload["esom_map"]),
            meta=dict(payload.get("meta", {})),
        )
        bank._rebuild_latent_matrix()
        return bank

    def check_encoder_compatibility(self, encoder_fingerprint: str | None) -> bool:
        """Check if a given encoder fingerprint is compatible with this bank.

        Returns True if compatible, False if mismatched.
        Logs a warning on mismatch; raises ValueError if require_encoder_match is True.
        """
        if self.encoder_fingerprint is None or encoder_fingerprint is None:
            return True
        if self.encoder_fingerprint == encoder_fingerprint:
            return True
        message = (
            f"Encoder fingerprint mismatch: bank={self.encoder_fingerprint}, "
            f"item={encoder_fingerprint}. Memory items may be in an incompatible latent space."
        )
        if self.require_encoder_match:
            raise ValueError(message)
        logger.warning(message)
        return False

    def _validate_item(self, item: MemoryItem) -> None:
        if item.feature_contract_id != self.feature_contract_id:
            raise ValueError(
                f"Memory item feature contract mismatch: {item.feature_contract_id} vs {self.feature_contract_id}"
            )
        if item.encoder_fingerprint is not None:
            self.check_encoder_compatibility(item.encoder_fingerprint)
        latent = item.latent.detach().cpu().to(dtype=torch.float32)
        if latent.dim() != 1 or latent.shape[0] != self.latent_dim:
            raise ValueError(f"Expected latent shape ({self.latent_dim},), got {tuple(latent.shape)}")
        item.latent = latent
        if item.regime_weights is not None:
            item.regime_weights = item.regime_weights.detach().cpu().to(dtype=torch.float32)

    def _rebuild_latent_matrix(self) -> None:
        if not self.items:
            self.latent_matrix = torch.empty((0, self.latent_dim), dtype=torch.float32)
            return
        self.latent_matrix = torch.stack(
            [item.latent.detach().cpu().to(dtype=torch.float32) for item in self.items],
            dim=0,
        )
