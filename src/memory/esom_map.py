from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.core.config import MemoryConfig
from src.memory.schemas import MemoryUnit
from src.memory.similarity import pairwise_similarity


@dataclass
class GraphEdge:
    """GNG-inspired edge between two memory units in the topology graph."""
    source_id: str
    target_id: str
    similarity: float
    age: int = 0
    creation_step: int = 0
    last_activated_step: int = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "similarity": self.similarity,
            "age": self.age,
            "creation_step": self.creation_step,
            "last_activated_step": self.last_activated_step,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GraphEdge":
        return cls(
            source_id=str(payload["source_id"]),
            target_id=str(payload["target_id"]),
            similarity=float(payload["similarity"]),
            age=int(payload.get("age", 0)),
            creation_step=int(payload.get("creation_step", 0)),
            last_activated_step=int(payload.get("last_activated_step", 0)),
        )


@dataclass
class ESOMNode:
    unit_id: str
    status: str
    support_count: int
    neighbor_ids: list[str]
    neighbor_similarities: list[float]
    local_density: float
    growth_score: float
    coordinate: tuple[float, float]
    local_error: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "status": self.status,
            "support_count": self.support_count,
            "neighbor_ids": list(self.neighbor_ids),
            "neighbor_similarities": list(self.neighbor_similarities),
            "local_density": self.local_density,
            "growth_score": self.growth_score,
            "coordinate": list(self.coordinate),
            "local_error": self.local_error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ESOMNode":
        return cls(
            unit_id=str(payload["unit_id"]),
            status=str(payload["status"]),
            support_count=int(payload["support_count"]),
            neighbor_ids=[str(value) for value in payload["neighbor_ids"]],
            neighbor_similarities=[float(value) for value in payload["neighbor_similarities"]],
            local_density=float(payload["local_density"]),
            growth_score=float(payload["growth_score"]),
            coordinate=(float(payload["coordinate"][0]), float(payload["coordinate"][1])),
            local_error=float(payload.get("local_error", 0.0)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass
class ESOMMap:
    nodes: list[ESOMNode]
    edges: list[GraphEdge]
    neighbor_k: int
    metric: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "neighbor_k": self.neighbor_k,
            "metric": self.metric,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ESOMMap":
        return cls(
            nodes=[ESOMNode.from_payload(node) for node in payload.get("nodes", [])],
            edges=[GraphEdge.from_payload(edge) for edge in payload.get("edges", [])],
            neighbor_k=int(payload["neighbor_k"]),
            metric=str(payload["metric"]),
        )

    def neighbor_ids_for(self, unit_id: str) -> list[str]:
        for node in self.nodes:
            if node.unit_id == unit_id:
                return list(node.neighbor_ids)
        return []

    def edges_for(self, unit_id: str) -> list[GraphEdge]:
        return [edge for edge in self.edges if edge.source_id == unit_id or edge.target_id == unit_id]

    def edge_between(self, unit_a: str, unit_b: str) -> GraphEdge | None:
        for edge in self.edges:
            if (edge.source_id == unit_a and edge.target_id == unit_b) or (
                edge.source_id == unit_b and edge.target_id == unit_a
            ):
                return edge
        return None

    @property
    def graph_statistics(self) -> dict[str, float]:
        if not self.nodes:
            return {"node_count": 0, "edge_count": 0, "avg_density": 0.0, "avg_edge_age": 0.0}
        avg_density = sum(node.local_density for node in self.nodes) / len(self.nodes)
        avg_edge_age = sum(edge.age for edge in self.edges) / max(len(self.edges), 1)
        avg_local_error = sum(node.local_error for node in self.nodes) / len(self.nodes)
        return {
            "node_count": float(len(self.nodes)),
            "edge_count": float(len(self.edges)),
            "avg_density": avg_density,
            "avg_edge_age": avg_edge_age,
            "avg_local_error": avg_local_error,
        }

    def age_edges(self, *, max_age: int = 100) -> list[GraphEdge]:
        """Age all edges by 1, return and remove those exceeding max_age."""
        expired: list[GraphEdge] = []
        kept: list[GraphEdge] = []
        for edge in self.edges:
            edge.age += 1
            if edge.age > max_age:
                expired.append(edge)
            else:
                kept.append(edge)
        self.edges = kept
        return expired

    def activate_edge(self, unit_a: str, unit_b: str, *, similarity: float, step: int) -> GraphEdge:
        """Create or refresh an edge between two units (GNG-style competitive Hebbian)."""
        existing = self.edge_between(unit_a, unit_b)
        if existing is not None:
            existing.age = 0
            existing.similarity = similarity
            existing.last_activated_step = step
            return existing
        new_edge = GraphEdge(
            source_id=unit_a,
            target_id=unit_b,
            similarity=similarity,
            age=0,
            creation_step=step,
            last_activated_step=step,
        )
        self.edges.append(new_edge)
        return new_edge


def build_esom_map(units: list[MemoryUnit], config: MemoryConfig, *, current_step: int = 0) -> ESOMMap | None:
    if not units:
        return None
    prototypes = torch.stack([unit.prototype.detach().cpu().to(dtype=torch.float32) for unit in units], dim=0)
    _, similarities = pairwise_similarity(
        prototypes,
        prototypes,
        metric=config.similarity_metric,
        temperature=config.attention_temperature,
    )
    if similarities.dim() == 1:
        similarities = similarities.unsqueeze(0)
    nodes: list[ESOMNode] = []
    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str]] = set()
    neighbor_k = min(config.topology_neighbor_k, max(len(units) - 1, 1))
    for index, unit in enumerate(units):
        row = similarities[index].clone()
        row[index] = -1.0
        top_values, top_indices = torch.topk(row, k=neighbor_k)
        neighbor_ids = [units[neighbor_index].unit_id for neighbor_index in top_indices.tolist()]
        neighbor_similarities = [float(value.item()) for value in top_values]
        local_density = sum(max(value, 0.0) for value in neighbor_similarities) / max(len(neighbor_similarities), 1)
        growth_score = max(
            0.0,
            (1.0 - local_density) * (1.0 - min(max(unit.maturity_score, 0.0), 1.0) + unit.outcome_dispersion),
        )
        # Use saturation to modulate growth: high saturation -> lower growth score
        saturation_modulator = max(0.0, 1.0 - unit.saturation_score)
        growth_score *= saturation_modulator
        prototype = unit.prototype.detach().cpu().reshape(-1)
        coordinate = (
            float(prototype[0].item()) if prototype.numel() > 0 else 0.0,
            float(prototype[1].item()) if prototype.numel() > 1 else 0.0,
        )
        nodes.append(
            ESOMNode(
                unit_id=unit.unit_id,
                status=unit.status,
                support_count=unit.support_count,
                neighbor_ids=neighbor_ids,
                neighbor_similarities=neighbor_similarities,
                local_density=local_density,
                growth_score=growth_score,
                coordinate=coordinate,
                local_error=unit.local_error_accumulator,
                metadata={
                    "utility_score": unit.utility_score,
                    "outcome_dispersion": unit.outcome_dispersion,
                    "saturation_score": unit.saturation_score,
                    "plasticity_factor": unit.plasticity_factor,
                },
            )
        )
        # Build explicit edges for GNG topology
        for neighbor_rank, neighbor_idx in enumerate(top_indices.tolist()):
            neighbor_unit = units[neighbor_idx]
            edge_key = tuple(sorted([unit.unit_id, neighbor_unit.unit_id]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                edges.append(
                    GraphEdge(
                        source_id=unit.unit_id,
                        target_id=neighbor_unit.unit_id,
                        similarity=float(top_values[neighbor_rank].item()),
                        age=0,
                        creation_step=current_step,
                        last_activated_step=current_step,
                    )
                )
    return ESOMMap(nodes=nodes, edges=edges, neighbor_k=neighbor_k, metric=config.similarity_metric)
