"""Minimal memory bank (post-Great-Simplification).

Stores encoded states + outcomes, supports append/extend, cosine-kNN lookup.
ESOM / clustering / dynamic units / calibration / maintenance / fusion are gone.

Callers get: bank.append(item), len(bank), bank.items, bank.latent_matrix,
bank.query_top_k_simple(query_latent, k), bank.summary(), bank.meta.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

import torch

from src.features.feature_pipeline import FeatureContract
from src.memory.schemas import MemoryItem

logger = logging.getLogger(__name__)


def feature_contract_fingerprint(contract: FeatureContract) -> str:
    payload = json.dumps(contract.to_dict(), sort_keys=True)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def compute_encoder_fingerprint(checkpoint_meta: dict[str, Any]) -> str:
    """Fingerprint the encoder weights so stale banks can be detected."""
    relevant_keys: dict[str, Any] = {}
    for key in ("checkpoint_version", "model_kind", "model_config", "epoch", "training_loss"):
        if key in checkpoint_meta:
            relevant_keys[key] = checkpoint_meta[key]
    payload = json.dumps(relevant_keys, sort_keys=True, default=str)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemoryBank:
    feature_contract: FeatureContract
    latent_dim: int
    similarity_metric: str = "cosine"
    max_items: int = 5000
    feature_contract_id: str | None = None
    encoder_fingerprint: str | None = None
    require_encoder_match: bool = False
    items: list[MemoryItem] = field(default_factory=list)
    latent_matrix: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float32)
    )
    meta: dict[str, Any] = field(default_factory=dict)
    _latent_matrix_normalized: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.feature_contract_id is None:
            self.feature_contract_id = feature_contract_fingerprint(self.feature_contract)
        if not self.items:
            self.latent_matrix = torch.empty((0, self.latent_dim), dtype=torch.float32)
            return
        self._rebuild_latent_matrix()

    def __len__(self) -> int:
        return len(self.items)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, item: MemoryItem) -> bool:
        """Append a single item. Returns True on accept, False on soft reject."""
        if not self._validate_item(item):
            return False
        if len(self.items) >= self.max_items:
            self.items.pop(0)
            if self.latent_matrix.shape[0] > 0:
                self.latent_matrix = self.latent_matrix[1:].contiguous()
        self.items.append(item)
        row = item.latent.detach().cpu().to(dtype=torch.float32).reshape(1, -1)
        if self.latent_matrix.shape[0] == 0 or self.latent_matrix.shape[1] != self.latent_dim:
            self.latent_matrix = row
        else:
            self.latent_matrix = torch.cat([self.latent_matrix, row], dim=0)
        self._latent_matrix_normalized = None
        return True

    def extend(self, items: list[MemoryItem]) -> int:
        """Append multiple items. Returns count of items actually added."""
        added = 0
        for item in items:
            if self.append(item):
                added += 1
        return added

    def query_top_k_simple(
        self,
        query_latent: torch.Tensor,
        k: int = 5,
    ) -> list[MemoryItem]:
        """Cosine-kNN lookup over stored items.

        Returns the top-k MemoryItems whose normalized latents are most similar
        to the query. Passive scaffold: diagnostic only, does not influence trade
        decisions (wiring lives in MemoryAwarePolicy).
        """
        if not self.items or self.latent_matrix.shape[0] == 0:
            return []
        keys = self._ensure_latent_matrix_normalized()
        query = torch.nn.functional.normalize(
            query_latent.detach().cpu().to(dtype=torch.float32).reshape(1, -1),
            dim=-1,
        )
        sims = (query @ keys.T).squeeze(0)
        top = min(int(k), sims.shape[0])
        if top <= 0:
            return []
        _, indices = torch.topk(sims, k=top, dim=-1)
        return [self.items[int(i)] for i in indices.tolist()]

    def summary(self) -> dict[str, Any]:
        return {
            "item_count": len(self.items),
            "latent_dim": self.latent_dim,
            "max_items": self.max_items,
            "similarity_metric": self.similarity_metric,
            "feature_contract_id": self.feature_contract_id,
            "encoder_fingerprint": self.encoder_fingerprint,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    # Bumped from any pre-Great-Simplification value; old payloads with
    # prototypes/units/esom_map fields are silently ignored on load
    # (only the keys we read here matter).
    PAYLOAD_VERSION = 2

    def to_payload(self) -> dict[str, Any]:
        """Serialize to a dict suitable for ``torch.save``.

        ``latent_matrix`` and ``_latent_matrix_normalized`` are not stored:
        they are recomputed from ``items`` on load via ``_rebuild_latent_matrix``.
        """
        return {
            "version": self.PAYLOAD_VERSION,
            "feature_contract": self.feature_contract.to_dict(),
            "latent_dim": self.latent_dim,
            "similarity_metric": self.similarity_metric,
            "max_items": self.max_items,
            "feature_contract_id": self.feature_contract_id,
            "encoder_fingerprint": self.encoder_fingerprint,
            "require_encoder_match": self.require_encoder_match,
            "meta": dict(self.meta),
            "items": [item.to_payload() for item in self.items],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryBank":
        """Reconstruct a bank from a ``torch.save`` payload.

        Tolerant of legacy payloads: extra keys (prototypes, units,
        esom_map, calibration, …) are simply ignored — the minimal
        bank only knows about items.
        """
        contract = FeatureContract.from_dict(payload["feature_contract"])
        items = [MemoryItem.from_payload(p) for p in payload.get("items", [])]
        bank = cls(
            feature_contract=contract,
            latent_dim=int(payload["latent_dim"]),
            similarity_metric=str(payload.get("similarity_metric", "cosine")),
            max_items=int(payload.get("max_items", 5000)),
            feature_contract_id=payload.get("feature_contract_id"),
            encoder_fingerprint=payload.get("encoder_fingerprint"),
            require_encoder_match=bool(payload.get("require_encoder_match", False)),
            items=items,
            meta=dict(payload.get("meta", {})),
        )
        bank._rebuild_latent_matrix()
        return bank

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_item(self, item: MemoryItem) -> bool:
        if item.latent.numel() == 0:
            return False
        if item.latent.shape[-1] != self.latent_dim:
            logger.warning(
                "memory_item_rejected reason=latent_dim_mismatch expected=%d got=%d",
                self.latent_dim,
                item.latent.shape[-1],
            )
            return False
        if (
            self.encoder_fingerprint
            and item.encoder_fingerprint
            and self.encoder_fingerprint != item.encoder_fingerprint
        ):
            # Always skip-on-mismatch to avoid silently contaminating the bank
            # with items from a different encoder. ``require_encoder_match``
            # only controls whether we also hard-raise in strict pipelines.
            logger.warning(
                "Encoder fingerprint mismatch: bank=%s item=%s; skipping item",
                self.encoder_fingerprint,
                item.encoder_fingerprint,
            )
            return False
        return True

    def _rebuild_latent_matrix(self) -> None:
        if not self.items:
            self.latent_matrix = torch.empty((0, self.latent_dim), dtype=torch.float32)
            self._latent_matrix_normalized = None
            return
        rows = [
            item.latent.detach().cpu().to(dtype=torch.float32).reshape(-1)
            for item in self.items
        ]
        self.latent_matrix = torch.stack(rows, dim=0)
        self._latent_matrix_normalized = None

    def _ensure_latent_matrix_normalized(self) -> torch.Tensor:
        if self._latent_matrix_normalized is not None and (
            self._latent_matrix_normalized.shape == self.latent_matrix.shape
        ):
            return self._latent_matrix_normalized
        if self.similarity_metric == "dot":
            self._latent_matrix_normalized = self.latent_matrix.clone()
        else:
            self._latent_matrix_normalized = torch.nn.functional.normalize(
                self.latent_matrix, dim=-1
            )
        return self._latent_matrix_normalized
