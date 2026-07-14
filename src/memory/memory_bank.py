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
from datetime import datetime
from hashlib import sha1
from typing import Any

import torch

from src.features.feature_pipeline import FeatureContract
from src.memory.schemas import MemoryItem

logger = logging.getLogger(__name__)


def feature_contract_fingerprint(contract: FeatureContract) -> str:
    payload = json.dumps(contract.to_dict(), sort_keys=True)
    return sha1(payload.encode("utf-8")).hexdigest()[:16]


def _item_epoch(item: MemoryItem) -> int:
    """Epoch seconds of an item's timestamp (0 for items without one)."""
    timestamp = item.timestamp
    if timestamp is None:
        return 0
    return int(timestamp.timestamp())


def compute_encoder_fingerprint(checkpoint_meta: dict[str, Any]) -> str:
    """Fingerprint the encoder so stale banks/calibrators can be detected.

    2026-07-14 collision fix: the fingerprint previously reduced to
    (version, kind, model_config) — the "epoch"/"training_loss" keys it
    hoped to hash never existed in meta (the save path writes
    "train_loss"), and neither the feature contract nor the weights
    entered the hash. Two same-config trainings therefore collided even
    across DIFFERENT feature contracts (observed: the ⟲2 44-dim and ⟲3
    52-dim candidates shared 4575582c9cd97795), silently defeating both
    the Platt calibrator binding and memory.require_encoder_match.
    "feature_contract" and "weights_digest" (sha1 of the state dict,
    stamped by build_checkpoint_meta) now anchor the identity. Old
    checkpoints keep their stored meta["encoder_fingerprint"] — this
    function only runs at save time or as a legacy fallback.
    """
    relevant_keys: dict[str, Any] = {}
    for key in (
        "checkpoint_version",
        "model_kind",
        "model_config",
        "feature_contract",
        "weights_digest",
        "epoch",
        "training_loss",
    ):
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
    # Epoch-second timestamps parallel to ``latent_matrix`` rows, so temporal
    # filtering in retrieval is a vectorized mask instead of a Python loop.
    _timestamps_epoch: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.int64), repr=False
    )

    def __post_init__(self) -> None:
        if self.feature_contract_id is None:
            self.feature_contract_id = feature_contract_fingerprint(self.feature_contract)
        if not self.items:
            self.latent_matrix = torch.empty((0, self.latent_dim), dtype=torch.float32)
            self._timestamps_epoch = torch.empty((0,), dtype=torch.int64)
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
            if self._timestamps_epoch.shape[0] > 0:
                self._timestamps_epoch = self._timestamps_epoch[1:].contiguous()
        self.items.append(item)
        row = item.latent.detach().cpu().to(dtype=torch.float32).reshape(1, -1)
        ts_row = torch.tensor([_item_epoch(item)], dtype=torch.int64)
        if self.latent_matrix.shape[0] == 0 or self.latent_matrix.shape[1] != self.latent_dim:
            self.latent_matrix = row
            self._timestamps_epoch = ts_row
        else:
            self.latent_matrix = torch.cat([self.latent_matrix, row], dim=0)
            self._timestamps_epoch = torch.cat([self._timestamps_epoch, ts_row], dim=0)
        self._latent_matrix_normalized = None
        return True

    def extend(self, items: list[MemoryItem]) -> int:
        """Append multiple items in one bulk operation.

        Validates each item, stacks all accepted latents once, and does a
        single ``torch.cat`` + a single FIFO trim — O(N) per call. (The old
        implementation looped over ``append``, whose per-item full-matrix
        ``torch.cat`` made large builds O(N²): ~200 GB of cumulative memcpy
        at 20k items.) Returns the count of items actually added.
        """
        accepted = [item for item in items if self._validate_item(item)]
        if not accepted:
            return 0
        rows = torch.stack(
            [item.latent.detach().cpu().to(dtype=torch.float32).reshape(-1) for item in accepted],
            dim=0,
        )
        ts_rows = torch.tensor([_item_epoch(item) for item in accepted], dtype=torch.int64)
        if self.latent_matrix.shape[0] == 0 or self.latent_matrix.shape[1] != self.latent_dim:
            self.latent_matrix = rows
            self._timestamps_epoch = ts_rows
            self.items = list(accepted)
        else:
            self.latent_matrix = torch.cat([self.latent_matrix, rows], dim=0)
            self._timestamps_epoch = torch.cat([self._timestamps_epoch, ts_rows], dim=0)
            self.items.extend(accepted)
        overflow = len(self.items) - self.max_items
        if overflow > 0:
            self.items = self.items[overflow:]
            self.latent_matrix = self.latent_matrix[overflow:].contiguous()
            self._timestamps_epoch = self._timestamps_epoch[overflow:].contiguous()
        self._latent_matrix_normalized = None
        return len(accepted)

    def query_top_k_simple(
        self,
        query_latent: torch.Tensor,
        k: int = 5,
        *,
        exclude_after: datetime | None = None,
        exclude_item: tuple[str, datetime] | None = None,
    ) -> list[MemoryItem]:
        """Cosine-kNN lookup over stored items.

        ``exclude_after`` drops every item whose timestamp is at or after the
        cutoff — this is the temporal guarantee the backtest engine relies on
        (a bank built over the full history must not leak future outcomes
        into a decision at bar t). ``exclude_item`` additionally drops the
        query's own (symbol, timestamp) so a stored copy of the current state
        can never be its own nearest neighbor.
        """
        if not self.items or self.latent_matrix.shape[0] == 0:
            return []
        keys = self._ensure_latent_matrix_normalized()
        query = torch.nn.functional.normalize(
            query_latent.detach().cpu().to(dtype=torch.float32).reshape(1, -1),
            dim=-1,
        )
        sims = (query @ keys.T).squeeze(0)
        if exclude_after is not None:
            cutoff_epoch = int(exclude_after.timestamp())
            sims = sims.masked_fill(self._timestamps_epoch >= cutoff_epoch, float("-inf"))
        if exclude_item is not None:
            symbol, timestamp = exclude_item
            ts_epoch = int(timestamp.timestamp())
            for index in (self._timestamps_epoch == ts_epoch).nonzero(as_tuple=True)[0].tolist():
                if self.items[index].symbol == symbol:
                    sims[index] = float("-inf")
        top = min(int(k), sims.shape[0])
        if top <= 0:
            return []
        values, indices = torch.topk(sims, k=top, dim=-1)
        return [
            self.items[int(i)]
            for value, i in zip(values.tolist(), indices.tolist())
            if value != float("-inf")
        ]

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
            self._timestamps_epoch = torch.empty((0,), dtype=torch.int64)
            self._latent_matrix_normalized = None
            return
        rows = [
            item.latent.detach().cpu().to(dtype=torch.float32).reshape(-1)
            for item in self.items
        ]
        self.latent_matrix = torch.stack(rows, dim=0)
        self._timestamps_epoch = torch.tensor(
            [_item_epoch(item) for item in self.items], dtype=torch.int64
        )
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
