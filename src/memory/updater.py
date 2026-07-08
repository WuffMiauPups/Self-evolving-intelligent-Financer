from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.core.config import MemoryConfig
from src.data.schemas import MarketBar, bars_to_tensor
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.outcomes import build_risk_summary, compute_outcome_summary
from src.memory.schemas import MemoryItem, PendingMemoryState
from src.models.state_encoder import EncodedState, ModelStateEncoder

logger = logging.getLogger(__name__)


@dataclass
class AssociativeMemoryUpdater:
    """Observe encoded states, finalize outcomes, append to bank.

    Post-Great-Simplification this is a passive scaffold: there is no
    prototype clustering, no ESOM map, no dynamic-unit maintenance and
    therefore no background refresher. ``finalize_ready`` just appends
    the new items to the bank and returns them.
    """
    bank: MemoryBank
    config: MemoryConfig
    pending_by_symbol: dict[str, list[PendingMemoryState]] = field(default_factory=lambda: defaultdict(list))
    observed_count: int = 0
    finalized_count: int = 0

    def observe(self, encoded_state: EncodedState, *, action_annotation: str | None = None) -> None:
        if self.config.runtime_update_mode == "disabled":
            return
        pending_items = self.pending_by_symbol[encoded_state.symbol]
        if any(item.timestamp == encoded_state.timestamp for item in pending_items):
            return
        pending_items.append(
            PendingMemoryState(
                symbol=encoded_state.symbol,
                timestamp=encoded_state.timestamp,
                feature_contract_id=encoded_state.feature_contract_id,
                latent=encoded_state.latent.detach().cpu(),
                regime_weights=None
                if encoded_state.regime_weights is None
                else encoded_state.regime_weights.detach().cpu(),
                branch_confidences=dict(encoded_state.branch_confidences),
                branch_norms=dict(encoded_state.branch_norms),
                action_annotation=action_annotation,
                encoder_fingerprint=encoded_state.encoder_fingerprint,
                metadata=dict(encoded_state.metadata),
            )
        )
        self.observed_count += 1

    def finalize_ready(self, symbol: str, bars: Sequence[MarketBar]) -> list[MemoryItem]:
        if self.config.runtime_update_mode != "append_when_ready":
            return []
        pending_items = self.pending_by_symbol.get(symbol, [])
        if not pending_items:
            return []
        timestamps_to_index = {bar.timestamp: index for index, bar in enumerate(bars)}
        max_horizon = max(self.config.forward_horizons)
        finalized: list[MemoryItem] = []
        remaining: list[PendingMemoryState] = []
        for pending in pending_items:
            index = timestamps_to_index.get(pending.timestamp)
            if index is None or index + max_horizon >= len(bars):
                remaining.append(pending)
                continue
            outcome = compute_outcome_summary(
                bars,
                index,
                horizons=self.config.forward_horizons,
                path_label_rules=self.config.path_label_rules,
            )
            item = MemoryItem(
                item_id=f"{pending.symbol}-{int(pending.timestamp.timestamp())}",
                symbol=pending.symbol,
                timestamp=pending.timestamp,
                latent=pending.latent.detach().cpu(),
                feature_contract_id=pending.feature_contract_id,
                regime_weights=None if pending.regime_weights is None else pending.regime_weights.detach().cpu(),
                branch_confidences=dict(pending.branch_confidences),
                branch_norms=dict(pending.branch_norms),
                outcome=outcome,
                risk=build_risk_summary(outcome),
                action_annotation=pending.action_annotation,
                encoder_fingerprint=pending.encoder_fingerprint,
                metadata=dict(pending.metadata),
            )
            self.bank.append(item)
            finalized.append(item)
            self.finalized_count += 1
        self.pending_by_symbol[symbol] = remaining
        # Post-Great-Simplification: appends are already done by
        # ``self.bank.append(item)`` above. No prototype rebuild, no
        # ESOM refresh, no background worker — the minimal bank has
        # no organization layer to refresh.
        return finalized

    def reset(self) -> None:
        self.pending_by_symbol.clear()
        self.observed_count = 0
        self.finalized_count = 0

    def summary(self) -> dict[str, int]:
        return {
            "observed_count": self.observed_count,
            "finalized_count": self.finalized_count,
            "pending_count": sum(len(items) for items in self.pending_by_symbol.values()),
        }

    def save_pending(self, path: str | Path) -> int:
        """Persist pending observations to disk for restart safety.

        Returns the number of pending items saved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        all_pending: list[dict] = []
        for symbol, items in self.pending_by_symbol.items():
            for item in items:
                all_pending.append({
                    "symbol": item.symbol,
                    "timestamp": item.timestamp,
                    "feature_contract_id": item.feature_contract_id,
                    "latent": item.latent.detach().cpu(),
                    "regime_weights": None if item.regime_weights is None else item.regime_weights.detach().cpu(),
                    "branch_confidences": dict(item.branch_confidences),
                    "branch_norms": dict(item.branch_norms),
                    "action_annotation": item.action_annotation,
                    "encoder_fingerprint": item.encoder_fingerprint,
                    "metadata": dict(item.metadata),
                })
        payload = {
            "pending_items": all_pending,
            "observed_count": self.observed_count,
            "finalized_count": self.finalized_count,
        }
        torch.save(payload, path)
        logger.info("pending_observations_saved path=%s count=%d", str(path), len(all_pending))
        return len(all_pending)

    def load_pending(self, path: str | Path) -> int:
        """Restore pending observations from disk.

        Returns the number of pending items restored.
        """
        path = Path(path)
        if not path.exists():
            return 0
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                logger.warning("cache_load_legacy_format path=%s", str(path))
                payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning("pending_observations_load_failed path=%s error=%s", str(path), str(exc))
            return 0
        restored = 0
        for item_data in payload.get("pending_items", []):
            pending = PendingMemoryState(
                symbol=str(item_data["symbol"]),
                timestamp=item_data["timestamp"],
                feature_contract_id=str(item_data["feature_contract_id"]),
                latent=item_data["latent"].detach().cpu(),
                regime_weights=None if item_data["regime_weights"] is None else item_data["regime_weights"].detach().cpu(),
                branch_confidences=dict(item_data["branch_confidences"]),
                branch_norms=dict(item_data["branch_norms"]),
                action_annotation=item_data.get("action_annotation"),
                encoder_fingerprint=item_data.get("encoder_fingerprint"),
                metadata=dict(item_data.get("metadata", {})),
            )
            # Avoid duplicates
            existing = self.pending_by_symbol[pending.symbol]
            if not any(p.timestamp == pending.timestamp for p in existing):
                existing.append(pending)
                restored += 1
        self.observed_count += payload.get("observed_count", 0)
        self.finalized_count += payload.get("finalized_count", 0)
        logger.info("pending_observations_restored path=%s count=%d", str(path), restored)
        return restored


# Chunk size for the batched encoder forward. 2048 is a safe default for
# both CPU and GPU — brain-model per-sample memory is tiny (short sequences,
# small branch dims), so peak RAM stays well under a GB even at this size.
# Larger chunks give strictly better throughput on a GPU (up to VRAM limit)
# and diminishing-but-still-positive returns on CPU. Override with
# SETAI_MEMORY_BUILD_ENCODE_CHUNK=N to push it higher on a strong GPU,
# or lower it to 256/512 on low-RAM systems.
_DEFAULT_ENCODE_CHUNK = 2048


def build_memory_bank_from_bars(
    *,
    bars_by_symbol: dict[str, Sequence[MarketBar]],
    pipeline: FeaturePipeline,
    encoder: ModelStateEncoder,
    memory_config: MemoryConfig,
    warmup_bars: int,
    encode_chunk_size: int | None = None,
    embargo_bars: int = 0,
) -> MemoryBank:
    """Build an associative memory bank by replaying historical bars.

    Hot-path notes:
      - ``pipeline.build`` is called per-sample (unchanged — it reads a
        rolling window and is cheap relative to the model forward).
      - ``encoder.encode`` is the expensive step: a full brain forward pass.
        We collect snapshots in chunks of ``encode_chunk_size`` and hand them
        to ``encoder.encode_batch`` so the model runs one large matmul per
        chunk instead of thousands of single-sample forwards. On multi-year
        1-minute bar data this collapses the encoder phase from per-sample
        O(N) model invocations to O(N / chunk) — typically a 10–50× speedup.
      - Everything else (outcome summary, per-symbol iteration, append order)
        is unchanged, so the produced bank is identical to the pre-batch
        version (modulo floating-point associativity in the encoder forward).
    """
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol cannot be empty")
    sample_symbol = next(iter(bars_by_symbol))
    sample_bars = list(bars_by_symbol[sample_symbol])
    if len(sample_bars) <= warmup_bars + max(memory_config.forward_horizons):
        raise ValueError("Not enough bars to build associative memory")

    if encode_chunk_size is None:
        import os as _os
        try:
            encode_chunk_size = int(_os.environ.get("SETAI_MEMORY_BUILD_ENCODE_CHUNK", _DEFAULT_ENCODE_CHUNK))
        except ValueError:
            encode_chunk_size = _DEFAULT_ENCODE_CHUNK
        encode_chunk_size = max(1, encode_chunk_size)
    logger.info("memory_build_encode_chunk_size=%d", encode_chunk_size)

    # First-pass probe to discover the latent dim (no allocation of the full
    # bank until we know this). Single-sample call is fine here — it only
    # runs once.
    sample_snapshot = pipeline.build(sample_bars[: warmup_bars + 1], symbol=sample_symbol)
    sample_encoded = encoder.encode(sample_snapshot)
    # Stamp the bank with the encoder's fingerprint at construction so that
    # check_encoder_compatibility has a real reference for every appended
    # item.  Without this stamp, banks built offline carry no fingerprint and
    # any subsequent require_encoder_match check is silently bypassed.
    bank = MemoryBank(
        feature_contract=sample_snapshot.contract,
        latent_dim=sample_encoded.latent.shape[0],
        similarity_metric=memory_config.similarity_metric,
        max_items=memory_config.max_items,
        encoder_fingerprint=encoder.encoder_fingerprint,
        require_encoder_match=memory_config.require_encoder_match,
        meta={
            "builder": "offline",
            "encoder_fingerprint": encoder.encoder_fingerprint,
        },
    )

    # Per-symbol stratified cap so every symbol gets representation in the
    # bank instead of the last-encoded symbol filling it via FIFO eviction.
    # With ``max_items=100_000`` and 13 symbols this caps each symbol at
    # ~7,692 randomly-sampled bars — enough for per-symbol calibration
    # profiles while keeping the bank small enough that ``build_dynamic_units``
    # finishes in a few minutes instead of ~30 min on 745K items.
    per_symbol_limit: int | None = None
    if memory_config.max_items > 0 and len(bars_by_symbol) > 0:
        per_symbol_limit = max(1, memory_config.max_items // len(bars_by_symbol))
        logger.info(
            "memory_build_per_symbol_limit=%d n_symbols=%d max_items=%d",
            per_symbol_limit, len(bars_by_symbol), memory_config.max_items,
        )

    total_samples = 0
    for symbol, bars in bars_by_symbol.items():
        max_index = len(bars) - max(memory_config.forward_horizons) - embargo_bars
        if max_index <= warmup_bars:
            continue
        # --- Phase 1: build feature snapshots + pending metadata for the symbol
        # (lightweight, vectorized rolling pipeline, CPU-bound). We queue them
        # and hand them to the encoder in chunks to amortize the forward cost.
        #
        # ``pipeline.build_batch`` runs every rolling feature reduction once
        # across the full bar history via ``torch.unfold``; ``snapshot_at``
        # then extracts per-bar snapshots in O(F) — no re-slice, no
        # re-reduction.  Biggest offline-build cold-start win.
        batched_snapshots = pipeline.build_batch(
            list(bars), symbol=symbol, temporal_window=warmup_bars
        )
        eligible_indices = list(range(warmup_bars, max_index))
        # Deterministic per-symbol stratified downsample.
        if per_symbol_limit is not None and len(eligible_indices) > per_symbol_limit:
            # Seed per-symbol so runs are reproducible but each symbol
            # gets an independent sample. ``hash(symbol)`` is NOT
            # deterministic across Python runs (PYTHONHASHSEED is random
            # unless fixed), so we use a stable hash (md5 → int) instead.
            import hashlib as _hashlib
            _sym_digest = int(
                _hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8], 16
            )
            g = torch.Generator().manual_seed(42 + (_sym_digest % 1_000_000))
            perm = torch.randperm(len(eligible_indices), generator=g)[:per_symbol_limit]
            sampled = sorted(eligible_indices[i] for i in perm.tolist())
            eligible_indices = sampled
        pending_snapshots = []
        pending_indices = []
        for index in eligible_indices:
            snapshot = batched_snapshots.snapshot_at(index)
            pending_snapshots.append(snapshot)
            pending_indices.append(index)
            if len(pending_snapshots) >= encode_chunk_size:
                _flush_encode_chunk(
                    bank=bank,
                    bars=bars,
                    pending_snapshots=pending_snapshots,
                    pending_indices=pending_indices,
                    encoder=encoder,
                    memory_config=memory_config,
                )
                total_samples += len(pending_snapshots)
                pending_snapshots = []
                pending_indices = []
        if pending_snapshots:
            _flush_encode_chunk(
                bank=bank,
                bars=bars,
                pending_snapshots=pending_snapshots,
                pending_indices=pending_indices,
                encoder=encoder,
                memory_config=memory_config,
            )
            total_samples += len(pending_snapshots)
        logger.info(
            "memory_build_symbol_done symbol=%s samples=%d running_total=%d",
            symbol,
            len(eligible_indices),
            total_samples,
        )

    logger.info("phase_timing phase=build_memory_done items=%d", len(bank))
    bank.meta["builder"] = "offline_complete"
    return bank


def _flush_encode_chunk(
    *,
    bank: MemoryBank,
    bars: Sequence[MarketBar],
    pending_snapshots: list,
    pending_indices: list[int],
    encoder: ModelStateEncoder,
    memory_config: MemoryConfig,
) -> None:
    """Encode a batch of snapshots in one forward pass and append the
    resulting MemoryItems to the bank in one bulk ``extend`` call.

    Using ``bank.extend`` instead of a per-item ``bank.append`` loop amortizes
    the O(N) latent-matrix stack across the whole chunk — the single biggest
    remaining hot spot on the offline memory-build path.
    """
    encoded_states = encoder.encode_batch(pending_snapshots)
    new_items: list[MemoryItem] = []
    for snapshot, encoded, index in zip(pending_snapshots, encoded_states, pending_indices):
        outcome = compute_outcome_summary(
            bars,
            index,
            horizons=memory_config.forward_horizons,
            path_label_rules=memory_config.path_label_rules,
        )
        new_items.append(
            MemoryItem(
                item_id=f"{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
                symbol=snapshot.symbol,
                timestamp=snapshot.timestamp,
                latent=encoded.latent.detach().cpu(),
                feature_contract_id=encoded.feature_contract_id,
                regime_weights=None if encoded.regime_weights is None else encoded.regime_weights.detach().cpu(),
                branch_confidences=dict(encoded.branch_confidences),
                branch_norms=dict(encoded.branch_norms),
                outcome=outcome,
                risk=build_risk_summary(outcome),
                encoder_fingerprint=encoded.encoder_fingerprint,
                metadata=dict(encoded.metadata),
            )
        )
    bank.extend(new_items)
