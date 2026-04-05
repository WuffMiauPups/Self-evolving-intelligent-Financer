from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from src.core.config import MemoryConfig
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.outcomes import build_risk_summary, compute_outcome_summary
from src.memory.schemas import MemoryItem, PendingMemoryState
from src.models.state_encoder import EncodedState, ModelStateEncoder

logger = logging.getLogger(__name__)


@dataclass
class AssociativeMemoryUpdater:
    bank: MemoryBank
    config: MemoryConfig
    pending_by_symbol: dict[str, list[PendingMemoryState]] = field(default_factory=lambda: defaultdict(list))
    observed_count: int = 0
    finalized_count: int = 0
    last_refresh_report: dict[str, object] = field(default_factory=dict)

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
        if finalized:
            self.bank.rebuild_prototypes(
                min_similarity=self.config.min_similarity,
                min_cluster_size=self.config.min_cluster_size,
                temperature=self.config.attention_temperature,
            )
            self.last_refresh_report = self.bank.refresh_organization(self.config)
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
            "last_refresh_report": dict(self.last_refresh_report),
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


def build_memory_bank_from_bars(
    *,
    bars_by_symbol: dict[str, Sequence[MarketBar]],
    pipeline: FeaturePipeline,
    encoder: ModelStateEncoder,
    memory_config: MemoryConfig,
    warmup_bars: int,
) -> MemoryBank:
    if not bars_by_symbol:
        raise ValueError("bars_by_symbol cannot be empty")
    sample_symbol = next(iter(bars_by_symbol))
    sample_bars = list(bars_by_symbol[sample_symbol])
    if len(sample_bars) <= warmup_bars + max(memory_config.forward_horizons):
        raise ValueError("Not enough bars to build associative memory")
    sample_snapshot = pipeline.build(sample_bars[: warmup_bars + 1], symbol=sample_symbol)
    sample_encoded = encoder.encode(sample_snapshot)
    bank = MemoryBank(
        feature_contract=sample_snapshot.contract,
        latent_dim=sample_encoded.latent.shape[0],
        similarity_metric=memory_config.similarity_metric,
        max_items=memory_config.max_items,
        require_encoder_match=memory_config.require_encoder_match,
        meta={"builder": "offline"},
    )
    for symbol, bars in bars_by_symbol.items():
        max_index = len(bars) - max(memory_config.forward_horizons)
        for index in range(warmup_bars, max_index):
            snapshot = pipeline.build(bars[: index + 1], symbol=symbol)
            encoded = encoder.encode(snapshot)
            outcome = compute_outcome_summary(
                bars,
                index,
                horizons=memory_config.forward_horizons,
                path_label_rules=memory_config.path_label_rules,
            )
            bank.append(
                MemoryItem(
                    item_id=f"{symbol}-{int(snapshot.timestamp.timestamp())}",
                    symbol=symbol,
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
    bank.rebuild_prototypes(
        min_similarity=memory_config.min_similarity,
        min_cluster_size=memory_config.min_cluster_size,
        temperature=memory_config.attention_temperature,
    )
    bank.refresh_organization(memory_config)
    bank.meta["builder"] = "offline_complete"
    return bank
