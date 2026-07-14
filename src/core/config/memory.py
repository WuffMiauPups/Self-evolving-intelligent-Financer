from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.exceptions import ConfigurationError


class MemoryConfig(BaseModel):
    """Passive memory bank configuration (post-Great-Simplification).

    The memory subsystem stores encoded states + realized outcomes and
    serves cosine-kNN lookups. Every field here drives real behavior:

      - Bank lifecycle: ``enabled``, ``persist_path``, ``max_items``,
        ``similarity_metric``, ``rebuild_on_startup``.
      - Encoder compatibility: ``latent_dim``, ``require_encoder_match``.
      - Retrieval: ``top_k`` neighbors per query.
      - Offline build + outcome labeling: ``build_from_real_checkpoint``,
        ``build_history_bars``, ``forward_horizons``, ``path_label_rules``.
      - Online observation queue: ``runtime_update_mode``,
        ``persist_pending_observations``, ``pending_store_path``,
        ``save_debounce_sec``, ``pending_max_per_symbol``.
      - Bounded influence (MASTER_PLAN §6.2): ``influence_mode`` plus the
        ``damping_*`` knobs — memory can only SHRINK a trade, never grow,
        flip, or veto one.

    The old Hopfield/ESOM/dynamic-unit/calibration/fusion knob graveyard
    (and the inert DecisionFusionConfig / TopologyGrowthConfig stubs) was
    deleted; tests/test_config_hygiene.py pins the allowed field set so
    dead knobs cannot silently regrow.
    """

    model_config = ConfigDict(extra="forbid")

    # enabled=False stays the bare-code default deliberately (documented
    # §4.6.5 divergence): a bare config must not switch subsystems on.
    # The other values are YAML-aligned.
    enabled: bool = False
    build_from_real_checkpoint: bool = True
    latent_dim: int | None = None
    top_k: int = 5
    similarity_metric: str = "cosine"
    max_items: int = 20000
    rebuild_on_startup: bool = False
    persist_path: str = "artifacts/market_memory.pt"
    forward_horizons: list[int] = Field(default_factory=lambda: [1, 3, 6])
    path_label_rules: dict[str, float] = Field(
        default_factory=lambda: {
            "breakout_return_threshold": 0.003,
            "mean_reversion_return_threshold": 0.0015,
            "flat_return_threshold": 0.00075,
            "time_to_move_threshold": 0.002,
        }
    )
    runtime_update_mode: str = "pending_only"
    build_history_bars: int = 320
    # --- Encoder versioning ---
    require_encoder_match: bool = True
    # --- Pending observation persistence ---
    persist_pending_observations: bool = True
    pending_store_path: str = "artifacts/pending_observations.pt"
    # Debounce for runtime memory persistence: the bank/pending stores are
    # saved at most once per interval (plus once at graceful shutdown)
    # instead of synchronously on the per-symbol decision hot path.
    save_debounce_sec: float = 300.0
    # Cap on queued pending observations per symbol (oldest dropped, loudly).
    pending_max_per_symbol: int = 2048
    # --- Bounded memory influence (Phase 2 §6.2) ---
    # "off": memory stays purely diagnostic. "damping": kNN outcome stats
    # apply a size multiplier in [damping_min_multiplier, 1.0] to the
    # proposal's target fraction — memory can only SHRINK a trade, never
    # grow one, flip its direction, veto it, or touch confidence. Keep off
    # until the counterfactual credit gate shows > 0 on OOS folds.
    influence_mode: str = "off"
    damping_min_multiplier: float = 0.5
    # Fewer action-aligned neighbor outcomes than this → no influence.
    damping_min_neighbors: int = 5
    # Mean action-aligned neighbor return at which damping saturates to the
    # floor (e.g. 0.002 = neighbors averaging -20bps against the action).
    damping_full_at_return: float = 0.002

    @model_validator(mode="after")
    def validate_memory_config(self) -> "MemoryConfig":
        if self.top_k <= 0:
            raise ConfigurationError("memory.top_k must be positive")
        if self.max_items <= 0:
            raise ConfigurationError("memory.max_items must be positive")
        if not self.forward_horizons or any(horizon <= 0 for horizon in self.forward_horizons):
            raise ConfigurationError("memory.forward_horizons must contain positive integers")
        if self.runtime_update_mode not in {"disabled", "pending_only", "append_when_ready"}:
            raise ConfigurationError(
                "memory.runtime_update_mode must be one of: disabled, pending_only, append_when_ready"
            )
        if self.similarity_metric not in {"cosine", "euclidean", "attention"}:
            raise ConfigurationError("memory.similarity_metric must be one of: cosine, euclidean, attention")
        if self.save_debounce_sec < 0.0:
            raise ConfigurationError("memory.save_debounce_sec must be non-negative")
        if self.pending_max_per_symbol <= 0:
            raise ConfigurationError("memory.pending_max_per_symbol must be positive")
        if self.influence_mode not in {"off", "damping"}:
            raise ConfigurationError("memory.influence_mode must be one of: off, damping")
        if not 0.0 < self.damping_min_multiplier <= 1.0:
            raise ConfigurationError("memory.damping_min_multiplier must be within (0, 1]")
        if self.damping_min_neighbors <= 0:
            raise ConfigurationError("memory.damping_min_neighbors must be positive")
        if self.damping_full_at_return <= 0.0:
            raise ConfigurationError("memory.damping_full_at_return must be positive")
        return self
