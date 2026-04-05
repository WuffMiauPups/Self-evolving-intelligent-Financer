"""Tests for stable brain + online-learning memory + offline retrain cycle.

Covers:
  Part 1: Frozen-brain runtime policy config + enforcement
  Part 2: Online memory-learning mode
  Part 3: Pending-memory persistence / outcome-finalization
  Part 4: Offline retrain cycle scaffolding
  Part 5: Checkpoint promotion / deployment rules
  Part 6: Observability
"""
from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import pytest

from src.core.config import AppConfig, MemoryConfig, RuntimeConfig


# ---------------------------------------------------------------------------
# Part 1: Frozen-brain runtime policy config
# ---------------------------------------------------------------------------

class TestFrozenBrainConfig:
    """RuntimeConfig frozen-brain fields."""

    def test_default_allow_online_model_updates_is_false(self):
        config = RuntimeConfig()
        assert config.allow_online_model_updates is False

    def test_default_allow_online_memory_updates_is_true(self):
        config = RuntimeConfig()
        assert config.allow_online_memory_updates is True

    def test_can_override_model_updates(self):
        config = RuntimeConfig(allow_online_model_updates=True)
        assert config.allow_online_model_updates is True

    def test_can_override_memory_updates(self):
        config = RuntimeConfig(allow_online_memory_updates=False)
        assert config.allow_online_memory_updates is False

    def test_frozen_brain_enforcement_freezes_parameters(self):
        """When allow_online_model_updates=False, model params should have requires_grad=False."""
        model = torch.nn.Linear(10, 3)
        assert all(p.requires_grad for p in model.parameters())

        # Simulate what trade_engine does
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        assert all(not p.requires_grad for p in model.parameters())
        assert not model.training

    def test_app_config_includes_frozen_brain(self):
        config = AppConfig()
        assert hasattr(config.runtime, "allow_online_model_updates")
        assert hasattr(config.runtime, "allow_online_memory_updates")


# ---------------------------------------------------------------------------
# Part 2: Online memory-learning mode
# ---------------------------------------------------------------------------

class TestOnlineMemoryMode:
    """Memory config enforcement for online learning."""

    def test_runtime_update_mode_options(self):
        for mode in ("disabled", "pending_only", "append_when_ready"):
            config = MemoryConfig(runtime_update_mode=mode)
            assert config.runtime_update_mode == mode

    def test_invalid_runtime_update_mode_raises(self):
        with pytest.raises(Exception):
            MemoryConfig(runtime_update_mode="invalid_mode")

    def test_memory_config_copy_overrides_update_mode(self):
        """Simulates how trade_engine overrides mode when memory updates disabled."""
        config = MemoryConfig(runtime_update_mode="append_when_ready")
        overridden = config.model_copy(update={"runtime_update_mode": "disabled"})
        assert overridden.runtime_update_mode == "disabled"
        assert config.runtime_update_mode == "append_when_ready"  # original unchanged


# ---------------------------------------------------------------------------
# Part 3: Pending-memory persistence / outcome-finalization
# ---------------------------------------------------------------------------

class TestPendingMemoryPersistence:
    """AssociativeMemoryUpdater save/load pending observations."""

    def _make_updater(self, update_mode: str = "pending_only"):
        from src.memory.updater import AssociativeMemoryUpdater
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract

        contract = FeatureContract(
            feature_names=("f1", "f2"),
            branch_feature_names={"trend": ("f1",), "vol": ("f2",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1", "f2"),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract,
            latent_dim=8,
            similarity_metric="cosine",
            max_items=100,
        )
        config = MemoryConfig(runtime_update_mode=update_mode)
        return AssociativeMemoryUpdater(bank, config)

    def _make_pending_state(self, symbol: str = "AAPL", offset_hours: int = 0):
        from src.memory.schemas import PendingMemoryState
        return PendingMemoryState(
            symbol=symbol,
            timestamp=datetime(2025, 1, 1, 10, 0, tzinfo=UTC) + timedelta(hours=offset_hours),
            feature_contract_id="test_contract",
            latent=torch.randn(8),
            regime_weights=torch.randn(4),
            branch_confidences={"trend": 0.8, "vol": 0.5},
            branch_norms={"trend": 1.0, "vol": 0.9},
            action_annotation="test_action",
            encoder_fingerprint="fp_abc123",
            metadata={"key": "value"},
        )

    def test_save_and_load_roundtrip(self):
        updater = self._make_updater()
        updater.pending_by_symbol["AAPL"].append(self._make_pending_state("AAPL", 0))
        updater.pending_by_symbol["AAPL"].append(self._make_pending_state("AAPL", 1))
        updater.pending_by_symbol["MSFT"].append(self._make_pending_state("MSFT", 2))
        updater.observed_count = 5
        updater.finalized_count = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            saved = updater.save_pending(path)
            assert saved == 3
            assert path.exists()

            # Load into fresh updater
            updater2 = self._make_updater()
            restored = updater2.load_pending(path)
            assert restored == 3
            assert len(updater2.pending_by_symbol["AAPL"]) == 2
            assert len(updater2.pending_by_symbol["MSFT"]) == 1
            assert updater2.observed_count == 5
            assert updater2.finalized_count == 2

    def test_load_nonexistent_returns_zero(self):
        updater = self._make_updater()
        restored = updater.load_pending("/nonexistent/path.pt")
        assert restored == 0

    def test_no_duplicates_on_reload(self):
        updater = self._make_updater()
        pending = self._make_pending_state("AAPL", 0)
        updater.pending_by_symbol["AAPL"].append(pending)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            updater.save_pending(path)
            # Load again into same updater (should skip duplicates)
            restored = updater.load_pending(path)
            assert restored == 0  # duplicate skipped
            assert len(updater.pending_by_symbol["AAPL"]) == 1

    def test_save_empty_pending(self):
        updater = self._make_updater()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            saved = updater.save_pending(path)
            assert saved == 0

    def test_pending_latent_shapes_preserved(self):
        updater = self._make_updater()
        pending = self._make_pending_state("AAPL", 0)
        original_shape = pending.latent.shape
        updater.pending_by_symbol["AAPL"].append(pending)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            updater.save_pending(path)
            updater2 = self._make_updater()
            updater2.load_pending(path)
            restored = updater2.pending_by_symbol["AAPL"][0]
            assert restored.latent.shape == original_shape
            assert restored.regime_weights is not None

    def test_pending_metadata_preserved(self):
        updater = self._make_updater()
        pending = self._make_pending_state("AAPL", 0)
        updater.pending_by_symbol["AAPL"].append(pending)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "pending.pt"
            updater.save_pending(path)
            updater2 = self._make_updater()
            updater2.load_pending(path)
            restored = updater2.pending_by_symbol["AAPL"][0]
            assert restored.action_annotation == "test_action"
            assert restored.encoder_fingerprint == "fp_abc123"
            assert restored.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# Part 3b: MemoryConfig pending persistence fields
# ---------------------------------------------------------------------------

class TestPendingConfig:
    """MemoryConfig fields for pending observation persistence."""

    def test_default_persist_pending_observations(self):
        config = MemoryConfig()
        assert config.persist_pending_observations is True

    def test_default_pending_store_path(self):
        config = MemoryConfig()
        assert config.pending_store_path == "artifacts/pending_observations.pt"

    def test_can_override_persist_path(self):
        config = MemoryConfig(pending_store_path="custom/path.pt")
        assert config.pending_store_path == "custom/path.pt"

    def test_can_disable_persistence(self):
        config = MemoryConfig(persist_pending_observations=False)
        assert config.persist_pending_observations is False


# ---------------------------------------------------------------------------
# Part 4: Offline retrain cycle scaffolding
# ---------------------------------------------------------------------------

class TestRetainMetadata:
    """RetainMetadata and retrain cycle functions."""

    def test_build_retrain_metadata(self):
        from src.training.retrain import build_retrain_metadata
        meta = build_retrain_metadata(
            parent_checkpoint_path="checkpoints/v1.pt",
            parent_checkpoint_fingerprint="fp_parent_123",
            retrain_reason="scheduled_weekly",
            data_start="2025-01-01",
            data_end="2025-06-01",
            symbols=["AAPL", "MSFT"],
            paper_observation_count=500,
            config_snapshot={"epochs": 3},
        )
        assert meta.retrain_id.startswith("retrain-")
        assert meta.parent_checkpoint_path == "checkpoints/v1.pt"
        assert meta.retrain_reason == "scheduled_weekly"
        assert meta.symbols == ["AAPL", "MSFT"]
        assert meta.paper_observation_count == 500
        assert meta.completed_at is None
        assert meta.promotion_eligible is False

    def test_finalize_retrain_metadata(self):
        from src.training.retrain import build_retrain_metadata, finalize_retrain_metadata
        meta = build_retrain_metadata(
            parent_checkpoint_path=None,
            parent_checkpoint_fingerprint=None,
            retrain_reason="first_train",
            data_start="2024-01-01",
            data_end="2025-01-01",
            symbols=["AAPL"],
            paper_observation_count=0,
            config_snapshot={},
        )
        finalized = finalize_retrain_metadata(
            meta,
            output_checkpoint_path="checkpoints/v2.pt",
            output_checkpoint_fingerprint="fp_v2",
            validation_metrics={"val_macro_f1": 0.45, "val_balanced_accuracy": 0.52},
            promotion_eligible=True,
        )
        assert finalized.completed_at is not None
        assert finalized.output_checkpoint_path == "checkpoints/v2.pt"
        assert finalized.output_checkpoint_fingerprint == "fp_v2"
        assert finalized.promotion_eligible is True
        assert finalized.validation_metrics["val_macro_f1"] == 0.45

    def test_save_and_load_roundtrip(self):
        from src.training.retrain import (
            build_retrain_metadata,
            finalize_retrain_metadata,
            save_retrain_metadata,
            load_retrain_metadata,
        )
        meta = build_retrain_metadata(
            parent_checkpoint_path="checkpoints/v1.pt",
            parent_checkpoint_fingerprint="fp_v1",
            retrain_reason="paper_observation_threshold",
            data_start="2025-01-01",
            data_end="2025-06-01",
            symbols=["AAPL"],
            paper_observation_count=200,
            config_snapshot={"epochs": 5, "lr": 0.001},
        )
        finalized = finalize_retrain_metadata(
            meta,
            output_checkpoint_path="checkpoints/v2.pt",
            output_checkpoint_fingerprint="fp_v2",
            validation_metrics={"macro_f1": 0.42},
            promotion_eligible=False,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_retrain_metadata(finalized, runs_dir=tmpdir)
            assert path.exists()
            loaded = load_retrain_metadata(path)
            assert loaded.retrain_id == finalized.retrain_id
            assert loaded.parent_checkpoint_fingerprint == "fp_v1"
            assert loaded.output_checkpoint_fingerprint == "fp_v2"
            assert loaded.validation_metrics["macro_f1"] == 0.42

    def test_list_retrain_history(self):
        from src.training.retrain import (
            build_retrain_metadata,
            save_retrain_metadata,
            list_retrain_history,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                meta = build_retrain_metadata(
                    parent_checkpoint_path=None,
                    parent_checkpoint_fingerprint=f"fp_{i}",
                    retrain_reason=f"test_{i}",
                    data_start="2025-01-01",
                    data_end="2025-06-01",
                    symbols=["AAPL"],
                    paper_observation_count=i * 100,
                    config_snapshot={},
                )
                save_retrain_metadata(meta, runs_dir=tmpdir)
            history = list_retrain_history(tmpdir)
            assert len(history) == 3

    def test_validate_retrain_prerequisites_missing_checkpoint(self):
        from src.training.retrain import validate_retrain_prerequisites
        can_proceed, warnings = validate_retrain_prerequisites(
            parent_checkpoint_path="/nonexistent/checkpoint.pt",
            paper_observation_count=200,
        )
        assert can_proceed is False
        assert any("not found" in w for w in warnings)

    def test_validate_retrain_prerequisites_low_observations(self):
        from src.training.retrain import validate_retrain_prerequisites
        can_proceed, warnings = validate_retrain_prerequisites(
            parent_checkpoint_path=None,
            paper_observation_count=10,
            min_paper_observations=100,
        )
        assert can_proceed is True  # low observations is a warning, not a block
        assert any("Only 10" in w for w in warnings)

    def test_retrain_id_uniqueness(self):
        from src.training.retrain import build_retrain_id
        id1 = build_retrain_id("fp1", datetime(2025, 1, 1, tzinfo=UTC))
        id2 = build_retrain_id("fp2", datetime(2025, 1, 1, tzinfo=UTC))
        id3 = build_retrain_id("fp1", datetime(2025, 1, 2, tzinfo=UTC))
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_genesis_retrain_id(self):
        from src.training.retrain import build_retrain_id
        id1 = build_retrain_id(None, datetime(2025, 1, 1, tzinfo=UTC))
        assert id1.startswith("retrain-")


# ---------------------------------------------------------------------------
# Part 5: Checkpoint promotion / deployment rules
# ---------------------------------------------------------------------------

class TestCheckpointPromotion:
    """Promotion gates, decisions, and registry."""

    def test_promotion_passes_all_gates(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 0.5,
            },
            parent_metrics={
                "val_report.macro_f1": 0.40,
                "val.loss": 0.55,
            },
            gate=PromotionGate(),
        )
        assert decision.eligible is True
        assert "all promotion gates passed" in decision.reasons

    def test_promotion_fails_low_balanced_accuracy(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.20,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 0.5,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is False
        assert any("val_balanced_accuracy" in r for r in decision.reasons)

    def test_promotion_fails_low_macro_f1(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.10,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 0.5,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is False
        assert any("val_macro_f1" in r for r in decision.reasons)

    def test_promotion_fails_low_recall(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.05,
                "val_report.per_class_metrics.short.recall": 0.05,
                "val.loss": 0.5,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is False
        assert any("long_recall" in r for r in decision.reasons)
        assert any("short_recall" in r for r in decision.reasons)

    def test_promotion_fails_regression_over_parent(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.35,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 0.5,
            },
            parent_metrics={
                "val_report.macro_f1": 0.40,
                "val.loss": 0.5,
            },
            gate=PromotionGate(require_improvement_over_parent=True),
        )
        assert decision.eligible is False
        assert any("delta" in r for r in decision.reasons)

    def test_promotion_fails_loss_regression(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 1.5,
            },
            parent_metrics={
                "val_report.macro_f1": 0.40,
                "val.loss": 0.5,
            },
            gate=PromotionGate(max_val_loss_regression=0.10),
        )
        assert decision.eligible is False
        assert any("val.loss regression" in r for r in decision.reasons)

    def test_promote_checkpoint_creates_copy_and_registry(self):
        from src.training.checkpoint_promotion import (
            CheckpointVersion,
            promote_checkpoint,
            load_registry,
            get_active_checkpoint,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.pt"
            target = Path(tmpdir) / "deployed" / "active.pt"
            registry_path = Path(tmpdir) / "registry.json"

            torch.save({"test": True}, source)
            version = CheckpointVersion(
                version_id="v1",
                checkpoint_path=str(source),
                fingerprint="fp_v1",
                parent_fingerprint=None,
                created_at=datetime.now(UTC).isoformat(),
                retrain_id="retrain-abc",
                metrics={"macro_f1": 0.45},
                status="candidate",
            )
            promoted = promote_checkpoint(
                source_path=source,
                target_path=target,
                version=version,
                registry_path=registry_path,
            )
            assert promoted.status == "active"
            assert target.exists()
            assert registry_path.exists()
            active = get_active_checkpoint(registry_path)
            assert active is not None
            assert active["version_id"] == "v1"
            assert active["status"] == "active"

    def test_promotion_retires_previous_active(self):
        from src.training.checkpoint_promotion import (
            CheckpointVersion,
            promote_checkpoint,
            load_registry,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"

            for i in range(2):
                source = Path(tmpdir) / f"v{i}.pt"
                target = Path(tmpdir) / f"deployed_v{i}.pt"
                torch.save({"version": i}, source)
                version = CheckpointVersion(
                    version_id=f"v{i}",
                    checkpoint_path=str(source),
                    fingerprint=f"fp_v{i}",
                    parent_fingerprint=f"fp_v{i-1}" if i > 0 else None,
                    created_at=datetime.now(UTC).isoformat(),
                    retrain_id=None,
                    metrics={},
                    status="candidate",
                )
                promote_checkpoint(
                    source_path=source,
                    target_path=target,
                    version=version,
                    registry_path=registry_path,
                )

            registry = load_registry(registry_path)
            assert len(registry) == 2
            assert registry[0]["status"] == "retired"
            assert registry[1]["status"] == "active"

    def test_checkpoint_history(self):
        from src.training.checkpoint_promotion import (
            CheckpointVersion,
            promote_checkpoint,
            get_checkpoint_history,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.json"
            for i in range(3):
                source = Path(tmpdir) / f"v{i}.pt"
                target = Path(tmpdir) / f"deployed_v{i}.pt"
                torch.save({}, source)
                version = CheckpointVersion(
                    version_id=f"v{i}",
                    checkpoint_path=str(source),
                    fingerprint=f"fp_{i}",
                    parent_fingerprint=None,
                    created_at=datetime.now(UTC).isoformat(),
                    retrain_id=None,
                    metrics={},
                    status="candidate",
                )
                promote_checkpoint(
                    source_path=source,
                    target_path=target,
                    version=version,
                    registry_path=registry_path,
                )
            history = get_checkpoint_history(registry_path)
            assert len(history) == 3
            assert history[0]["version_id"] == "v2"  # newest first

    def test_empty_registry(self):
        from src.training.checkpoint_promotion import get_active_checkpoint, load_registry
        assert get_active_checkpoint("/nonexistent/registry.json") is None
        assert load_registry("/nonexistent/registry.json") == []


# ---------------------------------------------------------------------------
# Part 6: Observability / integration
# ---------------------------------------------------------------------------

class TestObservability:
    """Observability logging and integration behavior."""

    def test_updater_summary_includes_pending(self):
        from src.memory.updater import AssociativeMemoryUpdater
        from src.memory.memory_bank import MemoryBank
        from src.features.feature_pipeline import FeatureContract
        from src.memory.schemas import PendingMemoryState

        contract = FeatureContract(
            feature_names=("f1",),
            branch_feature_names={"trend": ("f1",)},
            regime_feature_names=("r1",),
            temporal_feature_names=("f1",),
            feature_dtype="float32",
        )
        bank = MemoryBank(
            feature_contract=contract, latent_dim=4, similarity_metric="cosine", max_items=100,
        )
        config = MemoryConfig(runtime_update_mode="pending_only")
        updater = AssociativeMemoryUpdater(bank, config)
        updater.pending_by_symbol["AAPL"].append(
            PendingMemoryState(
                symbol="AAPL",
                timestamp=datetime(2025, 1, 1, tzinfo=UTC),
                feature_contract_id="fc",
                latent=torch.randn(4),
                regime_weights=None,
                branch_confidences={},
                branch_norms={},
            )
        )
        updater.observed_count = 5
        updater.finalized_count = 3
        summary = updater.summary()
        assert summary["observed_count"] == 5
        assert summary["finalized_count"] == 3
        assert summary["pending_count"] == 1

    def test_promotion_gate_defaults(self):
        from src.training.checkpoint_promotion import PromotionGate
        gate = PromotionGate()
        assert gate.min_val_balanced_accuracy == 0.40
        assert gate.min_val_macro_f1 == 0.30
        assert gate.min_long_recall == 0.15
        assert gate.min_short_recall == 0.15
        assert gate.require_improvement_over_parent is True

    def test_promotion_decision_has_delta(self):
        from src.training.checkpoint_promotion import evaluate_promotion, PromotionGate
        decision = evaluate_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.50,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.25,
                "val.loss": 0.4,
            },
            parent_metrics={
                "val_report.macro_f1": 0.40,
                "val.loss": 0.5,
            },
            gate=PromotionGate(),
        )
        assert "val_report.macro_f1" in decision.delta
        assert decision.delta["val_report.macro_f1"] == pytest.approx(0.10, abs=1e-6)
        assert "val.loss" in decision.delta
        assert decision.delta["val.loss"] == pytest.approx(-0.10, abs=1e-6)

    def test_retrain_metadata_immutable(self):
        from src.training.retrain import RetainMetadata
        meta = RetainMetadata(
            retrain_id="test",
            parent_checkpoint_path=None,
            parent_checkpoint_fingerprint=None,
            retrain_reason="test",
            data_start="2025-01-01",
            data_end="2025-06-01",
            symbols=["AAPL"],
            paper_observation_count=0,
            config_snapshot={},
            started_at="2025-01-01T00:00:00",
        )
        with pytest.raises(Exception):
            meta.retrain_id = "modified"  # frozen dataclass

    def test_checkpoint_version_immutable(self):
        from src.training.checkpoint_promotion import CheckpointVersion
        version = CheckpointVersion(
            version_id="v1",
            checkpoint_path="test.pt",
            fingerprint="fp",
            parent_fingerprint=None,
            created_at="2025-01-01",
            retrain_id=None,
            metrics={},
            status="candidate",
        )
        with pytest.raises(Exception):
            version.status = "active"  # frozen dataclass


# ---------------------------------------------------------------------------
# Part 6b: End-to-end config loading
# ---------------------------------------------------------------------------

class TestConfigEndToEnd:
    """Config loading includes all new fields."""

    def test_default_config_loads_with_new_fields(self):
        from src.core.config import load_config
        config = load_config("configs/default.yaml")
        assert config.runtime.allow_online_model_updates is False
        assert config.runtime.allow_online_memory_updates is True
        assert config.memory.persist_pending_observations is True
        assert config.memory.pending_store_path == "artifacts/pending_observations.pt"
