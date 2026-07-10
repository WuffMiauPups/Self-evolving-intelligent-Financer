"""Close-the-loop tests (MASTER_PLAN §4.5.3-7).

Covers:
  Task 3: per-metric retrain triggers, the persisted observation stream,
          and the trigger->retrain dry run with lineage recorded.
  Task 4: authoritative promotion — candidate staging in run_train,
          transactional promote (lock/backup/atomic registry), and the
          flat-vs-dotted metric-key adapter.
  Task 5: scorecard-judged promotion on stubbed BenchmarkRows.
  Task 6: forgetting-safe retraining (KD/L2 incumbent anchors).
  Task 7: backward-transfer floor in the gate.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.core.config import AppConfig, PromotionConfig, TrainingConfig, load_config
from src.core.exceptions import ConfigurationError
from src.training.checkpoint_promotion import (
    FLAT_TO_DOTTED_METRIC_KEYS,
    GATE_METRIC_KEYS,
    CheckpointVersion,
    PromotionGate,
    evaluate_promotion,
    load_registry,
    normalize_promotion_metrics,
    promote_checkpoint,
)
from src.training.replay_buffer import (
    DecisionObservation,
    ObservationBuffer,
    append_observation,
    load_observations,
)
from src.training.retrain import (
    DEFAULT_RETRAIN_TRIGGERS,
    check_retrain_trigger,
    check_retrain_trigger_from_observations,
    compute_outcome_metrics,
    load_retrain_metadata,
    run_triggered_retrain,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_run_train_module():
    spec = importlib.util.spec_from_file_location(
        "_close_loop_run_train", ROOT / "scripts" / "run_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_close_loop_run_train"] = module
    spec.loader.exec_module(module)
    return module


def _observation(
    index: int,
    pnl: float,
    *,
    symbol: str = "AAPL",
    action: int = 1,
    clamped: bool = False,
) -> DecisionObservation:
    return DecisionObservation(
        decision_id=f"dec-{index:05d}",
        symbol=symbol,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        features_ref="feat_fp_abc123",
        executed_action=action,
        realized_pnl=pnl,
        clamped=clamped,
    )


def _drifting_observations(count: int = 240) -> list[DecisionObservation]:
    """Older half wins, newer half loses — a clear win-rate collapse."""
    half = count // 2
    older = [_observation(i, 1.0 + 0.01 * (i % 5)) for i in range(half)]
    newer = [_observation(half + i, -1.0 - 0.01 * (i % 5)) for i in range(count - half)]
    return older + newer


# ---------------------------------------------------------------------------
# Task 3: per-metric retrain triggers
# ---------------------------------------------------------------------------


class TestRetrainTriggerThresholds:
    def test_sharpe_drop_over_threshold_fires(self):
        fired, reason = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.2, "win_rate": 0.5},
            baseline_metrics={"sharpe_ratio": 0.9, "win_rate": 0.5},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 0},
        )
        assert fired is True
        assert reason.startswith("sharpe_drift")

    def test_sharpe_drop_under_threshold_does_not_fire(self):
        # A 0.15 Sharpe drop is noise: with the per-metric 0.5 threshold it
        # must NOT fire (the old shared 0.15 threshold fired here).
        fired, reason = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.75, "win_rate": 0.5},
            baseline_metrics={"sharpe_ratio": 0.90, "win_rate": 0.5},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 0},
        )
        assert fired is False
        assert reason == "no_drift"

    def test_win_rate_drop_over_threshold_fires(self):
        # A 6pt win-rate drop is catastrophe: it must fire even though the
        # old shared 0.15 threshold ignored it.
        fired, reason = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.9, "win_rate": 0.44},
            baseline_metrics={"sharpe_ratio": 0.9, "win_rate": 0.50},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 0},
        )
        assert fired is True
        assert reason.startswith("win_rate_drift")

    def test_win_rate_drop_under_threshold_does_not_fire(self):
        fired, _ = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.9, "win_rate": 0.47},
            baseline_metrics={"sharpe_ratio": 0.9, "win_rate": 0.50},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 0},
        )
        assert fired is False

    def test_min_observations_blocks_trigger(self):
        fired, reason = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.0, "win_rate": 0.0},
            baseline_metrics={"sharpe_ratio": 2.0, "win_rate": 1.0},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200},
            observation_count=50,
        )
        assert fired is False
        assert reason.startswith("insufficient_observations")

    def test_default_thresholds_used_when_none(self):
        fired, _ = check_retrain_trigger(
            recent_metrics={"sharpe_ratio": 0.0},
            baseline_metrics={"sharpe_ratio": DEFAULT_RETRAIN_TRIGGERS["sharpe_drop"] + 0.1},
        )
        assert fired is True

    def test_config_carries_per_metric_thresholds(self):
        config = load_config("configs/default.yaml")
        assert config.training.retrain_triggers["sharpe_drop"] == pytest.approx(0.5)
        assert config.training.retrain_triggers["win_rate_drop"] == pytest.approx(0.05)
        assert config.training.retrain_triggers["min_observations"] == pytest.approx(200)

    def test_unknown_trigger_key_rejected(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(retrain_triggers={"sharpe_drop": 0.5, "vibes": 1.0})

    def test_negative_trigger_value_rejected(self):
        with pytest.raises(ConfigurationError):
            TrainingConfig(retrain_triggers={"sharpe_drop": -0.5})


class TestObservationStream:
    def test_jsonl_roundtrip(self, tmp_path: Path):
        path = tmp_path / "observations.jsonl"
        buffer = ObservationBuffer(capacity=10)
        buffer.add(_observation(0, 1.25, clamped=True))
        buffer.add(_observation(1, -0.5, action=-1))
        assert buffer.save(path) == 2

        loaded = load_observations(path)
        assert len(loaded) == 2
        assert loaded[0].decision_id == "dec-00000"
        assert loaded[0].clamped is True
        assert loaded[0].features_ref == "feat_fp_abc123"
        assert loaded[1].executed_action == -1
        assert loaded[1].realized_pnl == pytest.approx(-0.5)

    def test_corrupt_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "observations.jsonl"
        append_observation(path, _observation(0, 1.0))
        with path.open("a", encoding="utf-8") as fp:
            fp.write("{torn line\n")
        append_observation(path, _observation(1, 2.0))
        assert len(load_observations(path)) == 2

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert load_observations(tmp_path / "nope.jsonl") == []

    def test_buffer_is_bounded(self):
        buffer = ObservationBuffer(capacity=5)
        buffer.extend(_observation(i, 0.1) for i in range(20))
        assert len(buffer) == 5
        assert buffer.observations[0].decision_id == "dec-00015"

    def test_compute_outcome_metrics_excludes_flat(self):
        observations = [
            _observation(0, 1.0),
            _observation(1, -1.0),
            _observation(2, 0.0, action=0),  # FLAT: no trade
            _observation(3, 2.0),
        ]
        metrics = compute_outcome_metrics(observations)
        assert metrics["trade_count"] == 3
        assert metrics["win_rate"] == pytest.approx(2 / 3)


class TestTriggerDryRun:
    def test_stream_split_fires_on_drift(self):
        fired, reason = check_retrain_trigger_from_observations(
            _drifting_observations(240),
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200},
        )
        assert fired is True
        assert "drift" in reason

    def test_stream_without_drift_does_not_fire(self):
        stable = [_observation(i, 1.0 if i % 2 == 0 else -0.9) for i in range(240)]
        fired, reason = check_retrain_trigger_from_observations(
            stable,
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200},
        )
        assert fired is False

    def test_triggered_retrain_invokes_entry_point_with_lineage(self, tmp_path: Path):
        """N synthetic observations -> trigger fires -> the retrain entry
        point (a run_supervised_training stub) is invoked and RetainMetadata
        lineage is recorded."""
        checkpoint_path = tmp_path / "retrained.pt"
        calls: list[int] = []

        def train_fn():
            calls.append(1)
            checkpoint_path.write_bytes(b"retrained-weights")
            return {
                "checkpoint": str(checkpoint_path),
                "evaluation_reports": {"val": {"balanced_accuracy": 0.52, "macro_f1": 0.41}},
            }

        meta = run_triggered_retrain(
            observations=_drifting_observations(240),
            train_fn=train_fn,
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200},
            parent_checkpoint_path="checkpoints/brain.pt",
            parent_checkpoint_fingerprint="fp_incumbent",
            config_snapshot={"epochs": 3},
            runs_dir=str(tmp_path / "runs"),
        )
        assert calls == [1]
        assert meta is not None
        assert meta.retrain_reason.startswith(("win_rate_drift", "sharpe_drift"))
        assert meta.parent_checkpoint_fingerprint == "fp_incumbent"
        assert meta.paper_observation_count == 240
        assert meta.output_checkpoint_path == str(checkpoint_path)
        assert meta.output_checkpoint_fingerprint  # sha1 of the file
        assert meta.validation_metrics["val_balanced_accuracy"] == pytest.approx(0.52)
        assert meta.promotion_eligible is False  # the gate decides, not the trigger
        saved = list((tmp_path / "runs").glob("retrain-*.json"))
        assert len(saved) == 1
        assert load_retrain_metadata(saved[0]).retrain_id == meta.retrain_id

    def test_no_trigger_means_no_retrain(self, tmp_path: Path):
        calls: list[int] = []
        meta = run_triggered_retrain(
            observations=[_observation(i, 1.0) for i in range(20)],
            train_fn=lambda: calls.append(1) or {},
            thresholds={"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200},
            runs_dir=str(tmp_path),
        )
        assert meta is None
        assert calls == []


class TestRetrainTriggerCli:
    def test_cli_exits_zero_when_trigger_fires(self, tmp_path: Path):
        module = _load_run_train_module()
        path = tmp_path / "observations.jsonl"
        for observation in _drifting_observations(240):
            append_observation(path, observation)
        config = AppConfig()
        assert module._check_retrain_trigger_cli(config, str(path)) == 0

    def test_cli_exits_one_without_observations(self, tmp_path: Path):
        module = _load_run_train_module()
        config = AppConfig()
        assert module._check_retrain_trigger_cli(config, str(tmp_path / "missing.jsonl")) == 1


# ---------------------------------------------------------------------------
# Task 4: metric-key adapter + transactional promotion
# ---------------------------------------------------------------------------

GOOD_DOTTED_METRICS = {
    "val_report.balanced_accuracy": 0.55,
    "val_report.macro_f1": 0.45,
    "val_report.per_class_metrics.long.recall": 0.30,
    "val_report.per_class_metrics.short.recall": 0.25,
    "val.loss": 0.5,
}


class TestMetricKeyAdapter:
    def test_adapter_covers_every_gate_key(self):
        """Key-alignment contract: every dotted key the gate reads has a
        flat alias, so checkpoint-meta/gui flat metrics satisfy the gate."""
        assert set(GATE_METRIC_KEYS) == set(FLAT_TO_DOTTED_METRIC_KEYS.values())

    def test_checkpoint_service_flat_keys_have_aliases(self):
        # The flat schema produced by gui_services.checkpoint_service
        # (validate_checkpoint summary / checkpoint meta ``extra``).
        for flat_key in ("val_balanced_accuracy", "val_macro_f1", "val_loss"):
            assert flat_key in FLAT_TO_DOTTED_METRIC_KEYS

    def test_flat_metrics_normalized(self):
        normalized = normalize_promotion_metrics({"val_balanced_accuracy": 0.51})
        assert normalized["val_report.balanced_accuracy"] == pytest.approx(0.51)

    def test_dotted_keys_win_over_flat(self):
        normalized = normalize_promotion_metrics(
            {"val_balanced_accuracy": 0.10, "val_report.balanced_accuracy": 0.60}
        )
        assert normalized["val_report.balanced_accuracy"] == pytest.approx(0.60)

    def test_gate_accepts_flat_schema(self):
        """Before the adapter, flat metrics scored 0.0 on every gate key and
        were always refused — this pins the fix."""
        decision = evaluate_promotion(
            candidate_metrics={
                "val_balanced_accuracy": 0.55,
                "val_macro_f1": 0.45,
                "val_long_recall": 0.30,
                "val_short_recall": 0.25,
                "val_loss": 0.5,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is True


class TestTransactionalPromotion:
    def _version(self, version_id: str = "v1") -> CheckpointVersion:
        return CheckpointVersion(
            version_id=version_id,
            checkpoint_path="candidate.pt",
            fingerprint=f"fp_{version_id}",
            parent_fingerprint=None,
            created_at=datetime.now(UTC).isoformat(),
            retrain_id=None,
            metrics={},
            status="candidate",
        )

    def test_lock_file_blocks_concurrent_promotion(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new")
        registry = tmp_path / "registry.json"
        lock = tmp_path / "registry.json.lock"
        lock.write_text("12345")
        with pytest.raises(RuntimeError, match="concurrent promotion"):
            promote_checkpoint(
                source_path=source,
                target_path=tmp_path / "brain.pt",
                version=self._version(),
                registry_path=registry,
            )
        # A lock we failed to acquire is never deleted.
        assert lock.exists()

    def test_lock_released_after_success(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new")
        promote_checkpoint(
            source_path=source,
            target_path=tmp_path / "brain.pt",
            version=self._version(),
            registry_path=tmp_path / "registry.json",
        )
        assert not (tmp_path / "registry.json.lock").exists()

    def test_previous_target_backed_up(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new-weights")
        target = tmp_path / "brain.pt"
        target.write_bytes(b"old-weights")
        promote_checkpoint(
            source_path=source,
            target_path=target,
            version=self._version(),
            registry_path=tmp_path / "registry.json",
        )
        assert target.read_bytes() == b"new-weights"
        assert (tmp_path / "brain.pt.bak").read_bytes() == b"old-weights"

    def test_crash_between_file_swap_and_registry_stays_consistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Kill-injection: crash after the checkpoint swap but before the
        registry write. Nothing may be torn: the target is fully the new
        bytes, the .bak holds the old bytes, the registry is still valid
        JSON with the old content, and a retry converges both."""
        import src.training.checkpoint_promotion as cp

        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new-weights")
        target = tmp_path / "brain.pt"
        target.write_bytes(b"old-weights")
        registry = tmp_path / "registry.json"
        registry.write_text("[]")

        def crash(*args, **kwargs):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(cp, "_update_registry", crash)
        with pytest.raises(RuntimeError, match="simulated crash"):
            promote_checkpoint(
                source_path=source,
                target_path=target,
                version=self._version(),
                registry_path=registry,
            )
        # Consistent post-crash state: no torn files anywhere.
        assert target.read_bytes() == b"new-weights"
        assert (tmp_path / "brain.pt.bak").read_bytes() == b"old-weights"
        assert json.loads(registry.read_text()) == []  # valid, pre-crash content

        # Retry converges file + registry.
        monkeypatch.undo()
        promoted = promote_checkpoint(
            source_path=source,
            target_path=target,
            version=self._version(),
            registry_path=registry,
        )
        assert promoted.status == "active"
        entries = json.loads(registry.read_text())
        assert entries[-1]["status"] == "active"
        assert target.read_bytes() == b"new-weights"
        assert not (tmp_path / "registry.json.lock").exists()

    def test_corrupt_registry_backed_up_and_rebuilt(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new")
        registry = tmp_path / "registry.json"
        registry.write_text("{not valid json", encoding="utf-8")
        promoted = promote_checkpoint(
            source_path=source,
            target_path=tmp_path / "brain.pt",
            version=self._version(),
            registry_path=registry,
        )
        assert promoted.status == "active"
        entries = json.loads(registry.read_text())
        assert len(entries) == 1
        corrupt_backups = list(tmp_path.glob("registry.json.corrupt.*"))
        assert len(corrupt_backups) == 1
        assert corrupt_backups[0].read_text(encoding="utf-8") == "{not valid json"

    def test_registry_write_is_atomic_tmp_replace(self, tmp_path: Path):
        source = tmp_path / "candidate.pt"
        source.write_bytes(b"new")
        registry = tmp_path / "registry.json"
        promote_checkpoint(
            source_path=source,
            target_path=tmp_path / "brain.pt",
            version=self._version(),
            registry_path=registry,
        )
        # No tmp leftovers after a clean promotion.
        assert not list(tmp_path.glob("registry.json.tmp*"))
        assert not list(tmp_path.glob("brain.pt.promote_tmp*"))


class TestRunTrainCanonicalGate:
    """run_train stages canonical checkpoints as candidates and only
    promote_checkpoint may touch brain.pt (§4.5.4)."""

    def _config(self, tmp_path: Path) -> AppConfig:
        config = AppConfig()
        config.training.checkpoint_dir = str(tmp_path / "checkpoints")
        config.training.use_real_historical_data = False  # scorecard gate skips loudly
        return config

    def _stub_training(self, config: AppConfig, *, metrics: dict[str, float], run_id: str = "run_stub"):
        """A run_supervised_training stub that stages a candidate file the
        way isolate_checkpoint=True does."""

        def stub(cfg, *, use_baseline=False, run_id_arg=None, **kwargs):
            candidate_dir = Path(cfg.training.checkpoint_dir) / run_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate = candidate_dir / "brain.pt"
            candidate.write_bytes(b"candidate-weights")
            per_class = {
                "long": {"recall": metrics.get("long_recall", 0.30)},
                "short": {"recall": metrics.get("short_recall", 0.25)},
            }
            return {
                "checkpoint": str(candidate),
                "run_id": run_id,
                "history": SimpleNamespace(val=[SimpleNamespace(loss=metrics.get("val_loss", 0.5))]),
                "evaluation_reports": {
                    "val": {
                        "balanced_accuracy": metrics.get("balanced_accuracy", 0.55),
                        "macro_f1": metrics.get("macro_f1", 0.45),
                        "per_class_metrics": per_class,
                    }
                },
            }

        def wrapper(cfg, *, use_baseline=False, run_id=None, isolate_checkpoint=None):
            # The canonical path MUST stage isolated candidates.
            assert isolate_checkpoint is True
            return stub(cfg, use_baseline=use_baseline, run_id_arg=run_id)

        return wrapper

    def test_ungated_canonical_run_leaves_brain_untouched_and_stages_candidate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_run_train_module()
        config = self._config(tmp_path)
        canonical = Path(config.training.checkpoint_dir) / "brain.pt"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"incumbent-weights")

        # A lobotomized candidate: fails the classification floors.
        monkeypatch.setattr(
            module,
            "run_supervised_training",
            self._stub_training(config, metrics={"balanced_accuracy": 0.10, "macro_f1": 0.05}),
        )
        exit_code = module.run_canonical_training_with_gate(config, force_promote=False)
        assert exit_code == 0
        assert canonical.read_bytes() == b"incumbent-weights"  # untouched
        candidate = Path(config.training.checkpoint_dir) / "run_stub" / "brain.pt"
        assert candidate.read_bytes() == b"candidate-weights"  # staged
        assert not (Path(config.training.checkpoint_dir) / "registry.json").exists()

    def test_gated_promote_updates_file_and_registry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        module = _load_run_train_module()
        config = self._config(tmp_path)
        monkeypatch.setattr(
            module, "run_supervised_training", self._stub_training(config, metrics={})
        )
        exit_code = module.run_canonical_training_with_gate(config, force_promote=False)
        assert exit_code == 0
        canonical = Path(config.training.checkpoint_dir) / "brain.pt"
        assert canonical.read_bytes() == b"candidate-weights"
        registry = Path(config.training.checkpoint_dir) / "registry.json"
        entries = json.loads(registry.read_text())
        assert entries[-1]["status"] == "active"
        assert entries[-1]["version_id"] == "run_stub"
        assert entries[-1]["checkpoint_path"] == str(canonical)

    def test_force_promote_bypasses_gate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        module = _load_run_train_module()
        config = self._config(tmp_path)
        canonical = Path(config.training.checkpoint_dir) / "brain.pt"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(b"incumbent-weights")
        monkeypatch.setattr(
            module,
            "run_supervised_training",
            self._stub_training(config, metrics={"balanced_accuracy": 0.10, "macro_f1": 0.05}),
        )
        exit_code = module.run_canonical_training_with_gate(config, force_promote=True)
        assert exit_code == 0
        assert canonical.read_bytes() == b"candidate-weights"
        entries = json.loads((Path(config.training.checkpoint_dir) / "registry.json").read_text())
        assert "FORCED" in entries[-1]["notes"]


# ---------------------------------------------------------------------------
# Task 5: scorecard-judged promotion (stubbed BenchmarkRows)
# ---------------------------------------------------------------------------


def _benchmark_row(balanced_accuracy: float, expectancy: float, excess: float):
    return SimpleNamespace(
        balanced_accuracy=balanced_accuracy,
        expectancy=expectancy,
        excess_vs_buy_hold=excess,
    )


class TestScorecardGate:
    def _gate(self, **overrides) -> PromotionGate:
        return PromotionGate(require_improvement_over_parent=False, **overrides)

    def test_shuffled_label_candidate_refused(self):
        """A lobotomized candidate (worse balanced accuracy on the OOS
        benchmark) is refused even if classification floors pass."""
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            candidate_benchmark=_benchmark_row(0.34, 12.0, 0.02),
            incumbent_benchmark=_benchmark_row(0.52, 10.0, 0.01),
        )
        assert decision.eligible is False
        assert any("balanced_accuracy delta" in reason for reason in decision.reasons)

    def test_strictly_better_candidate_passes(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            candidate_benchmark=_benchmark_row(0.56, 14.0, 0.02),
            incumbent_benchmark=_benchmark_row(0.52, 10.0, 0.01),
        )
        assert decision.eligible is True
        assert decision.delta["benchmark.balanced_accuracy"] == pytest.approx(0.04)
        assert decision.delta["benchmark.expectancy"] == pytest.approx(4.0)

    def test_worse_expectancy_refused(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            candidate_benchmark=_benchmark_row(0.56, 8.0, 0.02),
            incumbent_benchmark=_benchmark_row(0.52, 10.0, 0.01),
        )
        assert decision.eligible is False
        assert any("expectancy delta" in reason for reason in decision.reasons)

    def test_losing_to_buy_and_hold_refused(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            candidate_benchmark=_benchmark_row(0.56, 14.0, -0.01),
            incumbent_benchmark=_benchmark_row(0.52, 10.0, 0.01),
        )
        assert decision.eligible is False
        assert any("buy-and-hold" in reason or "buy_hold" in reason for reason in decision.reasons)

    def test_no_incumbent_still_requires_beating_buy_and_hold(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            candidate_benchmark=_benchmark_row(0.56, 14.0, -0.02),
            incumbent_benchmark=None,
        )
        assert decision.eligible is False

    def test_floors_are_configurable(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(min_excess_vs_buy_hold=0.05),
            candidate_benchmark=_benchmark_row(0.56, 14.0, 0.02),
            incumbent_benchmark=_benchmark_row(0.52, 10.0, 0.01),
        )
        assert decision.eligible is False

    def test_without_benchmark_rows_classification_gates_still_apply(self):
        # Synthetic/test runs: no OOS benchmark — the gate falls back to
        # classification-only criteria (the run_train wiring logs the skip).
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
        )
        assert decision.eligible is True

    def test_promotion_config_defaults_load(self):
        config = load_config("configs/default.yaml")
        assert config.evaluation.promotion.min_balanced_accuracy_delta == 0.0
        assert config.evaluation.promotion.min_expectancy_delta == 0.0
        assert config.evaluation.promotion.min_excess_vs_buy_hold == 0.0
        assert config.evaluation.promotion.backward_transfer_floor == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Task 6: forgetting-safe retraining (KD/L2 anchors)
# ---------------------------------------------------------------------------


class _TinyBrain(torch.nn.Module):
    """Minimal model exposing the brain forward interface."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, *, branch_inputs, regime_input, full_features, temporal_tensor=None):
        return SimpleNamespace(action_logits=self.linear(full_features))


def _tiny_batches(seed: int = 0, batches: int = 8, batch_size: int = 16) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(batches):
        out.append(
            {
                "features": torch.randn(batch_size, 4, generator=generator),
                "branch_inputs": {},
                "regime_input": torch.zeros(batch_size, 1),
                "label": torch.randint(0, 3, (batch_size,), generator=generator),
            }
        )
    return out


def _make_trainer(model: torch.nn.Module, *, lr: float = 0.1, **anchor_kwargs):
    from src.training.trainer import Trainer

    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        device=torch.device("cpu"),
        use_amp=False,
        multitask_enabled=False,
        enable_future_return_targets=False,
        enable_risk_path_targets=False,
        augmentation_enabled=False,
        restore_best_on_finish=False,
        **anchor_kwargs,
    )


def _param_distance(model: torch.nn.Module, anchor_state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        total += float((parameter.detach() - anchor_state[name]).pow(2).sum())
    return total ** 0.5


def _kl_vs_anchor(model: torch.nn.Module, anchor: torch.nn.Module, batch: dict) -> float:
    with torch.no_grad():
        student = model(
            branch_inputs=batch["branch_inputs"],
            regime_input=batch["regime_input"],
            full_features=batch["features"],
        ).action_logits
        teacher = anchor(
            branch_inputs=batch["branch_inputs"],
            regime_input=batch["regime_input"],
            full_features=batch["features"],
        ).action_logits
    return float(
        torch.nn.functional.kl_div(
            torch.log_softmax(student, dim=-1),
            torch.softmax(teacher, dim=-1),
            reduction="batchmean",
        )
    )


class TestAnchoredRetraining:
    def test_l2_anchor_keeps_params_near_anchor(self):
        batches = _tiny_batches(seed=1)
        val = _tiny_batches(seed=2, batches=1)

        torch.manual_seed(7)
        free_model = _TinyBrain()
        anchor_state = {
            name: tensor.detach().clone() for name, tensor in free_model.state_dict().items()
        }
        _make_trainer(free_model).fit(batches, val, epochs=5)
        free_distance = _param_distance(free_model, anchor_state)

        torch.manual_seed(7)
        anchored_model = _TinyBrain()  # identical init
        # Large enough to dominate, small enough to stay stable with
        # lr=0.1 (SGD diverges once lr * 2 * lambda >= 1).
        _make_trainer(
            anchored_model,
            anchor_state_dict=anchor_state,
            proximal_lambda_l2=3.0,
        ).fit(batches, val, epochs=5)
        anchored_distance = _param_distance(anchored_model, anchor_state)

        assert free_distance > 0.0
        assert anchored_distance < free_distance * 0.5  # anchored stays much closer
        assert anchored_distance < 0.2  # and below an absolute bound

    def test_kd_term_decreases_kl_vs_anchor_outputs(self):
        torch.manual_seed(11)
        anchor_model = _TinyBrain()
        anchor_state = {
            name: tensor.detach().clone() for name, tensor in anchor_model.state_dict().items()
        }
        torch.manual_seed(23)
        student = _TinyBrain()  # different init than the anchor

        batches = _tiny_batches(seed=3)
        probe = _tiny_batches(seed=4, batches=1)[0]
        kl_before = _kl_vs_anchor(student, anchor_model, probe)

        trainer = _make_trainer(
            student,
            anchor_state_dict=anchor_state,
            distill_lambda_kd=1.0,
            # Isolate the KD term: zero out the classification loss.
            loss_weights={"classification": 0.0, "regression": 0.0, "future_return": 0.0, "risk_path": 0.0},
        )
        trainer.fit(batches, _tiny_batches(seed=5, batches=1), epochs=5)
        kl_after = _kl_vs_anchor(student, anchor_model, probe)

        assert kl_before > 0.0
        assert kl_after < kl_before

    def test_anchor_off_by_default(self):
        model = _TinyBrain()
        trainer = _make_trainer(model)
        assert trainer._anchor_model is None
        assert trainer._anchor_params == {}

    def test_anchor_config_fields(self):
        config = load_config("configs/default.yaml")
        assert config.training.anchor_checkpoint_path is None
        assert config.training.distill_lambda_kd == 0.0
        assert config.training.proximal_lambda_l2 == 0.0
        with pytest.raises(ConfigurationError):
            TrainingConfig(distill_lambda_kd=-0.1)
        with pytest.raises(ConfigurationError):
            TrainingConfig(proximal_lambda_l2=-0.1)


# ---------------------------------------------------------------------------
# Task 7: backward-transfer floor
# ---------------------------------------------------------------------------


class TestBackwardTransferFloor:
    def _gate(self, floor: float = 0.05) -> PromotionGate:
        return PromotionGate(require_improvement_over_parent=False, backward_transfer_floor=floor)

    def test_candidate_winning_recent_but_losing_old_slice_refused(self):
        """Wins the recent slice, loses an old slice by > floor -> refused."""
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(floor=0.05),
            slice_metrics={
                "recent_2026Q2": (0.58, 0.52),  # candidate wins recent
                "old_2024_regime": (0.42, 0.50),  # loses old by 0.08 > 0.05
            },
        )
        assert decision.eligible is False
        assert any("backward_transfer[old_2024_regime]" in reason for reason in decision.reasons)

    def test_degradation_within_floor_passes(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(floor=0.05),
            slice_metrics={
                "recent": (0.58, 0.52),
                "old": (0.47, 0.50),  # loses by 0.03 <= 0.05
            },
        )
        assert decision.eligible is True

    def test_floor_is_configurable(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(floor=0.10),
            slice_metrics={"old": (0.42, 0.50)},  # 0.08 <= 0.10
        )
        assert decision.eligible is True

    def test_no_slices_no_effect(self):
        decision = evaluate_promotion(
            candidate_metrics=dict(GOOD_DOTTED_METRICS),
            parent_metrics=None,
            gate=self._gate(),
            slice_metrics=None,
        )
        assert decision.eligible is True

    def test_config_floor_default_and_validation(self):
        assert PromotionConfig().backward_transfer_floor == pytest.approx(0.05)
        with pytest.raises(ConfigurationError):
            PromotionConfig(backward_transfer_floor=-0.01)
