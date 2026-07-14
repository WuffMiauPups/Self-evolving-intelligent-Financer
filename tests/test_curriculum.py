"""Anchored negatives + adaptive curriculum tests (MASTER_PLAN §6.5, roadmap §4.8-9).

Covers:
  - pair validity: economic-margin boundary math (round-trip cost + noise
    buffer converted bps-against-notional)
  - anchored selection: worst executed losers beyond the margin; noise-sized
    losses and FLAT decisions excluded
  - AnchoredNegativeStore: dedupe, permanence across retrains, bounding,
    torn-line tolerance
  - sample difficulty: extremes decay to 0, band edge is the maximum
  - CurriculumSampler: epoch progression, determinism, set_epoch composition
  - trainer wiring: fit() calls sampler.set_epoch per epoch
  - dataset seam: curriculum_enabled swaps the train DataLoader sampler
  - retrain wiring: run_triggered_retrain appends to the permanent store
"""
from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.core.config import AppConfig, TrainingConfig
from src.core.exceptions import ConfigurationError
from src.training.curriculum import (
    AnchoredNegativeStore,
    CurriculumSampler,
    compute_sample_difficulties,
    pair_is_valid,
    sample_difficulty,
    select_anchored_negatives,
)
from src.training.replay_buffer import DecisionObservation
from src.training.retrain import run_triggered_retrain


def _observation(
    index: int,
    pnl: float,
    *,
    symbol: str = "AAPL",
    action: int = 1,
) -> DecisionObservation:
    return DecisionObservation(
        decision_id=f"dec-{index:05d}",
        symbol=symbol,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        features_ref="feat_fp_abc123",
        executed_action=action,
        realized_pnl=pnl,
        clamped=False,
    )


def _drifting_observations(count: int = 240) -> list[DecisionObservation]:
    """Older half wins, newer half loses — fires the win-rate trigger."""
    half = count // 2
    older = [_observation(i, 1.0 + 0.01 * (i % 5)) for i in range(half)]
    newer = [_observation(half + i, -1.0 - 0.01 * (i % 5)) for i in range(count - half)]
    return older + newer


# ---------------------------------------------------------------------------
# Economic-margin pair validity
# ---------------------------------------------------------------------------


class TestPairValidity:
    def test_gap_above_margin_is_valid(self):
        # 10 bps cost + 5 bps buffer on 10_000 notional -> margin = 15.0
        assert pair_is_valid(20.0, 0.0, cost_bps=10.0, noise_buffer_bps=5.0, notional=10_000.0)

    def test_gap_exactly_at_margin_is_invalid(self):
        # The gap must EXCEED the margin, not merely reach it.
        assert not pair_is_valid(15.0, 0.0, cost_bps=10.0, noise_buffer_bps=5.0, notional=10_000.0)

    def test_gap_below_margin_is_invalid(self):
        assert not pair_is_valid(14.9, 0.0, cost_bps=10.0, noise_buffer_bps=5.0, notional=10_000.0)

    def test_bps_conversion_against_notional(self):
        # Same bps figures against unit notional -> margin 0.0015.
        assert pair_is_valid(0.0016, 0.0, cost_bps=10.0, noise_buffer_bps=5.0, notional=1.0)
        assert not pair_is_valid(0.0014, 0.0, cost_bps=10.0, noise_buffer_bps=5.0, notional=1.0)

    def test_symmetric_in_pair_order(self):
        kwargs = dict(cost_bps=10.0, noise_buffer_bps=5.0, notional=1.0)
        assert pair_is_valid(0.002, 0.0, **kwargs) == pair_is_valid(0.0, 0.002, **kwargs)

    def test_zero_margin_requires_any_nonzero_gap(self):
        assert pair_is_valid(0.0001, 0.0, cost_bps=0.0, noise_buffer_bps=0.0, notional=1.0)
        assert not pair_is_valid(0.5, 0.5, cost_bps=0.0, noise_buffer_bps=0.0, notional=1.0)


# ---------------------------------------------------------------------------
# Anchored-negative selection
# ---------------------------------------------------------------------------


class TestAnchoredSelection:
    # cost 10 bps + buffer 5 bps -> margin 0.0015 in return-fraction units
    KW = dict(cost_bps=10.0, noise_buffer_bps=5.0, max_count=10)

    def test_picks_worst_executed_losers_beyond_margin(self):
        observations = [
            _observation(0, 0.02),            # winner
            _observation(1, -0.005),          # genuine mistake
            _observation(2, -0.030),          # worst mistake
            _observation(3, -0.001),          # noise-sized loss (< margin)
            _observation(4, -0.010, action=-1),  # short-side mistake
        ]
        anchored = select_anchored_negatives(observations, **self.KW)
        assert [o.decision_id for o in anchored] == ["dec-00002", "dec-00004", "dec-00001"]

    def test_excludes_noise_sized_losses(self):
        observations = [_observation(0, -0.0014), _observation(1, -0.0010)]
        assert select_anchored_negatives(observations, **self.KW) == []

    def test_loss_exactly_at_margin_excluded(self):
        assert select_anchored_negatives([_observation(0, -0.0015)], **self.KW) == []
        assert len(select_anchored_negatives([_observation(0, -0.0016)], **self.KW)) == 1

    def test_excludes_flat_decisions(self):
        observations = [_observation(0, -0.05, action=0), _observation(1, -0.05)]
        anchored = select_anchored_negatives(observations, **self.KW)
        assert [o.decision_id for o in anchored] == ["dec-00001"]

    def test_excludes_winners(self):
        assert select_anchored_negatives([_observation(0, 0.05)], **self.KW) == []

    def test_max_count_truncates_to_worst(self):
        observations = [_observation(i, -0.01 * (i + 1)) for i in range(8)]
        anchored = select_anchored_negatives(
            observations, cost_bps=10.0, noise_buffer_bps=5.0, max_count=3
        )
        assert [o.decision_id for o in anchored] == ["dec-00007", "dec-00006", "dec-00005"]

    def test_max_count_zero_returns_empty(self):
        assert select_anchored_negatives(
            [_observation(0, -1.0)], cost_bps=0.0, noise_buffer_bps=0.0, max_count=0
        ) == []


# ---------------------------------------------------------------------------
# AnchoredNegativeStore: dedupe / permanence / bounding / torn lines
# ---------------------------------------------------------------------------


class TestAnchoredNegativeStore:
    def test_roundtrip(self, tmp_path: Path):
        store = AnchoredNegativeStore(tmp_path / "anchored.jsonl")
        assert store.append_new([_observation(0, -0.5), _observation(1, -0.7)]) == 2
        loaded = store.load()
        assert [o.decision_id for o in loaded] == ["dec-00000", "dec-00001"]
        assert loaded[1].realized_pnl == pytest.approx(-0.7)

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        assert AnchoredNegativeStore(tmp_path / "nope.jsonl").load() == []

    def test_dedupes_by_decision_id_across_calls(self, tmp_path: Path):
        store = AnchoredNegativeStore(tmp_path / "anchored.jsonl")
        assert store.append_new([_observation(0, -0.5)]) == 1
        assert store.append_new([_observation(0, -0.5), _observation(1, -0.6)]) == 1
        assert [o.decision_id for o in store.load()] == ["dec-00000", "dec-00001"]

    def test_dedupes_within_one_batch(self, tmp_path: Path):
        store = AnchoredNegativeStore(tmp_path / "anchored.jsonl")
        assert store.append_new([_observation(0, -0.5), _observation(0, -0.5)]) == 1
        assert len(store.load()) == 1

    def test_permanence_across_retrains(self, tmp_path: Path):
        """A fresh store instance (a later retrain cycle) sees everything
        earlier cycles anchored — the file is the permanence mechanism."""
        path = tmp_path / "anchored.jsonl"
        AnchoredNegativeStore(path).append_new([_observation(0, -0.5)])
        second_cycle = AnchoredNegativeStore(path)
        second_cycle.append_new([_observation(1, -0.9)])
        assert [o.decision_id for o in second_cycle.load()] == ["dec-00000", "dec-00001"]

    def test_bounded_drops_oldest(self, tmp_path: Path):
        store = AnchoredNegativeStore(tmp_path / "anchored.jsonl", max_anchored=5)
        store.append_new([_observation(i, -0.5) for i in range(3)])
        store.append_new([_observation(3 + i, -0.5) for i in range(5)])
        loaded = store.load()
        assert len(loaded) == 5
        assert [o.decision_id for o in loaded] == [f"dec-{i:05d}" for i in range(3, 8)]

    def test_torn_lines_skipped_on_load(self, tmp_path: Path):
        path = tmp_path / "anchored.jsonl"
        store = AnchoredNegativeStore(path)
        store.append_new([_observation(0, -0.5)])
        with path.open("a", encoding="utf-8") as fp:
            fp.write("{torn line\n")
        store.append_new([_observation(1, -0.6)])
        loaded = store.load()
        assert [o.decision_id for o in loaded] == ["dec-00000", "dec-00001"]

    def test_invalid_max_anchored_rejected(self, tmp_path: Path):
        with pytest.raises(ValueError):
            AnchoredNegativeStore(tmp_path / "anchored.jsonl", max_anchored=0)


# ---------------------------------------------------------------------------
# Sample difficulty
# ---------------------------------------------------------------------------


class TestSampleDifficulty:
    EPS = 0.001

    def test_band_edge_is_maximally_ambiguous(self):
        assert sample_difficulty(self.EPS, self.EPS) == pytest.approx(1.0)
        assert sample_difficulty(-self.EPS, self.EPS) == pytest.approx(1.0)

    def test_far_above_band_is_unambiguous(self):
        # |r| = 1000x the band edge -> difficulty = 1/1000
        assert sample_difficulty(1.0, self.EPS) == pytest.approx(0.001)

    def test_deep_inside_flat_band_is_unambiguous(self):
        assert sample_difficulty(0.0, self.EPS) < 1.0e-8
        assert sample_difficulty(self.EPS * 1.0e-3, self.EPS) == pytest.approx(0.001)

    def test_log_symmetric_around_band_edge(self):
        # 2x above the edge and 2x inside it are equally ambiguous.
        assert sample_difficulty(2 * self.EPS, self.EPS) == pytest.approx(0.5)
        assert sample_difficulty(0.5 * self.EPS, self.EPS) == pytest.approx(0.5)

    def test_sign_symmetric(self):
        assert sample_difficulty(-0.0007, self.EPS) == pytest.approx(
            sample_difficulty(0.0007, self.EPS)
        )

    def test_monotone_toward_band_edge_from_both_sides(self):
        below = [sample_difficulty(r, self.EPS) for r in (0.0002, 0.0005, 0.0008)]
        assert below == sorted(below)
        above = [sample_difficulty(r, self.EPS) for r in (0.0012, 0.002, 0.01)]
        assert above == sorted(above, reverse=True)

    def test_bounded_zero_one(self):
        for magnitude in (0.0, 1.0e-9, 0.0005, self.EPS, 0.01, 100.0):
            difficulty = sample_difficulty(magnitude, self.EPS)
            assert 0.0 <= difficulty <= 1.0

    def test_non_positive_epsilon_means_no_band(self):
        assert sample_difficulty(0.5, 0.0) == 0.0
        assert sample_difficulty(0.5, -1.0) == 0.0

    def test_vectorized_matches_scalar(self):
        returns = [-0.01, -0.001, 0.0, 0.0005, 0.001, 0.02]
        vector = compute_sample_difficulties(torch.tensor(returns), self.EPS)
        for value, expected_input in zip(vector.tolist(), returns):
            assert value == pytest.approx(
                sample_difficulty(expected_input, self.EPS), rel=1.0e-5
            )


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


class TestCurriculumSampler:
    def _difficulties(self, n: int = 20) -> torch.Tensor:
        # Strictly increasing: index i is harder than index i-1.
        return torch.linspace(0.05, 1.0, n)

    def test_early_epoch_excludes_hard_samples(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        sampler.set_epoch(0)
        yielded = list(sampler)
        assert len(yielded) == 10
        assert set(yielded) == set(range(10))  # only the 10 easiest indices
        assert 19 not in yielded  # the hardest sample is excluded

    def test_final_epoch_includes_all(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        sampler.set_epoch(2)
        assert sorted(sampler) == list(range(20))
        sampler.set_epoch(9)  # ramp stays at 1.0 past full_epoch
        assert sorted(sampler) == list(range(20))

    def test_threshold_ramps_between_start_and_full(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        counts = []
        for epoch in (0, 1, 2):
            sampler.set_epoch(epoch)
            counts.append(len(sampler))
        assert counts[0] < counts[1] < counts[2]
        assert counts[2] == 20

    def test_len_reflects_included_subset(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        sampler.set_epoch(0)
        assert len(sampler) == len(list(sampler)) == 10

    def test_deterministic_across_same_seed_runs(self):
        a = CurriculumSampler(self._difficulties(), start_quantile=0.5, full_epoch=3, seed=11)
        b = CurriculumSampler(self._difficulties(), start_quantile=0.5, full_epoch=3, seed=11)
        for epoch in range(4):
            a.set_epoch(epoch)
            b.set_epoch(epoch)
            assert list(a) == list(b)

    def test_set_epoch_changes_composition_and_order(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        assert set(first) != set(second)  # ramp admitted new samples
        assert set(first) < set(second)  # ...without dropping easy ones

    def test_permutation_differs_between_epochs_at_full_inclusion(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=1.0, full_epoch=1, seed=7
        )
        sampler.set_epoch(0)
        first = list(sampler)
        sampler.set_epoch(1)
        second = list(sampler)
        assert sorted(first) == sorted(second) == list(range(20))
        assert first != second  # seeded with seed + epoch

    def test_start_quantile_one_includes_everything_immediately(self):
        sampler = CurriculumSampler(self._difficulties(), start_quantile=1.0, full_epoch=3)
        sampler.set_epoch(0)
        assert len(sampler) == 20

    def test_works_with_dataloader(self):
        sampler = CurriculumSampler(
            self._difficulties(), start_quantile=0.5, full_epoch=2, seed=7
        )
        loader = torch.utils.data.DataLoader(list(range(20)), batch_size=4, sampler=sampler)
        sampler.set_epoch(0)
        seen = [int(i) for batch in loader for i in batch]
        assert sorted(seen) == list(range(10))
        sampler.set_epoch(2)
        seen = [int(i) for batch in loader for i in batch]
        assert sorted(seen) == list(range(20))

    def test_invalid_parameters_rejected(self):
        with pytest.raises(ValueError):
            CurriculumSampler(self._difficulties(), start_quantile=0.0)
        with pytest.raises(ValueError):
            CurriculumSampler(self._difficulties(), start_quantile=1.5)
        with pytest.raises(ValueError):
            CurriculumSampler(self._difficulties(), full_epoch=0)
        with pytest.raises(ValueError):
            CurriculumSampler(torch.empty(0))


# ---------------------------------------------------------------------------
# Config fields
# ---------------------------------------------------------------------------


class TestCurriculumConfig:
    def test_defaults_off_and_neutral(self):
        config = TrainingConfig()
        assert config.curriculum_enabled is False
        assert config.curriculum_start_quantile == pytest.approx(0.5)
        assert config.curriculum_full_epoch == 3
        assert config.anchored_negatives_enabled is False
        assert config.anchored_negatives_path == "runs/anchored_negatives.jsonl"
        assert config.anchored_negatives_max == 5000
        assert config.anchored_noise_buffer_bps == pytest.approx(5.0)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"curriculum_start_quantile": 0.0},
            {"curriculum_start_quantile": 1.5},
            {"curriculum_full_epoch": 0},
            {"anchored_negatives_max": 0},
            {"anchored_noise_buffer_bps": -1.0},
        ],
    )
    def test_invalid_values_rejected(self, overrides: dict):
        with pytest.raises(ConfigurationError):
            TrainingConfig(**overrides)


# ---------------------------------------------------------------------------
# Trainer wiring: fit() advances epoch-aware samplers
# ---------------------------------------------------------------------------


class _TinyBrain(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, *, branch_inputs, regime_input, full_features, temporal_tensor=None):
        return SimpleNamespace(action_logits=self.linear(full_features))


def _tiny_batches(seed: int = 0, batches: int = 2, batch_size: int = 8) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "features": torch.randn(batch_size, 4, generator=generator),
            "branch_inputs": {},
            "regime_input": torch.zeros(batch_size, 1),
            "label": torch.randint(0, 3, (batch_size,), generator=generator),
        }
        for _ in range(batches)
    ]


def _make_trainer(model: torch.nn.Module):
    from src.training.trainer import Trainer

    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        device=torch.device("cpu"),
        use_amp=False,
        multitask_enabled=False,
        enable_future_return_targets=False,
        enable_risk_path_targets=False,
        augmentation_enabled=False,
        restore_best_on_finish=False,
    )


class _EpochRecordingSampler:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _LoaderWithSampler(list):
    """Minimal stand-in for a DataLoader: iterable batches + .sampler."""

    def __init__(self, batches: list[dict], sampler) -> None:
        super().__init__(batches)
        self.sampler = sampler


class TestTrainerSetEpochWiring:
    def test_fit_calls_set_epoch_each_epoch(self):
        sampler = _EpochRecordingSampler()
        loader = _LoaderWithSampler(_tiny_batches(seed=1), sampler)
        _make_trainer(_TinyBrain()).fit(loader, _tiny_batches(seed=2, batches=1), epochs=3)
        assert sampler.epochs == [0, 1, 2]

    def test_fit_tolerates_loaders_without_sampler(self):
        # Plain batch lists (used throughout the suite) have no .sampler.
        history = _make_trainer(_TinyBrain()).fit(
            _tiny_batches(seed=1), _tiny_batches(seed=2, batches=1), epochs=2
        )
        assert len(history.train) == 2


# ---------------------------------------------------------------------------
# Dataset seam: _finalize_training_bundle swaps in the CurriculumSampler
# ---------------------------------------------------------------------------


def _columnar(n: int, *, horizon: int, epsilon: float) -> "object":
    from src.training.sample_columns import ColumnarSamples

    generator = torch.Generator().manual_seed(3)
    # Returns straddling the band edge so difficulties are non-degenerate.
    returns = torch.linspace(-4.0 * epsilon, 4.0 * epsilon, n)
    labels = torch.where(
        returns > epsilon,
        torch.tensor(2),
        torch.where(returns < -epsilon, torch.tensor(0), torch.tensor(1)),
    ).to(torch.long)
    base = datetime(2026, 1, 1, tzinfo=UTC)
    return ColumnarSamples(
        features=torch.randn(n, 4, generator=generator),
        label=labels,
        regime_input=torch.randn(n, 2, generator=generator),
        temporal_tensor=torch.randn(n, 2, 8, generator=generator),
        branch_inputs={"trend": torch.randn(n, 3, generator=generator)},
        path_label=torch.zeros(n, dtype=torch.long),
        regression_targets={"primary_return": returns.clone()},
        future_returns={horizon: returns.clone()},
        symbols=["AAPL"] * n,
        timestamps=[base + timedelta(minutes=i) for i in range(n)],
        direction_names=[("short", "flat", "long")[int(l)] for l in labels],
        path_names=["chop"] * n,
    )


class TestDatasetSeam:
    def _bundle(self, *, curriculum_enabled: bool):
        from src.core.logging import get_logger
        from src.training.data_module import _finalize_training_bundle

        config = AppConfig()
        config.training.curriculum_enabled = curriculum_enabled
        config.training.batch_size = 8
        horizon = config.training.effective_primary_horizon
        epsilon = config.training.label_epsilon * math.sqrt(max(horizon, 1)) + (
            config.training.transaction_cost_bps / 10_000.0
        )
        samples_by_split = {
            "train": _columnar(40, horizon=horizon, epsilon=epsilon),
            "val": _columnar(12, horizon=horizon, epsilon=epsilon),
            "test": _columnar(12, horizon=horizon, epsilon=epsilon),
        }
        return _finalize_training_bundle(
            config=config,
            feature_contract=SimpleNamespace(),
            samples_by_split=samples_by_split,
            split_ranges={"train": {}, "val": {}, "test": {}},
            logger=get_logger("test_curriculum"),
        )

    def test_curriculum_enabled_installs_sampler(self):
        bundle = self._bundle(curriculum_enabled=True)
        sampler = bundle.train_loader.sampler
        assert isinstance(sampler, CurriculumSampler)
        sampler.set_epoch(0)
        early_count = len(sampler)
        sampler.set_epoch(sampler.full_epoch)
        assert early_count < len(sampler) == 40

    def test_curriculum_disabled_keeps_shuffled_loader(self):
        bundle = self._bundle(curriculum_enabled=False)
        assert not isinstance(bundle.train_loader.sampler, CurriculumSampler)


# ---------------------------------------------------------------------------
# Retrain wiring: the triggered retrain feeds the permanent store
# ---------------------------------------------------------------------------


THRESHOLDS = {"sharpe_drop": 0.5, "win_rate_drop": 0.05, "min_observations": 200}


class TestRetrainAnchoredWiring:
    def _training_config(self, tmp_path: Path, *, enabled: bool = True, max_anchored: int = 50):
        return TrainingConfig(
            anchored_negatives_enabled=enabled,
            anchored_negatives_path=str(tmp_path / "anchored.jsonl"),
            anchored_negatives_max=max_anchored,
            transaction_cost_bps=1.0,
            anchored_noise_buffer_bps=5.0,
        )

    def _train_fn(self, tmp_path: Path):
        checkpoint_path = tmp_path / "retrained.pt"

        def train_fn():
            checkpoint_path.write_bytes(b"retrained-weights")
            return {
                "checkpoint": str(checkpoint_path),
                "evaluation_reports": {"val": {"balanced_accuracy": 0.52}},
            }

        return train_fn

    def test_triggered_retrain_appends_to_store(self, tmp_path: Path):
        training_config = self._training_config(tmp_path)
        meta = run_triggered_retrain(
            observations=_drifting_observations(240),
            train_fn=self._train_fn(tmp_path),
            thresholds=THRESHOLDS,
            runs_dir=str(tmp_path / "runs"),
            training_config=training_config,
        )
        assert meta is not None
        store = AnchoredNegativeStore(
            training_config.anchored_negatives_path,
            max_anchored=training_config.anchored_negatives_max,
        )
        anchored = store.load()
        # 120 beyond-margin losers, capped at anchored_negatives_max.
        assert len(anchored) == 50
        assert all(o.realized_pnl < 0 for o in anchored)
        assert all(o.executed_action != 0 for o in anchored)

    def test_rerun_dedupes_against_prior_cycle(self, tmp_path: Path):
        training_config = self._training_config(tmp_path, max_anchored=500)
        for _ in range(2):
            run_triggered_retrain(
                observations=_drifting_observations(240),
                train_fn=self._train_fn(tmp_path),
                thresholds=THRESHOLDS,
                runs_dir=str(tmp_path / "runs"),
                training_config=training_config,
            )
        anchored = AnchoredNegativeStore(
            training_config.anchored_negatives_path, max_anchored=500
        ).load()
        assert len(anchored) == 120  # same decision_ids, anchored once

    def test_disabled_flag_leaves_store_untouched(self, tmp_path: Path):
        training_config = self._training_config(tmp_path, enabled=False)
        meta = run_triggered_retrain(
            observations=_drifting_observations(240),
            train_fn=self._train_fn(tmp_path),
            thresholds=THRESHOLDS,
            runs_dir=str(tmp_path / "runs"),
            training_config=training_config,
        )
        assert meta is not None
        assert not Path(training_config.anchored_negatives_path).exists()

    def test_no_training_config_is_backwards_compatible(self, tmp_path: Path):
        meta = run_triggered_retrain(
            observations=_drifting_observations(240),
            train_fn=self._train_fn(tmp_path),
            thresholds=THRESHOLDS,
            runs_dir=str(tmp_path / "runs"),
        )
        assert meta is not None

    def test_untriggered_retrain_anchors_nothing(self, tmp_path: Path):
        training_config = self._training_config(tmp_path)
        stable = [_observation(i, 1.0 if i % 2 == 0 else -0.9) for i in range(240)]
        meta = run_triggered_retrain(
            observations=stable,
            train_fn=self._train_fn(tmp_path),
            thresholds=THRESHOLDS,
            runs_dir=str(tmp_path / "runs"),
            training_config=training_config,
        )
        assert meta is None
        assert not Path(training_config.anchored_negatives_path).exists()
