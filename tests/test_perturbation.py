"""Phase 2 §6.4-A — perturbed-R² signal diagnostic."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import torch

from src.core.config import FeatureConfig, ModelConfig
from src.evaluation.perturbation import (
    PerturbationPoint,
    PerturbedSignalModel,
    _observations,
    _wrap_policy_model,
    run_perturbation_curve,
)
from src.models.factory import build_model_from_contract

_T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def _bars(n: int = 200, symbol: str = "AAPL"):
    from src.data.schemas import MarketBar

    return [
        MarketBar(
            symbol=symbol,
            timestamp=_T0 + timedelta(minutes=5 * i),
            open=100.0 + 2.0 * torch.sin(torch.tensor(i / 7.0)).item(),
            high=101.0 + 2.0 * torch.sin(torch.tensor(i / 7.0)).item(),
            low=99.0 + 2.0 * torch.sin(torch.tensor(i / 7.0)).item(),
            close=100.5 + 2.0 * torch.sin(torch.tensor(i / 7.0)).item(),
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(n)
    ]


def _snapshot_and_contract():
    from src.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))
    snapshot = pipeline.build(_bars(80), symbol="AAPL")
    return pipeline, snapshot


def _decoupled_model(contract):
    torch.manual_seed(3)
    model = build_model_from_contract(
        model_kind="brain",
        model_config=ModelConfig(
            decoupled_heads=True, return_head_horizons=[15], use_cnn_encoder=False
        ),
        feature_contract=contract,
    )
    model.eval()
    return model


class TestPerturbedSignalModel:
    def _forward(self, model, contract, batch: int = 6):
        torch.manual_seed(11)
        return model(
            branch_inputs={},
            regime_input=torch.zeros(batch, contract.regime_dim),
            full_features=torch.randn(batch, contract.feature_dim),
        )

    def test_r2_one_is_bit_identical_to_base(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        wrapped = PerturbedSignalModel(base, signal_r2=1.0, seed=7)
        torch.manual_seed(11)
        base_out = base(
            branch_inputs={},
            regime_input=torch.zeros(4, snapshot.contract.regime_dim),
            full_features=torch.randn(4, snapshot.contract.feature_dim),
        )
        wrapped_out = self._forward(wrapped, snapshot.contract, batch=4)
        assert torch.equal(base_out.action_logits, wrapped_out.action_logits)
        assert torch.equal(base_out.confidence, wrapped_out.confidence)

    def test_r2_zero_destroys_signal_but_keeps_scale(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        wrapped = PerturbedSignalModel(base, signal_r2=0.0, seed=7)
        torch.manual_seed(11)
        base_out = base(
            branch_inputs={},
            regime_input=torch.zeros(256, snapshot.contract.regime_dim),
            full_features=torch.randn(256, snapshot.contract.feature_dim),
        )
        wrapped_out = self._forward(wrapped, snapshot.contract, batch=256)
        base_z = torch.nn.functional.normalize(
            base_out.action_logits - base_out.action_logits.mean(-1, keepdim=True), dim=-1
        ).flatten()
        mixed_z = torch.nn.functional.normalize(
            wrapped_out.action_logits - wrapped_out.action_logits.mean(-1, keepdim=True), dim=-1
        ).flatten()
        corr = torch.corrcoef(torch.stack([base_z, mixed_z]))[0, 1].item()
        assert abs(corr) < 0.2  # no shared signal
        # Magnitude comparable to the base logits (softmax is shift-invariant,
        # so per-row means are irrelevant; only the spread matters).
        base_std = base_out.action_logits.std().item()
        mixed_std = wrapped_out.action_logits.std().item()
        assert 0.5 * base_std < mixed_std < 2.0 * base_std

    def test_empirical_r2_matches_requested_level(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        wrapped = PerturbedSignalModel(base, signal_r2=0.5, seed=7)
        torch.manual_seed(11)
        features = torch.randn(2048, snapshot.contract.feature_dim)
        base_out = base(
            branch_inputs={},
            regime_input=torch.zeros(2048, snapshot.contract.regime_dim),
            full_features=features,
        )
        torch.manual_seed(11)
        wrapped_out = wrapped(
            branch_inputs={},
            regime_input=torch.zeros(2048, snapshot.contract.regime_dim),
            full_features=features,
        )

        def zrows(t):
            return ((t - t.mean(-1, keepdim=True)) / t.std(-1, keepdim=True).clamp_min(1e-8)).flatten()

        corr = torch.corrcoef(torch.stack([zrows(base_out.action_logits), zrows(wrapped_out.action_logits)]))[0, 1]
        assert corr.item() ** 2 == pytest.approx(0.5, abs=0.08)

    def test_decoupled_confidence_recomputed_from_perturbed_logits(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        wrapped = PerturbedSignalModel(base, signal_r2=0.1, seed=7)
        out = self._forward(wrapped, snapshot.contract)
        expected = torch.softmax(out.action_logits, dim=-1).amax(dim=-1, keepdim=True)
        assert torch.allclose(out.confidence, expected)

    def test_same_seed_is_deterministic(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        a = PerturbedSignalModel(base, signal_r2=0.5, seed=42)
        b = PerturbedSignalModel(base, signal_r2=0.5, seed=42)
        out_a = self._forward(a, snapshot.contract)
        out_b = self._forward(b, snapshot.contract)
        assert torch.equal(out_a.action_logits, out_b.action_logits)

    def test_attribute_delegation(self) -> None:
        _, snapshot = _snapshot_and_contract()
        base = _decoupled_model(snapshot.contract)
        wrapped = PerturbedSignalModel(base, signal_r2=0.5, seed=7)
        assert wrapped.decoupled_heads is True
        assert wrapped.memory_dim == base.memory_dim
        with pytest.raises(ValueError):
            PerturbedSignalModel(base, signal_r2=1.5, seed=7)


class TestPolicyWrapping:
    def test_wraps_nested_memory_policy(self) -> None:
        from types import SimpleNamespace

        inner = SimpleNamespace(model=torch.nn.Linear(2, 2))
        outer = SimpleNamespace(base_policy=inner)
        _wrap_policy_model(outer, signal_r2=0.5, seed=1)
        assert isinstance(inner.model, PerturbedSignalModel)

    def test_refuses_policy_without_model(self) -> None:
        from types import SimpleNamespace

        with pytest.raises(TypeError):
            _wrap_policy_model(SimpleNamespace(), signal_r2=0.5, seed=1)


class TestObservations:
    def _points(self, sharpes: dict[float, float]) -> list[PerturbationPoint]:
        return [
            PerturbationPoint(signal_r2=r2, metrics={"sharpe_ratio": s})
            for r2, s in sharpes.items()
        ]

    def test_leakage_flagged(self) -> None:
        notes = _observations(self._points({1.0: 2.0, 0.5: 1.9, 0.01: 1.8}))
        assert any("LEAKAGE" in n for n in notes)

    def test_structure_flagged(self) -> None:
        notes = _observations(self._points({1.0: 0.02, 0.5: 0.01, 0.01: 0.00}))
        assert any("STRUCTURE" in n for n in notes)

    def test_healthy_curve(self) -> None:
        notes = _observations(self._points({1.0: 1.5, 0.5: 0.7, 0.01: 0.05}))
        assert notes == ["curve shape consistent with signal-driven returns"]


def test_curve_runs_and_r2_one_matches_unperturbed_backtest() -> None:
    """End-to-end: the harness produces one point per level over identical
    bars, and the r2=1.0 point reproduces an unperturbed run exactly."""
    from src.backtest.engine import BacktestEngine
    from src.core.config import AppConfig
    from src.core.seeds import seed_everything
    from src.features.feature_pipeline import FeaturePipeline
    from src.risk.risk_manager import RiskManager
    from src.strategies.brain_policy import BrainPolicy

    _, snapshot = _snapshot_and_contract()
    model = _decoupled_model(snapshot.contract)
    config = AppConfig()
    config.backtest.fill_timing = "next_open"
    bars = {"AAPL": _bars(200)}

    def builder(arm_config, arm_bars, pipeline):
        return BrainPolicy(
            model=model, device=torch.device("cpu"), feature_contract=snapshot.contract
        )

    report = run_perturbation_curve(
        config, bars, builder, r2_levels=(1.0, 0.0), seed=99
    )
    assert [p.signal_r2 for p in report.points] == [1.0, 0.0]
    assert report.observations  # advisory notes always emitted

    seed_everything(99)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    plain = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy())).run(
        bars, builder(config, bars, pipeline)
    )
    full_signal_point = report.points[0]
    assert full_signal_point.metrics["total_return"] == pytest.approx(
        float(plain.metrics.get("total_return", 0.0)), abs=1e-12
    )
    assert full_signal_point.metrics["fill_count"] == pytest.approx(
        float(plain.metrics.get("fill_count", 0.0))
    )
