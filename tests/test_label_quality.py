"""Label-quality and training-accuracy tests (plan §4.1).

Pins: (1) the flat band scales with sqrt(horizon) and includes the
round-trip cost floor, (2) direction loss is magnitude-invariant,
(3) best-epoch selection uses the configured metric, (4) the retired
force_directional_flip flag has exactly one config source consumed
identically by backtest and runtime construction.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.core.config import load_config
from src.memory.schemas import OutcomeSummary
from src.training.losses import direction_classification_loss
from src.training.targets import build_training_targets


def _outcome(primary_return: float, horizon: int = 6) -> OutcomeSummary:
    return OutcomeSummary(
        future_returns={1: primary_return / 2.0, horizon: primary_return},
        max_favorable_excursion=abs(primary_return),
        max_adverse_excursion=-abs(primary_return) / 2.0,
        realized_volatility=0.001,
        drawdown=-0.001,
        path_label="chop",
        direction_class=0,
    )


def _label(primary_return: float, *, epsilon: float, cost_bps: float, horizon: int = 6) -> str:
    targets = build_training_targets(
        _outcome(primary_return, horizon=horizon),
        label_epsilon=epsilon,
        regression_targets=["primary_return"],
        primary_horizon=horizon,
        transaction_cost_bps=cost_bps,
    )
    return targets.direction_name


# ── Cost floor ──────────────────────────────────────────────────────────


def test_move_smaller_than_round_trip_cost_labels_flat() -> None:
    # cost = 10 bps -> 0.001; +8 bps move can never be worth trading.
    assert _label(0.0008, epsilon=0.0, cost_bps=10.0) == "flat"
    assert _label(0.0012, epsilon=0.0, cost_bps=10.0) == "long"
    assert _label(-0.0012, epsilon=0.0, cost_bps=10.0) == "short"


# ── sqrt(horizon) scaling ───────────────────────────────────────────────


def test_flat_band_scales_with_sqrt_horizon() -> None:
    # epsilon is a 1-bar scale; at H=6 the band is epsilon*sqrt(6) ≈ 2.449e-3.
    eps = 1.0e-3
    assert _label(0.002, epsilon=eps, cost_bps=0.0, horizon=6) == "flat"  # < eps*sqrt(6)
    assert _label(0.003, epsilon=eps, cost_bps=0.0, horizon=6) == "long"  # > eps*sqrt(6)
    # At H=1 the same 0.002 move IS directional (band = eps).
    assert _label(0.002, epsilon=eps, cost_bps=0.0, horizon=1) == "long"


# ── Direction loss magnitude invariance ────────────────────────────────


def test_direction_loss_is_magnitude_invariant() -> None:
    """Two equally-correct samples must contribute equal direction loss
    regardless of their outcome magnitude — the loss takes no return
    input at all, and batch reduction is a plain mean."""
    logits = torch.tensor([[0.2, 0.1, 1.5], [0.2, 0.1, 1.5]])
    labels = torch.tensor([2, 2])  # both long, one from a 10x larger move
    batch_loss = direction_classification_loss(logits, labels)
    single_loss = direction_classification_loss(logits[:1], labels[:1])
    torch.testing.assert_close(batch_loss, single_loss)


# ── Model selection metric ─────────────────────────────────────────────


def test_selection_score_prefers_configured_metric() -> None:
    from src.training.trainer import Trainer

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    make = lambda metric: Trainer(  # noqa: E731
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        use_amp=False,
        model_selection_metric=metric,
    )
    low_loss_flat_hedger = SimpleNamespace(loss=0.5, balanced_accuracy=0.34, macro_f1=0.2)
    higher_loss_but_accurate = SimpleNamespace(loss=0.9, balanced_accuracy=0.52, macro_f1=0.5)

    by_acc = make("val_balanced_accuracy")
    assert by_acc._selection_score(higher_loss_but_accurate) > by_acc._selection_score(low_loss_flat_hedger)

    by_loss = make("val_loss")
    assert by_loss._selection_score(low_loss_flat_hedger) > by_loss._selection_score(higher_loss_but_accurate)


def test_model_selection_metric_validated() -> None:
    from src.core.config import TrainingConfig
    from src.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        TrainingConfig(model_selection_metric="vibes")


# ── force_directional_flip: one source, both consumers ─────────────────


def test_flip_flag_default_off_and_single_source() -> None:
    config = load_config("configs/default.yaml")
    assert config.runtime.force_directional_flip is False
    assert not hasattr(config.backtest, "force_directional_flip_enabled")

    # Source contract: both construction paths consume the SAME config
    # field. Crude but effective — a divergence (one path reading a
    # different knob) is exactly the backtest/live parity bug this
    # retires, and it cannot survive this check.
    root = Path(__file__).resolve().parents[1]
    expected = "force_directional_flip=config.runtime.force_directional_flip"
    for consumer in ("src/backtest/policy_factory.py", "src/execution/runtime_bootstrap.py"):
        source = (root / consumer).read_text(encoding="utf-8")
        assert expected in source, f"{consumer} does not consume runtime.force_directional_flip"
    assert "force_directional_flip_enabled" not in (root / "src/backtest/policy_factory.py").read_text(
        encoding="utf-8"
    )


def test_default_training_config_single_imbalance_mechanism() -> None:
    config = load_config("configs/default.yaml")
    active = [
        config.training.class_weight_mode != "none",
        config.training.use_weighted_sampler,
        config.training.classification_loss == "focal",
    ]
    assert sum(active) == 1, (
        "exactly one class-imbalance correction may be active by default; "
        f"got weights={active[0]} sampler={active[1]} focal={active[2]}"
    )
    assert config.training.model_selection_metric == "val_balanced_accuracy"
    assert config.training.transaction_cost_bps > 0.0


def test_label_cost_matches_backtest_costs() -> None:
    """§4.4.2: labels and fills price the same economic reality.

    The FLAT-band cost floor must equal the round trip a backtested fill
    actually pays (fee + slippage per side, spread excluded because it is
    per-bar data, not config), and the EV gate's risk costs must be the
    same numbers the backtest charges. One reality, three consumers.
    """
    config = load_config("configs/default.yaml")
    round_trip_bps = 2.0 * (config.backtest.fee_bps + config.backtest.slippage_bps)
    assert config.training.transaction_cost_bps == round_trip_bps
    assert config.risk.fee_bps == config.backtest.fee_bps
    assert config.risk.slippage_bps == config.backtest.slippage_bps
