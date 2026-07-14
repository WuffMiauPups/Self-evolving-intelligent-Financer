"""Strategy arbitration layer tests (MASTER_PLAN §6.3).

The contract these tests pin: every registered strategy proposes on every
snapshot; the highest-weight directional candidate wins and is scaled by
its budget fraction (the SOFT cap); realized fill PnL shifts the
outcome-weighted allocation, but the weight floor means a losing strategy
is throttled, never starved; ``strategy_id`` threads honestly from
proposal through OrderIntent to FillRecord; and the arbiter delegates the
brain's memory surface so memory keeps learning under arbitration.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
import torch

from src.backtest.engine import BacktestEngine
from src.core.config import ArbitrationConfig, load_config
from src.core.exceptions import ConfigurationError
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.risk.constraints import TradeProposal
from src.risk.ledger import FillRecord
from src.risk.risk_manager import RiskManager
from src.strategies.arbiter import StrategyArbiter, build_arbiter
from src.strategies.base import BatchTradingPolicy, MemoryCapablePolicy
from src.strategies.memory_policy import MemoryAwarePolicy

T0 = datetime(2026, 7, 7, 15, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# fixtures: deterministic stub strategies (EV comfortably clears the gate:
# 0.9*30 - 0.1*10 - cost ≈ 22 bps > 0, mirroring tests/test_risk_correctness)
# ---------------------------------------------------------------------------


class _StubStrategy:
    """Fixed-action primitive: only implements ``propose``."""

    def __init__(self, action: int, target_fraction: float = 0.02, confidence: float = 0.9):
        self._action = action
        self._fraction = target_fraction
        self._confidence = confidence

    def propose(self, snapshot) -> tuple[TradeProposal, dict[str, object]]:
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=self._action,
            confidence=self._confidence,
            target_fraction=self._fraction,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id=f"stub-{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
            expected_gain_bps=30.0,
            expected_loss_bps=10.0,
        )
        return proposal, {}


class _StubBrain(_StubStrategy):
    """Stub brain: adds propose_batch + state resets (BatchTradingPolicy)."""

    def __init__(self, action: int = 0, target_fraction: float = 0.0, confidence: float = 0.9):
        super().__init__(action, target_fraction, confidence)
        self.batch_calls = 0

    def propose_batch(self, snapshots) -> list[tuple[TradeProposal, dict[str, object]]]:
        self.batch_calls += 1
        return [self.propose(snapshot) for snapshot in snapshots]

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        return reason

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        return reason


def _snapshot(symbol: str = "AAPL") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, timestamp=T0, last_price=100.0, spread_bps=2.0)


def _arbiter(
    *,
    brain: object | None = None,
    primitives: list[tuple[str, object]] | None = None,
    budgets: dict[str, float] | None = None,
) -> StrategyArbiter:
    return StrategyArbiter(
        brain=brain if brain is not None else _StubBrain(),
        primitives=primitives if primitives is not None else [],
        budget_fractions=budgets
        or {"brain": 1.0, "momentum": 0.5, "mean_reversion": 0.5},
        ema_alpha=0.1,
        weight_floor=0.05,
    )


# ---------------------------------------------------------------------------
# strategy_id threading: proposal -> intent -> fill through a real backtest
# ---------------------------------------------------------------------------


def test_strategy_id_threads_proposal_to_intent_to_fill() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.symbols = ["AAPL"]
    config.runtime.warmup_bars = 32
    config.memory.enabled = False
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    engine = BacktestEngine(config, pipeline, RiskManager(config.risk))
    # Flat brain, always-long momentum stub: every fill must credit momentum.
    arbiter = _arbiter(
        primitives=[("momentum", _StubStrategy(action=1, target_fraction=0.02))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    report = engine.run({"AAPL": generate_synthetic_bars("AAPL", 180)}, arbiter)
    assert report.fills, "arbitrated backtest produced no fills"
    assert {fill.strategy_id for fill in report.fills} == {"momentum"}
    # The engine wired realized outcomes back into the arbiter's EMA.
    assert arbiter.pnl_ema["momentum"] != 0.0


# ---------------------------------------------------------------------------
# arbitration semantics
# ---------------------------------------------------------------------------


def test_highest_weight_directional_candidate_wins_with_budget_scaling() -> None:
    arbiter = _arbiter(
        primitives=[
            ("momentum", _StubStrategy(action=1, target_fraction=0.10)),
            ("mean_reversion", _StubStrategy(action=-1, target_fraction=-0.10)),
        ],
    )
    # Reward mean_reversion so its weight rises above momentum's.
    arbiter.record_fill_outcome("mean_reversion", 100.0)
    proposal, diagnostics = arbiter.propose(_snapshot())
    assert proposal.strategy_id == "mean_reversion"
    assert proposal.action == -1
    # SOFT CAP: target fraction scaled by the strategy's budget fraction.
    assert proposal.target_fraction == pytest.approx(-0.10 * 0.5)
    arbitration = diagnostics["arbitration"]
    assert arbitration["chosen"] == "mean_reversion"
    assert arbitration["budget_fraction"] == pytest.approx(0.5)
    assert arbitration["candidate_actions"] == {
        "brain": 0,
        "momentum": 1,
        "mean_reversion": -1,
    }
    assert arbitration["weights"]["mean_reversion"] > arbitration["weights"]["momentum"]


def test_equal_weight_ties_resolve_to_registry_order() -> None:
    # Fresh arbiter: all weights equal; brain registered first wins the tie
    # and keeps its full budget (1.0).
    arbiter = _arbiter(
        brain=_StubBrain(action=1, target_fraction=0.04),
        primitives=[("momentum", _StubStrategy(action=1, target_fraction=0.10))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    proposal, diagnostics = arbiter.propose(_snapshot())
    assert proposal.strategy_id == "brain"
    assert proposal.target_fraction == pytest.approx(0.04)
    assert diagnostics["arbitration"]["chosen"] == "brain"


def test_all_flat_returns_brain_proposal() -> None:
    arbiter = _arbiter(
        primitives=[("momentum", _StubStrategy(action=0, target_fraction=0.0))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    proposal, diagnostics = arbiter.propose(_snapshot())
    assert proposal.action == 0
    assert proposal.strategy_id == "brain"
    assert proposal.target_fraction == 0.0
    assert diagnostics["arbitration"]["chosen"] == "brain"
    assert diagnostics["arbitration"]["candidate_actions"]["momentum"] == 0


def test_propose_batch_arbitrates_every_snapshot_via_brain_batch() -> None:
    brain = _StubBrain(action=0)
    arbiter = _arbiter(
        brain=brain,
        primitives=[("momentum", _StubStrategy(action=1, target_fraction=0.02))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    results = arbiter.propose_batch([_snapshot("AAPL"), _snapshot("MSFT")])
    assert brain.batch_calls == 1  # brain batched once for both symbols
    assert [proposal.strategy_id for proposal, _ in results] == ["momentum", "momentum"]
    assert all("arbitration" in diagnostics for _, diagnostics in results)


# ---------------------------------------------------------------------------
# outcome-weighted allocation: winners rise, losers floor but never starve
# ---------------------------------------------------------------------------


def test_record_fill_outcome_shifts_weights_toward_winners() -> None:
    arbiter = _arbiter(
        primitives=[("momentum", _StubStrategy(action=1, target_fraction=0.10))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    assert arbiter.weights() == {"brain": pytest.approx(0.5), "momentum": pytest.approx(0.5)}
    arbiter.record_fill_outcome("momentum", 50.0)
    assert arbiter.pnl_ema["momentum"] == pytest.approx(0.1 * 50.0)  # EMA step
    weights = arbiter.weights()
    assert weights["momentum"] > weights["brain"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_losses_throttle_to_floor_but_never_starve() -> None:
    arbiter = _arbiter(
        primitives=[
            ("momentum", _StubStrategy(action=1, target_fraction=0.10)),
            ("mean_reversion", _StubStrategy(action=0, target_fraction=0.0)),
        ],
    )
    for _ in range(50):
        arbiter.record_fill_outcome("momentum", -100.0)
        arbiter.record_fill_outcome("mean_reversion", 100.0)
    weights = arbiter.weights()
    # A losing strategy's score collapses to exactly the floor, never zero.
    floor_score = 0.05
    expected_total = (
        floor_score  # brain: ema 0
        + floor_score  # momentum: negative ema clamped to 0
        + max(arbiter.pnl_ema["mean_reversion"], 0.0) + floor_score
    )
    assert weights["momentum"] == pytest.approx(floor_score / expected_total)
    assert weights["momentum"] > 0.0
    assert weights["mean_reversion"] > weights["brain"] >= weights["momentum"]
    # ...and the floored strategy still proposes AND can still win when it
    # is the only directional candidate — throttled, not starved.
    proposal, diagnostics = arbiter.propose(_snapshot())
    assert proposal.strategy_id == "momentum"
    assert diagnostics["arbitration"]["chosen"] == "momentum"


def test_unknown_strategy_outcome_is_ignored() -> None:
    arbiter = _arbiter(
        primitives=[("momentum", _StubStrategy(action=1))],
        budgets={"brain": 1.0, "momentum": 0.5},
    )
    before = arbiter.weights()
    arbiter.record_fill_outcome("hft_scalper", 1_000.0)  # not registered
    assert arbiter.weights() == before


# ---------------------------------------------------------------------------
# memory delegation: MemoryCapablePolicy passes through the arbiter
# ---------------------------------------------------------------------------


def test_memory_capable_brain_keeps_learning_under_arbitration() -> None:
    observed: list[tuple[object, str | None]] = []

    class _Updater:
        def observe(self, encoded_state, *, action_annotation=None):
            observed.append((encoded_state, action_annotation))

    memory_policy = MemoryAwarePolicy.__new__(MemoryAwarePolicy)
    memory_policy.updater = _Updater()
    memory_policy.last_encoded_by_symbol = {"AAPL": object()}

    arbiter = _arbiter(brain=memory_policy)
    assert isinstance(arbiter, MemoryCapablePolicy)
    assert arbiter.updater is memory_policy.updater
    annotation = "action=1 approved=1 side=buy qty=1.0000 resized=0 strategy=brain"
    arbiter.record_snapshot("AAPL", action_annotation=annotation)
    assert observed and observed[0][1] == annotation
    # set_temporal_cutoff lands on the brain member (backtest leak shield).
    arbiter.set_temporal_cutoff(T0)
    assert memory_policy._temporal_cutoff == T0


def test_bare_brain_arbiter_is_not_memory_capable() -> None:
    arbiter = _arbiter(brain=_StubBrain())
    assert isinstance(arbiter, BatchTradingPolicy)
    assert not isinstance(arbiter, MemoryCapablePolicy)
    with pytest.raises(AttributeError):
        _ = arbiter.updater


# ---------------------------------------------------------------------------
# config validation
# ---------------------------------------------------------------------------


def test_arbitration_config_defaults_are_disabled_paper_first() -> None:
    config = ArbitrationConfig()
    assert config.enabled is False
    assert config.strategies == ["brain", "momentum", "mean_reversion"]
    assert config.budget_fractions["brain"] == 1.0


def test_unknown_strategy_name_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(strategies=["brain", "hft_scalper"])


def test_missing_brain_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(strategies=["momentum", "mean_reversion"])


@pytest.mark.parametrize("bad_fraction", [0.0, -0.5, 1.5])
def test_out_of_range_budget_fraction_rejected(bad_fraction: float) -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(
            budget_fractions={
                "brain": bad_fraction,
                "momentum": 0.5,
                "mean_reversion": 0.5,
            }
        )


def test_registered_strategy_without_budget_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(budget_fractions={"brain": 1.0})


@pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.5])
def test_out_of_range_ema_alpha_rejected(bad_alpha: float) -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(ema_alpha=bad_alpha)


@pytest.mark.parametrize("bad_floor", [0.0, -0.05, 0.6])
def test_out_of_range_weight_floor_rejected(bad_floor: float) -> None:
    with pytest.raises(ConfigurationError):
        ArbitrationConfig(weight_floor=bad_floor)


# ---------------------------------------------------------------------------
# FillRecord JSONL backward compatibility
# ---------------------------------------------------------------------------


def _fill(**overrides) -> FillRecord:
    base = dict(
        timestamp=T0,
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        price=100.0,
        fee=0.1,
        realized_pnl=5.0,
        client_order_id="cid-1",
    )
    base.update(overrides)
    return FillRecord(**base)


def test_fill_record_round_trips_strategy_id() -> None:
    fill = _fill(strategy_id="momentum")
    revived = FillRecord.from_dict(json.loads(fill.to_json_line()))
    assert revived == fill
    assert revived.strategy_id == "momentum"


def test_fill_record_tolerates_pre_arbitration_lines() -> None:
    legacy_line = json.loads(_fill().to_json_line())
    del legacy_line["strategy_id"]  # a ledger line written before §6.3
    revived = FillRecord.from_dict(legacy_line)
    assert revived.strategy_id == "brain"
    assert revived.symbol == "AAPL"


# ---------------------------------------------------------------------------
# runtime wiring: enabled wraps, disabled leaves the policy untouched
# ---------------------------------------------------------------------------


class _LoggerSpy:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **kwargs) -> None:
        self.events.append((event, kwargs))


def test_runtime_disabled_config_builds_without_arbiter() -> None:
    from src.execution.runtime_bootstrap import maybe_wrap_arbitration

    config = load_config("configs/default.yaml")
    assert config.arbitration.enabled is False  # paper-only default
    policy = _StubBrain()
    logger = _LoggerSpy()
    assert maybe_wrap_arbitration(policy, config, logger=logger) is policy
    assert logger.events == []


def test_runtime_enabled_config_wraps_policy_in_arbiter() -> None:
    from src.execution.runtime_bootstrap import maybe_wrap_arbitration

    config = load_config("configs/default.yaml")
    config.arbitration.enabled = True
    policy = _StubBrain()
    logger = _LoggerSpy()
    wrapped = maybe_wrap_arbitration(policy, config, logger=logger)
    assert isinstance(wrapped, StrategyArbiter)
    assert wrapped.brain is policy
    assert wrapped.strategy_ids() == ["brain", "momentum", "mean_reversion"]
    event, payload = logger.events[0]
    assert event == "arbitration_ready"
    assert payload["strategies"] == ["brain", "momentum", "mean_reversion"]
    assert payload["budgets"]["momentum"] == pytest.approx(0.5)


def test_build_arbiter_instantiates_real_primitives_in_config_order() -> None:
    from src.strategies.baseline_policy import BaselineMomentumPolicy
    from src.strategies.baselines import MeanReversionPolicy

    arbiter = build_arbiter(_StubBrain(), ArbitrationConfig())
    assert [strategy_id for strategy_id, _ in arbiter.primitives] == [
        "momentum",
        "mean_reversion",
    ]
    assert isinstance(arbiter.primitives[0][1], BaselineMomentumPolicy)
    assert isinstance(arbiter.primitives[1][1], MeanReversionPolicy)
