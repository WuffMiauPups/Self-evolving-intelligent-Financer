"""Confidence semantics fix (CONFIDENCE_AUDIT.md, fixed 2026-07-21).

Proposal confidence must be P(resolved action class) computed from the
action logits in ``_build_proposal`` -- the shared funnel both ``propose``
and ``propose_batch`` route through. Pre-fix, legacy checkpoints emitted a
sigmoid head trained as an MFE/volatility/stability regressor (measured
rank vs realized outcomes: rho -0.009 over 1,313 OOS round trips -- never
P(win)), and decoupled checkpoints emitted the ARGMAX-class probability,
which names the wrong class under a forced flip.

A Platt calibrator fitted on the pre-fix quantity must never apply to
post-fix proposals: the loader refuses artifacts without the
``confidence_semantics`` stamp, and fit() never self-stamps -- the caller
must attest the quantity.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.core.config.risk import RiskConfig
from src.features.feature_pipeline import FeatureSnapshot
from src.models.base import ModelOutput
from src.monitoring.decision_log import DecisionEvent
from src.risk.portfolio_state import PortfolioState
from src.risk.risk_manager import RiskManager
from src.strategies.brain_policy import BrainPolicy
from src.training.platt_calibration import (
    CONFIDENCE_SEMANTICS,
    PlattCalibrator,
    load_calibrator_for_checkpoint,
)

T = datetime(2026, 7, 21, 15, 0, tzinfo=UTC)

LONG_LOGITS = [0.1, 0.2, 3.0]   # short, flat, long
SHORT_LOGITS = [3.0, 0.2, 0.1]
FLAT_LOGITS = [0.1, 3.0, 0.2]


def _softmax_prob(logits: list[float], index: int) -> float:
    peak = max(logits)
    exps = [math.exp(v - peak) for v in logits]
    return exps[index] / sum(exps)


def _policy(**overrides) -> BrainPolicy:
    return BrainPolicy(
        model=SimpleNamespace(risk_head=None),
        device=torch.device("cpu"),
        **overrides,
    )


def _snapshot(symbol: str = "NVDA") -> SimpleNamespace:
    return SimpleNamespace(symbol=symbol, timestamp=T, last_price=100.0, spread_bps=1.0)


def _build(policy: BrainPolicy, logits: list[float], head_confidence: float = 0.9):
    return policy._build_proposal(
        snapshot=_snapshot(),
        logit_values=logits,
        confidence=head_confidence,
        target_fraction=0.05,
        soma_activation=1.0,
        branch_confidence_dict={},
        soma_weights=[1.0],
        regime_weights=[],
        state_reset_reason=None,
        expected_gain_bps=100.0,
        expected_loss_bps=200.0,
    )


def test_legacy_head_confidence_is_overridden() -> None:
    """THE fix: a legacy sigmoid-head value (here 0.9) must not reach the
    proposal -- confidence is P(resolved class) from the logits."""
    proposal, _ = _build(_policy(), LONG_LOGITS, head_confidence=0.9)
    assert proposal.action == 1
    assert proposal.confidence == pytest.approx(_softmax_prob(LONG_LOGITS, 2))
    assert proposal.confidence != pytest.approx(0.9)


def test_short_confidence_is_short_class_probability() -> None:
    proposal, _ = _build(_policy(), SHORT_LOGITS, head_confidence=0.42)
    assert proposal.action == -1
    assert proposal.confidence == pytest.approx(_softmax_prob(SHORT_LOGITS, 0))


def test_flat_confidence_is_flat_class_probability() -> None:
    proposal, _ = _build(_policy(), FLAT_LOGITS, head_confidence=0.42)
    assert proposal.action == 0
    assert proposal.confidence == pytest.approx(_softmax_prob(FLAT_LOGITS, 1))


def test_decoupled_argmax_equivalence() -> None:
    """For decoupled checkpoints (head conf = amax softmax) with no flip,
    the fix is numerically a no-op: P(resolved) == P(argmax)."""
    amax_conf = _softmax_prob(LONG_LOGITS, 2)
    proposal, _ = _build(_policy(), LONG_LOGITS, head_confidence=amax_conf)
    assert proposal.confidence == pytest.approx(amax_conf)


def test_forced_flip_uses_resolved_class_not_argmax() -> None:
    """Under force_directional_flip, FLAT argmax resolves to a direction.
    Confidence must be P(resolved direction), NOT P(flat) = amax."""
    policy = _policy(force_directional_flip=True)
    flat_but_short_leaning = [2.0, 3.0, 0.1]  # flat wins, short > long
    proposal, _ = _build(policy, flat_but_short_leaning)
    assert proposal.action == -1
    assert proposal.confidence == pytest.approx(_softmax_prob(flat_but_short_leaning, 0))
    assert proposal.confidence != pytest.approx(_softmax_prob(flat_but_short_leaning, 1))


def test_diagnostics_record_head_confidence() -> None:
    """The replaced head value stays observable for the semantics
    transition; the proposal carries the as-consumed value."""
    proposal, diagnostics = _build(_policy(), LONG_LOGITS, head_confidence=0.9)
    assert diagnostics["head_confidence"] == pytest.approx(0.9)
    assert proposal.confidence == pytest.approx(_softmax_prob(LONG_LOGITS, 2))


def test_override_lives_only_in_the_shared_funnel_source_contract() -> None:
    """The softmax override must exist exactly once, inside _build_proposal."""
    source = (
        Path(__file__).resolve().parents[1] / "src" / "strategies" / "brain_policy.py"
    ).read_text(encoding="utf-8")
    override_line = "confidence = exps[action_index] / sum(exps)"
    assert source.count(override_line) == 1
    assert override_line in source.split("def _build_proposal", 1)[1].split("def propose", 1)[0]


# ---- End-to-end through the packing paths ---------------------------------

E2E_LOGITS = [0.4, 0.2, 1.3]  # short, flat, long -> LONG argmax
E2E_HEAD_CONF = 0.9           # distinct from every softmax prob of E2E_LOGITS


class _E2EStubModel:
    """Real ModelOutput producer so propose()/propose_batch() exercise the
    scalar pack/unpack plumbing the unit tests bypass."""

    risk_head = object()

    def __call__(
        self,
        *,
        branch_inputs,
        regime_input,
        memory_state=None,
        full_features=None,
        temporal_tensor=None,
    ) -> ModelOutput:
        batch = regime_input.shape[0]
        return ModelOutput(
            action_logits=torch.tensor([E2E_LOGITS]).repeat(batch, 1),
            confidence=torch.full((batch, 1), E2E_HEAD_CONF),
            target_fraction=torch.zeros(batch, 1),
            fused_representation=torch.zeros(batch, 8),
            branch_embeddings={},
            branch_confidences={},
            soma_weights=torch.zeros(batch, 1),
            soma_activation=torch.full((batch, 1), 1.0),
            risk_predictions={
                "mfe": torch.full((batch, 1), 0.0020),
                "mae": torch.full((batch, 1), -0.0010),
            },
        )


def _e2e_snapshot(symbol: str = "AAPL") -> FeatureSnapshot:
    dims = dict(f=7, r=4, c=3, t=12, d=5)
    return FeatureSnapshot(
        symbol=symbol,
        timestamp=T,
        last_price=100.0,
        spread_bps=1.0,
        features=torch.full((dims["f"],), 0.5),
        branch_inputs={"trend": torch.full((dims["d"],), 0.25)},
        regime_input=torch.full((dims["r"],), -0.5),
        temporal_tensor=torch.full((dims["c"], dims["t"]), 1e-3),
        feature_names=[f"f{i}" for i in range(dims["f"])],
        branch_feature_names={"trend": [f"d{i}" for i in range(dims["d"])]},
        regime_feature_names=[f"r{i}" for i in range(dims["r"])],
        temporal_feature_names=[f"t{i}" for i in range(dims["c"])],
    )


def test_end_to_end_propose_and_batch_carry_resolved_class_probability() -> None:
    """Pins the whole pipeline: a pack/unpack slice bug in propose() or
    propose_batch() would corrupt logit_values and this catches it -- the
    _build_proposal unit tests cannot."""
    policy = BrainPolicy(model=_E2EStubModel(), device=torch.device("cpu"))
    expected = _softmax_prob(E2E_LOGITS, 2)

    proposal, diagnostics = policy.propose(_e2e_snapshot())
    assert proposal.action == 1
    assert proposal.confidence == pytest.approx(expected, abs=1e-6)
    assert diagnostics["head_confidence"] == pytest.approx(E2E_HEAD_CONF, abs=1e-6)

    results = policy.propose_batch([_e2e_snapshot("MSFT"), _e2e_snapshot("NVDA")])
    for batch_proposal, batch_diag in results:
        assert batch_proposal.confidence == pytest.approx(expected, abs=1e-6)
        assert batch_diag["head_confidence"] == pytest.approx(E2E_HEAD_CONF, abs=1e-6)


# ---- Composition with the EV gate ------------------------------------------


def test_policy_built_short_approved_only_with_correct_conf_and_legs() -> None:
    """The confidence override and the EV-leg swap compose at the gate:
    ev = P(short)*down - (1-P(short))*up - cost. Costs are chosen so the
    correct composition approves while a weak short is rejected -- and the
    weak case would be WRONGLY approved if the pre-fix head confidence
    (0.9) fed the gate, pinning both fixes at the money path."""
    policy = _policy()
    proposal, _ = _build(policy, SHORT_LOGITS)  # post-swap: gain=200 down-move
    p_short = _softmax_prob(SHORT_LOGITS, 0)
    assert proposal.confidence == pytest.approx(p_short)

    manager = RiskManager(RiskConfig(fee_bps=10.0, slippage_bps=10.0))
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    # cost = 10 fee + 10 slippage + 1 spread = 21 bps.
    # Correct composition: p~0.90 -> 0.90*200 - 0.10*100 - 21 ~= +149 bps.
    decision = manager.evaluate(proposal, portfolio, evaluation_time=T)
    assert "negative_ev" not in decision.reasons

    # A weak short (near-tied logits, p~0.37) must be rejected:
    # 0.37*200 - 0.63*100 - 21 ~= -10 bps. With the PRE-fix head value
    # (0.9 fed straight through) it would be 0.9*200 - 0.1*100 - 21 = +149
    # and wrongly approved.
    proposal_weak, _ = _build(_policy(), [1.0, 0.9, 0.8], head_confidence=0.9)
    decision_weak = manager.evaluate(proposal_weak, portfolio, evaluation_time=T)
    assert "negative_ev" in decision_weak.reasons


def test_nan_confidence_fails_closed_at_the_gate() -> None:
    """A non-finite logit yields NaN through the softmax; the policy zeroes
    it and the gate must reject (NaN <= 0 is False -- the gate needs the
    isfinite check for proposals built elsewhere)."""
    proposal, _ = _build(_policy(), [float("inf"), 0.2, 0.1])
    assert proposal.confidence == 0.0  # fail-closed at the source

    manager = RiskManager(RiskConfig(fee_bps=10.0, slippage_bps=10.0))
    portfolio = PortfolioState(cash=100_000.0, equity=100_000.0)
    decision = manager.evaluate(proposal, portfolio, evaluation_time=T)
    assert "negative_ev" in decision.reasons


# ---- Ledger row records the as-consumed value + semantics marker -----------


def test_decision_event_records_as_consumed_confidence_and_semantics() -> None:
    proposal, _ = _build(_policy(), LONG_LOGITS, head_confidence=0.9)
    event = DecisionEvent(
        ts=T,
        symbol=proposal.symbol,
        request_id=proposal.request_id,
        proposed_action=int(proposal.action),
        confidence=float(proposal.confidence),
        target_fraction=float(proposal.target_fraction),
        approved=True,
        confidence_semantics=CONFIDENCE_SEMANTICS,
    )
    row = json.loads(event.to_json_line())
    assert row["confidence"] == pytest.approx(_softmax_prob(LONG_LOGITS, 2))
    assert row["confidence_semantics"] == CONFIDENCE_SEMANTICS


# ---- Platt semantics guard -------------------------------------------------


def _write_artifact(tmp_path: Path, **overrides) -> Path:
    calibrator = PlattCalibrator(
        a=0.87,
        b=-0.36,
        is_fitted=True,
        n_samples=1000,
        encoder_fingerprint="fp1234",
        checkpoint_filename="brain.pt",
        **overrides,
    )
    checkpoint = tmp_path / "brain.pt"
    checkpoint.write_bytes(b"stub")
    calibrator.save(tmp_path / "platt_calibration.json")
    return checkpoint


def test_loader_refuses_artifact_without_semantics_stamp(tmp_path) -> None:
    """Artifacts written before the fix calibrated the legacy sigmoid head --
    they must degrade to the uncalibrated passthrough."""
    checkpoint = _write_artifact(tmp_path)  # confidence_semantics=None
    loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp1234")
    assert not loaded.is_fitted


def test_loader_accepts_current_semantics(tmp_path) -> None:
    checkpoint = _write_artifact(tmp_path, confidence_semantics=CONFIDENCE_SEMANTICS)
    loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp1234")
    assert loaded.is_fitted
    assert loaded.confidence_semantics == CONFIDENCE_SEMANTICS


def test_fit_with_attestation_round_trips_and_loads(tmp_path) -> None:
    """fit() stamps ONLY what the caller attests; the stamp round-trips
    through save/load and satisfies the loader."""
    calibrator = PlattCalibrator()
    rng = np.random.default_rng(7)
    conf = rng.uniform(0.3, 0.9, size=200)
    calibrator.fit(
        conf,
        (conf - 0.5) + rng.normal(0, 0.1, size=200),
        semantics=CONFIDENCE_SEMANTICS,
    )
    assert calibrator.is_fitted
    assert calibrator.confidence_semantics == CONFIDENCE_SEMANTICS
    calibrator.encoder_fingerprint = "fp1234"
    checkpoint = tmp_path / "brain.pt"
    checkpoint.write_bytes(b"stub")
    calibrator.save(tmp_path / "platt_calibration.json")
    loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp1234")
    assert loaded.is_fitted


def test_fit_without_attestation_is_unstamped_and_refused(tmp_path) -> None:
    """fit() cannot know its input's provenance, so it never self-stamps --
    an unattested artifact must be refused at load time (the guard that
    keeps a mixed/unknown-quantity fit out of the gate)."""
    calibrator = PlattCalibrator()
    rng = np.random.default_rng(11)
    conf = rng.uniform(0.3, 0.9, size=200)
    calibrator.fit(conf, (conf - 0.5) + rng.normal(0, 0.1, size=200))
    assert calibrator.is_fitted
    assert calibrator.confidence_semantics is None
    calibrator.encoder_fingerprint = "fp1234"
    checkpoint = tmp_path / "brain.pt"
    checkpoint.write_bytes(b"stub")
    calibrator.save(tmp_path / "platt_calibration.json")
    loaded = load_calibrator_for_checkpoint(checkpoint, expected_fingerprint="fp1234")
    assert not loaded.is_fitted
