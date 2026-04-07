"""Tests for the confidence-aware soma gating fix.

Locks in the runtime behavior change made in src/models/soma.py and the
matching regularization loss in src/training/losses.py:

  - The soma gate now bakes log(branch_confidence) into the softmax exponent,
    so a branch with much-lower confidence cannot win the highest gate weight
    when the raw gate logits are equal.
  - soma_gating_confidence_loss returns 1 - E[soma_weight * branch_confidence]
    and is minimized when soma puts all weight on the most-confident branch.

Both protect against the regression where the runtime brain was assigning
~0.65 soma weight to the regime branch (confidence ≈ 0.42) while suppressing
the volume branch (confidence ≈ 0.97).
"""
from __future__ import annotations

import torch

from src.models.soma import SomaAggregator
from src.training.losses import soma_gating_confidence_loss


def _make_soma(branch_dim: int = 4, regime_dim: int = 3, memory_dim: int = 2) -> SomaAggregator:
    torch.manual_seed(0)
    return SomaAggregator(
        branch_dim=branch_dim,
        regime_dim=regime_dim,
        memory_dim=memory_dim,
        hidden_dim=8,
        dropout=0.0,
    )


def test_soma_gating_respects_confidence_when_logits_are_equal() -> None:
    """If we zero out the gate_layer, the only thing differentiating branches
    is their confidence, and the softmax must put more mass on the higher one."""
    soma = _make_soma()
    # Force the learned gate to contribute zero so confidence is the *only*
    # signal driving the softmax. Without the fix, the weights would be
    # uniform; with the fix, they should track confidence directly.
    with torch.no_grad():
        soma.gate_layer.weight.zero_()
        soma.gate_layer.bias.zero_()

    branch_count = 4
    branch_embeddings = torch.zeros(1, branch_count, 4)
    regime_embedding = torch.zeros(1, 3)
    memory_state = torch.zeros(1, 2)
    branch_confidences = torch.tensor([[[0.10], [0.20], [0.30], [0.95]]])

    out = soma(
        branch_embeddings=branch_embeddings,
        branch_confidences=branch_confidences,
        regime_embedding=regime_embedding,
        memory_state=memory_state,
    )

    weights = out.weights[0]
    # Highest-confidence branch should win.
    assert int(weights.argmax()) == 3
    # Lowest-confidence branch should lose.
    assert int(weights.argmin()) == 0
    # Weights should be monotonic in confidence when raw logits are equal.
    assert weights[0] < weights[1] < weights[2] < weights[3]
    # Weights still sum to 1.
    assert torch.isclose(weights.sum(), torch.tensor(1.0), atol=1.0e-5)


def test_soma_gating_normalizes_to_branch_confidence_distribution() -> None:
    """With zero gate logits, softmax(0 + log p) = p / sum(p). This is the
    exact mathematical property we rely on for the fix."""
    soma = _make_soma()
    with torch.no_grad():
        soma.gate_layer.weight.zero_()
        soma.gate_layer.bias.zero_()

    confidences = torch.tensor([[[0.1], [0.4], [0.2], [0.3]]])
    out = soma(
        branch_embeddings=torch.zeros(1, 4, 4),
        branch_confidences=confidences,
        regime_embedding=torch.zeros(1, 3),
        memory_state=torch.zeros(1, 2),
    )

    expected = confidences.squeeze(-1).squeeze(0) / confidences.sum()
    assert torch.allclose(out.weights[0], expected, atol=1.0e-5)


def test_soma_gating_does_not_explode_on_zero_confidence() -> None:
    """log(0) would be -inf; the fix uses clamp(min=1e-6). Verify no NaN/inf
    propagates even when one branch has effectively zero confidence."""
    soma = _make_soma()
    confidences = torch.tensor([[[0.0], [0.5], [0.5], [0.5]]])
    out = soma(
        branch_embeddings=torch.randn(1, 4, 4),
        branch_confidences=confidences,
        regime_embedding=torch.zeros(1, 3),
        memory_state=torch.zeros(1, 2),
    )
    assert torch.isfinite(out.weights).all()
    assert torch.isclose(out.weights.sum(), torch.tensor(1.0), atol=1.0e-5)


def test_gating_loss_zero_when_perfectly_confident() -> None:
    soma_weights = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    confidences = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    loss = soma_gating_confidence_loss(soma_weights, confidences)
    assert float(loss) == 0.0


def test_gating_loss_high_when_inverted() -> None:
    """The runtime pathology: high weight on the *lowest*-confidence branch.
    Loss should be near 1.0 in that case."""
    soma_weights = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    confidences = torch.tensor([[0.95, 0.95, 0.95, 0.05]])
    loss = soma_gating_confidence_loss(soma_weights, confidences)
    assert float(loss) > 0.9


def test_gating_loss_gradient_flows_into_soma_weights() -> None:
    """Sanity: the loss is differentiable wrt soma_weights so backprop reshapes
    the gate."""
    soma_weights = torch.tensor([[0.25, 0.25, 0.25, 0.25]], requires_grad=True)
    confidences = torch.tensor([[0.1, 0.2, 0.3, 0.95]])
    loss = soma_gating_confidence_loss(soma_weights, confidences)
    loss.backward()
    assert soma_weights.grad is not None
    # Gradient on the highest-confidence branch should be the *most negative*
    # (push that weight up to reduce loss).
    g = soma_weights.grad[0]
    assert int(g.argmin()) == 3
