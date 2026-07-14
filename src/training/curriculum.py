"""Anchored negatives + adaptive curriculum for retrain dataset construction.

MASTER_PLAN §6.5 / RESEARCH_ROADMAP §4 items 8-9:

- *Economic-margin filtering*: a contrast pair (or a "mistake" worth
  anchoring on) only carries signal when the outcome gap exceeds what
  round-trip transaction costs plus a noise buffer could explain.
  ``pair_is_valid`` / ``select_anchored_negatives`` implement that filter.
- *Anchored negatives*: the worst genuinely-wrong executed decisions are
  persisted permanently across retrains (``AnchoredNegativeStore``) so
  every future dataset build can re-include the model's realized mistakes.
- *Adaptive curriculum*: per-sample difficulty is distance from the label
  band edge (``sample_difficulty``); ``CurriculumSampler`` feeds easy,
  unambiguous samples first and ramps up to the full dataset over epochs.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch

from src.training.replay_buffer import DecisionObservation, load_observations

logger = logging.getLogger(__name__)

# Floor for |primary_return| so the log-ratio difficulty is defined at
# exactly-zero returns (deep inside the flat band -> difficulty ~ 0).
_MIN_RETURN_MAGNITUDE = 1.0e-12


def pair_is_valid(
    pnl_a: float,
    pnl_b: float,
    *,
    cost_bps: float,
    noise_buffer_bps: float,
    notional: float,
) -> bool:
    """Economic-margin filter for a contrast pair (roadmap §4 item 8).

    A pair of outcomes is a valid training contrast only when the gap
    between them exceeds the round-trip transaction cost plus a noise
    buffer — otherwise the "better" outcome is indistinguishable from
    execution noise.  ``cost_bps`` follows the repo convention
    (``training.transaction_cost_bps``): it is the full round-trip cost in
    basis points.  Both bps figures are converted against ``notional``.
    """
    margin = (cost_bps + noise_buffer_bps) / 10_000.0 * notional
    return abs(pnl_a - pnl_b) > margin


def select_anchored_negatives(
    observations: list[DecisionObservation],
    *,
    cost_bps: float,
    noise_buffer_bps: float,
    max_count: int,
) -> list[DecisionObservation]:
    """Pick the worst genuinely-wrong EXECUTED decisions.

    A decision qualifies as an anchored negative when:
    - it was executed (``executed_action != 0``; FLAT places no trade), and
    - its realized loss magnitude exceeds the cost + noise-buffer margin —
      losses inside the margin are noise, not mistakes.

    ``realized_pnl`` is interpreted in return-fraction units against unit
    notional (matching ``training.transaction_cost_bps / 10_000`` in
    ``targets.py``).  Returns at most ``max_count`` observations, worst
    (most negative ``realized_pnl``) first.
    """
    if max_count <= 0:
        return []
    margin = (cost_bps + noise_buffer_bps) / 10_000.0
    losers = [
        observation
        for observation in observations
        if int(observation.executed_action) != 0
        and float(observation.realized_pnl) < -margin
    ]
    losers.sort(key=lambda observation: float(observation.realized_pnl))
    return losers[:max_count]


class AnchoredNegativeStore:
    """JSONL persistence for anchored negatives — PERMANENT across retrains.

    Anchored negatives are retained permanently: each retrain cycle appends
    its newly-identified mistakes here, and the accumulated set survives
    into every future retrain's dataset construction.  This store IS the
    permanence mechanism — nothing else re-persists these observations, so
    the file must never be treated as a per-run scratch artifact.

    Mirrors the ledger/replay-buffer JSONL patterns: append-friendly
    writes, torn/corrupt lines skipped on load.  ``append_new`` dedupes by
    ``decision_id``; the store is bounded at ``max_anchored`` rows with the
    oldest entries dropped first (a snapshot rewrite, like
    ``ObservationBuffer.save``).
    """

    def __init__(self, path: str | Path, *, max_anchored: int = 5_000) -> None:
        if max_anchored <= 0:
            raise ValueError("AnchoredNegativeStore max_anchored must be positive")
        self.path = Path(path)
        self.max_anchored = max_anchored

    def load(self) -> list[DecisionObservation]:
        """Read the anchored set (ledger-style: torn lines skipped)."""
        return load_observations(self.path)[-self.max_anchored:]

    def append_new(self, observations: Iterable[DecisionObservation]) -> int:
        """Append observations not already anchored; returns the new count.

        Dedupes by ``decision_id`` against both the persisted set and
        within the incoming batch.  When the combined set exceeds
        ``max_anchored`` the file is rewritten keeping only the newest
        ``max_anchored`` rows (oldest dropped).
        """
        existing = load_observations(self.path)
        known_ids = {observation.decision_id for observation in existing}
        fresh: list[DecisionObservation] = []
        for observation in observations:
            if observation.decision_id in known_ids:
                continue
            known_ids.add(observation.decision_id)
            fresh.append(observation)
        if not fresh:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        combined = existing + fresh
        if len(combined) > self.max_anchored:
            combined = combined[-self.max_anchored:]
            with self.path.open("w", encoding="utf-8") as fp:
                for observation in combined:
                    fp.write(observation.to_json_line() + "\n")
        else:
            with self.path.open("a", encoding="utf-8") as fp:
                for observation in fresh:
                    fp.write(observation.to_json_line() + "\n")
        logger.info(
            "anchored_negatives_appended path=%s newly_anchored=%d anchored_total=%d",
            str(self.path), len(fresh), len(combined),
        )
        return len(fresh)


def sample_difficulty(primary_return: float, effective_epsilon: float) -> float:
    """Difficulty of one sample in [0, 1] from distance to the label band edge.

    The direction label flips at ``|primary_return| == effective_epsilon``
    (the flat-band edge from ``targets.py``), so ambiguity is maximal there
    and vanishes as the return moves far above the band or deep inside it:

        difficulty = exp(-|ln(|r| / eps)|)   ==   min(|r|/eps, eps/|r|)

    This is smooth and symmetric in log-space around the band edge,
    monotone increasing toward it from both sides, exactly 1.0 at the edge,
    and decays to 0 at both extremes.  ``|r|`` is clamped away from zero so
    an exactly-flat return is well-defined (difficulty ~ 0).  A non-positive
    epsilon means there is no flat band — every sample is unambiguous (0.0).
    """
    if effective_epsilon <= 0.0:
        return 0.0
    magnitude = max(abs(primary_return), _MIN_RETURN_MAGNITUDE)
    return math.exp(-abs(math.log(magnitude / effective_epsilon)))


def compute_sample_difficulties(
    primary_returns: torch.Tensor, effective_epsilon: float
) -> torch.Tensor:
    """Vectorized :func:`sample_difficulty` over a (N,) returns column."""
    returns = primary_returns.detach().to("cpu", torch.float32).flatten()
    if effective_epsilon <= 0.0:
        return torch.zeros_like(returns)
    ratio = returns.abs().clamp_min(_MIN_RETURN_MAGNITUDE) / effective_epsilon
    return torch.exp(-ratio.log().abs())


class CurriculumSampler(torch.utils.data.Sampler[int]):
    """Deterministic epoch-progressive sampler over per-sample difficulties.

    At epoch ``e`` only samples with difficulty at or below the current
    inclusion threshold are yielded; the threshold is the difficulty value
    at quantile ``q(e)``, where ``q`` ramps linearly from
    ``start_quantile`` at epoch 0 to 1.0 at ``full_epoch`` (and stays at
    1.0 afterwards).  Excluded indices are simply not yielded that epoch.
    Iteration order is a seeded permutation (``torch.Generator`` seeded
    with ``seed + epoch``), so two runs with the same seed and epoch yield
    identical sequences.  ``len()`` reflects the included subset.

    Call :meth:`set_epoch` at the start of every epoch (the trainer does
    this automatically, mirroring the ``DistributedSampler`` convention).
    """

    def __init__(
        self,
        difficulties: torch.Tensor | list[float],
        *,
        start_quantile: float = 0.5,
        full_epoch: int = 3,
        seed: int = 0,
    ) -> None:
        if not 0.0 < start_quantile <= 1.0:
            raise ValueError("CurriculumSampler start_quantile must be within (0, 1]")
        if full_epoch < 1:
            raise ValueError("CurriculumSampler full_epoch must be >= 1")
        self.difficulties = torch.as_tensor(
            difficulties, dtype=torch.float32
        ).detach().to("cpu").flatten()
        if len(self.difficulties) == 0:
            raise ValueError("CurriculumSampler requires at least one sample")
        self.start_quantile = float(start_quantile)
        self.full_epoch = int(full_epoch)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch (0-based), advancing the inclusion ramp."""
        self.epoch = int(epoch)

    def _inclusion_quantile(self) -> float:
        progress = min(self.epoch / self.full_epoch, 1.0)
        return self.start_quantile + (1.0 - self.start_quantile) * progress

    def _included_indices(self) -> torch.Tensor:
        quantile = self._inclusion_quantile()
        if quantile >= 1.0:
            return torch.arange(len(self.difficulties))
        threshold = torch.quantile(self.difficulties, quantile)
        # Non-empty by construction: threshold >= min(difficulties) because
        # quantile >= start_quantile > 0.
        return torch.nonzero(self.difficulties <= threshold, as_tuple=False).flatten()

    def __iter__(self) -> Iterator[int]:
        included = self._included_indices()
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(included), generator=generator)
        return iter(included[permutation].tolist())

    def __len__(self) -> int:
        return len(self._included_indices())
