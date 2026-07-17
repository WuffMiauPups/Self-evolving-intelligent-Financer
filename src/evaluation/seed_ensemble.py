"""Seed-ensemble policy (§9.4 arm 2 — the §7.3 pull-forward, exercised).

Majority-vote ensemble over same-architecture policies trained with
different seeds. Rationale: the measured noise floor (0.92pt val BA)
means a material share of any single seed's decisions is seed
idiosyncrasy; voting keeps only the direction the seeds agree on, which
is the cheapest variance-reduction available — the three floor
checkpoints already exist, so this arm costs zero training.

Merge rule per snapshot:
- action: strict majority (≥2 of 3); no majority → FLAT (disagreement is
  exactly the seed-noise this arm exists to suppress).
- target_fraction: mean over the members that voted the winning action
  (0.0 on a FLAT verdict).
- confidence / EV legs: mean over ALL members (the risk gate should see
  the ensemble's honest uncertainty, not the winners' bravado).

Each member keeps its own recurrent state; the wrapper is stateless.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from statistics import mean

from src.risk.constraints import TradeProposal


@dataclass
class SeedEnsemblePolicy:
    members: list

    def propose_batch(self, snapshots) -> list[tuple[TradeProposal, dict[str, object]]]:
        member_results = [member.propose_batch(snapshots) for member in self.members]
        merged: list[tuple[TradeProposal, dict[str, object]]] = []
        for index in range(len(snapshots)):
            proposals = [results[index][0] for results in member_results]
            merged.append(self._merge(proposals))
        return merged

    def propose(self, snapshot) -> tuple[TradeProposal, dict[str, object]]:
        return self.propose_batch([snapshot])[0]

    def reset_symbol_state(self, symbol: str, *, reason: str | None = None) -> None:
        for member in self.members:
            if hasattr(member, "reset_symbol_state"):
                member.reset_symbol_state(symbol, reason=reason)

    def _merge(
        self, proposals: list[TradeProposal]
    ) -> tuple[TradeProposal, dict[str, object]]:
        votes = Counter(int(p.action) for p in proposals)
        winner, count = votes.most_common(1)[0]
        majority = len(proposals) // 2 + 1
        if count < majority:
            winner = 0  # 3-way disagreement IS seed noise — stand down
        voters = [p for p in proposals if int(p.action) == winner] or proposals
        template = proposals[0]
        merged = replace(
            template,
            action=winner,
            confidence=mean(p.confidence for p in proposals),
            target_fraction=(
                0.0 if winner == 0 else mean(p.target_fraction for p in voters)
            ),
            expected_gain_bps=mean(p.expected_gain_bps for p in proposals),
            expected_loss_bps=mean(p.expected_loss_bps for p in proposals),
        )
        return merged, {
            "votes": dict(votes),
            "unanimous": count == len(proposals),
            "branch_confidences": {},
            "soma_weights": [],
            "soma_activation": 1.0,
        }
