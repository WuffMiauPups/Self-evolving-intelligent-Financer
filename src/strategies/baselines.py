"""Additional baseline strategies for ablation comparison.

These provide simple reference strategies that any meaningful trading
system should outperform.  All implement the same ``propose(snapshot)``
interface as ``BaselineMomentumPolicy``.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.features.feature_pipeline import FeatureSnapshot
from src.risk.constraints import TradeProposal


def heuristic_ev_bps(snapshot: FeatureSnapshot, *, floor_bps: float = 20.0) -> float:
    """Volatility-scaled symmetric EV magnitude for rule baselines.

    Baselines have no risk head, so their proposals carried
    expected_gain/loss_bps of 0 and the EV gate rejected every trade —
    making every benchmark number fiction.  A symmetric estimate keeps
    them honest: EV reduces to (2·confidence − 1)·V − costs, so the gate
    still prices conviction against costs instead of auto-rejecting.
    """
    try:
        index = snapshot.feature_names.index("volatility_10")
        per_bar_vol = abs(float(snapshot.features[index].item()))
    except (ValueError, IndexError):
        per_bar_vol = 0.0
    return max(per_bar_vol * 10_000.0, floor_bps)


@dataclass
class BuyAndHoldPolicy:
    """Always go LONG on every symbol at every bar.

    This is the simplest possible benchmark: it measures whether the
    model can outperform simply buying and holding the universe.
    """

    max_fraction: float = 0.1

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        return reason

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        return reason

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]:
        ev_bps = heuristic_ev_bps(snapshot)
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=1,
            confidence=1.0,
            target_fraction=self.max_fraction,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id=f"buyhold-{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
            metadata={"strategy": "buy_and_hold"},
            expected_gain_bps=ev_bps,
            expected_loss_bps=ev_bps,
        )
        return proposal, proposal.metadata


@dataclass
class MeanReversionPolicy:
    """Simple mean-reversion baseline: fade momentum.

    When momentum is positive (price trending up), go SHORT expecting
    reversion.  When momentum is negative, go LONG.  Only trades when
    volatility is below a cap to avoid fading strong trends.
    """

    reversion_threshold: float = 0.002
    vol_cap: float = 0.03
    max_fraction: float = 0.1

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        return reason

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        return reason

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]:
        values = {
            name: float(snapshot.features[index].item())
            for index, name in enumerate(snapshot.feature_names)
        }
        momentum = values.get("momentum_5", 0.0)
        volatility = values.get("volatility_10", 0.0)
        action = 0
        target_fraction = 0.0
        confidence = min(max(abs(momentum) * 150.0, 0.0), 1.0)
        if volatility < self.vol_cap and momentum > self.reversion_threshold:
            action = -1  # SHORT: fade positive momentum
            target_fraction = -self.max_fraction
        elif volatility < self.vol_cap and momentum < -self.reversion_threshold:
            action = 1  # LONG: fade negative momentum
            target_fraction = self.max_fraction
        ev_bps = heuristic_ev_bps(snapshot)
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=action,
            confidence=confidence,
            target_fraction=target_fraction,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id=f"meanrev-{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
            metadata={"momentum": momentum, "volatility": volatility, "strategy": "mean_reversion"},
            expected_gain_bps=ev_bps,
            expected_loss_bps=ev_bps,
        )
        return proposal, proposal.metadata
