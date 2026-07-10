from __future__ import annotations

from dataclasses import dataclass

from src.features.feature_pipeline import FeatureSnapshot
from src.risk.constraints import TradeProposal


@dataclass
class BaselineMomentumPolicy:
    momentum_threshold: float = 0.001
    max_fraction: float = 0.1

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]:
        values = {
            name: float(snapshot.features[index].item())
            for index, name in enumerate(snapshot.feature_names)
        }
        momentum = values.get("momentum_5", 0.0)
        volatility = values.get("volatility_10", 0.0)
        action = 0
        target_fraction = 0.0
        confidence = min(max(abs(momentum) * 200.0, 0.0), 1.0)
        if volatility < 0.02 and momentum > self.momentum_threshold:
            action = 1
            target_fraction = self.max_fraction
        elif volatility < 0.02 and momentum < -self.momentum_threshold:
            action = -1
            target_fraction = -self.max_fraction
        from src.strategies.baselines import heuristic_ev_bps

        ev_bps = heuristic_ev_bps(snapshot)
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=action,
            confidence=confidence,
            target_fraction=target_fraction,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id=f"baseline-{snapshot.symbol}-{int(snapshot.timestamp.timestamp())}",
            metadata={"momentum": momentum, "volatility": volatility},
            expected_gain_bps=ev_bps,
            expected_loss_bps=ev_bps,
        )
        return proposal, proposal.metadata
