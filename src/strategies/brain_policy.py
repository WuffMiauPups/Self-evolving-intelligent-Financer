from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field

import torch

from src.core.constants import ACTION_TO_SIGN, Action
from src.features.feature_pipeline import FeatureContract, FeatureSnapshot
from src.models.base import TradingModelProtocol
from src.risk.constraints import TradeProposal
from src.utils.torch_utils import ensure_batch_dimension, to_tensor


@dataclass
class BrainPolicy:
    model: TradingModelProtocol
    device: torch.device
    feature_contract: FeatureContract | None = None
    max_state_gap_sec: int | None = None
    memory_state_by_symbol: dict[str, torch.Tensor] = field(default_factory=dict)
    request_seq_by_symbol: dict[str, int] = field(default_factory=dict)
    last_timestamp_by_symbol: dict[str, datetime] = field(default_factory=dict)

    def reset_symbol_state(self, symbol: str, *, reason: str = "manual_reset") -> str:
        self.memory_state_by_symbol.pop(symbol, None)
        self.request_seq_by_symbol.pop(symbol, None)
        self.last_timestamp_by_symbol.pop(symbol, None)
        return reason

    def reset_all_state(self, *, reason: str = "manual_reset") -> str:
        self.memory_state_by_symbol.clear()
        self.request_seq_by_symbol.clear()
        self.last_timestamp_by_symbol.clear()
        return reason

    def _maybe_reset_for_boundary(self, symbol: str, timestamp: datetime) -> str | None:
        last_timestamp = self.last_timestamp_by_symbol.get(symbol)
        if last_timestamp is None:
            return None
        if timestamp < last_timestamp:
            return self.reset_symbol_state(symbol, reason="non_monotonic_timestamp")
        if self.max_state_gap_sec is not None:
            gap_seconds = (timestamp - last_timestamp).total_seconds()
            if gap_seconds > self.max_state_gap_sec:
                return self.reset_symbol_state(symbol, reason="state_gap_exceeded")
        return None

    def propose(self, snapshot: FeatureSnapshot) -> tuple[TradeProposal, dict[str, object]]:
        if self.feature_contract is not None:
            self.feature_contract.validate_snapshot(snapshot, context=f"policy[{snapshot.symbol}]")
        state_reset_reason = self._maybe_reset_for_boundary(snapshot.symbol, snapshot.timestamp)
        branch_inputs = {
            name: ensure_batch_dimension(to_tensor(value, device=self.device, dtype=torch.float32))
            for name, value in snapshot.branch_inputs.items()
        }
        regime_input = ensure_batch_dimension(to_tensor(snapshot.regime_input, device=self.device, dtype=torch.float32))
        features = ensure_batch_dimension(to_tensor(snapshot.features, device=self.device, dtype=torch.float32))
        memory_state = self.memory_state_by_symbol.get(snapshot.symbol)
        with torch.no_grad():
            output = self.model(
                branch_inputs=branch_inputs,
                regime_input=regime_input,
                memory_state=memory_state,
                full_features=features,
                temporal_tensor=ensure_batch_dimension(
                    to_tensor(snapshot.temporal_tensor, device=self.device, dtype=torch.float32)
                ),
            )
        if output.memory_state is None:
            self.memory_state_by_symbol.pop(snapshot.symbol, None)
        else:
            self.memory_state_by_symbol[snapshot.symbol] = output.memory_state.detach()
        self.last_timestamp_by_symbol[snapshot.symbol] = snapshot.timestamp
        logits = output.action_logits.squeeze(0)
        action_index = int(logits.argmax(dim=-1).item())
        # Force directional trade: if model picks FLAT, fall back to the
        # strongest directional logit (SHORT vs LONG).
        if Action(action_index) == Action.FLAT:
            short_logit = float(logits[int(Action.SHORT)].item())
            long_logit = float(logits[int(Action.LONG)].item())
            action_index = int(Action.LONG) if long_logit >= short_logit else int(Action.SHORT)
        action = ACTION_TO_SIGN[Action(action_index)]
        confidence = float(output.confidence.squeeze(0).item())
        target_fraction = float(output.target_fraction.squeeze(0).item())
        if abs(target_fraction) < 1e-6:
            target_fraction = 0.05 * action
        elif action != 0 and (target_fraction == 0.0 or (target_fraction > 0) != (action > 0)):
            target_fraction = abs(target_fraction) * action
        sequence_id = self.request_seq_by_symbol.get(snapshot.symbol, 0) + 1
        self.request_seq_by_symbol[snapshot.symbol] = sequence_id
        proposal = TradeProposal(
            symbol=snapshot.symbol,
            timestamp=snapshot.timestamp,
            action=action,
            confidence=confidence,
            target_fraction=target_fraction,
            price=snapshot.last_price,
            spread_bps=snapshot.spread_bps,
            request_id=f"{snapshot.symbol}-{action_index}-{int(snapshot.timestamp.timestamp())}-{sequence_id}",
            metadata={
                "soma_activation": float(output.soma_activation.squeeze(0).item()),
            },
        )
        diagnostics = {
            "branch_confidences": {
                name: float(value.squeeze(0).item()) for name, value in output.branch_confidences.items()
            },
            "soma_weights": output.soma_weights.squeeze(0).detach().cpu().tolist(),
            "soma_activation": float(output.soma_activation.squeeze(0).item()),
            "regime_weights": output.regime_weights.squeeze(0).detach().cpu().tolist()
            if output.regime_weights is not None
            else [],
            "state_reset_reason": state_reset_reason,
        }
        return proposal, diagnostics
