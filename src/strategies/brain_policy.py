from __future__ import annotations

import math
from datetime import datetime
from dataclasses import dataclass, field

import torch

from src.core.constants import ACTION_TO_SIGN, Action
from src.core.device import maybe_autocast
from src.core.logging import get_logger
from src.features.feature_pipeline import FeatureContract, FeatureSnapshot
from src.features.normalization import NormalizationStats
from src.models.base import ModelOutput, TradingModelProtocol
from src.risk.constraints import TradeProposal
from src.utils.torch_utils import ensure_batch_dimension, to_tensor

_diag_logger = get_logger("brain_policy")
# Bounded diagnostic: log raw inference state for the first N calls so we can
# compare paper-trading output against validation-time output without flooding
# logs. Set to 0 to disable.
_DIAG_MAX_CALLS = 10


def _extract_ev_signals_single(
    risk_predictions: dict[str, torch.Tensor] | None,
) -> tuple[float, float]:
    """Extract (expected_gain_bps, expected_loss_bps) from a single-sample
    risk-head output. Returns (0.0, 0.0) when the risk head is absent.

    MFE/MAE are predicted as raw fractions (e.g. 0.025 = 2.5%). MAE is
    typically negative (adverse excursion); we use abs() for the EV-gate
    loss magnitude. Fractions are multiplied by 10000 to get basis points.
    """
    if not risk_predictions:
        return 0.0, 0.0
    mfe = risk_predictions.get("mfe")
    mae = risk_predictions.get("mae")
    gain_fraction = float(mfe.detach().reshape(-1)[0].item()) if mfe is not None else 0.0
    loss_fraction = float(mae.detach().reshape(-1)[0].item()) if mae is not None else 0.0
    return gain_fraction * 10_000.0, abs(loss_fraction) * 10_000.0


def _extract_ev_signals_batch(
    risk_predictions: dict[str, torch.Tensor] | None,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    """Batched variant of ``_extract_ev_signals_single``. Single host sync.

    Each EV leg is handled independently, matching the single-sample
    path: a missing ``mfe`` (or ``mae``) zeroes only its own leg instead
    of zeroing both.
    """
    zeros = [0.0] * batch_size
    if not risk_predictions:
        return zeros, zeros
    mfe = risk_predictions.get("mfe")
    mae = risk_predictions.get("mae")
    if mfe is None and mae is None:
        return zeros, zeros
    present = [t.detach().reshape(batch_size) for t in (mfe, mae) if t is not None]
    values = iter(torch.stack(present, dim=0).cpu().tolist())
    gain_fractions = next(values) if mfe is not None else zeros
    loss_fractions = next(values) if mae is not None else zeros
    gain_bps = [float(v) * 10_000.0 for v in gain_fractions]
    loss_bps = [abs(float(v)) * 10_000.0 for v in loss_fractions]
    return gain_bps, loss_bps


def _reassemble_bucketed_outputs(
    outputs: list[ModelOutput],
    order: list[int],
) -> ModelOutput:
    """Recombine per-length-bucket ``ModelOutput``s into original batch order.

    Used only by ``BrainPolicy.propose_batch`` (§4.3.5).  ``order[j]``
    is the original batch row of the j-th row of the concatenated
    bucket outputs.  Optional fields are assumed consistently
    present/absent across buckets — they come from the same model and
    forward signature.
    """
    batch_size = len(order)
    perm = torch.tensor(order, dtype=torch.long)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(batch_size, dtype=torch.long)

    def gather(values: list[torch.Tensor | None]) -> torch.Tensor | None:
        if values[0] is None:
            return None
        combined = torch.cat(values, dim=0)
        return combined[inverse.to(combined.device)]

    def gather_dict(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {key: gather([d[key] for d in dicts]) for key in dicts[0]}

    return ModelOutput(
        action_logits=gather([o.action_logits for o in outputs]),
        confidence=gather([o.confidence for o in outputs]),
        target_fraction=gather([o.target_fraction for o in outputs]),
        fused_representation=gather([o.fused_representation for o in outputs]),
        branch_embeddings=gather_dict([o.branch_embeddings for o in outputs]),
        branch_confidences=gather_dict([o.branch_confidences for o in outputs]),
        soma_weights=gather([o.soma_weights for o in outputs]),
        soma_activation=gather([o.soma_activation for o in outputs]),
        regime_weights=gather([o.regime_weights for o in outputs]),
        value=gather([o.value for o in outputs]),
        memory_state=gather([o.memory_state for o in outputs]),
        temporal_embedding=gather([o.temporal_embedding for o in outputs]),
        risk_predictions=(
            gather_dict([o.risk_predictions for o in outputs])
            if outputs[0].risk_predictions is not None
            else None
        ),
        horizon_returns=(
            gather_dict([o.horizon_returns for o in outputs])
            if outputs[0].horizon_returns is not None
            else None
        ),
    )


@dataclass
class BrainPolicy:
    model: TradingModelProtocol
    device: torch.device
    feature_contract: FeatureContract | None = None
    # Train-split z-score stats from the checkpoint (C2 fix): every model
    # input is normalized with these before the forward, exactly as in
    # training. None = pre-fix checkpoint (raw inputs, undefined behavior).
    normalization: NormalizationStats | None = None
    # §7.0.4: per-symbol mfe/mae de-normalization scales from checkpoint
    # meta (training.risk_target_per_symbol_norm). None = checkpoint
    # trained on raw excursions — EV legs pass through unchanged.
    risk_target_scales: dict[str, dict[str, float]] | None = None
    max_state_gap_sec: int | None = None
    force_directional_flip: bool = False
    default_target_fraction: float = 0.05
    # FP16 autocast enable for inference. ``None`` auto-enables on CUDA and
    # skips on CPU; callers that want explicit control (e.g. to disable even
    # on CUDA) can pass ``True`` / ``False`` directly.
    use_amp: bool | None = None
    # §4.4.4 fail-fast: when True, refuse to wrap a model without a risk
    # head — its (0, 0) EV signals make the EV gate reject every trade,
    # producing a silent zero-trade session. Production construction paths
    # (policy_factory, runtime_bootstrap) pass risk.ev_gate_requires_signals
    # (default true); direct construction (tests, tools) defaults to off.
    ev_gate_requires_signals: bool = False
    memory_state_by_symbol: dict[str, torch.Tensor] = field(default_factory=dict)
    request_seq_by_symbol: dict[str, int] = field(default_factory=dict)
    last_timestamp_by_symbol: dict[str, datetime] = field(default_factory=dict)
    _diag_call_counter: int = 0

    def __post_init__(self) -> None:
        if self.ev_gate_requires_signals and getattr(self.model, "risk_head", None) is None:
            from src.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "BrainPolicy: model has no risk head, so the EV gate would "
                "receive (0, 0) signals and reject every trade (silent "
                "zero-trade session). Retrain with model.separate_risk_heads "
                "or set risk.ev_gate_requires_signals=false to accept an "
                "inert EV gate for an explicit experiment."
            )

    def _denormalize_ev(
        self, symbol: str, gain_bps: float, loss_bps: float
    ) -> tuple[float, float]:
        """Undo §7.0.4 per-symbol risk-target normalization on the EV legs.

        A checkpoint trained with ``risk_target_per_symbol_norm`` predicts
        excursions in per-symbol units (typical ≈ 1.0); multiplying back by
        the symbol's train-split scale restores raw fractions so the EV
        gate's contract is unchanged. Pre-normalization checkpoints
        (``risk_target_scales is None``) and unknown symbols pass through.
        """
        if self.risk_target_scales is None:
            return gain_bps, loss_bps
        scales = self.risk_target_scales.get(symbol)
        if not scales:
            return gain_bps, loss_bps
        return gain_bps * float(scales.get("mfe", 1.0)), loss_bps * float(scales.get("mae", 1.0))

    def _amp_enabled(self) -> bool:
        """Resolve whether autocast is active for this policy's device.

        Defaults to ``True`` on CUDA / ``False`` on CPU when ``use_amp`` was
        not passed explicitly. ``maybe_autocast`` further guards against a
        non-CUDA device so explicitly passing ``use_amp=True`` on CPU is a
        safe no-op rather than an error.
        """
        if self.use_amp is None:
            return self.device.type == "cuda"
        return bool(self.use_amp)

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

    def _build_proposal(
        self,
        snapshot: FeatureSnapshot,
        logit_values: list[float],
        confidence: float,
        target_fraction: float,
        soma_activation: float,
        branch_confidence_dict: dict[str, float],
        soma_weights: list[float],
        regime_weights: list[float],
        state_reset_reason: str | None,
        expected_gain_bps: float = 0.0,
        expected_loss_bps: float = 0.0,
    ) -> tuple[TradeProposal, dict[str, object]]:
        """Shared action-resolution → proposal → diagnostics logic."""
        action_index = int(max(range(len(logit_values)), key=lambda i: logit_values[i]))
        if self.force_directional_flip and Action(action_index) == Action.FLAT:
            short_logit = float(logit_values[int(Action.SHORT)])
            long_logit = float(logit_values[int(Action.LONG)])
            action_index = int(Action.LONG) if long_logit >= short_logit else int(Action.SHORT)
        action = ACTION_TO_SIGN[Action(action_index)]
        # Confidence semantics fix (2026-07-21, CONFIDENCE_AUDIT.md): the
        # proposal's confidence is P(resolved action class) from the action
        # logits — the quantity the EV gate consumes as win probability.
        # Pre-fix, legacy checkpoints emitted a sigmoid head trained as an
        # MFE/volatility/stability regressor (never P(win); measured rank
        # vs realized outcomes: rho -0.009 over 1,313 OOS round trips) and
        # decoupled checkpoints emitted the ARGMAX-class probability, which
        # names the wrong class under a forced flip. Computed exactly once,
        # here, where the resolved action is known; the raw head value
        # stays visible in diagnostics as head_confidence.
        head_confidence = confidence
        peak = max(logit_values)
        exps = [math.exp(value - peak) for value in logit_values]
        confidence = exps[action_index] / sum(exps)
        if not math.isfinite(confidence):
            # Non-finite logits (fp16 overflow under autocast saturates to
            # inf; inf - inf = NaN through the softmax) must fail CLOSED:
            # NaN would sail through the EV gate because `NaN <= 0` is
            # False. Zero confidence makes the gate's EV strictly
            # -loss - cost < 0, vetoing the garbage proposal.
            confidence = 0.0
        # §7.0.4 addendum (2026-07-20): MFE/MAE are PRICE-PATH up/down
        # excursions, so the extracted legs mean "gain/loss for a LONG".
        # The gate's contract is "gain/loss FOR THE PROPOSED TRADE" — a
        # short's gain is the down-move. Swap exactly once, here, where the
        # resolved action is known; both propose() and propose_batch()
        # funnel through this method. Pre-fix, 6,425 OOS short proposals
        # passed the pre-cost EV inequality at 0.3% vs 48.4% corrected.
        if action == -1:
            expected_gain_bps, expected_loss_bps = expected_loss_bps, expected_gain_bps
        if action == 0:
            target_fraction = 0.0
        elif abs(target_fraction) < 1e-6:
            target_fraction = self.default_target_fraction * action
        elif target_fraction == 0.0 or (target_fraction > 0) != (action > 0):
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
            metadata={"soma_activation": soma_activation},
            expected_gain_bps=expected_gain_bps,
            expected_loss_bps=expected_loss_bps,
        )
        diagnostics: dict[str, object] = {
            "branch_confidences": branch_confidence_dict,
            "soma_weights": soma_weights,
            "soma_activation": soma_activation,
            "regime_weights": regime_weights,
            "state_reset_reason": state_reset_reason,
            "expected_gain_bps": expected_gain_bps,
            "expected_loss_bps": expected_loss_bps,
            "head_confidence": head_confidence,
        }
        return proposal, diagnostics

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
        temporal_tensor = ensure_batch_dimension(
            to_tensor(snapshot.temporal_tensor, device=self.device, dtype=torch.float32)
        )
        if self.normalization is not None:
            branch_inputs = {
                name: self.normalization.apply_branch(name, value)
                for name, value in branch_inputs.items()
            }
            regime_input = self.normalization.apply_regime(regime_input)
            features = self.normalization.apply_features(features)
            temporal_tensor = self.normalization.apply_temporal(temporal_tensor)
        memory_state = self.memory_state_by_symbol.get(snapshot.symbol)
        with torch.no_grad(), maybe_autocast(self.device, self._amp_enabled()):
            output = self.model(
                branch_inputs=branch_inputs,
                regime_input=regime_input,
                memory_state=memory_state,
                full_features=features,
                temporal_tensor=temporal_tensor,
            )
        if output.memory_state is None:
            self.memory_state_by_symbol.pop(snapshot.symbol, None)
        else:
            self.memory_state_by_symbol[snapshot.symbol] = output.memory_state.detach()
        self.last_timestamp_by_symbol[snapshot.symbol] = snapshot.timestamp

        # ------------------------------------------------------------
        # Single-host-transfer scalar block.
        #
        # The previous implementation called ``.item()`` ten times per
        # proposal: once on argmax, twice on per-action logits, once on
        # confidence, once on target_fraction, once on soma_activation
        # (twice — proposal metadata + diagnostics), once each on every
        # branch_confidence value, plus ``.cpu().tolist()`` on
        # soma_weights and regime_weights. On CUDA each ``.item()`` is a
        # device→host sync, so the per-bar inner loop hit ten serialized
        # syncs. Pack the scalars we need into a single 1-D tensor and
        # call ``.tolist()`` once. Soma/regime weight vectors are still
        # transferred but in a single ``.cpu()`` call each.
        # ------------------------------------------------------------
        logits = output.action_logits.squeeze(0)  # (3,)
        confidence_t = output.confidence.squeeze(0).reshape(-1)[0]
        target_fraction_t = output.target_fraction.squeeze(0).reshape(-1)[0]
        soma_activation_t = output.soma_activation.squeeze(0).reshape(-1)[0]
        # Pack into a single tensor: [logit_short, logit_flat, logit_long,
        #                             confidence, target_fraction, soma_activation]
        scalar_block = torch.cat(
            [
                logits.detach().reshape(-1),
                confidence_t.detach().reshape(1),
                target_fraction_t.detach().reshape(1),
                soma_activation_t.detach().reshape(1),
            ],
            dim=0,
        )
        scalars = scalar_block.cpu().tolist()
        logit_values = scalars[: logits.shape[0]]
        confidence = float(scalars[logits.shape[0]])
        target_fraction = float(scalars[logits.shape[0] + 1])
        soma_activation = float(scalars[logits.shape[0] + 2])

        # Branch confidences: single host transfer.
        if output.branch_confidences:
            branch_names = list(output.branch_confidences.keys())
            branch_conf_tensor = torch.stack(
                [
                    output.branch_confidences[name].reshape(-1)[0].detach()
                    for name in branch_names
                ],
                dim=0,
            )
            branch_conf_values = branch_conf_tensor.cpu().tolist()
            branch_confidence_dict = {
                name: float(value)
                for name, value in zip(branch_names, branch_conf_values)
            }
        else:
            branch_confidence_dict = {}

        if self._diag_call_counter < _DIAG_MAX_CALLS:
            self._diag_call_counter += 1
            feats_cpu = snapshot.features.detach().cpu().float()
            reg_cpu = snapshot.regime_input.detach().cpu().float()
            _diag_logger.info(
                "inference_diag",
                path="propose",
                call=self._diag_call_counter,
                symbol=snapshot.symbol,
                action_logits=[float(x) for x in logit_values],
                # Raw head output — proposals carry P(resolved class); the
                # field name says so to avoid mixing the two quantities.
                head_confidence=confidence,
                target_fraction=target_fraction,
                soma_activation=soma_activation,
                branch_confidences=branch_confidence_dict,
                features_stats={
                    "shape": list(feats_cpu.shape),
                    "mean": float(feats_cpu.mean()),
                    "std": float(feats_cpu.std()),
                    "min": float(feats_cpu.min()),
                    "max": float(feats_cpu.max()),
                    "zero_frac": float((feats_cpu == 0).float().mean()),
                    "nan_count": int(torch.isnan(feats_cpu).sum()),
                },
                regime_stats={
                    "shape": list(reg_cpu.shape),
                    "mean": float(reg_cpu.mean()),
                    "std": float(reg_cpu.std()),
                    "values": [float(x) for x in reg_cpu.flatten().tolist()],
                },
            )

        expected_gain_bps, expected_loss_bps = _extract_ev_signals_single(output.risk_predictions)
        expected_gain_bps, expected_loss_bps = self._denormalize_ev(
            snapshot.symbol, expected_gain_bps, expected_loss_bps
        )

        return self._build_proposal(
            snapshot=snapshot,
            logit_values=logit_values,
            confidence=confidence,
            target_fraction=target_fraction,
            soma_activation=soma_activation,
            branch_confidence_dict=branch_confidence_dict,
            soma_weights=output.soma_weights.squeeze(0).detach().cpu().tolist(),
            regime_weights=output.regime_weights.squeeze(0).detach().cpu().tolist()
            if output.regime_weights is not None
            else [],
            state_reset_reason=state_reset_reason,
            expected_gain_bps=expected_gain_bps,
            expected_loss_bps=expected_loss_bps,
        )

    def propose_batch(
        self, snapshots: list[FeatureSnapshot]
    ) -> list[tuple[TradeProposal, dict[str, object]]]:
        """Run a single model forward for N snapshots and return per-symbol
        ``(TradeProposal, diagnostics)`` tuples in the same order.

        This is the runtime multi-symbol fast path.  Every symbol in
        ``snapshots`` contributes one row of a ``batch_size = N``
        forward, so the GPU runs one large matmul instead of N small
        ones.  Per-symbol GRU hidden states are stacked on entry and
        split back out on exit — each symbol keeps its own recurrent
        state exactly as if ``propose`` had been called N times in
        sequence.

        Safety invariants preserved:

        * Every branch's dimensionality check is still enforced by
          ``BrainTradingModel.forward``.
        * Per-symbol ``feature_contract.validate_snapshot`` runs
          before batching.
        * Per-symbol state-gap / non-monotonic-timestamp resets run
          before batching so a gap on symbol A does not corrupt
          symbol B's recurrent state in the same batch.
        * The force-directional flat→directional flip runs per-row
          with the same thresholds as the per-symbol path.
        """
        if not snapshots:
            return []

        device = self.device
        dtype = torch.float32
        batch_size = len(snapshots)

        # --- Validation + per-symbol state reset checks --------------------
        state_reset_reasons: list[str | None] = []
        for snapshot in snapshots:
            if self.feature_contract is not None:
                self.feature_contract.validate_snapshot(
                    snapshot, context=f"policy[{snapshot.symbol}]"
                )
            state_reset_reasons.append(
                self._maybe_reset_for_boundary(snapshot.symbol, snapshot.timestamp)
            )

        # --- Stack branch_inputs across snapshots --------------------------
        branch_names = list(snapshots[0].branch_inputs.keys())
        branch_inputs_batch: dict[str, torch.Tensor] = {}
        for name in branch_names:
            rows = []
            for snapshot in snapshots:
                t = ensure_batch_dimension(
                    to_tensor(snapshot.branch_inputs[name], device=device, dtype=dtype)
                )
                rows.append(t.squeeze(0))
            branch_inputs_batch[name] = torch.stack(rows, dim=0)

        regime_input_batch = torch.stack(
            [
                ensure_batch_dimension(
                    to_tensor(snapshot.regime_input, device=device, dtype=dtype)
                ).squeeze(0)
                for snapshot in snapshots
            ],
            dim=0,
        )
        features_batch = torch.stack(
            [
                ensure_batch_dimension(
                    to_tensor(snapshot.features, device=device, dtype=dtype)
                ).squeeze(0)
                for snapshot in snapshots
            ],
            dim=0,
        )

        # Temporal tensors may differ in time-dim when symbols haven't
        # reached the same buffer length (rare at steady state).  They
        # are forwarded in per-length sub-batches below — never padded.
        temporal_slices = [
            ensure_batch_dimension(
                to_tensor(snapshot.temporal_tensor, device=device, dtype=dtype)
            ).squeeze(0)
            for snapshot in snapshots
        ]
        if self.normalization is not None:
            temporal_slices = [self.normalization.apply_temporal(s) for s in temporal_slices]
        if self.normalization is not None:
            branch_inputs_batch = {
                name: self.normalization.apply_branch(name, value)
                for name, value in branch_inputs_batch.items()
            }
            regime_input_batch = self.normalization.apply_regime(regime_input_batch)
            features_batch = self.normalization.apply_features(features_batch)

        # --- Per-symbol previous memory state → (N, memory_dim) ------------
        memory_dim = getattr(self.model, "memory_dim", None)
        memory_rows: list[torch.Tensor] = []
        has_any_state = False
        for snapshot in snapshots:
            prior = self.memory_state_by_symbol.get(snapshot.symbol)
            if prior is not None:
                has_any_state = True
            memory_rows.append(prior)
        if has_any_state and memory_dim is not None:
            resolved_rows: list[torch.Tensor] = []
            for row in memory_rows:
                if row is None:
                    resolved_rows.append(torch.zeros(memory_dim, device=device, dtype=dtype))
                else:
                    tensor = row.to(device=device, dtype=dtype)
                    if tensor.dim() == 2:
                        tensor = tensor.squeeze(0)
                    resolved_rows.append(tensor)
            memory_state_batch: torch.Tensor | None = torch.stack(resolved_rows, dim=0)
        else:
            # No prior state for any snapshot — let the model initialize
            # itself (matches the per-symbol propose behaviour when this
            # symbol has never been seen).  When only *some* symbols have
            # state but none of the paths above fired, we still zero-fill
            # the missing ones above so this branch only runs on a pure
            # cold batch.
            memory_state_batch = None

        # --- Batched forward, length-bucketed (§4.3.5) ---------------------
        # The old approach left-zero-padded shorter temporal tensors to
        # the batch max and convolved the padding: eval-mode BatchNorm
        # maps pad zeros to a nonzero bias that the next conv smears
        # into real positions, and the adaptive average pool divides by
        # the padded length — so a symbol's proposal depended on which
        # other symbols shared its batch.  Masked pooling cannot fix
        # this (the contamination happens inside every residual block,
        # not just at the pool), so instead we forward one sub-batch
        # per distinct temporal length with no padding at all: each
        # row's computation is then identical to its single-sample
        # ``propose`` forward regardless of batch composition
        # (tests/test_model_input_hardening.py pins the equality).
        lengths = [slice_.shape[-1] for slice_ in temporal_slices]
        unique_lengths = sorted(set(lengths))
        with torch.no_grad(), maybe_autocast(self.device, self._amp_enabled()):
            if len(unique_lengths) == 1:
                output = self.model(
                    branch_inputs=branch_inputs_batch,
                    regime_input=regime_input_batch,
                    memory_state=memory_state_batch,
                    full_features=features_batch,
                    temporal_tensor=torch.stack(temporal_slices, dim=0),
                )
            else:
                order: list[int] = []
                bucket_outputs: list[ModelOutput] = []
                for length in unique_lengths:
                    rows = [i for i, l in enumerate(lengths) if l == length]
                    order.extend(rows)
                    bucket_outputs.append(
                        self.model(
                            branch_inputs={
                                name: value[rows]
                                for name, value in branch_inputs_batch.items()
                            },
                            regime_input=regime_input_batch[rows],
                            memory_state=(
                                memory_state_batch[rows]
                                if memory_state_batch is not None
                                else None
                            ),
                            full_features=features_batch[rows],
                            temporal_tensor=torch.stack(
                                [temporal_slices[i] for i in rows], dim=0
                            ),
                        )
                    )
                output = _reassemble_bucketed_outputs(bucket_outputs, order)

        # --- Update per-symbol memory state -------------------------------
        if output.memory_state is None:
            for snapshot in snapshots:
                self.memory_state_by_symbol.pop(snapshot.symbol, None)
        else:
            next_memory = output.memory_state.detach()  # (N, memory_dim)
            for i, snapshot in enumerate(snapshots):
                self.memory_state_by_symbol[snapshot.symbol] = next_memory[i]

        for snapshot in snapshots:
            self.last_timestamp_by_symbol[snapshot.symbol] = snapshot.timestamp

        # --- Single host transfer for all scalar outputs ------------------
        # Match ``propose``'s packed-tensor-to-tolist pattern but across
        # the batch dim so we pay 1 host sync for the whole batch.
        logits_all = output.action_logits.detach()  # (N, 3)
        confidence_all = output.confidence.detach().reshape(batch_size)
        target_all = output.target_fraction.detach().reshape(batch_size)
        soma_all = output.soma_activation.detach().reshape(batch_size)
        # Concatenate: [logits (N*3), confidence (N), target (N), soma (N)]
        scalar_block = torch.cat(
            [
                logits_all.reshape(-1),
                confidence_all,
                target_all,
                soma_all,
            ],
            dim=0,
        )
        scalars = scalar_block.cpu().tolist()
        logits_flat = scalars[: batch_size * 3]
        confidences = scalars[batch_size * 3 : batch_size * 4]
        target_fractions = scalars[batch_size * 4 : batch_size * 5]
        soma_activations = scalars[batch_size * 5 : batch_size * 6]

        soma_weights_all = output.soma_weights.detach().cpu().tolist()
        regime_weights_all: list[list[float]] | None = None
        if output.regime_weights is not None:
            regime_weights_all = output.regime_weights.detach().cpu().tolist()

        branch_confidence_tensors: dict[str, list[float]] = {}
        for name, tensor in output.branch_confidences.items():
            # Each tensor is (N, 1) or (N,); reshape to (N,) then tolist.
            branch_confidence_tensors[name] = tensor.detach().reshape(batch_size).cpu().tolist()

        if self._diag_call_counter < _DIAG_MAX_CALLS:
            self._diag_call_counter += 1
            # Use the first snapshot as the representative sample for
            # feature/regime stats — cheaper than per-symbol and enough to
            # answer the OOD question.
            repr_snap = snapshots[0]
            feats_cpu = repr_snap.features.detach().cpu().float()
            reg_cpu = repr_snap.regime_input.detach().cpu().float()
            _diag_logger.info(
                "inference_diag",
                path="propose_batch",
                call=self._diag_call_counter,
                batch_size=batch_size,
                symbols=[s.symbol for s in snapshots],
                action_logits_per_symbol=[
                    logits_flat[i * 3 : (i + 1) * 3] for i in range(batch_size)
                ],
                # Raw head outputs — proposals carry P(resolved class); the
                # field name says so to avoid mixing the two quantities.
                head_confidences=confidences,
                target_fractions=target_fractions,
                soma_activations=soma_activations,
                branch_confidences_first_symbol={
                    name: float(values[0]) for name, values in branch_confidence_tensors.items()
                },
                features_stats_first_symbol={
                    "shape": list(feats_cpu.shape),
                    "mean": float(feats_cpu.mean()),
                    "std": float(feats_cpu.std()),
                    "min": float(feats_cpu.min()),
                    "max": float(feats_cpu.max()),
                    "zero_frac": float((feats_cpu == 0).float().mean()),
                    "nan_count": int(torch.isnan(feats_cpu).sum()),
                },
                regime_stats_first_symbol={
                    "shape": list(reg_cpu.shape),
                    "mean": float(reg_cpu.mean()),
                    "std": float(reg_cpu.std()),
                    "values": [float(x) for x in reg_cpu.flatten().tolist()],
                },
                first_symbol=repr_snap.symbol,
            )

        gain_bps_list, loss_bps_list = _extract_ev_signals_batch(output.risk_predictions, batch_size)

        # --- Build per-symbol proposals via shared helper --------------------
        results: list[tuple[TradeProposal, dict[str, object]]] = []
        for i, snapshot in enumerate(snapshots):
            branch_confidence_dict = {
                name: float(values[i]) for name, values in branch_confidence_tensors.items()
            }
            gain_bps, loss_bps = self._denormalize_ev(
                snapshot.symbol, gain_bps_list[i], loss_bps_list[i]
            )
            results.append(self._build_proposal(
                snapshot=snapshot,
                logit_values=logits_flat[i * 3 : (i + 1) * 3],
                confidence=float(confidences[i]),
                target_fraction=float(target_fractions[i]),
                soma_activation=float(soma_activations[i]),
                branch_confidence_dict=branch_confidence_dict,
                soma_weights=soma_weights_all[i],
                regime_weights=regime_weights_all[i] if regime_weights_all is not None else [],
                state_reset_reason=state_reset_reasons[i],
                expected_gain_bps=gain_bps,
                expected_loss_bps=loss_bps,
            ))
        return results
