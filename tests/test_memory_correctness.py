"""Phase 2 §6.2-A — memory correctness.

Pins the four reviewed defects:
- the backtest temporal cutoff was stored but never applied to retrieval
  (future outcomes could surface as neighbors when replaying history);
- the query's own stored (symbol, timestamp) could be its own neighbor;
- ``MemoryBank.extend`` was a per-append O(N²) loop while its caller's
  docstring claimed amortization;
- pending observations persisted datetime objects (forcing the unsafe
  torch.load fallback forever), grew without bound, and the bank was
  saved synchronously on the decision hot path.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import torch

from src.core.config import FeatureConfig, MemoryConfig
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank, feature_contract_fingerprint
from src.memory.outcomes import build_risk_summary
from src.memory.schemas import MemoryItem, OutcomeSummary
from src.memory.updater import AssociativeMemoryUpdater
from src.models.state_encoder import EncodedState

_T0 = datetime(2026, 6, 1, 14, 30, tzinfo=UTC)


def _bars(n: int = 80, symbol: str = "AAPL") -> list[MarketBar]:
    return [
        MarketBar(
            symbol=symbol,
            timestamp=_T0 + timedelta(minutes=5 * i),
            open=100.0 + 0.1 * i,
            high=100.5 + 0.1 * i,
            low=99.5 + 0.1 * i,
            close=100.2 + 0.1 * i,
            volume=1000.0 + i,
            spread=0.05,
        )
        for i in range(n)
    ]


def _contract():
    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))
    return pipeline.build(_bars(), symbol="AAPL").contract


def _outcome(primary: float = 0.001) -> OutcomeSummary:
    return OutcomeSummary(
        future_returns={1: primary},
        max_favorable_excursion=abs(primary) * 2,
        max_adverse_excursion=-abs(primary),
        realized_volatility=0.01,
        drawdown=0.0,
        path_label="chop",
        direction_class="flat",
        time_to_move=1,
    )


def _item(
    idx: int,
    *,
    symbol: str = "AAPL",
    timestamp: datetime | None = None,
    latent: torch.Tensor | None = None,
    primary: float = 0.001,
) -> MemoryItem:
    return MemoryItem(
        item_id=f"{symbol}-{idx}",
        symbol=symbol,
        timestamp=timestamp or (_T0 + timedelta(minutes=5 * idx)),
        latent=latent if latent is not None else torch.randn(8),
        feature_contract_id="cid",
        regime_weights=None,
        branch_confidences={},
        branch_norms={},
        outcome=_outcome(primary),
        risk=build_risk_summary(_outcome(primary)),
    )


def _bank(max_items: int = 100) -> MemoryBank:
    contract = _contract()
    return MemoryBank(
        feature_contract=contract,
        latent_dim=8,
        max_items=max_items,
        feature_contract_id=feature_contract_fingerprint(contract),
    )


# ---------------------------------------------------------------------------
# Temporal cutoff + self-exclusion in retrieval
# ---------------------------------------------------------------------------


class TestTemporalCutoff:
    def test_items_at_or_after_cutoff_never_surface(self) -> None:
        bank = _bank()
        base = torch.ones(8)
        # Three items, increasingly similar to the query but later in time.
        bank.append(_item(0, timestamp=_T0, latent=base + 0.30 * torch.randn(8)))
        bank.append(_item(1, timestamp=_T0 + timedelta(days=1), latent=base.clone()))
        bank.append(_item(2, timestamp=_T0 + timedelta(days=2), latent=base.clone()))
        cutoff = _T0 + timedelta(days=1)
        matches = bank.query_top_k_simple(base, k=3, exclude_after=cutoff)
        assert [m.item_id for m in matches] == ["AAPL-0"]

    def test_without_cutoff_all_items_eligible(self) -> None:
        bank = _bank()
        base = torch.ones(8)
        for i in range(3):
            bank.append(_item(i, latent=base.clone()))
        assert len(bank.query_top_k_simple(base, k=3)) == 3

    def test_query_own_item_is_excluded(self) -> None:
        bank = _bank()
        latent = torch.ones(8)
        own_ts = _T0 + timedelta(minutes=50)
        bank.append(_item(0, symbol="AAPL", timestamp=own_ts, latent=latent.clone()))
        bank.append(_item(1, symbol="MSFT", timestamp=own_ts, latent=latent.clone()))
        matches = bank.query_top_k_simple(
            latent, k=2, exclude_item=("AAPL", own_ts)
        )
        # The identical MSFT-at-same-timestamp item stays; own copy is gone.
        assert [m.symbol for m in matches] == ["MSFT"]

    def test_cutoff_survives_persistence_round_trip(self) -> None:
        bank = _bank()
        base = torch.ones(8)
        bank.append(_item(0, timestamp=_T0, latent=base.clone()))
        bank.append(_item(1, timestamp=_T0 + timedelta(days=2), latent=base.clone()))
        reloaded = MemoryBank.from_payload(bank.to_payload())
        matches = reloaded.query_top_k_simple(
            base, k=2, exclude_after=_T0 + timedelta(days=1)
        )
        assert [m.item_id for m in matches] == ["AAPL-0"]


class TestPolicyRetrievalHonorsCutoff:
    def test_memory_policy_passes_cutoff_and_self_exclusion(self) -> None:
        from src.strategies.memory_policy import MemoryAwarePolicy

        bank = _bank()
        latent = torch.ones(8)
        now = _T0 + timedelta(days=1)
        bank.append(_item(0, timestamp=_T0, latent=latent.clone()))          # past → eligible
        bank.append(_item(1, timestamp=now, latent=latent.clone()))          # own state → excluded
        bank.append(_item(2, timestamp=now + timedelta(days=1), latent=latent.clone()))  # future → excluded
        policy = MemoryAwarePolicy(
            base_policy=SimpleNamespace(), encoder=SimpleNamespace(), bank=bank
        )
        policy.set_temporal_cutoff(now)
        encoded = EncodedState(
            symbol="AAPL", timestamp=now, latent=latent, feature_contract_id="cid"
        )
        matches = policy._retrieve_neighbors(encoded)
        assert [m.item_id for m in matches] == ["AAPL-0"]


# ---------------------------------------------------------------------------
# Vectorized extend
# ---------------------------------------------------------------------------


class TestVectorizedExtend:
    def test_extend_equals_sequential_append(self) -> None:
        items = [_item(i) for i in range(50)]
        via_extend = _bank()
        via_append = _bank()
        added = via_extend.extend(items)
        for item in items:
            via_append.append(item)
        assert added == 50
        assert [i.item_id for i in via_extend.items] == [i.item_id for i in via_append.items]
        assert torch.allclose(via_extend.latent_matrix, via_append.latent_matrix)
        assert torch.equal(via_extend._timestamps_epoch, via_append._timestamps_epoch)

    def test_extend_applies_fifo_trim_once(self) -> None:
        bank = _bank(max_items=10)
        bank.extend([_item(i) for i in range(25)])
        assert len(bank) == 10
        assert [i.item_id for i in bank.items] == [f"AAPL-{i}" for i in range(15, 25)]
        assert bank.latent_matrix.shape[0] == 10
        assert bank._timestamps_epoch.shape[0] == 10

    def test_extend_rejects_bad_items_and_counts_accepted(self) -> None:
        bank = _bank()
        good = _item(0)
        wrong_dim = _item(1, latent=torch.randn(3))
        assert bank.extend([good, wrong_dim]) == 1
        assert len(bank) == 1

    def test_retrieval_consistent_after_mixed_append_extend(self) -> None:
        bank = _bank()
        bank.append(_item(0, latent=torch.ones(8)))
        bank.extend([_item(i, latent=torch.ones(8)) for i in range(1, 4)])
        bank.append(_item(4, latent=torch.ones(8)))
        assert len(bank.query_top_k_simple(torch.ones(8), k=10)) == 5


# ---------------------------------------------------------------------------
# Updater lifecycle: pending caps + safe persistence format
# ---------------------------------------------------------------------------


def _encoded(idx: int, symbol: str = "AAPL") -> EncodedState:
    return EncodedState(
        symbol=symbol,
        timestamp=_T0 + timedelta(minutes=5 * idx),
        latent=torch.randn(8),
        feature_contract_id="cid",
    )


class TestUpdaterLifecycle:
    def test_pending_queue_is_bounded(self) -> None:
        config = MemoryConfig(runtime_update_mode="pending_only", pending_max_per_symbol=5)
        updater = AssociativeMemoryUpdater(bank=_bank(), config=config)
        for i in range(12):
            updater.observe(_encoded(i))
        pending = updater.pending_by_symbol["AAPL"]
        assert len(pending) == 5
        # Oldest were dropped; the newest observations survive.
        assert pending[0].timestamp == _T0 + timedelta(minutes=5 * 7)

    def test_save_pending_loads_under_weights_only(self, tmp_path) -> None:
        config = MemoryConfig(runtime_update_mode="pending_only")
        updater = AssociativeMemoryUpdater(bank=_bank(), config=config)
        updater.observe(_encoded(0))
        updater.observe(_encoded(1, symbol="MSFT"))
        path = tmp_path / "pending.pt"
        assert updater.save_pending(path) == 2
        # The strict loader must succeed directly — datetime objects in the
        # payload used to force the unsafe weights_only=False fallback.
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert len(payload["pending_items"]) == 2
        assert all(isinstance(p["timestamp"], str) for p in payload["pending_items"])

        restored = AssociativeMemoryUpdater(bank=_bank(), config=config)
        assert restored.load_pending(path) == 2
        assert restored.pending_by_symbol["AAPL"][0].timestamp == _T0


# ---------------------------------------------------------------------------
# Build provenance stamps
# ---------------------------------------------------------------------------


def test_build_memory_bank_stamps_provenance() -> None:
    from src.memory.updater import build_memory_bank_from_bars

    bars = _bars(70)
    pipeline = FeaturePipeline(FeatureConfig(), device=torch.device("cpu"))

    class _FakeEncoder:
        encoder_fingerprint = "PROV_FP"

        def encode(self, snapshot):
            return self.encode_batch([snapshot])[0]

        def encode_batch(self, snapshots):
            return [
                EncodedState(
                    symbol=s.symbol,
                    timestamp=s.timestamp,
                    latent=torch.randn(8),
                    feature_contract_id="cid",
                    encoder_fingerprint="PROV_FP",
                )
                for s in snapshots
            ]

    config = MemoryConfig(forward_horizons=[1, 3], max_items=1000)
    bank = build_memory_bank_from_bars(
        bars_by_symbol={"AAPL": bars},
        pipeline=pipeline,
        encoder=_FakeEncoder(),
        memory_config=config,
        warmup_bars=40,
        embargo_bars=2,
    )
    assert bank.meta["builder_warmup_bars"] == 40
    assert bank.meta["builder_embargo_bars"] == 2
    assert bank.meta["builder_outcome_horizon"] == 3
    # Last decision bar: index len(bars) - horizon - embargo - 1 = 64.
    assert bank.meta["builder_built_through_ts"] == bars[64].timestamp.isoformat()
    assert bank.encoder_fingerprint == "PROV_FP"


# ---------------------------------------------------------------------------
# Debounced hot-path persistence
# ---------------------------------------------------------------------------


class TestDebouncedPersistence:
    def _ctx_and_policy(self, tmp_path, monkeypatch, *, debounce: float):
        from src.execution import tick_processor

        saves: list[str] = []
        monkeypatch.setattr(
            tick_processor, "save_memory_bank", lambda path, bank: saves.append("bank")
        )

        class _FakePolicy:
            updater = SimpleNamespace(save_pending=lambda path: saves.append("pending"))

            def prepare_symbol_history(self, symbol, bars):
                return []

            def record_snapshot(self, symbol):
                return None

            def metrics_snapshot(self):
                return {}

            def set_temporal_cutoff(self, cutoff):
                return None

            def propose(self, snapshot):
                raise NotImplementedError

            def propose_batch(self, snapshots):
                raise NotImplementedError

            def reset_symbol_state(self, symbol, *, reason="x"):
                return reason

            def reset_all_state(self, *, reason="x"):
                return reason

        config = SimpleNamespace(
            memory=SimpleNamespace(
                save_debounce_sec=debounce,
                persist_path=str(tmp_path / "bank.pt"),
                pending_store_path=str(tmp_path / "pending.pt"),
                persist_pending_observations=True,
            )
        )
        ctx = SimpleNamespace(
            policy=_FakePolicy(),
            config=config,
            memory_bank=object(),
            memory_bank_dirty=False,
            last_memory_persist_monotonic=0.0,
        )
        return ctx, saves

    def test_saves_are_debounced_not_per_tick(self, tmp_path, monkeypatch) -> None:
        from src.execution.tick_processor import _maybe_persist_memory

        ctx, saves = self._ctx_and_policy(tmp_path, monkeypatch, debounce=3600.0)
        logger = SimpleNamespace(warning=lambda *a, **k: None)
        ctx.memory_bank_dirty = True
        _maybe_persist_memory(ctx, logger=logger)  # first call persists
        assert saves == ["bank", "pending"]
        assert ctx.memory_bank_dirty is False
        ctx.memory_bank_dirty = True
        for _ in range(5):  # further ticks inside the window: no IO
            _maybe_persist_memory(ctx, logger=logger)
        assert saves == ["bank", "pending"]
        assert ctx.memory_bank_dirty is True  # still dirty, awaiting window

    def test_clean_bank_skips_bank_save_but_persists_pending(self, tmp_path, monkeypatch) -> None:
        from src.execution.tick_processor import _maybe_persist_memory

        ctx, saves = self._ctx_and_policy(tmp_path, monkeypatch, debounce=0.0)
        logger = SimpleNamespace(warning=lambda *a, **k: None)
        _maybe_persist_memory(ctx, logger=logger)
        assert saves == ["pending"]
