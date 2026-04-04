from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

from src.core.config import load_config
from src.data.alpaca_client import MockBrokerClient
from src.data.cache import HistoricalBarCache
from src.data.dataset import TrainingSample
from src.data.historical_data import HistoricalDataService, generate_synthetic_bars
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.outcomes import compute_outcome_summary
from src.memory.storage import load_memory_bank
from src.training.checkpoints import load_checkpoint_payload
from src.training.class_balance import build_class_weights, compute_target_diagnostics
from src.training.data_module import build_real_data_training_bundle
from src.training.supervised import run_supervised_training
from src.training.targets import build_training_targets


class CountingMockClient(MockBrokerClient):
    def __init__(self) -> None:
        super().__init__()
        self.range_calls: list[tuple[datetime, datetime]] = []

    def get_historical_bars_range(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        self.range_calls.append((start, end))
        return super().get_historical_bars_range(symbol=symbol, timeframe=timeframe, start=start, end=end)


def test_historical_cache_roundtrip_and_incremental_hits(tmp_path: Path) -> None:
    client = CountingMockClient()
    service = HistoricalDataService(client=client, cache=HistoricalBarCache(tmp_path))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(minutes=20)
    bars_first, report_first = service.load_bars_range("AAPL", timeframe="1Min", start=start, end=end)
    bars_second, report_second = service.load_bars_range("AAPL", timeframe="1Min", start=start, end=end)
    cache_only_service = HistoricalDataService(client=None, cache=HistoricalBarCache(tmp_path))
    bars_third, report_third = cache_only_service.load_bars_range("AAPL", timeframe="1Min", start=start, end=end)
    assert len(bars_first) == len(bars_second) > 0
    assert len(bars_third) == len(bars_second)
    assert report_first.cache_hit is False
    assert report_second.cache_hit is True
    assert report_third.cache_hit is True
    assert len(client.range_calls) == 1


def test_real_data_bundle_prevents_time_leakage_between_splits() -> None:
    config = load_config("configs/default.yaml")
    config.runtime.warmup_bars = 16
    config.training.sequence_length = 16
    config.training.forward_horizons = [1, 3]
    config.training.train_split = 0.6
    config.training.val_split = 0.2
    config.training.test_split = 0.2
    config.data.symbols = ["AAPL", "MSFT"]
    bars_by_symbol = {
        "AAPL": generate_synthetic_bars("AAPL", 180),
        "MSFT": generate_synthetic_bars("MSFT", 180),
    }
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
    bundle = build_real_data_training_bundle(config=config, bars_by_symbol=bars_by_symbol, pipeline=pipeline)
    for symbol in config.data.symbols:
        train_timestamps = [sample.timestamp for sample in bundle.samples_by_split["train"] if sample.symbol == symbol]
        val_timestamps = [sample.timestamp for sample in bundle.samples_by_split["val"] if sample.symbol == symbol]
        test_timestamps = [sample.timestamp for sample in bundle.samples_by_split["test"] if sample.symbol == symbol]
        assert train_timestamps
        assert val_timestamps
        assert test_timestamps
        assert max(train_timestamps) < min(val_timestamps)
        assert max(val_timestamps) < min(test_timestamps)


def test_real_target_generation_uses_future_paths_not_runtime_guards() -> None:
    bars = [
        MarketBar("AAPL", datetime(2026, 1, 1, 10, 0, tzinfo=UTC) + timedelta(minutes=index), 100.0 + index, 101.0 + index, 99.0 + index, close, 1000.0)
        for index, close in enumerate([100.0, 101.0, 103.0, 104.0, 105.0])
    ]
    outcome = compute_outcome_summary(
        bars,
        1,
        horizons=[1, 3],
        path_label_rules={
            "breakout_return_threshold": 0.01,
            "mean_reversion_return_threshold": 0.005,
            "flat_return_threshold": 0.001,
            "time_to_move_threshold": 0.002,
        },
    )
    targets = build_training_targets(
        outcome,
        label_epsilon=0.001,
        regression_targets=["primary_return", "drawdown", "stability"],
    )
    assert targets.direction_name == "long"
    assert targets.primary_return > 0.03
    assert "drawdown" in targets.regression_targets
    assert "stability" in targets.regression_targets


def test_class_balance_diagnostics_and_weights_surface_flat_dominance() -> None:
    samples = [
        TrainingSample(
            features=torch.zeros(2),
            branch_inputs={"trend": torch.zeros(1)},
            regime_input=torch.zeros(1),
            temporal_tensor=torch.zeros(2, 4),
            label=torch.tensor(1, dtype=torch.long),
            symbol="AAPL",
            direction_name="flat",
            regression_targets={"primary_return": 0.0},
        )
        for _ in range(8)
    ] + [
        TrainingSample(
            features=torch.zeros(2),
            branch_inputs={"trend": torch.zeros(1)},
            regime_input=torch.zeros(1),
            temporal_tensor=torch.zeros(2, 4),
            label=torch.tensor(2, dtype=torch.long),
            symbol="MSFT",
            direction_name="long",
            regression_targets={"primary_return": 0.01},
        )
    ]
    diagnostics = compute_target_diagnostics(
        samples,
        split="train",
        minority_class_floor_warning=0.15,
        label_epsilon=0.001,
        inspect_target_boundaries=True,
    )
    weights = build_class_weights(samples, mode="balanced", cap=3.0)
    assert diagnostics.flat_rate > 0.8
    assert diagnostics.warnings
    assert diagnostics.class_ratios["flat"] > 0.8
    assert weights is not None
    assert weights.shape[0] == 3


def test_checkpoint_metadata_and_memory_prebuild_from_training(tmp_path: Path) -> None:
    config = load_config("configs/default.yaml")
    config.logging.file_enabled = False
    config.training.epochs = 1
    config.training.batch_size = 8
    config.training.sequence_length = 16
    config.runtime.warmup_bars = 16
    config.training.forward_horizons = [1, 3]
    config.training.save_memory_after_training = True
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.memory.persist_path = str(tmp_path / "memory.pt")
    config.data.symbols = ["AAPL", "MSFT"]
    result = run_supervised_training(config)
    checkpoint_payload = load_checkpoint_payload(result["checkpoint"])
    meta = checkpoint_payload["meta"]
    assert "data_config" in meta
    assert "training_config" in meta
    assert "evaluation_reports" in meta
    assert "collapse_warnings" in meta
    assert "training_split_ranges" in meta
    assert meta["symbol_universe"] == ["AAPL", "MSFT"]
    assert "confusion_matrix" in meta["evaluation_reports"]["test"]
    assert "per_symbol_accuracy" in meta["evaluation_reports"]["test"]
    assert "per_class_metrics" in meta["evaluation_reports"]["test"]
    assert "long_short_confusion" in meta["evaluation_reports"]["test"]
    bank = load_memory_bank(config.memory.persist_path)
    assert bank.summary()["item_count"] > 0
    assert bank.feature_contract.to_dict() == meta["feature_contract"]
