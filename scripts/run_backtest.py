from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.engine import BacktestEngine
from src.core.config import load_config
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import HistoricalDataService, generate_synthetic_bars, parse_date_or_datetime
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.memory_bank import MemoryBank
from src.memory.retrieval import AssociativeRetriever
from src.memory.storage import load_memory_bank, save_memory_bank
from src.memory.updater import AssociativeMemoryUpdater
from src.models.factory import resolve_inference_model
from src.models.state_encoder import ModelStateEncoder
from src.risk.risk_manager import RiskManager
from src.strategies.baseline_policy import BaselineMomentumPolicy
from src.strategies.brain_policy import BrainPolicy
from src.strategies.memory_policy import MemoryAwarePolicy


def _print_report(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label}: total_return={metrics['total_return']:.4f} "
        f"max_drawdown={metrics['max_drawdown']:.4f} "
        f"win_rate={metrics['win_rate']:.4f} "
        f"turnover={metrics['turnover']:.4f} "
        f"fills={int(metrics['fill_count'])}"
    )


def _load_bars_by_symbol(config, *, use_real_data: bool, allow_mock: bool) -> dict[str, list[MarketBar]]:
    if not use_real_data:
        return {
            symbol: generate_synthetic_bars(symbol, 240)
            for symbol in config.runtime.symbols
        }
    if credentials_available():
        client = AlpacaClient(paper=config.broker.paper, data_feed=config.broker.data_feed)
    elif allow_mock:
        client = MockBrokerClient(paper=True, data_feed="mock")
    else:
        client = None
    service = HistoricalDataService(
        client=client,
        cache=HistoricalBarCache(config.data.cache_dir),
        timezone_normalization=config.data.timezone_normalization,
        max_missing_gap_handling=config.data.max_missing_gap_handling,
    )
    start = parse_date_or_datetime(
        config.data.start_date,
        default=datetime.now(UTC) - timedelta(days=365 * 2),
    )
    end = parse_date_or_datetime(config.data.end_date, default=datetime.now(UTC))
    bars_by_symbol, _ = service.load_bars_for_symbols(
        config.data.symbols,
        timeframe=config.data.timeframe,
        start=start,
        end=end,
        force_refresh=config.data.force_refresh,
    )
    return bars_by_symbol


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal backtest")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load for the neural model")
    parser.add_argument(
        "--use-memory",
        action="store_true",
        help="Enable associative market memory decision support during the backtest.",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Backtest on cached/fetched real historical bars instead of synthetic bars.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow explicit mock bar generation for --real-data when Alpaca credentials are unavailable.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/default.yaml"],
        help="Config file(s), later files override earlier files.",
    )
    args = parser.parse_args()
    config = load_config(*args.config)
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    if args.use_memory:
        config.memory.enabled = True
        if config.memory.runtime_update_mode == "pending_only":
            config.memory.runtime_update_mode = "append_when_ready"
    use_real_data = args.real_data or config.training.use_real_historical_data
    bars_by_symbol = _load_bars_by_symbol(config, use_real_data=use_real_data, allow_mock=args.allow_mock)
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))

    baseline_engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
    baseline_report = baseline_engine.run(bars_by_symbol, BaselineMomentumPolicy())
    _print_report("baseline", baseline_report.metrics)

    sample_symbol = next(iter(bars_by_symbol))
    sample_snapshot = pipeline.build(
        bars_by_symbol[sample_symbol][: config.runtime.warmup_bars + 1],
        symbol=sample_symbol,
    )
    artifacts = resolve_inference_model(
        model_config=config.model,
        feature_config=config.features,
        feature_contract=sample_snapshot.contract,
        device=torch.device("cpu"),
        default_model_kind="brain",
    )
    brain_policy = BrainPolicy(
        model=artifacts.model,
        device=torch.device("cpu"),
        feature_contract=artifacts.feature_contract,
        max_state_gap_sec=config.runtime.state_reset_gap_sec,
    )
    if config.memory.enabled:
        encoder = ModelStateEncoder(
            model=artifacts.model,
            device=torch.device("cpu"),
            feature_contract=artifacts.feature_contract,
            expected_latent_dim=config.memory.latent_dim,
        )
        sample_encoded = encoder.encode(sample_snapshot)
        memory_bank: MemoryBank
        persist_path = Path(config.memory.persist_path)
        if persist_path.exists() and not config.memory.rebuild_on_startup:
            memory_bank = load_memory_bank(persist_path, map_location="cpu")
            memory_bank.feature_contract.assert_matches(
                artifacts.feature_contract,
                context=f"memory_bank[{persist_path.name}]",
            )
            if config.memory.dynamic_growth_enabled and not memory_bank.memory_units:
                memory_bank.refresh_organization(config.memory)
                save_memory_bank(persist_path, memory_bank)
        else:
            memory_bank = MemoryBank(
                feature_contract=artifacts.feature_contract,
                latent_dim=sample_encoded.latent.shape[0],
                similarity_metric=config.memory.similarity_metric,
                max_items=config.memory.max_items,
                require_encoder_match=config.memory.require_encoder_match,
                meta={"source": "backtest_runtime"},
            )
        brain_policy = MemoryAwarePolicy(
            base_policy=brain_policy,
            encoder=encoder,
            retriever=AssociativeRetriever(memory_bank, config.memory),
            updater=AssociativeMemoryUpdater(memory_bank, config.memory),
            require_strong_match_for_trade=config.memory.require_strong_match_for_trade,
            fusion_config=config.memory.decision_fusion,
        )
    brain_engine = BacktestEngine(config, pipeline, RiskManager(config.risk.model_copy()))
    brain_report = brain_engine.run(bars_by_symbol, brain_policy)
    label = f"{artifacts.model_kind}_{'checkpoint' if config.model.checkpoint_path else 'untrained'}"
    _print_report(label, brain_report.metrics)
    if config.memory.enabled:
        print(
            "memory:"
            f" item_count={int(brain_report.metrics.get('memory_item_count', 0.0))}"
            f" cluster_count={int(brain_report.metrics.get('memory_cluster_count', 0.0))}"
            f" unit_count={int(brain_report.metrics.get('memory_unit_count', 0.0))}"
            f" esom_nodes={int(brain_report.metrics.get('memory_esom_node_count', 0.0))}"
            f" weak_match_rate={brain_report.metrics.get('memory_weak_match_rate', 0.0):.4f}"
            f" avg_similarity={brain_report.metrics.get('memory_average_similarity', 0.0):.4f}"
            f" avg_confidence={brain_report.metrics.get('memory_average_confidence', 0.0):.4f}"
            f" veto_rate={brain_report.metrics.get('memory_veto_rate', 0.0):.4f}"
            f" dampened_rate={brain_report.metrics.get('memory_dampened_rate', 0.0):.4f}"
            f" quarantined={int(brain_report.metrics.get('memory_quarantined_unit_count', 0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
