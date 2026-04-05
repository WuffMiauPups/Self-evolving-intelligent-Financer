from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.data.alpaca_client import AlpacaClient, MockBrokerClient, credentials_available
from src.data.cache import HistoricalBarCache
from src.data.historical_data import HistoricalDataService, generate_synthetic_bars, parse_date_or_datetime
from src.data.schemas import MarketBar
from src.features.feature_pipeline import FeaturePipeline
from src.memory.encoder_eval import evaluate_encoder_quality
from src.memory.storage import save_memory_bank
from src.memory.updater import build_memory_bank_from_bars
from src.models.factory import resolve_inference_model
from src.models.state_encoder import ModelStateEncoder


def _load_bars_by_symbol(
    config,
    *,
    use_real_data: bool,
    bar_override: int | None,
    allow_mock: bool,
) -> dict[str, list[MarketBar]]:
    if use_real_data:
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

    bar_count = bar_override or (
        config.memory.build_history_bars + max(config.memory.forward_horizons) + config.runtime.warmup_bars
    )
    return {
        symbol: generate_synthetic_bars(symbol, bar_count)
        for symbol in config.runtime.symbols
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an associative market memory bank")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to load for the neural encoder/model")
    parser.add_argument(
        "--from-checkpoint",
        dest="checkpoint",
        help="Alias for --checkpoint.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Optional synthetic bar count override for the offline memory build.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/default.yaml"],
        help="Config file(s), later files override earlier files.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run encoder/memory quality evaluation after building the bank.",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Build memory from cached/fetched real historical bars.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow mock historical generation when --real-data is requested without Alpaca credentials.",
    )
    args = parser.parse_args()
    config = load_config(*args.config)
    config.memory.enabled = True
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    use_real_data = args.real_data or config.training.use_real_historical_data
    if use_real_data and config.memory.build_from_real_checkpoint and not config.model.checkpoint_path:
        raise RuntimeError("memory.build_from_real_checkpoint=true requires --from-checkpoint/--checkpoint for real-data memory builds")
    bars_by_symbol = _load_bars_by_symbol(
        config,
        use_real_data=use_real_data,
        bar_override=args.bars,
        allow_mock=args.allow_mock,
    )
    pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
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
    encoder = ModelStateEncoder(
        model=artifacts.model,
        device=torch.device("cpu"),
        feature_contract=artifacts.feature_contract,
        expected_latent_dim=config.memory.latent_dim,
    )
    memory_config = config.memory.model_copy(
        update={
            "build_from_real_checkpoint": use_real_data or config.memory.build_from_real_checkpoint,
            "forward_horizons": list(config.training.forward_horizons),
            "path_label_rules": dict(config.training.path_label_rules),
        }
    )
    bank = build_memory_bank_from_bars(
        bars_by_symbol=bars_by_symbol,
        pipeline=pipeline,
        encoder=encoder,
        memory_config=memory_config,
        warmup_bars=config.runtime.warmup_bars,
    )
    save_memory_bank(config.memory.persist_path, bank)
    print(
        "memory_build_complete"
        f" path={config.memory.persist_path}"
        f" items={len(bank)}"
        f" clusters={len(bank.prototypes)}"
        f" units={len(bank.memory_units)}"
        f" mature_units={sum(1 for unit in bank.memory_units if unit.status == 'mature')}"
        f" esom_nodes={0 if bank.esom_map is None else len(bank.esom_map.nodes)}"
        f" latent_dim={bank.latent_dim}"
        f" source={'real' if use_real_data else 'synthetic'}"
    )
    if args.evaluate:
        report = evaluate_encoder_quality(bank)
        organization_report = bank.meta.get("organization_report", {})
        health = organization_report.get("health", {})
        summary = bank.summary()
        print(
            "memory_eval"
            f" nn_similarity={report.nearest_neighbor_similarity:.4f}"
            f" nn_consistency={report.nearest_neighbor_directional_consistency:.4f}"
            f" outcome_dispersion={report.average_outcome_dispersion:.4f}"
            f" unit_compactness={report.unit_compactness:.4f}"
            f" unit_separation={report.unit_separation:.4f}"
            f" retrieval_coverage={report.retrieval_coverage:.4f}"
            f" weak_match_rate={report.weak_match_rate:.4f}"
            f" conflict_rate={report.conflict_rate:.4f}"
            f" avg_calibrated_confidence={report.average_calibrated_confidence:.4f}"
            f" mature_candidate_ratio={float(health.get('mature_candidate_ratio', 0.0)):.4f}"
            f" topology_density={float(health.get('topology_density', 0.0)):.4f}"
            f" avg_saturation={float(health.get('average_saturation', 0.0)):.4f}"
            f" avg_plasticity={float(health.get('average_plasticity', 0.0)):.4f}"
            f" conflict_heavy_units={int(health.get('conflict_heavy_unit_count', 0))}"
            f" graph_edges={int(health.get('graph_edge_count', 0))}"
            f" create_vs_update={float(health.get('create_vs_update_ratio', 0.0)):.4f}"
            f" quarantined={int(health.get('quarantined_units', 0))}"
            f" symbol_profiles={int(summary.get('symbol_calibration_profile_count', 0))}"
            f" regime_profiles={int(summary.get('regime_calibration_profile_count', 0))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
