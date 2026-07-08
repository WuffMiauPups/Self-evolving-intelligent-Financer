from __future__ import annotations

import argparse
import sys
import time
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
from src.memory.bank_cache import (
    compute_bank_fingerprint,
    load_cached_bank,
    save_cached_bank,
)
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
        "--real-data",
        action="store_true",
        help="Build memory from cached/fetched real historical bars.",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Allow mock historical generation when --real-data is requested without Alpaca credentials.",
    )
    parser.add_argument(
        "--no-bank-cache",
        action="store_true",
        help="Skip the on-disk bank cache and force a fresh build even when a "
             "matching cached bank exists. Implies no cache write either.",
    )
    args = parser.parse_args()
    config = load_config(*args.config)
    # Configure logging so ``logger.info(...)`` calls from memory modules
    # (updater, memory_bank, topology_growth, etc.) are visible to whoever
    # is driving the subprocess.  Without this, stdlib loggers have no
    # handlers and all the phase_timing / progress logs are silently
    # dropped — which makes the build look hung for ~20 min.
    from src.core.logging import configure_logging
    configure_logging(
        level=config.logging.level,
        json_logs=config.logging.json,
        file_enabled=False,  # subprocess stdout is the primary channel
        file_path=None,
    )
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
    # Device selection: the encode phase is the dominant cost of building
    # a multi-symbol bank (one full brain forward per chunk of samples).
    # On a 40-symbol × 4-year × 5-minute dataset the encoder alone churns
    # through ~3 M samples — running it on CPU takes ~20 minutes while
    # running it on a 4070-class GPU completes in ~1-2 minutes.  Use CUDA
    # when available; fall back to CPU only if torch reports no CUDA.
    # The FeaturePipeline stays on CPU because its rolling reductions are
    # already vectorized via torch.unfold and copying bars to the GPU for
    # a single-pass reduction would cost more than it saves.
    build_device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
        device=build_device,
        default_model_kind="brain",
    )
    encoder = ModelStateEncoder(
        model=artifacts.model,
        device=build_device,
        feature_contract=artifacts.feature_contract,
        expected_latent_dim=config.memory.latent_dim,
        encoder_fingerprint=artifacts.encoder_fingerprint,
    )
    print(f"memory_build_device={build_device}")
    # forward_horizons / path_label_rules are auto-synced training -> memory
    # by AppConfig.normalize_data_defaults; only the real-data flag merges here.
    memory_config = config.memory.model_copy(
        update={
            "build_from_real_checkpoint": use_real_data or config.memory.build_from_real_checkpoint,
        }
    )

    # Bank cache fast-path: identical (bars + memory/feature config + checkpoint
    # + warmup) inputs deterministically produce the same bank, so reload from
    # disk when possible. --no-bank-cache forces a fresh build.
    bank = None
    bank_fingerprint: str | None = None
    if not args.no_bank_cache:
        try:
            bank_fingerprint = compute_bank_fingerprint(
                config=config,
                bars_by_symbol=bars_by_symbol,
                warmup_bars=config.runtime.warmup_bars,
                checkpoint_path=config.model.checkpoint_path,
            )
            bank = load_cached_bank(fingerprint=bank_fingerprint)
        except Exception as exc:  # noqa: BLE001
            print(f"bank_cache_fingerprint_failed error={exc}")
            bank_fingerprint = None
    if bank is None:
        bank = build_memory_bank_from_bars(
            bars_by_symbol=bars_by_symbol,
            pipeline=pipeline,
            encoder=encoder,
            memory_config=memory_config,
            warmup_bars=config.runtime.warmup_bars,
        )
        if bank_fingerprint is not None and not args.no_bank_cache:
            _t0 = time.perf_counter()
            print(f"phase_timing phase=save_cached_bank_start items={len(bank)}", flush=True)
            save_cached_bank(fingerprint=bank_fingerprint, bank=bank)
            print(f"phase_timing phase=save_cached_bank_done elapsed_sec={time.perf_counter() - _t0:.2f}", flush=True)
    _t_save = time.perf_counter()
    print(f"phase_timing phase=save_memory_bank_start items={len(bank)} path={config.memory.persist_path}", flush=True)
    save_memory_bank(config.memory.persist_path, bank)
    print(f"phase_timing phase=save_memory_bank_done elapsed_sec={time.perf_counter() - _t_save:.2f}", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
