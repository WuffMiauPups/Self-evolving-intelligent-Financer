"""Cold-start micro-benchmark.

Measures the wall-clock time of ``build_real_data_training_bundle`` with:
  (a) no cache present  (cold build)
  (b) cache present      (warm reload)

Uses synthetic bars so it runs offline and stays fast enough to be a
regression benchmark. The ratio between (a) and (b) is the speedup that
a real multi-year training run will see on the second launch onwards.

Usage:
    python scripts/bench_cold_start.py
    python scripts/bench_cold_start.py --bars 20000 --symbols AAPL,MSFT
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.training.data_module import build_real_data_training_bundle
from src.training.sample_cache import DEFAULT_CACHE_DIR


def _build_bars(symbols: list[str], count: int) -> dict:
    return {symbol: generate_synthetic_bars(symbol, count) for symbol in symbols}


def _clear_cache() -> None:
    if DEFAULT_CACHE_DIR.exists():
        shutil.rmtree(DEFAULT_CACHE_DIR, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=8000, help="bars per symbol")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT")
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    config = load_config("configs/default.yaml")
    # Force deterministic, non-fast real path for the benchmark.
    config.training.use_real_historical_data = True
    config.training.fast_mode.enabled = False
    config.data.symbols = symbols
    config.runtime.symbols = symbols

    bars = _build_bars(symbols, args.bars)
    total_bars = sum(len(v) for v in bars.values())
    print(f"bench_cold_start: symbols={symbols} bars_per_symbol={args.bars} total={total_bars}")

    pipeline = FeaturePipeline(config.features)

    # Run 1: cold (no cache)
    _clear_cache()
    t0 = time.perf_counter()
    bundle_cold = build_real_data_training_bundle(
        config=config, bars_by_symbol=bars, pipeline=pipeline
    )
    cold_s = time.perf_counter() - t0
    n_train = len(bundle_cold.samples_by_split["train"])
    n_val = len(bundle_cold.samples_by_split["val"])
    n_test = len(bundle_cold.samples_by_split["test"])
    print(
        f"cold_build:  {cold_s:6.2f}s  train={n_train} val={n_val} test={n_test}"
    )

    # Run 2: warm (cache hit)
    t0 = time.perf_counter()
    bundle_warm = build_real_data_training_bundle(
        config=config, bars_by_symbol=bars, pipeline=pipeline
    )
    warm_s = time.perf_counter() - t0
    print(
        f"warm_reload: {warm_s:6.2f}s  train={len(bundle_warm.samples_by_split['train'])}"
    )

    if warm_s > 0:
        print(f"speedup:     {cold_s / warm_s:6.1f}x  ({(1 - warm_s / cold_s) * 100:5.1f}% reduction)")

    # Sanity: the two bundles should contain the same number of samples.
    for split in ("train", "val", "test"):
        a = len(bundle_cold.samples_by_split[split])
        b = len(bundle_warm.samples_by_split[split])
        assert a == b, f"split {split}: cold={a} warm={b}"
    print("sample_count_match: ok")

    if not args.keep_cache:
        _clear_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
