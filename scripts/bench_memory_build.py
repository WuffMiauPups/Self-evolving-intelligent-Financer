"""Micro-benchmark the memory-build encoder path.

Compares chunk=1 (per-sample, the old behavior) against chunk=2048 (new
default). Runs on synthetic bars so it is self-contained and quick.

Usage:
    python scripts/bench_memory_build.py
    python scripts/bench_memory_build.py --bars 2000 --chunks 1,256,1024,2048
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config
from src.data.historical_data import generate_synthetic_bars
from src.features.feature_pipeline import FeaturePipeline
from src.memory.updater import build_memory_bank_from_bars
from src.models.factory import build_model_from_contract
from src.models.state_encoder import ModelStateEncoder


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=int, default=1500)
    parser.add_argument("--chunks", type=str, default="1,256,1024,2048")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    chunk_sizes = [int(c) for c in args.chunks.split(",") if c.strip()]

    cfg = load_config("configs/default.yaml")
    cfg.data.symbols = symbols
    cfg.runtime.symbols = symbols
    bars = {s: generate_synthetic_bars(s, args.bars) for s in symbols}

    pipeline = FeaturePipeline(cfg.features)
    warmup = pipeline.build(bars[symbols[0]][:100], symbol=symbols[0])
    model = build_model_from_contract(
        model_kind="brain",
        model_config=cfg.model,
        feature_contract=warmup.contract,
    )
    encoder = ModelStateEncoder(
        model=model,
        device=torch.device("cpu"),
        feature_contract=warmup.contract,
        expected_latent_dim=cfg.memory.latent_dim,
    )

    eligible = sum(max(0, len(v) - cfg.runtime.warmup_bars - max(cfg.memory.forward_horizons)) for v in bars.values())
    print(f"bench_memory_build: symbols={symbols} bars/sym={args.bars} total_samples~{eligible}")

    results: list[tuple[int, float, int]] = []
    for chunk in chunk_sizes:
        t0 = time.perf_counter()
        bank = build_memory_bank_from_bars(
            bars_by_symbol=bars,
            pipeline=pipeline,
            encoder=encoder,
            memory_config=cfg.memory,
            warmup_bars=cfg.runtime.warmup_bars,
            encode_chunk_size=chunk,
        )
        dt = time.perf_counter() - t0
        results.append((chunk, dt, len(bank)))
        print(f"chunk={chunk:>5}  build={dt:7.2f}s  items={len(bank):>6}")

    if len(results) >= 2:
        base_chunk, base_time, _ = results[0]
        print()
        print("relative speedups vs chunk=1:")
        for chunk, dt, _ in results:
            speedup = base_time / dt if dt > 0 else 0
            print(f"  chunk={chunk:>5}  {speedup:6.1f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
