"""Pre-launch checks for the unattended paper week (MASTER_PLAN §9.2).

Fast, offline GO/NO-GO gate run by ``launch_paper_week.bat`` before the
trader starts: config interlock, the week's hardening flags, credentials,
and artifact integrity (canonical brain + fingerprint-matched memory
bank). Exit 0 = GO, 1 = NO-GO with the failing checks listed.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed, detail))


def main() -> int:
    from src.core.config import load_config
    from src.data.alpaca_client import credentials_available

    try:
        config = load_config("configs/default.yaml", "configs/paper_trade.yaml")
        check("config loads (default + paper_trade)", True)
    except Exception as exc:  # noqa: BLE001
        check("config loads (default + paper_trade)", False, str(exc)[:120])
        _report()
        return 1

    check(
        "paper interlock (paper=true, live_enabled=false)",
        bool(config.broker.paper) and not bool(config.broker.live_enabled),
    )
    check("kill_switch off", not bool(config.risk.kill_switch))
    check("VOO out of runtime.symbols (7.1.2)", "VOO" not in config.runtime.symbols)
    check(
        "failed-submit backoff armed (7.1.1)",
        float(config.execution.submit_failure_cooldown_sec) > 0,
        f"{config.execution.submit_failure_cooldown_sec}s",
    )
    check("once-per-bar gate on", bool(config.runtime.decide_once_per_bar))
    check(
        "stale_data_sec exceeds bar period",
        float(config.risk.stale_data_sec) >= 300.0,
        f"{config.risk.stale_data_sec}s",
    )
    check(
        "Alpaca credentials present (no mock)",
        credentials_available(),
        "PK... paper key from .env",
    )

    checkpoint = Path(config.model.checkpoint_path)
    check(f"canonical checkpoint exists ({checkpoint})", checkpoint.exists())
    bank_path = Path(config.memory.persist_path)
    if checkpoint.exists() and bank_path.exists():
        try:
            import torch

            meta = torch.load(checkpoint, map_location="cpu", weights_only=False).get("meta", {})
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            expected = str(meta.get("encoder_fingerprint"))
            stored = str(bank.get("encoder_fingerprint"))
            check(
                "memory bank fingerprint matches checkpoint",
                expected == stored,
                f"bank={stored[:8]} vs brain={expected[:8]}",
            )
        except Exception as exc:  # noqa: BLE001
            check("memory bank fingerprint matches checkpoint", False, str(exc)[:120])
    else:
        check(
            "memory bank present (rebuild otherwise: scripts/build_memory.py --real-data)",
            bank_path.exists(),
        )

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    check("disk free > 5 GB", free_gb > 5.0, f"{free_gb:.1f} GB")

    return _report()


def _report() -> int:
    print("\n  MONDAY PREFLIGHT (MASTER_PLAN 9.2)")
    failed = 0
    for name, passed, detail in CHECKS:
        mark = "OK " if passed else "FAIL"
        suffix = f"  ({detail})" if detail else ""
        print(f"   [{mark}] {name}{suffix}")
        failed += 0 if passed else 1
    print(f"\n  {'GO - start the week.' if failed == 0 else f'NO-GO: {failed} check(s) failed.'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
