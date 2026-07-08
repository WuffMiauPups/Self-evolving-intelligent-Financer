"""GUI service layer for the SETAI desktop control center.

This package provides a thin, safe abstraction between the PySide6 desktop
application and the existing trading backend. It is the ONLY layer the GUI
should call into — never the backend modules directly.

Design rules:
- All calls return plain dataclasses or dicts (no torch tensors, no live model refs).
- Long-running operations are split into "prepare" + "run" so the GUI can run
  the heavy part in a worker thread.
- The paper-trading runtime is managed via subprocess (never in-process), because
  src/execution/trade_engine.run_trade_loop uses asyncio.run() and would block.
- No service mutates global state (logging, torch device) outside what the
  underlying backend already does.
- Risk authority is preserved: the GUI can request a kill-switch, but the
  underlying RiskManager remains the final authority during execution.

Communication pattern:
  backend runtime → JSON status file → service reads file → DTO → GUI tab
  GUI action → service function → backend API or subprocess → result DTO

Service categories:
  - **Readers** (stateless functions, read JSON status files):
    equity_service, positions_service, risk_service, log_service, config_service
  - **Orchestrators** (stateless functions, invoke backend computation):
    backtest_service, experiment_service, memory_service, checkpoint_service,
    checkpoint_manager
  - **Process managers** (singleton classes with RLock, manage subprocesses):
    runtime_service (get_runtime_service), training_service (get_training_service)
"""
from src.gui_services.state import (
    BacktestResultRow,
    CheckpointInfo,
    MemoryStatus,
    RiskStatus,
    RuntimeStatus,
    ServiceError,
)

__all__ = [
    "BacktestResultRow",
    "CheckpointInfo",
    "MemoryStatus",
    "RiskStatus",
    "RuntimeStatus",
    "ServiceError",
]
