from __future__ import annotations

from pathlib import Path

from src.core.logging import configure_logging, get_logger


def test_runtime_logging_writes_events_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    configure_logging("INFO", False, file_enabled=True, file_path=str(log_path))
    logger = get_logger("test_logger")
    logger.info("order_submit_attempt", symbol="AAPL", quantity=1.0)
    content = log_path.read_text(encoding="utf-8")
    assert "order_submit_attempt" in content
    assert "AAPL" in content
