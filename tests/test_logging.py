from __future__ import annotations

import io
from pathlib import Path

from src.core.logging import TeeStream, configure_logging, get_logger


def test_runtime_logging_writes_events_to_file(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.log"
    configure_logging("INFO", False, file_enabled=True, file_path=str(log_path))
    logger = get_logger("test_logger")
    logger.info("order_submit_attempt", symbol="AAPL", quantity=1.0)
    content = log_path.read_text(encoding="utf-8")
    assert "order_submit_attempt" in content
    assert "AAPL" in content


def test_tee_stream_survives_cp1252_stdout(tmp_path: Path) -> None:
    # Redirected stdout on Windows defaults to cp1252, which cannot encode
    # characters like the arrows structlog's console renderer emits.  A log
    # line must degrade, never raise (it crashed fetch_historical.py).
    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    utf8_file = (tmp_path / "log.txt").open("w", encoding="utf-8")
    tee = TeeStream(narrow, utf8_file)
    tee.write("progress 2022-01-01 → 2026-07-13\n")
    tee.flush()
    utf8_file.close()
    narrow.seek(0)
    assert "\\u2192" in narrow.buffer.getvalue().decode("cp1252")
    assert "→" in (tmp_path / "log.txt").read_text(encoding="utf-8")
