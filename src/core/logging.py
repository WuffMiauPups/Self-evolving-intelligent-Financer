from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO
from typing import Any

try:
    import structlog
except ImportError:  # pragma: no cover - exercised in smoke environments without optional deps
    structlog = None


_LOG_FILE_HANDLE: TextIO | None = None


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            try:
                stream.write(data)
            except UnicodeEncodeError:
                # Redirected stdout on Windows defaults to cp1252, which
                # cannot encode every character a log line may contain.
                # A log write must never take down the process — degrade
                # to backslash-escaped ASCII instead.
                stream.write(data.encode("ascii", errors="backslashreplace").decode("ascii"))
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


class StdlibEventLogger:
    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def info(self, event: str, /, **kwargs: Any) -> None:
        self._logger.info("%s %s", event, kwargs if kwargs else "")

    def debug(self, event: str, /, **kwargs: Any) -> None:
        self._logger.debug("%s %s", event, kwargs if kwargs else "")

    def warning(self, event: str, /, **kwargs: Any) -> None:
        self._logger.warning("%s %s", event, kwargs if kwargs else "")

    def exception(self, event: str, /, **kwargs: Any) -> None:
        self._logger.exception("%s %s", event, kwargs if kwargs else "")


def _build_output_stream(file_enabled: bool, file_path: str | None) -> TextIO:
    global _LOG_FILE_HANDLE
    # When stdout is redirected to a file/pipe on Windows it defaults to
    # cp1252; log lines with non-Latin-1 characters would raise.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    if _LOG_FILE_HANDLE is not None:
        try:
            _LOG_FILE_HANDLE.close()
        except Exception:
            pass
        _LOG_FILE_HANDLE = None
    if not file_enabled or not file_path:
        return sys.stdout
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE_HANDLE = path.open("a", encoding="utf-8")
    return TeeStream(sys.stdout, _LOG_FILE_HANDLE)


def configure_logging(
    level: str = "INFO",
    json_logs: bool = False,
    *,
    file_enabled: bool = False,
    file_path: str | None = None,
) -> None:
    output_stream = _build_output_stream(file_enabled, file_path)
    logging.basicConfig(
        format="%(message)s",
        stream=output_stream,
        level=getattr(logging, level.upper(), logging.INFO),
        force=True,
    )
    if structlog is None:
        return
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=output_stream),
    )


def get_logger(name: str = "seif") -> Any:
    if structlog is None:
        return StdlibEventLogger(name)
    return structlog.get_logger(name)
