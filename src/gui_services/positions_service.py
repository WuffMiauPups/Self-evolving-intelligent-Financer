"""Thin GUI service over ``runs/positions_status.json``.

The trade loop writes a fresh snapshot of open positions + account state
every tick via ``PositionsStatusWriter``.  The ``PositionsTab`` polls
this service every ~5 s to render the table.  We deliberately do NOT
talk to the broker from the GUI process — everything flows through the
on-disk snapshot so:

- the GUI stays decoupled from Alpaca credentials,
- a stopped runtime just means "no fresh snapshot" (not a crash),
- we never double-count fills by reading the broker twice.

Returns plain ``PositionsStatusSnapshot`` dataclasses (or ``None`` when
the runtime has not produced a snapshot yet).  No Qt imports.
"""
from __future__ import annotations

from pathlib import Path

from src.monitoring.positions_status import (
    DEFAULT_POSITIONS_STATUS_PATH,
    PositionsStatusSnapshot,
    PositionsStatusWriter,
)


def get_positions_snapshot(
    path: Path | str = DEFAULT_POSITIONS_STATUS_PATH,
) -> PositionsStatusSnapshot | None:
    """Return the latest snapshot or ``None`` if no file / unreadable.

    Never raises — a missing file is expected when the runtime hasn't
    started yet, and a partial/corrupt file is a transient race condition
    the reader should tolerate.
    """
    return PositionsStatusWriter.read_latest(path)
