"""Equity / performance data provider for the GUI.

Reads the structured equity log written by the runtime trade loop
(``src/monitoring/equity_log.py``) and exposes it as plain
``EquityPoint`` / ``EquitySeries`` dataclasses.

This service deliberately does **not** import Qt or any plotting code —
the chart widget calls into here and renders the result.
"""
from __future__ import annotations

from pathlib import Path

from src.gui_services.state import EquityPoint, EquitySeries
from src.monitoring.equity_log import DEFAULT_EQUITY_LOG_PATH, EquityLogWriter

# Hard cap so we never load an unbounded file into the GUI process.
_DEFAULT_MAX_POINTS = 5000


def get_live_equity_series(
    *,
    path: Path | str = DEFAULT_EQUITY_LOG_PATH,
    max_points: int = _DEFAULT_MAX_POINTS,
) -> EquitySeries:
    """Read the live equity log and return the latest ``max_points`` samples.

    Returns an EquitySeries with ``source="live"`` if the log exists,
    ``source="none"`` if it does not (with a clear note explaining why
    nothing is plotted).
    """
    p = Path(path)
    if not p.exists():
        return EquitySeries(
            points=[],
            source="none",
            label="Live equity",
            note=(
                "No live equity log yet. Start the runtime (Run → Start) — "
                "the chart will populate as the trade loop ticks."
            ),
        )

    points: list[EquityPoint] = []
    try:
        file_size = p.stat().st_size
        # Each equity log line is ~200 bytes. For 5000 points we need ~1MB.
        # Read only the tail to avoid scanning the full file.
        max_tail_bytes = max_points * 256  # generous estimate
        with p.open("rb") as fp:
            if file_size > max_tail_bytes:
                fp.seek(file_size - max_tail_bytes)
                fp.readline()  # discard partial first line
            data = fp.read().decode("utf-8", errors="replace")
        for line in data.splitlines():
            if not line.strip():
                continue
            snap = EquityLogWriter.parse_line(line)
            if snap is None:
                continue
            points.append(
                EquityPoint(
                    timestamp=snap.ts,
                    equity=snap.equity,
                    realized_pnl=snap.realized_pnl,
                    unrealized_pnl=snap.unrealized_pnl,
                    fills=snap.fills,
                )
            )
    except OSError as exc:
        return EquitySeries(
            points=[],
            source="none",
            label="Live equity",
            note=f"Could not read equity log: {exc}",
        )

    if max_points and len(points) > max_points:
        points = points[-max_points:]

    return EquitySeries(
        points=points,
        source="live",
        label="Live equity (paper runtime)",
        note=(
            "Plotting realized + mark-to-market equity from "
            "src/monitoring/equity_log.py (one snapshot per runtime tick)."
        ),
    )


