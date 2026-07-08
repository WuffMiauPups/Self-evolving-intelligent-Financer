"""Background worker primitives for the desktop GUI.

All long-running service calls go through one of these classes so the Qt
event loop never blocks.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(str)
    finished = Signal(object)  # result
    failed = Signal(str)        # operator-friendly error


class CallableWorker(QRunnable):
    """Run an arbitrary callable in a thread pool.

    The callable receives a `progress(msg: str)` keyword if it accepts one.
    """

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:  # noqa: D401
        self.signals.started.emit()
        try:
            # Inject progress callback if the callee supports it.
            kwargs = dict(self.kwargs)
            try:
                import inspect

                sig = inspect.signature(self.fn)
                if "progress_callback" in sig.parameters:
                    kwargs.setdefault("progress_callback", self.signals.progress.emit)
            except (TypeError, ValueError):
                pass
            result = self.fn(*self.args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(tb)  # full detail to stderr/console for debugging
            self.signals.failed.emit(str(exc))
            return
        self.signals.finished.emit(result)
