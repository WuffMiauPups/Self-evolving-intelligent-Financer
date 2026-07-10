"""Best-effort webhook alerting for critical events.

Delivery is asynchronous: ``send`` only enqueues onto a bounded queue that
a daemon worker thread drains with the actual HTTP POST.  The previous
implementation ran a synchronous ``urlopen`` (up to ``timeout_sec``) on the
caller's thread — placing a network round-trip on the trade loop's
rejection hot path.  ``send`` now never blocks on the network, never
raises, and drops the OLDEST queued alert when the queue is full (the most
recent event is the one the operator needs).
"""
from __future__ import annotations

import json
import queue
import threading
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class AlertEvent:
    severity: str  # "info", "warning", "critical"
    event_type: str  # "kill_switch", "circuit_breaker", "account_blocked"
    message: str
    timestamp: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class WebhookAlerter:
    """Best-effort HTTP POST to a configured webhook URL, off the hot path.

    ``send`` enqueues and returns immediately; a lazily-started daemon
    worker thread performs the POST.  Queue is bounded (``max_queue``);
    on overflow the oldest pending alert is dropped in favour of the new
    one.  Nothing in this class ever raises into the caller.
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        timeout_sec: float = 5.0,
        max_queue: int = 64,
    ):
        self.webhook_url = webhook_url
        self.timeout_sec = timeout_sec
        self._queue: queue.Queue[AlertEvent] = queue.Queue(maxsize=max(1, int(max_queue)))
        self._worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def send(self, event: AlertEvent) -> bool:
        """Enqueue ``event`` for asynchronous delivery.

        Returns True when the event was queued (delivery itself remains
        best-effort), False when no webhook is configured or the queue
        could not accept the event.  Never blocks on the network and
        never raises.
        """
        if not self.webhook_url:
            return False
        try:
            self._ensure_worker()
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                # Drop-oldest: evict one stale alert, then retry once.
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(event)
                except queue.Full:
                    return False
            return True
        except Exception:
            return False

    def flush(self, timeout_sec: float = 5.0) -> bool:
        """Wait until every queued alert has been processed (tests/shutdown).

        Returns True when the queue drained within ``timeout_sec``.
        """
        deadline = threading.Event()
        timer = threading.Timer(timeout_sec, deadline.set)
        timer.daemon = True
        timer.start()
        try:
            with self._queue.all_tasks_done:
                while self._queue.unfinished_tasks:
                    if deadline.is_set():
                        return False
                    self._queue.all_tasks_done.wait(0.05)
            return True
        except Exception:
            return False
        finally:
            timer.cancel()

    # ------------------------------------------------------------------
    # worker internals
    # ------------------------------------------------------------------
    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._drain_forever,
                    daemon=True,
                    name="webhook-alerter",
                )
                self._worker.start()

    def _drain_forever(self) -> None:
        while True:
            event = self._queue.get()
            try:
                self._post(event)
            except Exception:
                pass  # best-effort: delivery failures are silently dropped
            finally:
                self._queue.task_done()

    def _post(self, event: AlertEvent) -> bool:
        """Synchronous webhook POST — runs on the worker thread only."""
        if not self.webhook_url:
            return False
        try:
            payload = json.dumps({
                "severity": event.severity,
                "event_type": event.event_type,
                "message": event.message,
                "timestamp": event.timestamp,
            }).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=self.timeout_sec)
            return True
        except Exception:
            return False
