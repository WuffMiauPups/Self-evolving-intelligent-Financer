"""WebhookAlerter must stay off the trade loop's hot path (MASTER_PLAN 4.7).

``send`` only enqueues; a daemon worker performs the HTTP POST. The queue is
bounded and drops the OLDEST alert on overflow. Nothing raises into callers.
"""
from __future__ import annotations

import queue
import threading

from src.monitoring.alerting import AlertEvent, WebhookAlerter


def _event(message: str = "test") -> AlertEvent:
    return AlertEvent(severity="critical", event_type="kill_switch", message=message)


def test_send_without_url_returns_false() -> None:
    alerter = WebhookAlerter(webhook_url=None)
    assert alerter.send(_event()) is False


def test_send_enqueues_and_worker_delivers(monkeypatch) -> None:
    alerter = WebhookAlerter(webhook_url="http://example.invalid/hook")
    posted: list[AlertEvent] = []
    delivered = threading.Event()

    def fake_post(event: AlertEvent) -> bool:
        posted.append(event)
        delivered.set()
        return True

    monkeypatch.setattr(alerter, "_post", fake_post)
    assert alerter.send(_event("hello")) is True
    assert delivered.wait(timeout=5.0), "worker thread never delivered the alert"
    assert alerter.flush(timeout_sec=5.0) is True
    assert [event.message for event in posted] == ["hello"]


def test_send_does_not_block_on_slow_delivery(monkeypatch) -> None:
    """A hung webhook must not stall the caller — send returns immediately."""
    alerter = WebhookAlerter(webhook_url="http://example.invalid/hook")
    release = threading.Event()
    monkeypatch.setattr(alerter, "_post", lambda event: release.wait(timeout=10.0))
    try:
        for index in range(5):
            assert alerter.send(_event(f"e{index}")) is True
    finally:
        release.set()
    assert alerter.flush(timeout_sec=10.0) is True


def test_queue_overflow_drops_oldest(monkeypatch) -> None:
    alerter = WebhookAlerter(webhook_url="http://example.invalid/hook", max_queue=2)
    # Keep the worker from starting so the queue actually fills.
    monkeypatch.setattr(alerter, "_ensure_worker", lambda: None)
    assert alerter.send(_event("first")) is True
    assert alerter.send(_event("second")) is True
    assert alerter.send(_event("third")) is True  # evicts "first"
    remaining = []
    while True:
        try:
            remaining.append(alerter._queue.get_nowait().message)
        except queue.Empty:
            break
    assert remaining == ["second", "third"]


def test_delivery_failure_never_raises(monkeypatch) -> None:
    alerter = WebhookAlerter(webhook_url="http://example.invalid/hook")

    def broken_post(event: AlertEvent) -> bool:
        raise RuntimeError("network down")

    monkeypatch.setattr(alerter, "_post", broken_post)
    assert alerter.send(_event()) is True
    # Worker swallows the failure and marks the task done.
    assert alerter.flush(timeout_sec=5.0) is True


def test_post_returns_false_on_unreachable_host() -> None:
    """_post itself is best-effort: bad host -> False, no exception."""
    alerter = WebhookAlerter(
        webhook_url="http://127.0.0.1:1/unroutable", timeout_sec=0.2
    )
    assert alerter._post(_event()) is False
