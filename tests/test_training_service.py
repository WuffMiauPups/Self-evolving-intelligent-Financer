"""Tests for TrainingService — subprocess spawn + status + stop.

We don't actually run scripts/run_train.py in these tests (that's a multi-
minute process with data fetching).  Instead we point at a tiny dummy
script so the subprocess lifecycle is exercised without the heavy
training logic.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.gui_services.state import RuntimeState, ServiceError
from src.gui_services.training_service import (
    TrainingService,
    TrainingStatus,
    _TRAINING_OVERRIDE_YAML,
)


@pytest.fixture
def dummy_train_script(tmp_path: Path) -> Path:
    """Write a tiny script that prints some fake epoch lines then exits.

    Uses the `-u` unbuffered flag in the subprocess invocation
    (TrainingService passes `-u`) so we don't need to worry about
    stdout buffering at the dummy-script level.
    """
    script = tmp_path / "scripts" / "run_train.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        "import time\n"
        "# dummy stand-in for scripts/run_train.py (ignores all CLI args)\n"
        "print('training started', flush=True)\n"
        "for i in range(1, 3):\n"
        "    time.sleep(0.05)\n"
        "    print(f'epoch {i}/2 loss=0.1', flush=True)\n"
        "print('training finished', flush=True)\n",
        encoding="utf-8",
    )
    return script


def test_start_spawns_subprocess_and_captures_output(
    dummy_train_script: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    svc = TrainingService(project_root=dummy_train_script.parent.parent)
    svc.start(
        preset=None,
        fast=False,
        use_real_data=False,
        epochs=2,
        output_checkpoint=str(tmp_path / "out.pt"),
        save_memory=False,
    )
    # Wait for the dummy script to finish (< 1s is plenty)
    deadline = time.time() + 5.0
    while svc.is_running() and time.time() < deadline:
        time.sleep(0.02)
    status = svc.get_status()
    assert status.state == RuntimeState.STOPPED
    assert status.pid is None
    assert any("training started" in line for line in status.recent_lines)
    assert any("epoch 2/2" in line for line in status.recent_lines)
    # Epoch progress parser should have caught the latest epoch line.
    assert status.epoch_progress == "epoch 2/2"


def test_double_start_raises(
    dummy_train_script: Path, tmp_path: Path
) -> None:
    svc = TrainingService(project_root=dummy_train_script.parent.parent)
    svc.start(
        output_checkpoint=str(tmp_path / "out.pt"),
        use_real_data=False,
        save_memory=False,
    )
    try:
        with pytest.raises(ServiceError, match="already running"):
            svc.start(
                output_checkpoint=str(tmp_path / "out2.pt"),
                use_real_data=False,
                save_memory=False,
            )
    finally:
        # Ensure the first subprocess finishes before the test exits.
        deadline = time.time() + 5.0
        while svc.is_running() and time.time() < deadline:
            time.sleep(0.02)


def test_stop_when_not_running_is_safe() -> None:
    svc = TrainingService()
    status = svc.stop()
    assert isinstance(status, TrainingStatus)
    assert status.state == RuntimeState.STOPPED


def test_override_yaml_is_written(
    dummy_train_script: Path, tmp_path: Path
) -> None:
    svc = TrainingService(project_root=dummy_train_script.parent.parent)
    svc.start(
        output_checkpoint=str(tmp_path / "exp_custom.pt"),
        use_real_data=False,
        save_memory=False,
    )
    override_path = dummy_train_script.parent.parent / _TRAINING_OVERRIDE_YAML
    assert override_path.exists()
    # It should contain our output checkpoint path
    text = override_path.read_text()
    assert "exp_custom.pt" in text
    # Drain the subprocess
    deadline = time.time() + 5.0
    while svc.is_running() and time.time() < deadline:
        time.sleep(0.02)


def test_epoch_parser_handles_no_matches() -> None:
    parsed = TrainingService._parse_epoch_progress(["hello", "world", "no epochs here"])
    assert parsed == ""


def test_epoch_parser_returns_last_match() -> None:
    lines = [
        "starting",
        "epoch 1/5 loss=0.5",
        "epoch 2/5 loss=0.4",
        "epoch 3/5 loss=0.3",
        "some unrelated log line",
    ]
    assert TrainingService._parse_epoch_progress(lines) == "epoch 3/5"
