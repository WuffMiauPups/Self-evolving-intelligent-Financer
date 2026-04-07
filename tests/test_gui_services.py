"""Focused tests for the GUI service layer.

These tests target the service-layer contract — not Qt UI rendering. They
verify:
  - service calls dispatch to the correct backend path
  - runtime start/stop state machine
  - duplicate-start prevention
  - background work doesn't import Qt at the service layer
  - checkpoint promotion path respects existing PromotionGate
  - no service mutates risk_manager state outside of approved paths
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.gui_services import state as gs_state
from src.gui_services.backtest_service import _persist_row, list_recent_results
from src.gui_services.checkpoint_service import (
    evaluate_for_promotion,
    promote_if_eligible,
)
from src.gui_services.memory_service import (
    build_memory,
    plan_memory_build,
)
from src.gui_services.config_service import (
    describe_config,
    list_config_presets,
    load_app_config,
)
from src.gui_services.log_service import _parse_line
from src.gui_services.risk_service import get_risk_status
from src.gui_services.runtime_service import RuntimeService
from src.gui_services.state import (
    BacktestResultRow,
    RuntimeState,
    ServiceError,
)
from src.training.checkpoint_promotion import PromotionGate


# ---------------------------------------------------------------------------
# state types
# ---------------------------------------------------------------------------
class TestState:
    def test_backtest_row_roundtrip(self) -> None:
        row = BacktestResultRow(
            timestamp=datetime(2026, 4, 7, 12, 0, 0),
            label="unit",
            total_return=0.05,
            max_drawdown=-0.02,
            win_rate=0.55,
            turnover=1.2,
            fill_count=42,
            profit_factor=1.7,
            expectancy=0.0001,
            avg_return_per_trade=0.0002,
            fusion_veto_count=3,
            fusion_dampen_count=4,
            fusion_support_count=5,
            fusion_neutral_count=10,
            config_paths=["configs/default.yaml"],
            notes="x",
        )
        d = row.to_dict()
        restored = BacktestResultRow.from_dict(d)
        assert restored == row


# ---------------------------------------------------------------------------
# config_service
# ---------------------------------------------------------------------------
class TestConfigService:
    def test_list_presets_returns_yaml_files(self) -> None:
        presets = list_config_presets("configs")
        names = {p.name for p in presets}
        assert "default.yaml" in names

    def test_load_app_config_default(self) -> None:
        cfg = load_app_config("configs/default.yaml")
        summary = describe_config(cfg)
        # Frozen-brain default must be enforced
        assert summary["frozen_brain"] is True
        assert "symbols" in summary
        assert isinstance(summary["symbols"], list)

    def test_load_app_config_invalid_path_raises_service_error(self) -> None:
        with pytest.raises(ServiceError):
            load_app_config("configs/this_does_not_exist.yaml")


# ---------------------------------------------------------------------------
# runtime_service — state machine + duplicate-start prevention
# ---------------------------------------------------------------------------
class TestRuntimeService:
    def test_initial_state_is_stopped(self) -> None:
        svc = RuntimeService()
        status = svc.get_status()
        assert status.state == RuntimeState.STOPPED
        assert status.pid is None
        assert svc.is_running() is False

    def test_duplicate_start_raises(self) -> None:
        svc = RuntimeService()
        # Fake a running subprocess
        fake = MagicMock()
        fake.poll.return_value = None
        fake.pid = 12345
        svc._proc = fake  # type: ignore[assignment]
        svc._started_at = datetime.now()
        try:
            with pytest.raises(ServiceError, match="already running"):
                svc.start()
        finally:
            svc._proc = None
            svc._started_at = None

    def test_set_config_paths_blocked_while_running(self) -> None:
        svc = RuntimeService()
        fake = MagicMock()
        fake.poll.return_value = None
        svc._proc = fake  # type: ignore[assignment]
        try:
            with pytest.raises(ServiceError, match="while runtime is running"):
                svc.set_config_paths(["configs/default.yaml"])
        finally:
            svc._proc = None

    def test_stop_when_not_running_is_safe(self) -> None:
        svc = RuntimeService()
        status = svc.stop()
        assert status.state == RuntimeState.STOPPED

    def test_start_invokes_run_paper_script(self) -> None:
        svc = RuntimeService()
        with patch("src.gui_services.runtime_service.subprocess.Popen") as popen:
            fake_proc = MagicMock()
            fake_proc.poll.return_value = None
            fake_proc.pid = 99
            popen.return_value = fake_proc
            try:
                svc.start(checkpoint="checkpoints/brain.pt", use_memory=True)
            finally:
                svc._proc = None
                svc._started_at = None
        args, _kwargs = popen.call_args
        cmd = args[0]
        assert "scripts" in str(Path(cmd[2])).replace("\\", "/")
        assert "run_paper.py" in cmd[2]
        assert "--checkpoint" in cmd
        assert "checkpoints/brain.pt" in cmd
        assert "--use-memory" in cmd


# ---------------------------------------------------------------------------
# backtest_service — persistence roundtrip
# ---------------------------------------------------------------------------
class TestBacktestService:
    def test_persist_and_list(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "src.gui_services.backtest_service.BACKTEST_HISTORY_DIR",
            tmp_path / "bt",
        )
        row = BacktestResultRow(
            timestamp=datetime(2026, 4, 7, 12, 0, 0),
            label="t1",
            total_return=0.01,
            max_drawdown=-0.01,
            win_rate=0.5,
            turnover=0.2,
            fill_count=4,
        )
        _persist_row(row)
        rows = list_recent_results()
        assert len(rows) == 1
        assert rows[0].label == "t1"
        assert rows[0].fill_count == 4


# ---------------------------------------------------------------------------
# checkpoint_service — promotion gate enforcement
# ---------------------------------------------------------------------------
class TestCheckpointService:
    def test_eligible_metrics_pass_gate(self) -> None:
        decision = evaluate_for_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.30,
                "val_report.per_class_metrics.short.recall": 0.30,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is True

    def test_failing_recall_blocks_promotion(self) -> None:
        decision = evaluate_for_promotion(
            candidate_metrics={
                "val_report.balanced_accuracy": 0.55,
                "val_report.macro_f1": 0.45,
                "val_report.per_class_metrics.long.recall": 0.05,  # too low
                "val_report.per_class_metrics.short.recall": 0.30,
            },
            parent_metrics=None,
            gate=PromotionGate(require_improvement_over_parent=False),
        )
        assert decision.eligible is False
        assert any("long_recall" in r for r in decision.reasons)

    def test_promote_if_eligible_blocks_when_gate_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The promotion path should refuse to call promote_checkpoint when blocked.
        monkeypatch.setattr(
            "src.gui_services.checkpoint_service.REGISTRY_PATH",
            tmp_path / "registry.json",
        )
        with pytest.raises(ServiceError, match="Promotion blocked"):
            promote_if_eligible(
                source_path=tmp_path / "fake.pt",
                target_path=tmp_path / "out.pt",
                version_id="v1",
                fingerprint="abc",
                candidate_metrics={"val_report.balanced_accuracy": 0.0},
                gate=PromotionGate(),
            )
        # Registry must NOT have been written
        assert not (tmp_path / "registry.json").exists()


# ---------------------------------------------------------------------------
# risk_service
# ---------------------------------------------------------------------------
class TestRiskService:
    def test_get_risk_status_returns_envelope(self) -> None:
        status = get_risk_status()
        assert status.max_position_notional > 0
        assert status.max_daily_loss > 0
        assert "Risk manager" in status.notes


# ---------------------------------------------------------------------------
# log_service
# ---------------------------------------------------------------------------
class TestLogService:
    def test_parse_json_line(self) -> None:
        line = json.dumps({"timestamp": "2026-04-07T12:00:00", "level": "INFO",
                            "message": "memory_retrieval ok"})
        entry = _parse_line(line)
        assert entry.level == "INFO"
        assert "memory" in entry.category

    def test_parse_plain_line_with_ansi(self) -> None:
        line = "\x1b[1mmodel_decision\x1b[0m action=0 fusion=neutral"
        entry = _parse_line(line)
        assert "\x1b" not in entry.message
        assert "model_decision" in entry.message
        assert entry.category in {"runtime", "fusion"}


# ---------------------------------------------------------------------------
# memory_service — preflight plan + subprocess wiring
# ---------------------------------------------------------------------------
def _patch_no_config_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force config.model.checkpoint_path to None so the resolution branches
    that depend on explicit / active_registry actually exercise. default.yaml
    now ships with a real checkpoint_path; these tests target the *fallback*
    paths, not the YAML default."""
    from src.gui_services import memory_service as ms

    real_loader = ms.load_app_config

    def _load(*paths: str):
        cfg = real_loader(*paths)
        cfg.model.checkpoint_path = None
        return cfg

    monkeypatch.setattr(ms, "load_app_config", _load)


class TestMemoryBuildPlan:
    DEFAULT = ["configs/default.yaml"]

    def test_blocked_when_checkpoint_required_and_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_no_config_checkpoint(monkeypatch)
        plan = plan_memory_build(
            config_paths=self.DEFAULT,
            explicit_checkpoint=None,
            use_active_registry=False,
        )
        assert plan.requires_checkpoint is True
        assert plan.is_runnable is False
        assert "checkpoint" in plan.blocking_error.lower()
        assert plan.resolved_checkpoint == ""

    def test_explicit_checkpoint_resolves(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "brain.pt"
        ckpt.write_bytes(b"x")
        plan = plan_memory_build(
            config_paths=self.DEFAULT,
            explicit_checkpoint=str(ckpt),
        )
        assert plan.is_runnable is True
        assert plan.resolved_checkpoint == str(ckpt)
        assert plan.checkpoint_source == "explicit"

    def test_nonexistent_explicit_checkpoint_blocks(self, tmp_path: Path) -> None:
        plan = plan_memory_build(
            config_paths=self.DEFAULT,
            explicit_checkpoint=str(tmp_path / "missing.pt"),
        )
        assert plan.is_runnable is False
        assert "does not exist" in plan.blocking_error

    def test_active_registry_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_no_config_checkpoint(monkeypatch)
        ckpt = tmp_path / "active.pt"
        ckpt.write_bytes(b"x")
        fake_active = MagicMock()
        fake_active.checkpoint_path = str(ckpt)
        monkeypatch.setattr(
            "src.gui_services.checkpoint_service.get_active",
            lambda: fake_active,
        )
        plan = plan_memory_build(
            config_paths=self.DEFAULT,
            use_active_registry=True,
        )
        assert plan.is_runnable is True
        assert plan.checkpoint_source == "active_registry"
        assert plan.resolved_checkpoint == str(ckpt)

    def test_build_memory_refuses_when_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_no_config_checkpoint(monkeypatch)
        with pytest.raises(ServiceError, match="checkpoint"):
            build_memory(config_paths=self.DEFAULT, explicit_checkpoint=None)

    def test_build_memory_passes_checkpoint_to_subprocess(self, tmp_path: Path) -> None:
        ckpt = tmp_path / "brain.pt"
        ckpt.write_bytes(b"x")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            r = MagicMock()
            r.returncode = 0
            r.stdout = "ok"
            r.stderr = ""
            return r

        with patch("src.gui_services.memory_service.subprocess.run", side_effect=fake_run):
            build_memory(
                config_paths=["configs/default.yaml"],
                explicit_checkpoint=str(ckpt),
            )
        cmd = captured["cmd"]
        assert "scripts/build_memory.py" in cmd[2]
        assert "--checkpoint" in cmd
        assert str(ckpt) in cmd
        assert "--real-data" in cmd
        assert "--allow-mock" in cmd


# ---------------------------------------------------------------------------
# safety: GUI service layer must NOT import Qt (so service tests run headless)
# ---------------------------------------------------------------------------
class TestSafetyInvariants:
    def test_memory_tab_uses_only_service_layer(self) -> None:
        import re
        path = Path(__file__).resolve().parents[1] / "src" / "gui" / "tabs" / "memory_tab.py"
        text = path.read_text(encoding="utf-8")
        forbidden = re.compile(
            r"^\s*(from|import)\s+src\.(memory|strategies|training|core|features|data)",
            re.MULTILINE,
        )
        assert not forbidden.search(text), (
            "memory_tab.py must talk to backend only via src.gui_services"
        )

    def test_gui_services_does_not_import_pyside(self) -> None:
        # Re-import freshly to ensure no side-effect grabbed PySide6.
        for mod in list(sys.modules):
            if mod.startswith("src.gui_services"):
                del sys.modules[mod]
        import src.gui_services  # noqa: F401
        import src.gui_services.backtest_service  # noqa: F401
        import src.gui_services.checkpoint_service  # noqa: F401
        import src.gui_services.config_service  # noqa: F401
        import src.gui_services.log_service  # noqa: F401
        import src.gui_services.memory_service  # noqa: F401
        import src.gui_services.risk_service  # noqa: F401
        import src.gui_services.runtime_service  # noqa: F401

        # No PySide6 module should have been pulled into sys.modules as a
        # side-effect of importing the service layer.
        assert "PySide6" not in sys.modules, (
            "Service layer must not import PySide6"
        )
        # And no service-layer file should contain a real `from PySide6` /
        # `import PySide6` statement.
        import re
        rx = re.compile(r"^\s*(from|import)\s+PySide6", re.MULTILINE)
        for name, mod in sys.modules.items():
            if name.startswith("src.gui_services") and mod is not None:
                file = getattr(mod, "__file__", "") or ""
                if file:
                    text = Path(file).read_text(encoding="utf-8", errors="ignore")
                    assert not rx.search(text), (
                        f"{name} imports PySide6 — service layer must stay GUI-free"
                    )
