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

    def test_stop_when_not_running_is_safe(self) -> None:
        svc = RuntimeService()
        status = svc.stop()
        assert status.state == RuntimeState.STOPPED

    def test_start_invokes_run_paper_script(self, tmp_path: Path) -> None:
        """Checkpoint + memory-enable flow through a layered YAML override
        now (no more ``--checkpoint`` / ``--use-memory`` CLI flags).  Verify:
          - the spawned command points at run_paper.py
          - the final --config arg is the GUI override file
          - the override file contains the expected fields
        """
        svc = RuntimeService(project_root=tmp_path)
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
        # The last --config pair should point at the generated override file.
        last_config_idx = max(i for i, a in enumerate(cmd) if a == "--config")
        override_arg = cmd[last_config_idx + 1]
        assert override_arg.endswith("gui_runtime_override.yaml")
        override_path = Path(override_arg)
        assert override_path.exists()
        import yaml as _yaml
        with open(override_path, encoding="utf-8") as f:
            data = _yaml.safe_load(f)
        assert data["model"]["checkpoint_path"] == "checkpoints/brain.pt"
        assert data["memory"]["enabled"] is True


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

    def test_run_backtest_drops_symbols_with_zero_bars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_backtest must filter out symbols that return an empty bar list.

        This reproduces the "SQ has zero bars in recent window" failure:
        the Alpaca feed stops returning data for a ticker (delisted,
        renamed) and the backtest engine's ``build_batch`` / ``min_length``
        path crashes or silently no-ops when handed an empty sequence.
        The service layer must drop those symbols up front and keep going
        with the survivors.
        """
        from src.data.historical_data import generate_synthetic_bars
        from src.gui_services import backtest_service

        monkeypatch.setattr(
            "src.gui_services.backtest_service.BACKTEST_HISTORY_DIR",
            tmp_path / "bt",
        )

        # Fake ``_load_bars`` so we don't hit Alpaca or touch the real
        # historical cache.  One symbol comes back empty (the bug); the
        # other two come back healthy.
        def fake_load_bars(config, *, use_real_data: bool, allow_mock: bool):
            del use_real_data, allow_mock
            return {
                "AAPL": generate_synthetic_bars("AAPL", 220),
                "MSFT": generate_synthetic_bars("MSFT", 220),
                "SQ": [],  # delisted ticker — zero bars in window
            }

        monkeypatch.setattr(backtest_service, "_load_bars", fake_load_bars)

        seen_progress: list[str] = []
        row = backtest_service.run_backtest(
            config_paths=["configs/default.yaml"],
            use_real_data=False,
            policy_kind="baseline",
            label="test_empty_symbol_drop",
            progress_callback=seen_progress.append,
        )

        # The run must complete (not raise) and produce a row.
        assert isinstance(row, gs_state.BacktestResultRow)
        assert row.label == "test_empty_symbol_drop"

        # The progress callback must mention the drop so operators can
        # see which symbols got filtered.
        drop_messages = [msg for msg in seen_progress if "Dropping" in msg and "SQ" in msg]
        assert drop_messages, f"expected a drop progress message mentioning SQ; got {seen_progress}"

    def test_run_backtest_raises_when_all_symbols_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If every symbol is empty we can't run anything — raise a clear
        ServiceError instead of crashing deeper in the pipeline."""
        from src.gui_services import backtest_service

        monkeypatch.setattr(
            "src.gui_services.backtest_service.BACKTEST_HISTORY_DIR",
            tmp_path / "bt",
        )

        def fake_load_bars(config, *, use_real_data: bool, allow_mock: bool):
            del use_real_data, allow_mock
            return {"AAPL": [], "MSFT": [], "SQ": []}

        monkeypatch.setattr(backtest_service, "_load_bars", fake_load_bars)

        with pytest.raises(gs_state.ServiceError, match="returned zero bars"):
            backtest_service.run_backtest(
                config_paths=["configs/default.yaml"],
                use_real_data=False,
                policy_kind="baseline",
                label="test_all_empty",
            )


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

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            proc = MagicMock()
            proc.returncode = 0
            # stdout is iterated line-by-line by the streaming reader.
            proc.stdout = iter(["ok\n"])
            proc.wait = MagicMock(return_value=0)
            return proc

        with patch("src.gui_services.memory_service.subprocess.Popen", side_effect=fake_popen):
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


# ---------------------------------------------------------------------------
# Platt badge resolves the checkpoint-adjacent calibrator (wave-1 follow-up)
# ---------------------------------------------------------------------------
# The calibrator artifact is checkpoint-bound: it lives NEXT TO the
# checkpoint it was fitted for (src/training/platt_calibration.py). The GUI
# badge used to poll the legacy runs/platt_calibration.json location, which
# stays stale forever under the new layout.


class TestPlattBadgePath:
    def _service_with_summary(self, tmp_path: Path, monkeypatch, summary: dict) -> RuntimeService:
        svc = RuntimeService(project_root=tmp_path)
        monkeypatch.setattr(svc, "preview_config", lambda: dict(summary))
        return svc

    def test_badge_reads_checkpoint_adjacent_artifact(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)  # keep legacy relative paths inside tmp
        checkpoint = tmp_path / "checkpoints" / "brain.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        svc = self._service_with_summary(
            tmp_path,
            monkeypatch,
            {"platt_calibration_enabled": True, "checkpoint_path": str(checkpoint)},
        )
        assert svc.get_status().platt_calibration_fitted is False
        (checkpoint.parent / "platt_calibration.json").write_text("{}", encoding="utf-8")
        assert svc.get_status().platt_calibration_fitted is True

    def test_legacy_runs_location_no_longer_satisfies_badge(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        checkpoint = tmp_path / "checkpoints" / "brain.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        legacy = tmp_path / "runs" / "platt_calibration.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("{}", encoding="utf-8")
        svc = self._service_with_summary(
            tmp_path,
            monkeypatch,
            {"platt_calibration_enabled": True, "checkpoint_path": str(checkpoint)},
        )
        assert svc.get_status().platt_calibration_fitted is False

    def test_badge_false_when_disabled_or_no_checkpoint(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        svc = self._service_with_summary(
            tmp_path,
            monkeypatch,
            {"platt_calibration_enabled": True, "checkpoint_path": ""},
        )
        assert svc.get_status().platt_calibration_fitted is False
        svc2 = self._service_with_summary(
            tmp_path,
            monkeypatch,
            {"platt_calibration_enabled": False, "checkpoint_path": "checkpoints/brain.pt"},
        )
        assert svc2.get_status().platt_calibration_fitted is False


# ---------------------------------------------------------------------------
# _CachingBacktestEngine is the real BacktestEngine (wave-1 follow-up)
# ---------------------------------------------------------------------------
# The class used to carry a hand-copied fork of BacktestEngine.run that
# silently diverged (unsorted symbols, no fill_timing, no temporal cutoff,
# no decision log). It now reuses the base run() verbatim with a
# precomputed-features pipeline facade; these tests pin that equivalence.


class TestCachingBacktestEngineDedupe:
    def _setup(self):
        import torch

        from src.backtest.policy_factory import build_baseline_policy
        from src.data.historical_data import generate_synthetic_bars
        from src.features.feature_pipeline import FeaturePipeline

        config = load_app_config("configs/default.yaml")
        # Insertion order deliberately unsorted: the engine must sort.
        bars = {
            "MSFT": generate_synthetic_bars("MSFT", 200),
            "AAPL": generate_synthetic_bars("AAPL", 200),
        }
        pipeline = FeaturePipeline(config.features, device=torch.device("cpu"))
        cached = {
            symbol: pipeline.build_batch(
                symbol_bars, symbol=symbol, temporal_window=config.runtime.warmup_bars
            )
            for symbol, symbol_bars in bars.items()
        }
        return config, bars, pipeline, cached, build_baseline_policy

    def test_report_identical_to_base_engine(self) -> None:
        from src.backtest.engine import BacktestEngine
        from src.gui_services.experiment_service import _CachingBacktestEngine
        from src.risk.risk_manager import RiskManager

        config, bars, pipeline, cached, build_policy = self._setup()
        base_report = BacktestEngine(
            config, pipeline, RiskManager(config.risk.model_copy())
        ).run(bars, build_policy())
        caching_report = _CachingBacktestEngine(
            config, pipeline, RiskManager(config.risk.model_copy()), cached
        ).run(bars, build_policy())

        assert caching_report.equity_curve == base_report.equity_curve
        assert caching_report.metrics == base_report.metrics
        assert len(caching_report.fills) == len(base_report.fills)
        # The diverged copy produced no decision log at all.
        assert caching_report.decision_log == base_report.decision_log
        assert caching_report.decision_log, "expected a populated decision log"

    def test_cached_batches_are_served_not_rebuilt(self, monkeypatch) -> None:
        from src.gui_services.experiment_service import _CachingBacktestEngine
        from src.risk.risk_manager import RiskManager

        config, bars, pipeline, cached, build_policy = self._setup()
        rebuilds: list[str] = []
        original_build_batch = pipeline.build_batch

        def counting_build_batch(*args, **kwargs):
            rebuilds.append(kwargs.get("symbol", "?"))
            return original_build_batch(*args, **kwargs)

        monkeypatch.setattr(pipeline, "build_batch", counting_build_batch)
        engine = _CachingBacktestEngine(
            config, pipeline, RiskManager(config.risk.model_copy()), cached
        )
        report = engine.run(bars, build_policy())
        assert rebuilds == [], "cached feature batches must be served, not rebuilt"
        assert report.equity_curve

    def test_symbols_processed_in_sorted_order(self) -> None:
        from src.gui_services.experiment_service import _CachingBacktestEngine
        from src.risk.risk_manager import RiskManager

        config, bars, pipeline, cached, build_policy = self._setup()
        report = _CachingBacktestEngine(
            config, pipeline, RiskManager(config.risk.model_copy()), cached
        ).run(bars, build_policy())
        first_bar_index = report.decision_log[0]["bar_index"]
        first_bar_symbols = [
            entry["symbol"]
            for entry in report.decision_log
            if entry["bar_index"] == first_bar_index
        ]
        assert first_bar_symbols == sorted(first_bar_symbols)
