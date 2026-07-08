"""Import-smoke test for scripts/*.py.

Every script must be importable without executing its main entry point:
they all guard entry logic behind ``if __name__ == "__main__":``. Importing
catches stale imports (e.g. a refactor deleting a src module that a script
still imports) without running any heavy build logic.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"
SCRIPT_PATHS = sorted(SCRIPTS_DIR.glob("*.py"))


@pytest.fixture()
def _isolated_globals():
    """Undo process-global tweaks scripts make at import time.

    profile_backtest.py sets TORCHDYNAMO_DISABLE and monkeypatches
    ``torch.compile`` for its own run; keep that from leaking into the
    rest of the test session.
    """
    env_snapshot = dict(os.environ)
    path_snapshot = list(sys.path)
    compile_snapshot = torch.compile
    yield
    torch.compile = compile_snapshot
    sys.path[:] = path_snapshot
    os.environ.clear()
    os.environ.update(env_snapshot)


def test_scripts_discovered():
    assert SCRIPT_PATHS, f"no scripts found in {SCRIPTS_DIR}"


def test_all_scripts_have_main_guard():
    missing = [
        path.name
        for path in SCRIPT_PATHS
        if 'if __name__ == "__main__":' not in path.read_text(encoding="utf-8")
    ]
    assert not missing, f"scripts without __main__ guard: {missing}"


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_imports_without_running_main(script_path: Path, _isolated_globals):
    module_name = f"_scripts_smoke_{script_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        root_pkg = (exc.name or "").partition(".")[0]
        if root_pkg == "src":
            raise
        pytest.skip(
            f"{script_path.name}: optional dependency {exc.name!r} not installed"
        )
    finally:
        sys.modules.pop(module_name, None)
