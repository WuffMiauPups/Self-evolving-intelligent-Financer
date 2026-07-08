"""Tests for checkpoint_manager — list/snapshot/rename/tag/trash helpers.

These tests use ``tmp_path`` + monkeypatching to redirect the module's
CHECKPOINTS_DIR / TRASH_DIR / TAGS_PATH so they don't touch the real
checkpoints directory.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.gui_services import checkpoint_manager as cm
from src.gui_services.state import CheckpointFile, ServiceError


@pytest.fixture
def ckpt_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect checkpoint_manager to a tmp directory + create some files."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    monkeypatch.setattr(cm, "CHECKPOINTS_DIR", ckpt_dir)
    monkeypatch.setattr(cm, "TRASH_DIR", ckpt_dir / "_trash")
    monkeypatch.setattr(cm, "TAGS_PATH", ckpt_dir / "tags.json")
    monkeypatch.setattr(cm, "REGISTRY_PATH", ckpt_dir / "registry.json")
    # Create two dummy checkpoint files.
    (ckpt_dir / "brain.pt").write_bytes(b"\x00" * 128)
    (ckpt_dir / "brain_old.pt").write_bytes(b"\x00" * 64)
    return ckpt_dir


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------
class TestListCheckpointFiles:
    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        assert cm.list_checkpoint_files(tmp_path / "missing") == []

    def test_returns_metadata_for_each_file(self, ckpt_env: Path) -> None:
        files = cm.list_checkpoint_files(ckpt_env)
        assert len(files) == 2
        names = {f.path.name for f in files}
        assert names == {"brain.pt", "brain_old.pt"}
        # sorted by mtime desc — both were just created, so both sorted
        assert all(isinstance(f.mtime, datetime) for f in files)
        assert all(f.size_bytes > 0 for f in files)
        assert all(f.tag == "" for f in files)  # no tags assigned yet
        assert all(not f.is_active for f in files)  # no registry yet

    def test_trash_files_are_hidden(self, ckpt_env: Path) -> None:
        trash = ckpt_env / "_trash"
        trash.mkdir()
        (trash / "brain_old.pt").write_bytes(b"\x00")
        files = cm.list_checkpoint_files(ckpt_env)
        # The _trash directory is not recursed into
        assert all("_trash" not in f.path.parts for f in files)

    def test_tags_are_attached(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain.pt", "prod")
        files = cm.list_checkpoint_files(ckpt_env)
        tagged = next(f for f in files if f.path.name == "brain.pt")
        assert tagged.tag == "prod"

    def test_active_flag_reflects_registry(self, ckpt_env: Path) -> None:
        reg = [
            {
                "version_id": "v1",
                "checkpoint_path": str(ckpt_env / "brain.pt"),
                "fingerprint": "abc",
                "status": "active",
                "created_at": "2026-01-01T00:00:00",
                "metrics": {},
            }
        ]
        (ckpt_env / "registry.json").write_text(json.dumps(reg))
        files = cm.list_checkpoint_files(ckpt_env)
        active = [f for f in files if f.is_active]
        assert len(active) == 1
        assert active[0].path.name == "brain.pt"
        assert active[0].registry_version_id == "v1"


# ---------------------------------------------------------------------------
# tagging
# ---------------------------------------------------------------------------
class TestTagCheckpoint:
    def test_tag_creates_sidecar_file(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain.pt", "40-symbol")
        assert (ckpt_env / "tags.json").exists()
        tags = cm.load_tags()
        assert tags["brain.pt"] == "40-symbol"

    def test_empty_tag_clears_entry(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain.pt", "prod")
        cm.tag_checkpoint(ckpt_env / "brain.pt", "")
        assert "brain.pt" not in cm.load_tags()

    def test_whitespace_only_tag_clears_entry(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain.pt", "prod")
        cm.tag_checkpoint(ckpt_env / "brain.pt", "   ")
        assert "brain.pt" not in cm.load_tags()

    def test_corrupted_sidecar_returns_empty(self, ckpt_env: Path) -> None:
        (ckpt_env / "tags.json").write_text("not json")
        assert cm.load_tags() == {}


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------
class TestSnapshotCheckpoint:
    def test_snapshot_creates_timestamped_backup(self, ckpt_env: Path) -> None:
        src = ckpt_env / "brain.pt"
        dest = cm.snapshot_checkpoint(src)
        assert dest.exists()
        assert dest.parent == ckpt_env
        assert "backup" in dest.name
        assert dest.read_bytes() == src.read_bytes()

    def test_snapshot_missing_source_raises(self, ckpt_env: Path) -> None:
        with pytest.raises(ServiceError, match="not found"):
            cm.snapshot_checkpoint(ckpt_env / "missing.pt")

    def test_snapshot_with_tag_tags_destination(self, ckpt_env: Path) -> None:
        dest = cm.snapshot_checkpoint(ckpt_env / "brain.pt", tag="pre-retrain")
        assert cm.load_tags()[dest.name] == "pre-retrain"


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------
class TestRenameCheckpoint:
    def test_rename_moves_file_and_preserves_tag(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain.pt", "prod")
        dest = cm.rename_checkpoint(ckpt_env / "brain.pt", "brain_v2.pt")
        assert dest.exists()
        assert not (ckpt_env / "brain.pt").exists()
        tags = cm.load_tags()
        assert tags.get("brain_v2.pt") == "prod"
        assert "brain.pt" not in tags

    def test_rename_adds_pt_extension_if_missing(self, ckpt_env: Path) -> None:
        dest = cm.rename_checkpoint(ckpt_env / "brain.pt", "exp1")
        assert dest.name == "exp1.pt"

    def test_rename_refuses_path_separators(self, ckpt_env: Path) -> None:
        with pytest.raises(ServiceError, match="basename"):
            cm.rename_checkpoint(ckpt_env / "brain.pt", "sub/brain.pt")
        with pytest.raises(ServiceError, match="basename"):
            cm.rename_checkpoint(ckpt_env / "brain.pt", "sub\\brain.pt")

    def test_rename_refuses_empty_name(self, ckpt_env: Path) -> None:
        with pytest.raises(ServiceError, match="empty"):
            cm.rename_checkpoint(ckpt_env / "brain.pt", "   ")

    def test_rename_refuses_existing_target(self, ckpt_env: Path) -> None:
        with pytest.raises(ServiceError, match="already exists"):
            cm.rename_checkpoint(ckpt_env / "brain.pt", "brain_old.pt")

    def test_rename_refuses_active_checkpoint(self, ckpt_env: Path) -> None:
        reg = [
            {
                "version_id": "v1",
                "checkpoint_path": str(ckpt_env / "brain.pt"),
                "fingerprint": "abc",
                "status": "active",
                "created_at": "2026-01-01T00:00:00",
                "metrics": {},
            }
        ]
        (ckpt_env / "registry.json").write_text(json.dumps(reg))
        with pytest.raises(ServiceError, match="active"):
            cm.rename_checkpoint(ckpt_env / "brain.pt", "brain_v2.pt")


# ---------------------------------------------------------------------------
# trash
# ---------------------------------------------------------------------------
class TestMoveToTrash:
    def test_trash_moves_to_trash_dir(self, ckpt_env: Path) -> None:
        dest = cm.move_to_trash(ckpt_env / "brain_old.pt")
        assert dest.exists()
        assert dest.parent.name == "_trash"
        assert not (ckpt_env / "brain_old.pt").exists()

    def test_trash_collision_disambiguated(self, ckpt_env: Path) -> None:
        # Seed trash with a same-named file
        trash = ckpt_env / "_trash"
        trash.mkdir()
        (trash / "brain_old.pt").write_bytes(b"\x00")
        dest = cm.move_to_trash(ckpt_env / "brain_old.pt")
        assert dest.parent.name == "_trash"
        # The new file must not clobber the existing one
        assert dest.name != "brain_old.pt"
        assert (trash / "brain_old.pt").exists()

    def test_trash_refuses_active(self, ckpt_env: Path) -> None:
        reg = [
            {
                "version_id": "v1",
                "checkpoint_path": str(ckpt_env / "brain.pt"),
                "fingerprint": "abc",
                "status": "active",
                "created_at": "2026-01-01T00:00:00",
                "metrics": {},
            }
        ]
        (ckpt_env / "registry.json").write_text(json.dumps(reg))
        with pytest.raises(ServiceError, match="active"):
            cm.move_to_trash(ckpt_env / "brain.pt")

    def test_trash_drops_tag_sidecar_entry(self, ckpt_env: Path) -> None:
        cm.tag_checkpoint(ckpt_env / "brain_old.pt", "stale")
        cm.move_to_trash(ckpt_env / "brain_old.pt")
        assert "brain_old.pt" not in cm.load_tags()

    def test_trash_missing_source_raises(self, ckpt_env: Path) -> None:
        with pytest.raises(ServiceError, match="not found"):
            cm.move_to_trash(ckpt_env / "missing.pt")
