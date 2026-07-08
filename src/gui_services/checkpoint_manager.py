"""Checkpoint file management helpers.

Pure business-logic helpers used by the Checkpoints tab (snapshot, rename,
tag, soft-delete) and the Runtime tab's checkpoint picker.  Intentionally
Qt-free so they can be unit-tested without a Qt event loop.

Responsibilities:
  - list checkpoint files with metadata (size, mtime, tag, registry status)
  - snapshot a checkpoint to a timestamped backup file
  - rename a checkpoint on disk (refuses when the file is in use)
  - tag a checkpoint via a sidecar ``checkpoints/tags.json`` file
  - soft-delete: move to ``checkpoints/_trash/`` (never hard-remove)

The sidecar tags file is a flat ``{file_name: tag}`` dict keyed by the
basename, not the absolute path — so rename operations preserve the tag
(see ``rename_checkpoint`` which updates the sidecar in the same op).

None of these helpers touch the registry schema at
``src/training/checkpoint_promotion.py`` — the registry continues to be
the source of truth for *promoted* checkpoints.  These helpers are about
ad-hoc file management on disk.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.gui_services.state import CheckpointFile, ServiceError
from src.training.checkpoint_promotion import get_active_checkpoint, load_registry


CHECKPOINTS_DIR = Path("checkpoints")
TRASH_DIR = CHECKPOINTS_DIR / "_trash"
TAGS_PATH = CHECKPOINTS_DIR / "tags.json"
REGISTRY_PATH = CHECKPOINTS_DIR / "registry.json"


# ---------------------------------------------------------------------------
# listing
# ---------------------------------------------------------------------------
def list_checkpoint_files(directory: Path | str = CHECKPOINTS_DIR) -> list[CheckpointFile]:
    """List every ``*.pt`` file in ``directory`` (non-recursive), sorted by mtime desc.

    Skips files inside a ``_trash`` subfolder.  Cross-references the
    ``checkpoints/registry.json`` for active/version info and the sidecar
    tags file for operator-assigned labels.
    """
    root = Path(directory)
    if not root.exists():
        return []

    tags = load_tags()
    active_entry = get_active_checkpoint(REGISTRY_PATH) if REGISTRY_PATH.exists() else None
    active_path = (
        Path(active_entry["checkpoint_path"]).resolve()
        if active_entry and active_entry.get("checkpoint_path")
        else None
    )

    registry_entries = (
        load_registry(REGISTRY_PATH) if REGISTRY_PATH.exists() else []
    )
    by_resolved_path: dict[Path, dict] = {}
    for entry in registry_entries:
        p = entry.get("checkpoint_path")
        if not p:
            continue
        try:
            by_resolved_path[Path(p).resolve()] = entry
        except OSError:
            continue

    files: list[CheckpointFile] = []
    for path in root.glob("*.pt"):
        if not path.is_file():
            continue
        # Defensive: someone could symlink a trash file back in.
        if "_trash" in path.parts or "trash" in path.name.lower():
            continue
        try:
            stat = path.stat()
            resolved = path.resolve()
        except OSError:
            continue
        entry = by_resolved_path.get(resolved, {})
        is_active = active_path is not None and resolved == active_path
        files.append(
            CheckpointFile(
                path=path,
                size_bytes=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                tag=tags.get(path.name, ""),
                registry_version_id=entry.get("version_id", ""),
                is_active=is_active,
            )
        )
    files.sort(key=lambda f: f.mtime, reverse=True)
    return files


# ---------------------------------------------------------------------------
# tagging (sidecar JSON — checkpoints/tags.json)
# ---------------------------------------------------------------------------
def load_tags() -> dict[str, str]:
    if not TAGS_PATH.exists():
        return {}
    try:
        with open(TAGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    # Defensive: make sure we got a flat string-to-string map.
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, (str, int, float))}


def _save_tags(tags: dict[str, str]) -> None:
    TAGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TAGS_PATH, "w", encoding="utf-8") as f:
        json.dump(tags, f, indent=2, sort_keys=True)


def tag_checkpoint(path: Path | str, tag: str) -> None:
    """Assign (or clear, when tag is empty) a tag for ``path``."""
    p = Path(path)
    tags = load_tags()
    clean = tag.strip()
    if clean:
        tags[p.name] = clean
    else:
        tags.pop(p.name, None)
    _save_tags(tags)


# ---------------------------------------------------------------------------
# snapshot (copy to timestamped backup)
# ---------------------------------------------------------------------------
def snapshot_checkpoint(source: Path | str, *, tag: str | None = None) -> Path:
    """Copy ``source`` to ``checkpoints/<stem>_backup_<YYYYMMDD_HHMMSS>.pt``.

    If ``tag`` is given, the new file inherits the tag in the sidecar.
    Uses ``shutil.copy2`` so the mtime is preserved on the source file
    (the backup gets a fresh mtime, which is what we want for sorting).
    """
    src = Path(source)
    if not src.exists():
        raise ServiceError(f"Checkpoint not found: {src}")
    if not src.is_file():
        raise ServiceError(f"Not a file: {src}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = CHECKPOINTS_DIR / f"{src.stem}_backup_{ts}.pt"
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    # Set fresh mtime on the copy so it sorts above the source in the picker.
    dest.touch()
    if tag:
        tag_checkpoint(dest, tag)
    return dest


# ---------------------------------------------------------------------------
# rename
# ---------------------------------------------------------------------------
def rename_checkpoint(source: Path | str, new_name: str) -> Path:
    """Rename a checkpoint file (preserves tag + extension).

    Does NOT move the file to a different directory; ``new_name`` must be
    a basename (no path separators).  Refuses when:
      - the target already exists
      - ``new_name`` is empty or contains path separators
      - the file is currently marked as active in the registry
    """
    src = Path(source)
    clean = new_name.strip()
    if not clean:
        raise ServiceError("New name is empty.")
    if "/" in clean or "\\" in clean:
        raise ServiceError("New name must be a basename, not a path.")
    if not src.exists():
        raise ServiceError(f"Checkpoint not found: {src}")
    # Preserve the .pt extension if the user omitted it.
    if not clean.endswith(".pt"):
        clean = f"{clean}.pt"
    dest = src.with_name(clean)
    if dest.exists():
        raise ServiceError(f"Target already exists: {dest}")
    # Don't let operators rename the live-loaded checkpoint out from under
    # the running trade loop.  Active = registry.status=='active'.
    if _is_active_in_registry(src):
        raise ServiceError(
            "Refusing to rename the currently-active checkpoint. "
            "Stop the runtime and promote a different checkpoint first."
        )
    # Preserve the tag (sidecar is keyed by basename, so we have to migrate).
    tags = load_tags()
    tag = tags.pop(src.name, None)
    src.rename(dest)
    if tag:
        tags[dest.name] = tag
    if tag or tags != load_tags():
        _save_tags(tags)
    return dest


# ---------------------------------------------------------------------------
# trash (soft delete)
# ---------------------------------------------------------------------------
def move_to_trash(source: Path | str) -> Path:
    """Move a checkpoint to ``checkpoints/_trash/``.

    We never hard-delete: this protects the operator from catastrophically
    losing hours of training if they misclick.  The trash folder is not
    scanned by ``list_checkpoint_files`` so it won't show up in any picker.

    Refuses when the source is the currently-active checkpoint.  If a
    same-named file already exists in the trash (e.g., same basename
    moved twice), appends a timestamp to disambiguate.
    """
    src = Path(source)
    if not src.exists():
        raise ServiceError(f"Checkpoint not found: {src}")
    if _is_active_in_registry(src):
        raise ServiceError(
            "Refusing to trash the currently-active checkpoint. "
            "Promote a different checkpoint first."
        )
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    dest = TRASH_DIR / src.name
    if dest.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = TRASH_DIR / f"{src.stem}_{ts}{src.suffix}"
    src.rename(dest)
    # Drop the tag sidecar entry — a trashed file is no longer visible, so
    # a stale tag would just clutter the sidecar.
    tags = load_tags()
    if src.name in tags:
        tags.pop(src.name)
        _save_tags(tags)
    return dest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_active_in_registry(path: Path) -> bool:
    """True if ``path`` is the checkpoint marked active in registry.json."""
    if not REGISTRY_PATH.exists():
        return False
    entry = get_active_checkpoint(REGISTRY_PATH)
    if not entry:
        return False
    active_str = entry.get("checkpoint_path", "")
    if not active_str:
        return False
    try:
        return path.resolve() == Path(active_str).resolve()
    except OSError:
        return False
