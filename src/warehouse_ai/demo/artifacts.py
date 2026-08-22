"""Small deterministic artifact helpers shared by the prototype phases."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _jsonable(value: Any) -> Any:
    """Convert common record/container values to JSON-compatible values."""
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json_text(value: Any) -> str:
    """Serialize a value using the project's stable JSON conventions."""
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return UTF-8 bytes for :func:`canonical_json_text`."""
    return canonical_json_text(value).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest for a byte string."""
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading the whole artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_git_state(repository: Path) -> tuple[str, bool]:
    """Return ``(HEAD SHA, dirty-worktree)`` for a Git repository."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not capture Git state for {repository}") from exc

    if not commit:
        raise RuntimeError(f"Git repository {repository} has no HEAD commit")
    return commit, bool(status.strip())


def write_json_atomic(path: Path, model: BaseModel) -> None:
    """Atomically write one small Pydantic artifact as canonical JSON.

    The temporary file is created beside the destination so replacement is an
    atomic same-filesystem operation. A failed serialization never replaces an
    existing destination.
    """
    payload = canonical_json_bytes(model) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def append_jsonl(path: Path, model: BaseModel) -> None:
    """Append one canonical Pydantic record and flush it immediately.

    Decision logs are intentionally append-only. Opening the file for each
    record keeps the helper usable by an episode loop without holding a file
    descriptor for the whole run, and ``fsync`` makes a completed decision
    visible even if a later simulator or model call fails.
    """
    payload = canonical_json_bytes(model) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def create_run_directory(
    base_directory: Path,
    *,
    controller_name: str,
    created_at_utc: datetime | None = None,
    git_commit: str,
) -> Path:
    """Create a unique, machine-sortable directory for one demo run."""
    if not controller_name or not controller_name.strip():
        raise ValueError("controller_name must not be blank")
    if len(git_commit) < 8 or any(
        character not in "0123456789abcdef" for character in git_commit.lower()
    ):
        raise ValueError("git_commit must contain a hexadecimal commit identifier")

    timestamp = (created_at_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{controller_name.strip()}-{timestamp_text}-{git_commit[:8]}"
    run_directory = base_directory / run_id
    run_directory.mkdir(parents=True, exist_ok=False)
    return run_directory
