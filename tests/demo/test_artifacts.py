from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from warehouse_ai.demo.artifacts import (
    canonical_json_bytes,
    capture_git_state,
    sha256_file,
    write_json_atomic,
)


class _Artifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    values: tuple[int, ...]


def test_canonical_json_and_atomic_write_are_repeatable(tmp_path: Path) -> None:
    artifact = _Artifact(name="demo", values=(2, 1))
    destination = tmp_path / "nested" / "artifact.json"

    write_json_atomic(destination, artifact)
    first_bytes = destination.read_bytes()
    first_digest = sha256_file(destination)
    write_json_atomic(destination, artifact)

    assert destination.read_bytes() == first_bytes
    assert sha256_file(destination) == first_digest
    assert json.loads(first_bytes) == {"name": "demo", "values": [2, 1]}
    assert canonical_json_bytes(artifact) in first_bytes
    assert not list(destination.parent.glob("*.tmp"))


def test_capture_git_state_reports_current_repository() -> None:
    repository = Path(__file__).parents[2]

    commit, dirty = capture_git_state(repository)

    assert len(commit) == 40
    assert isinstance(dirty, bool)
