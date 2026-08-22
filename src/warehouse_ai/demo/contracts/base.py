"""Shared validation primitives for demo contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Annotated, Any, TypeVar

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _nonblank(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("value must not be blank")
    return normalized


def _sha256(value: str) -> str:
    normalized = _nonblank(value).lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("value must be a 64-character hexadecimal SHA-256")
    return normalized


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must include a timezone")
    return value


def _repository_path(value: Path) -> Path:
    if value.is_absolute() or PureWindowsPath(str(value)).is_absolute():
        raise ValueError("path must be repository-relative")
    return value


NonBlankStr = Annotated[str, AfterValidator(_nonblank)]
Sha256 = Annotated[str, AfterValidator(_sha256)]
UtcDateTime = Annotated[datetime, AfterValidator(_utc_datetime)]
RepositoryPath = Annotated[Path, AfterValidator(_repository_path)]


class DemoModel(BaseModel):
    """Strict immutable base for serialized prototype contracts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )


def validate_seed_partition(value: Sequence[int], name: str) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in value)
    if not seeds:
        raise ValueError(f"{name} must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must not contain duplicate seeds")
    return seeds


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load YAML configuration {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML configuration {path} must contain a mapping")
    return data


ConfigT = TypeVar("ConfigT", bound=DemoModel)


def load_yaml_config(path: Path, model_type: type[ConfigT]) -> ConfigT:
    """Load and strictly validate one YAML configuration model."""
    return model_type.model_validate(load_yaml_mapping(path))
