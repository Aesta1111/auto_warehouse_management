"""Typed joint macro-actions and validity masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AgentAction:
    """One agent's requested macro destination; zero means no new destination."""

    agent_id: str
    destination_id: int

    def __post_init__(self) -> None:
        if self.destination_id < 0:
            raise ValueError("destination_id must be non-negative")


@dataclass(frozen=True, slots=True)
class JointAction:
    """One simultaneous macro-action in canonical agent order."""

    actions: tuple[AgentAction, ...]


@dataclass(frozen=True, slots=True)
class ActionMask:
    """Per-agent valid destination mask with canonical agent ordering."""

    agent_ids: tuple[str, ...]
    values: NDArray[np.bool_]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.bool_)
        if values.ndim != 2:
            raise ValueError("action mask must have shape [agent, destination]")
        if values.shape[0] != len(self.agent_ids):
            raise ValueError("action mask row count must match agent_ids")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("action mask agent_ids must be unique")
        if values.shape[1] < 1 or not np.all(values[:, 0]):
            raise ValueError("destination zero must be valid for every agent")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    @property
    def destination_count(self) -> int:
        return int(self.values.shape[1])
