"""Typed controller-visible warehouse observations and numeric encoding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class AgentRole(str, Enum):
    """Roles controlled by TA-RWARE."""

    AGV = "agv"
    PICKER = "picker"


@dataclass(frozen=True, slots=True)
class Position:
    """A zero-based location in row-column order."""

    row: int
    column: int

    def as_features(self) -> tuple[float, float]:
        return float(self.row), float(self.column)


@dataclass(frozen=True, slots=True)
class AgentObservation:
    """Controller-visible state for one agent."""

    agent_id: str
    simulator_id: int
    role: AgentRole
    position: Position
    target: Position | None
    target_destination_id: int | None
    busy: bool
    carrying_shelf: bool = False
    carrying_requested_shelf: bool = False
    toggling_load: bool = False

    def __post_init__(self) -> None:
        if self.simulator_id < 1:
            raise ValueError("simulator_id must be positive")
        if self.role is AgentRole.PICKER and (
            self.carrying_shelf
            or self.carrying_requested_shelf
            or self.toggling_load
        ):
            raise ValueError("picker observations cannot contain AGV-only fields")
        if (self.target is None) != (self.target_destination_id is None):
            raise ValueError("target and target_destination_id must both be set or absent")


@dataclass(frozen=True, slots=True)
class ShelfObservation:
    """Observable state for one fixed shelf location."""

    destination_id: int
    position: Position
    occupied: bool
    requested: bool

    def __post_init__(self) -> None:
        if self.destination_id < 1:
            raise ValueError("destination_id must be positive")
        if self.requested and not self.occupied:
            raise ValueError("an empty shelf location cannot be requested")


@dataclass(frozen=True, slots=True)
class WarehouseObservation:
    """Canonical centralized observation presented to every controller."""

    step: int
    grid_rows: int
    grid_columns: int
    agents: tuple[AgentObservation, ...]
    shelves: tuple[ShelfObservation, ...]
    goal_destination_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.grid_rows < 1 or self.grid_columns < 1:
            raise ValueError("grid dimensions must be positive")
        simulator_ids = tuple(agent.simulator_id for agent in self.agents)
        if simulator_ids != tuple(sorted(simulator_ids)) or len(set(simulator_ids)) != len(
            simulator_ids
        ):
            raise ValueError("agents must have unique ascending simulator IDs")
        shelf_ids = tuple(shelf.destination_id for shelf in self.shelves)
        if shelf_ids != tuple(sorted(shelf_ids)) or len(set(shelf_ids)) != len(shelf_ids):
            raise ValueError("shelves must have unique ascending destination IDs")


class WarehouseObservationEncoder:
    """Encode canonical observations using TA-RWARE's global feature semantics."""

    def __init__(self, template: WarehouseObservation) -> None:
        self._agent_layout = tuple(
            (agent.simulator_id, agent.role) for agent in template.agents
        )
        self._shelf_ids = tuple(shelf.destination_id for shelf in template.shelves)
        self._feature_names = self._build_feature_names(template)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    def encode(self, observation: WarehouseObservation) -> NDArray[np.float32]:
        self._validate_layout(observation)
        features: list[float] = []
        for agent in observation.agents:
            prefix = f"agent.{agent.agent_id}"
            if agent.role is AgentRole.AGV:
                features.extend(
                    (
                        float(agent.carrying_shelf),
                        float(agent.carrying_requested_shelf),
                        float(agent.toggling_load),
                    )
                )
            features.extend(agent.position.as_features())
            features.extend(
                (0.0, 0.0) if agent.target is None else agent.target.as_features()
            )
        for shelf in observation.shelves:
            features.extend((float(shelf.occupied), float(shelf.requested)))
        return np.asarray(features, dtype=np.float32)

    def _validate_layout(self, observation: WarehouseObservation) -> None:
        layout = tuple(
            (agent.simulator_id, agent.role) for agent in observation.agents
        )
        shelf_ids = tuple(shelf.destination_id for shelf in observation.shelves)
        if layout != self._agent_layout or shelf_ids != self._shelf_ids:
            raise ValueError("observation layout differs from the encoder template")

    @staticmethod
    def _build_feature_names(observation: WarehouseObservation) -> tuple[str, ...]:
        names: list[str] = []
        for agent in observation.agents:
            prefix = f"agent.{agent.agent_id}"
            if agent.role is AgentRole.AGV:
                names.extend(
                    (
                        f"{prefix}.carrying_shelf",
                        f"{prefix}.carrying_requested_shelf",
                        f"{prefix}.toggling_load",
                    )
                )
            names.extend(
                (
                    f"{prefix}.row",
                    f"{prefix}.column",
                    f"{prefix}.target_row",
                    f"{prefix}.target_column",
                )
            )
        for shelf in observation.shelves:
            prefix = f"shelf_location.{shelf.destination_id}"
            names.extend((f"{prefix}.occupied", f"{prefix}.requested"))
        return tuple(names)
