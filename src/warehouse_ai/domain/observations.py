"""Typed controller-visible warehouse observations and numeric encoding."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator


class DomainModel(BaseModel):
    """Strict immutable base for serializable domain records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentRole(str, Enum):
    """Roles controlled by TA-RWARE."""

    AGV = "agv"
    PICKER = "picker"


class Position(DomainModel):
    """A zero-based location in row-column order."""

    row: NonNegativeInt
    column: NonNegativeInt

    def as_features(self) -> tuple[float, float]:
        """Return row and column in model-feature order."""
        return float(self.row), float(self.column)


class AgentObservation(DomainModel):
    """Controller-visible state for one agent."""

    agent_id: str
    simulator_id: PositiveInt
    role: AgentRole
    position: Position
    target: Position | None
    target_destination_id: PositiveInt | None
    busy: bool
    carrying_shelf: bool = False
    carrying_requested_shelf: bool = False
    toggling_load: bool = False

    @model_validator(mode="after")
    def validate_role_and_target(self) -> AgentObservation:
        """Enforce role-specific fields and target consistency."""
        if self.role is AgentRole.PICKER and (
            self.carrying_shelf
            or self.carrying_requested_shelf
            or self.toggling_load
        ):
            raise ValueError("picker observations cannot contain AGV-only fields")
        if (self.target is None) != (self.target_destination_id is None):
            raise ValueError("target and target_destination_id must both be set or absent")
        return self


class ShelfObservation(DomainModel):
    """Observable state for one fixed shelf location."""

    destination_id: PositiveInt
    position: Position
    occupied: bool
    requested: bool

    @model_validator(mode="after")
    def validate_request(self) -> ShelfObservation:
        """A request must refer to a shelf currently occupying the location."""
        if self.requested and not self.occupied:
            raise ValueError("an empty shelf location cannot be requested")
        return self


class WarehouseObservation(DomainModel):
    """Canonical centralized observation presented to every controller.

    ``encode`` returns one float32 vector with shape
    ``(7 * num_agvs + 4 * num_pickers + 2 * num_shelf_locations,)``.
    Agents are ordered by ascending simulator ID, followed by shelf locations
    ordered by destination ID. See ``docs/ENCODING.md`` for the complete
    feature table.

    Tuples are intentional here. Agent and shelf order defines model feature
    meaning, so allowing callers to mutate these collections after validation
    would make an observation's encoded representation unstable.
    """

    step: NonNegativeInt
    grid_rows: PositiveInt
    grid_columns: PositiveInt
    agents: tuple[AgentObservation, ...]
    shelves: tuple[ShelfObservation, ...]
    goal_destination_ids: tuple[PositiveInt, ...]

    @model_validator(mode="after")
    def validate_canonical_order(self) -> WarehouseObservation:
        """Require stable unique identifiers in model-feature order."""
        simulator_ids = tuple(agent.simulator_id for agent in self.agents)
        if simulator_ids != tuple(sorted(simulator_ids)) or len(set(simulator_ids)) != len(
            simulator_ids
        ):
            raise ValueError("agents must have unique ascending simulator IDs")
        shelf_ids = tuple(shelf.destination_id for shelf in self.shelves)
        if shelf_ids != tuple(sorted(shelf_ids)) or len(set(shelf_ids)) != len(shelf_ids):
            raise ValueError("shelves must have unique ascending destination IDs")
        if tuple(self.goal_destination_ids) != tuple(sorted(set(self.goal_destination_ids))):
            raise ValueError("goals must have unique ascending destination IDs")
        return self

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Names corresponding one-to-one with values returned by ``encode``."""
        names: list[str] = []
        for agent in self.agents:
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
        for shelf in self.shelves:
            prefix = f"shelf_location.{shelf.destination_id}"
            names.extend((f"{prefix}.occupied", f"{prefix}.requested"))
        return tuple(names)

    @property
    def encoded_shape(self) -> tuple[int]:
        """Shape of the one-dimensional vector returned by ``encode``."""
        return (len(self.feature_names),)

    def encode(self) -> NDArray[np.float32]:
        """Encode this observation as the deterministic model input vector.

        Boolean fields become 0.0 or 1.0. Coordinates remain unnormalized and
        use row-column order. An agent without a target receives target
        coordinates ``[0.0, 0.0]``, matching pinned TA-RWARE global-observation
        behavior. The action mask is deliberately not included; models receive
        it separately when selecting valid destinations.
        """
        features: list[float] = []
        for agent in self.agents:
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
        for shelf in self.shelves:
            features.extend((float(shelf.occupied), float(shelf.requested)))
        return np.asarray(features, dtype=np.float32)
