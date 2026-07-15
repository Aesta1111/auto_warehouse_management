"""Typed joint macro-actions and validity masks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    field_validator,
    model_validator,
)


class ActionModel(BaseModel):
    """Strict immutable base for action records."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class AgentAction(ActionModel):
    """One agent's requested macro destination; zero means no new destination."""

    agent_id: str
    destination_id: NonNegativeInt


class JointAction(ActionModel):
    """One simultaneous macro-action in canonical agent order.

    The tuple prevents action order from changing after validation. Each entry
    corresponds to the agent at the same index in ``WarehouseObservation``.
    """

    actions: tuple[AgentAction, ...]


class ActionRepair(ActionModel):
    """One explicit deterministic repair made for an SB3 joint action."""

    agent_id: str
    requested_destination_id: NonNegativeInt
    applied_destination_id: Literal[0] = 0
    reason: Literal["same_role_destination_conflict"]


class ProjectedJointAction(ActionModel):
    """Requested destinations and the validated action actually applied."""

    requested_destinations: tuple[NonNegativeInt, ...]
    action: JointAction
    repairs: tuple[ActionRepair, ...]


class ActionMask(ActionModel):
    """Per-agent valid destination mask with canonical agent ordering."""

    agent_ids: tuple[str, ...]
    values: NDArray[np.bool_]

    @field_validator("values", mode="before")
    @classmethod
    def coerce_boolean_array(cls, value: object) -> NDArray[np.bool_]:
        """Copy input into an immutable boolean NumPy array."""
        values = np.asarray(value, dtype=np.bool_).copy()
        values.setflags(write=False)
        return values

    @model_validator(mode="after")
    def validate_shape(self) -> ActionMask:
        """Validate mask axes and the always-valid no-op column."""
        if self.values.ndim != 2:
            raise ValueError("action mask must have shape [agent, destination]")
        if self.values.shape[0] != len(self.agent_ids):
            raise ValueError("action mask row count must match agent_ids")
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("action mask agent_ids must be unique")
        if self.values.shape[1] < 1 or not np.all(self.values[:, 0]):
            raise ValueError("destination zero must be valid for every agent")
        return self

    @property
    def destination_count(self) -> int:
        """Number of destination choices including no-op."""
        return int(self.values.shape[1])
