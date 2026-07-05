"""Conversion between pinned TA-RWARE objects and project-owned records."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from tarware.definitions import Action, AgentType, CollisionLayers

from warehouse_ai.domain import (
    ActionMask,
    AgentAction,
    AgentObservation,
    AgentRole,
    JointAction,
    Position,
    ShelfObservation,
    WarehouseObservation,
)


class InvalidJointActionError(ValueError):
    """Raised when a joint action cannot be safely submitted to TA-RWARE."""


def _agent_id(role: AgentRole, role_index: int) -> str:
    return f"{role.value}-{role_index}"


class TarwareObservationAdapter:
    """Read TA-RWARE's observable global state into canonical records."""

    def observe(self, warehouse: Any) -> WarehouseObservation:
        role_counts = {AgentRole.AGV: 0, AgentRole.PICKER: 0}
        agents: list[AgentObservation] = []
        requested_shelf_ids = {shelf.id for shelf in warehouse.request_queue}

        for agent in warehouse.agents:
            role = self._role(agent.type)
            stable_id = _agent_id(role, role_counts[role])
            role_counts[role] += 1
            target_id = int(agent.target) if agent.target else None
            target = (
                self._position(warehouse.action_id_to_coords_map[target_id])
                if target_id is not None
                else None
            )
            carrying_shelf = getattr(agent, "carrying_shelf", None)
            agents.append(
                AgentObservation(
                    agent_id=stable_id,
                    simulator_id=int(agent.id),
                    role=role,
                    position=Position(row=int(agent.y), column=int(agent.x)),
                    target=target,
                    target_destination_id=target_id,
                    busy=bool(agent.busy),
                    carrying_shelf=role is AgentRole.AGV
                    and carrying_shelf is not None,
                    carrying_requested_shelf=role is AgentRole.AGV
                    and carrying_shelf is not None
                    and carrying_shelf.id in requested_shelf_ids,
                    toggling_load=role is AgentRole.AGV
                    and agent.req_action == Action.TOGGLE_LOAD,
                )
            )

        goal_count = len(warehouse.goals)
        shelves: list[ShelfObservation] = []
        for destination_id in range(goal_count + 1, warehouse.action_size):
            row, column = warehouse.action_id_to_coords_map[destination_id]
            shelf_id = int(warehouse.grid[CollisionLayers.SHELVES, row, column])
            shelves.append(
                ShelfObservation(
                    destination_id=destination_id,
                    position=Position(row=int(row), column=int(column)),
                    occupied=shelf_id != 0,
                    requested=shelf_id in requested_shelf_ids,
                )
            )

        return WarehouseObservation(
            step=int(warehouse._cur_steps),
            grid_rows=int(warehouse.grid_size[0]),
            grid_columns=int(warehouse.grid_size[1]),
            agents=tuple(agents),
            shelves=tuple(shelves),
            goal_destination_ids=tuple(range(1, goal_count + 1)),
        )

    @staticmethod
    def _position(coords: Sequence[int]) -> Position:
        return Position(row=int(coords[0]), column=int(coords[1]))

    @staticmethod
    def _role(agent_type: AgentType) -> AgentRole:
        if agent_type is AgentType.AGV:
            return AgentRole.AGV
        if agent_type is AgentType.PICKER:
            return AgentRole.PICKER
        raise ValueError(f"unsupported TA-RWARE agent type: {agent_type!r}")


class TarwareActionAdapter:
    """Build masks and validate joint destination actions."""

    def action_mask(
        self,
        warehouse: Any,
        observation: WarehouseObservation,
    ) -> ActionMask:
        raw_mask = np.asarray(warehouse.compute_valid_action_masks(), dtype=np.bool_)
        expected_shape = (len(observation.agents), int(warehouse.action_size))
        if raw_mask.shape != expected_shape:
            raise RuntimeError(
                f"TA-RWARE returned mask shape {raw_mask.shape}, expected {expected_shape}"
            )
        raw_mask = raw_mask.copy()
        for row, agent in enumerate(observation.agents):
            if agent.busy:
                raw_mask[row, :] = False
                raw_mask[row, 0] = True
        return ActionMask(
            agent_ids=tuple(agent.agent_id for agent in observation.agents),
            values=raw_mask,
        )

    def to_tarware(
        self,
        action: JointAction,
        observation: WarehouseObservation,
        mask: ActionMask,
    ) -> list[int]:
        expected_ids = tuple(agent.agent_id for agent in observation.agents)
        actual_ids = tuple(item.agent_id for item in action.actions)
        if actual_ids != expected_ids:
            raise InvalidJointActionError(
                f"joint action agents {actual_ids} do not match canonical order {expected_ids}"
            )
        if mask.agent_ids != expected_ids:
            raise InvalidJointActionError("action mask agent ordering does not match observation")

        role_by_id = {agent.agent_id: agent.role for agent in observation.agents}
        selected_by_role: set[tuple[AgentRole, int]] = set()
        result: list[int] = []
        for row, item in enumerate(action.actions):
            if item.destination_id >= mask.destination_count:
                raise InvalidJointActionError(
                    f"{item.agent_id} destination {item.destination_id} is out of range"
                )
            if not bool(mask.values[row, item.destination_id]):
                raise InvalidJointActionError(
                    f"{item.agent_id} destination {item.destination_id} is masked"
                )
            if item.destination_id != 0:
                selection = (role_by_id[item.agent_id], item.destination_id)
                if selection in selected_by_role:
                    raise InvalidJointActionError(
                        "agents with the same role cannot select the same destination"
                    )
                selected_by_role.add(selection)
            result.append(item.destination_id)
        return result

    def no_op(self, observation: WarehouseObservation) -> JointAction:
        return JointAction(
            actions=tuple(
                AgentAction(agent_id=agent.agent_id, destination_id=0)
                for agent in observation.agents
            )
        )
