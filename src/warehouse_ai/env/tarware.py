"""Project-owned wrapper around the pinned TA-RWARE environment."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import gymnasium as gym
import numpy as np
import tarware  # noqa: F401 - importing registers TA-RWARE environments
from numpy.typing import ArrayLike, NDArray
from tarware.definitions import Action, AgentType, CollisionLayers
from tarware.warehouse import Warehouse

from warehouse_ai.domain import (
    ActionMask,
    ActionRepair,
    AgentAction,
    AgentObservation,
    AgentRole,
    JointAction,
    Position,
    ProjectedJointAction,
    ShelfObservation,
    WarehouseObservation,
)


class InvalidJointActionError(ValueError):
    """Raised when a model output or action cannot be safely submitted."""


def _agent_id(role: AgentRole, role_index: int) -> str:
    return f"{role.value}-{role_index}"


class TarwareEnvWrapper:
    """Expose TA-RWARE through project-owned observation and action classes.

    ``reset`` and ``step`` return ``WarehouseObservation`` instead of
    TA-RWARE's heterogeneous per-agent vectors. ``step`` accepts only a
    validated ``JointAction``. Use ``decode`` to turn PPO-style raw model output
    into that action.
    """

    def __init__(self, environment_id: str) -> None:
        self.environment_id = environment_id
        self._env = gym.make(environment_id, disable_env_checker=True)
        self._warehouse = cast(Warehouse, self._env.unwrapped)
        self._observation: WarehouseObservation | None = None

    @property
    def unwrapped(self) -> Warehouse:
        """Underlying TA-RWARE instance for compatibility tests only."""
        return self._warehouse

    @property
    def observation(self) -> WarehouseObservation:
        """Most recent observation, requiring ``reset`` first."""
        if self._observation is None:
            raise RuntimeError("reset must be called before reading observation")
        return self._observation

    @property
    def action_mask(self) -> ActionMask:
        """Current valid destinations, including project-level busy handling."""
        observation = self.observation
        raw_mask = np.asarray(
            self._warehouse.compute_valid_action_masks(),
            dtype=np.bool_,
        )
        expected_shape = (len(observation.agents), int(self._warehouse.action_size))
        if raw_mask.shape != expected_shape:
            raise RuntimeError(
                f"TA-RWARE returned mask shape {raw_mask.shape}, expected {expected_shape}"
            )
        for row, agent in enumerate(observation.agents):
            if agent.busy:
                raw_mask[row, :] = False
                raw_mask[row, 0] = True
        return ActionMask(
            agent_ids=tuple(agent.agent_id for agent in observation.agents),
            values=raw_mask,
        )

    def reset(self, *, seed: int) -> WarehouseObservation:
        """Reset with an explicit seed and return the canonical observation."""
        _ = self._warehouse.reset(seed=seed)
        self._observation = self._read_observation()
        return self._observation

    def decode(self, model_output: ArrayLike) -> JointAction:
        """Decode a raw model action vector into a validated ``JointAction``.

        The required shape is ``(num_agents,)``. Element ``i`` is the integer
        destination ID for observation agent ``i``. Agents use canonical
        observation order: ascending simulator ID, which is
        ``[agv-0, agv-1, agv-2, picker-0, picker-1]`` in the tiny environment.

        Destination zero means no-op, followed by goal IDs and then fixed
        shelf-location IDs. The tiny environment requires shape ``(5,)`` with
        every value in ``[0, 54]``. This method expects selected integer IDs,
        not logits or probabilities. Floating-point values, nested/batched
        arrays, masked destinations, and same-role destination conflicts are
        rejected.

        Returns:
            A validated ``JointAction`` ready for ``step``.
        """
        observation = self.observation
        raw = self._validate_raw_output(model_output)

        action = JointAction(
            actions=tuple(
                AgentAction(
                    agent_id=agent.agent_id,
                    destination_id=int(destination_id),
                )
                for agent, destination_id in zip(observation.agents, raw, strict=True)
            )
        )
        _ = self._validate(action, self.action_mask)
        return action

    def project(self, model_output: ArrayLike) -> ProjectedJointAction:
        """Project only cross-agent conflicts into safe, logged no-ops.

        This compatibility path exists for algorithms whose ``MultiDiscrete``
        masks cannot represent dependencies between action components. Shape,
        dtype, range, and ordinary per-agent mask violations are still errors.
        For a duplicate nonzero destination among agents with the same role,
        canonical order wins and every later selection becomes a no-op.
        """
        observation = self.observation
        raw = self._validate_raw_output(model_output)
        mask = self.action_mask
        selected_by_role: set[tuple[AgentRole, int]] = set()
        applied = raw.copy()
        repairs: list[ActionRepair] = []

        for row, (agent, destination) in enumerate(
            zip(observation.agents, raw, strict=True)
        ):
            destination_id = int(destination)
            if destination_id >= mask.destination_count:
                raise InvalidJointActionError(
                    f"{agent.agent_id} destination {destination_id} is out of range"
                )
            if not bool(mask.values[row, destination_id]):
                raise InvalidJointActionError(
                    f"{agent.agent_id} destination {destination_id} is masked"
                )
            selection = (agent.role, destination_id)
            if destination_id != 0 and selection in selected_by_role:
                applied[row] = 0
                repairs.append(
                    ActionRepair(
                        agent_id=agent.agent_id,
                        requested_destination_id=destination_id,
                        reason="same_role_destination_conflict",
                    )
                )
            elif destination_id != 0:
                selected_by_role.add(selection)

        action = self.decode(applied)
        return ProjectedJointAction(
            requested_destinations=tuple(int(value) for value in raw),
            action=action,
            repairs=tuple(repairs),
        )

    def no_op(self) -> JointAction:
        """Return the canonical all-agent no-op action."""
        return self.decode(np.zeros(len(self.observation.agents), dtype=np.int64))

    def step(
        self,
        action: JointAction,
    ) -> tuple[
        WarehouseObservation,
        list[float],
        list[bool],
        list[bool],
        dict[str, Any],
    ]:
        """Validate and apply one joint action, then return project state."""
        raw_action = self._validate(action, self.action_mask)
        _, rewards, terminated, truncated, info = self._warehouse.step(raw_action)
        self._observation = self._read_observation()
        return self._observation, rewards, terminated, truncated, info

    def render(self, mode: str = "human") -> Any:
        """Delegate rendering to the pinned TA-RWARE environment."""
        return self._warehouse.render(mode=mode)

    def close(self) -> None:
        """Release TA-RWARE rendering and environment resources."""
        self._env.close()

    def _validate_raw_output(self, model_output: ArrayLike) -> NDArray[Any]:
        observation = self.observation
        raw = np.asarray(model_output)
        expected_shape = (len(observation.agents),)
        if raw.shape != expected_shape:
            raise InvalidJointActionError(
                f"model output shape {raw.shape} does not match {expected_shape}"
            )
        if not np.issubdtype(raw.dtype, np.integer):
            raise InvalidJointActionError(
                "model output must contain integer destination IDs"
            )
        if np.any(raw < 0):
            raise InvalidJointActionError(
                "model output destination IDs cannot be negative"
            )
        return raw

    def _validate(self, action: JointAction, mask: ActionMask) -> list[int]:
        observation = self.observation
        expected_ids = tuple(agent.agent_id for agent in observation.agents)
        actual_ids = tuple(item.agent_id for item in action.actions)
        if actual_ids != expected_ids:
            raise InvalidJointActionError(
                f"joint action agents {actual_ids} do not match canonical order {expected_ids}"
            )
        if mask.agent_ids != expected_ids:
            raise InvalidJointActionError(
                "action mask agent ordering does not match observation"
            )

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
                selection = (role_by_id[item.agent_id], int(item.destination_id))
                if selection in selected_by_role:
                    raise InvalidJointActionError(
                        "agents with the same role cannot select the same destination"
                    )
                selected_by_role.add(selection)
            result.append(int(item.destination_id))
        return result

    def _read_observation(self) -> WarehouseObservation:
        warehouse = self._warehouse
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
                    carrying_shelf=role is AgentRole.AGV and carrying_shelf is not None,
                    carrying_requested_shelf=role is AgentRole.AGV
                    and carrying_shelf is not None
                    and carrying_shelf.id in requested_shelf_ids,
                    toggling_load=role is AgentRole.AGV
                    and agent.req_action == Action.TOGGLE_LOAD,
                )
            )

        goal_count = len(warehouse.goals)
        goal_positions = tuple(
            self._position(warehouse.action_id_to_coords_map[destination_id])
            for destination_id in range(1, goal_count + 1)
        )
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
            goal_positions=goal_positions,
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
