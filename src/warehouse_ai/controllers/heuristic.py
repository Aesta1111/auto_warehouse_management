"""Deterministic Manhattan-distance rule-based task-assignment heuristic."""

from __future__ import annotations

from warehouse_ai.domain import ActionMask, AgentRole, Position, WarehouseObservation


class HeuristicController:
    """Deterministic nearest-agent, Manhattan-distance assignment heuristic.

    Mirrors TA-RWARE's own pod-to-picker Kiva-style workflow using only
    fields every controller can see on ``WarehouseObservation`` for the
    current step: idle AGVs fetch the nearest requested shelf, AGVs carrying
    the requested shelf deliver it to the nearest free goal, AGVs returning
    an already-delivered shelf go to the nearest empty shelf location, and
    idle pickers move to whichever shelf location an AGV is actively
    working. Ties break by ascending destination ID, so the same observation
    always produces the same destinations.
    """

    def choose_destinations(
        self, observation: WarehouseObservation, mask: ActionMask
    ) -> list[int]:
        """Return one destination ID per agent, in canonical observation order."""
        goal_count = len(observation.goal_destination_ids)
        goal_position_by_id = dict(
            zip(
                observation.goal_destination_ids,
                observation.goal_positions,
                strict=False,
            )
        )
        shelf_position_by_id = {
            shelf.destination_id: shelf.position for shelf in observation.shelves
        }

        destinations = [0] * len(observation.agents)
        claimed_by_agvs: set[int] = set()
        active_shelf_missions: set[int] = set()

        agv_rows = [
            row
            for row, agent in enumerate(observation.agents)
            if agent.role is AgentRole.AGV
        ]
        picker_rows = [
            row
            for row, agent in enumerate(observation.agents)
            if agent.role is AgentRole.PICKER
        ]

        # Busy AGVs already picking up or returning a shelf keep that
        # location an active picker-assist target.
        for row in agv_rows:
            agent = observation.agents[row]
            if (
                agent.busy
                and not agent.carrying_shelf
                and agent.target_destination_id is not None
                and agent.target_destination_id > goal_count
            ):
                active_shelf_missions.add(agent.target_destination_id)

        # AGVs carrying the requested shelf: deliver to the nearest free goal.
        for row in agv_rows:
            agent = observation.agents[row]
            if agent.busy or not agent.carrying_requested_shelf:
                continue
            destination_id = self._nearest(
                row, observation, mask, goal_position_by_id, claimed_by_agvs
            )
            if destination_id is not None:
                destinations[row] = destination_id
                claimed_by_agvs.add(destination_id)

        # AGVs carrying an already-delivered shelf: return it to an empty slot.
        for row in agv_rows:
            agent = observation.agents[row]
            if agent.busy or destinations[row] != 0:
                continue
            if not agent.carrying_shelf or agent.carrying_requested_shelf:
                continue
            empty_positions = {
                shelf.destination_id: shelf.position
                for shelf in observation.shelves
                if not shelf.occupied
            }
            destination_id = self._nearest(
                row, observation, mask, empty_positions, claimed_by_agvs
            )
            if destination_id is not None:
                destinations[row] = destination_id
                claimed_by_agvs.add(destination_id)

        # Idle AGVs: fetch the nearest requested shelf.
        for row in agv_rows:
            agent = observation.agents[row]
            if agent.busy or destinations[row] != 0 or agent.carrying_shelf:
                continue
            requested_positions = {
                shelf.destination_id: shelf.position
                for shelf in observation.shelves
                if shelf.requested
            }
            destination_id = self._nearest(
                row, observation, mask, requested_positions, claimed_by_agvs
            )
            if destination_id is not None:
                destinations[row] = destination_id
                claimed_by_agvs.add(destination_id)
                active_shelf_missions.add(destination_id)

        # Idle pickers: help at whichever shelf location has an active AGV mission.
        claimed_by_pickers: set[int] = set()
        for row in picker_rows:
            agent = observation.agents[row]
            if agent.busy:
                continue
            mission_positions = {
                destination_id: shelf_position_by_id[destination_id]
                for destination_id in active_shelf_missions
                if destination_id in shelf_position_by_id
            }
            destination_id = self._nearest(
                row, observation, mask, mission_positions, claimed_by_pickers
            )
            if destination_id is not None:
                destinations[row] = destination_id
                claimed_by_pickers.add(destination_id)

        return destinations

    @staticmethod
    def _nearest(
        row: int,
        observation: WarehouseObservation,
        mask: ActionMask,
        candidates: dict[int, Position],
        claimed: set[int],
    ) -> int | None:
        """Return the closest unclaimed, masked-valid candidate destination."""
        origin = observation.agents[row].position.as_features()
        best_id: int | None = None
        best_key: tuple[float, int] | None = None
        for destination_id in sorted(candidates):
            if destination_id in claimed or destination_id >= mask.destination_count:
                continue
            if not bool(mask.values[row, destination_id]):
                continue
            target = candidates[destination_id].as_features()
            distance = abs(origin[0] - target[0]) + abs(origin[1] - target[1])
            key = (distance, destination_id)
            if best_key is None or key < best_key:
                best_key = key
                best_id = destination_id
        return best_id
