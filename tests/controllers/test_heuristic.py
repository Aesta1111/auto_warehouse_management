from __future__ import annotations

import numpy as np

from warehouse_ai.controllers import HeuristicController
from warehouse_ai.domain import (
    ActionMask,
    AgentObservation,
    AgentRole,
    Position,
    ShelfObservation,
    WarehouseObservation,
)


def _mask(agent_ids: tuple[str, ...], destination_count: int) -> ActionMask:
    return ActionMask(
        agent_ids=agent_ids,
        values=np.ones((len(agent_ids), destination_count), dtype=np.bool_),
    )


def test_heuristic_covers_the_full_pod_lifecycle_and_helps_pickers() -> None:
    observation = WarehouseObservation(
        step=0,
        grid_rows=10,
        grid_columns=30,
        agents=(
            AgentObservation(
                agent_id="agv-0",
                simulator_id=1,
                role=AgentRole.AGV,
                position=Position(row=0, column=0),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
            AgentObservation(
                agent_id="agv-1",
                simulator_id=2,
                role=AgentRole.AGV,
                position=Position(row=0, column=10),
                target=None,
                target_destination_id=None,
                busy=False,
                carrying_shelf=True,
                carrying_requested_shelf=True,
            ),
            AgentObservation(
                agent_id="agv-2",
                simulator_id=3,
                role=AgentRole.AGV,
                position=Position(row=0, column=20),
                target=None,
                target_destination_id=None,
                busy=False,
                carrying_shelf=True,
                carrying_requested_shelf=False,
            ),
            AgentObservation(
                agent_id="picker-0",
                simulator_id=4,
                role=AgentRole.PICKER,
                position=Position(row=5, column=5),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=10,
                position=Position(row=0, column=1),
                occupied=True,
                requested=True,
            ),
            ShelfObservation(
                destination_id=11,
                position=Position(row=0, column=2),
                occupied=True,
                requested=True,
            ),
            ShelfObservation(
                destination_id=12,
                position=Position(row=0, column=21),
                occupied=False,
                requested=False,
            ),
        ),
        goal_destination_ids=(1, 2),
        goal_positions=(Position(row=0, column=9), Position(row=0, column=11)),
    )
    mask = _mask(
        tuple(agent.agent_id for agent in observation.agents),
        destination_count=13,
    )

    destinations = HeuristicController().choose_destinations(observation, mask)

    assert destinations == [10, 1, 12, 10]


def test_heuristic_does_not_send_two_idle_agvs_to_the_same_shelf() -> None:
    observation = WarehouseObservation(
        step=0,
        grid_rows=5,
        grid_columns=5,
        agents=(
            AgentObservation(
                agent_id="agv-0",
                simulator_id=1,
                role=AgentRole.AGV,
                position=Position(row=0, column=0),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
            AgentObservation(
                agent_id="agv-1",
                simulator_id=2,
                role=AgentRole.AGV,
                position=Position(row=0, column=0),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=10,
                position=Position(row=0, column=1),
                occupied=True,
                requested=True,
            ),
        ),
        goal_destination_ids=(1,),
        goal_positions=(Position(row=0, column=4),),
    )
    mask = _mask(
        tuple(agent.agent_id for agent in observation.agents),
        destination_count=11,
    )

    destinations = HeuristicController().choose_destinations(observation, mask)

    assert destinations == [10, 0]


def test_heuristic_respects_the_action_mask() -> None:
    observation = WarehouseObservation(
        step=0,
        grid_rows=5,
        grid_columns=5,
        agents=(
            AgentObservation(
                agent_id="agv-0",
                simulator_id=1,
                role=AgentRole.AGV,
                position=Position(row=0, column=0),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=10,
                position=Position(row=0, column=1),
                occupied=True,
                requested=True,
            ),
        ),
        goal_destination_ids=(1,),
        goal_positions=(Position(row=0, column=4),),
    )
    values = np.ones((1, 11), dtype=np.bool_)
    values[0, 10] = False
    mask = ActionMask(agent_ids=("agv-0",), values=values)

    destinations = HeuristicController().choose_destinations(observation, mask)

    assert destinations == [0]
