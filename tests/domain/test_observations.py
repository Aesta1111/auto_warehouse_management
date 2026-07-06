from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from warehouse_ai.domain import (
    AgentObservation,
    AgentRole,
    Position,
    ShelfObservation,
    WarehouseObservation,
)


def observation() -> WarehouseObservation:
    return WarehouseObservation(
        step=0,
        grid_rows=5,
        grid_columns=6,
        agents=(
            AgentObservation(
                agent_id="agv-0",
                simulator_id=1,
                role=AgentRole.AGV,
                position=Position(row=1, column=2),
                target=Position(row=3, column=4),
                target_destination_id=2,
                busy=True,
                carrying_shelf=True,
                carrying_requested_shelf=True,
                toggling_load=False,
            ),
            AgentObservation(
                agent_id="picker-0",
                simulator_id=2,
                role=AgentRole.PICKER,
                position=Position(row=2, column=3),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=3,
                position=Position(row=1, column=1),
                occupied=True,
                requested=False,
            ),
        ),
        goal_destination_ids=(1, 2),
    )


def test_observation_encodes_stable_global_feature_order() -> None:
    state = observation()

    encoded = state.encode()

    np.testing.assert_array_equal(
        encoded,
        np.asarray(
            [1, 1, 0, 1, 2, 3, 4, 2, 3, 0, 0, 1, 0],
            dtype=np.float32,
        ),
    )
    assert encoded.dtype == np.float32
    assert state.encoded_shape == encoded.shape
    assert len(state.feature_names) == len(encoded)


def test_observation_is_serializable_and_collections_are_immutable() -> None:
    state = observation()

    dumped = state.model_dump(mode="json")

    assert dumped["agents"][0]["role"] == "agv"
    with pytest.raises(ValidationError, match="frozen"):
        state.step = 1
    with pytest.raises(TypeError):
        cast(list[AgentObservation], state.agents)[0] = state.agents[0]


def test_ordered_list_inputs_are_stored_as_tuples() -> None:
    state = observation()

    rebuilt = WarehouseObservation(
        **{
            **state.model_dump(),
            "agents": list(state.agents),
            "shelves": list(state.shelves),
            "goal_destination_ids": list(state.goal_destination_ids),
        }
    )

    assert isinstance(rebuilt.agents, tuple)
    assert isinstance(rebuilt.shelves, tuple)
    assert isinstance(rebuilt.goal_destination_ids, tuple)


def test_picker_rejects_agv_only_fields() -> None:
    with pytest.raises(ValidationError, match="AGV-only"):
        AgentObservation(
            agent_id="picker-0",
            simulator_id=1,
            role=AgentRole.PICKER,
            position=Position(row=0, column=0),
            target=None,
            target_destination_id=None,
            busy=False,
            carrying_shelf=True,
        )


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        Position.model_validate({"row": 0, "column": 0, "depth": 0})
