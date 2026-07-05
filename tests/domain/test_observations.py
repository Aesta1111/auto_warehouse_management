from dataclasses import replace

import numpy as np
import pytest

from warehouse_ai.domain import (
    AgentObservation,
    AgentRole,
    Position,
    ShelfObservation,
    WarehouseObservation,
    WarehouseObservationEncoder,
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
                position=Position(1, 2),
                target=Position(3, 4),
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
                position=Position(2, 3),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=3,
                position=Position(1, 1),
                occupied=True,
                requested=False,
            ),
        ),
        goal_destination_ids=(1, 2),
    )


def test_encoder_has_stable_global_feature_order() -> None:
    state = observation()
    encoder = WarehouseObservationEncoder(state)

    encoded = encoder.encode(state)

    np.testing.assert_array_equal(
        encoded,
        np.asarray(
            [1, 1, 0, 1, 2, 3, 4, 2, 3, 0, 0, 1, 0],
            dtype=np.float32,
        ),
    )
    assert encoded.dtype == np.float32
    assert len(encoder.feature_names) == len(encoded)


def test_encoder_rejects_a_different_layout() -> None:
    state = observation()
    encoder = WarehouseObservationEncoder(state)
    changed = replace(state, shelves=())

    with pytest.raises(ValueError, match="layout"):
        encoder.encode(changed)


def test_picker_rejects_agv_only_fields() -> None:
    with pytest.raises(ValueError, match="AGV-only"):
        AgentObservation(
            agent_id="picker-0",
            simulator_id=1,
            role=AgentRole.PICKER,
            position=Position(0, 0),
            target=None,
            target_destination_id=None,
            busy=False,
            carrying_shelf=True,
        )
