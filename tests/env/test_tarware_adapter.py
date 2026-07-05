from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import tarware  # noqa: F401 - register environments

from warehouse_ai.domain import (
    AgentAction,
    AgentRole,
    JointAction,
    WarehouseObservationEncoder,
)
from warehouse_ai.env import (
    InvalidJointActionError,
    TarwareActionAdapter,
    TarwareObservationAdapter,
)

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def warehouse():
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    instance = env.unwrapped
    instance.reset(seed=0)
    try:
        yield instance
    finally:
        env.close()


def test_observation_matches_tarware_global_vector(warehouse) -> None:
    adapter = TarwareObservationAdapter()
    observation = adapter.observe(warehouse)
    encoded = WarehouseObservationEncoder(observation).encode(observation)

    # TA-RWARE's global mapper puts the observing agent first. Agent 1 is
    # already first in our canonical order, so its vector is directly comparable.
    mapper_type = tarware.observation_map["global"]
    mapper = mapper_type(
        warehouse.num_agvs,
        warehouse.num_pickers,
        warehouse.grid_size,
        len(warehouse.action_id_to_coords_map) - len(warehouse.goals),
        False,
    )
    mapper.extract_environment_info(warehouse)
    expected = mapper.observation(warehouse.agents[0])

    np.testing.assert_array_equal(encoded, expected)
    assert encoded.shape == (125,)
    assert [agent.role for agent in observation.agents] == [
        AgentRole.AGV,
        AgentRole.AGV,
        AgentRole.AGV,
        AgentRole.PICKER,
        AgentRole.PICKER,
    ]


def test_seeded_reset_produces_identical_records_and_features(warehouse) -> None:
    adapter = TarwareObservationAdapter()
    first = adapter.observe(warehouse)
    warehouse.reset(seed=0)
    second = adapter.observe(warehouse)

    assert first == second
    encoder = WarehouseObservationEncoder(first)
    np.testing.assert_array_equal(encoder.encode(first), encoder.encode(second))


def test_action_adapter_round_trips_no_op(warehouse) -> None:
    observation = TarwareObservationAdapter().observe(warehouse)
    adapter = TarwareActionAdapter()
    mask = adapter.action_mask(warehouse, observation)

    assert mask.values.shape == (5, 55)
    assert adapter.to_tarware(adapter.no_op(observation), observation, mask) == [
        0,
        0,
        0,
        0,
        0,
    ]


def test_seeded_no_op_steps_replay_through_adapters(warehouse) -> None:
    observation_adapter = TarwareObservationAdapter()
    action_adapter = TarwareActionAdapter()

    def run_steps() -> tuple:
        records = []
        for _ in range(5):
            observation = observation_adapter.observe(warehouse)
            mask = action_adapter.action_mask(warehouse, observation)
            raw_action = action_adapter.to_tarware(
                action_adapter.no_op(observation),
                observation,
                mask,
            )
            _, rewards, terminated, truncated, info = warehouse.step(raw_action)
            records.append(
                (
                    observation_adapter.observe(warehouse),
                    tuple(rewards),
                    tuple(terminated),
                    tuple(truncated),
                    tuple(sorted(info.items())),
                )
            )
        return tuple(records)

    first = run_steps()
    warehouse.reset(seed=0)
    second = run_steps()

    assert first == second


def test_busy_agent_only_accepts_no_op(warehouse) -> None:
    observation_adapter = TarwareObservationAdapter()
    action_adapter = TarwareActionAdapter()
    observation = observation_adapter.observe(warehouse)
    raw_mask = warehouse.compute_valid_action_masks()
    destination = int(np.flatnonzero(raw_mask[0, 1:])[0] + 1)
    warehouse.step([destination, 0, 0, 0, 0])
    observation = observation_adapter.observe(warehouse)
    mask = action_adapter.action_mask(warehouse, observation)

    assert observation.agents[0].busy
    assert mask.values[0].tolist() == [True] + [False] * 54
    action = JointAction(
        actions=(
            AgentAction("agv-0", destination),
            AgentAction("agv-1", 0),
            AgentAction("agv-2", 0),
            AgentAction("picker-0", 0),
            AgentAction("picker-1", 0),
        )
    )
    with pytest.raises(InvalidJointActionError, match="masked"):
        action_adapter.to_tarware(action, observation, mask)


def test_joint_action_rejects_same_role_destination_conflict(warehouse) -> None:
    observation = TarwareObservationAdapter().observe(warehouse)
    adapter = TarwareActionAdapter()
    mask = adapter.action_mask(warehouse, observation)
    common = np.flatnonzero(mask.values[0] & mask.values[1])
    destination = int(common[common != 0][0])
    action = JointAction(
        actions=(
            AgentAction("agv-0", destination),
            AgentAction("agv-1", destination),
            AgentAction("agv-2", 0),
            AgentAction("picker-0", 0),
            AgentAction("picker-1", 0),
        )
    )

    with pytest.raises(InvalidJointActionError, match="same role"):
        adapter.to_tarware(action, observation, mask)


def test_joint_action_rejects_wrong_agent_order(warehouse) -> None:
    observation = TarwareObservationAdapter().observe(warehouse)
    adapter = TarwareActionAdapter()
    mask = adapter.action_mask(warehouse, observation)
    action = JointAction(
        actions=(
            AgentAction("agv-1", 0),
            AgentAction("agv-0", 0),
            AgentAction("agv-2", 0),
            AgentAction("picker-0", 0),
            AgentAction("picker-1", 0),
        )
    )

    with pytest.raises(InvalidJointActionError, match="canonical order"):
        adapter.to_tarware(action, observation, mask)
