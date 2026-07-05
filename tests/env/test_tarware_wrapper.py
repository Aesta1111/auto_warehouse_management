from __future__ import annotations

import numpy as np
import pytest
import tarware

from warehouse_ai.domain import AgentAction, AgentRole, JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def wrapped_env():
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


def test_observation_encode_matches_tarware_global_vector(wrapped_env) -> None:
    observation = wrapped_env.observation
    encoded = observation.encode()
    warehouse = wrapped_env.unwrapped

    # TA-RWARE global observations put the observer first. Agent 1 is already
    # first in canonical order, making its vector directly comparable.
    mapper_type = tarware.observation_map["global"]
    mapper = mapper_type(
        warehouse.num_agvs,
        warehouse.num_pickers,
        warehouse.grid_size,
        len(warehouse.action_id_to_coords_map) - len(warehouse.goals),
        False,
    )
    mapper.extract_environment_info(warehouse)

    np.testing.assert_array_equal(encoded, mapper.observation(warehouse.agents[0]))
    assert observation.encoded_shape == (125,)
    assert [agent.role for agent in observation.agents] == [
        AgentRole.AGV,
        AgentRole.AGV,
        AgentRole.AGV,
        AgentRole.PICKER,
        AgentRole.PICKER,
    ]


def test_seeded_reset_reproduces_records_and_features(wrapped_env) -> None:
    first = wrapped_env.observation
    second = wrapped_env.reset(seed=0)

    assert first == second
    np.testing.assert_array_equal(first.encode(), second.encode())


def test_decode_and_step_use_project_owned_types(wrapped_env) -> None:
    action = wrapped_env.decode(np.zeros(5, dtype=np.int64))

    observation, rewards, terminated, truncated, info = wrapped_env.step(action)

    assert isinstance(action, JointAction)
    assert observation.step == 1
    assert len(rewards) == len(terminated) == len(truncated) == 5
    assert "shelf_deliveries" in info


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (np.zeros((1, 5), dtype=np.int64), "shape"),
        (np.zeros(4, dtype=np.int64), "shape"),
        (np.zeros(5, dtype=np.float32), "integer"),
        (np.asarray([-1, 0, 0, 0, 0]), "negative"),
        (np.asarray([55, 0, 0, 0, 0]), "out of range"),
    ],
)
def test_decode_rejects_malformed_model_output(wrapped_env, output, message) -> None:
    with pytest.raises(InvalidJointActionError, match=message):
        wrapped_env.decode(output)


def test_busy_agent_only_accepts_no_op(wrapped_env) -> None:
    raw_mask = wrapped_env.unwrapped.compute_valid_action_masks()
    destination = int(np.flatnonzero(raw_mask[0, 1:])[0] + 1)
    wrapped_env.step(
        wrapped_env.decode(
            np.asarray([destination, 0, 0, 0, 0], dtype=np.int64)
        )
    )

    assert wrapped_env.observation.agents[0].busy
    assert wrapped_env.action_mask.values[0].tolist() == [True] + [False] * 54
    with pytest.raises(InvalidJointActionError, match="masked"):
        wrapped_env.decode(np.asarray([destination, 0, 0, 0, 0], dtype=np.int64))


def test_decode_rejects_same_role_destination_conflict(wrapped_env) -> None:
    mask = wrapped_env.action_mask.values
    common = np.flatnonzero(mask[0] & mask[1])
    destination = int(common[common != 0][0])

    with pytest.raises(InvalidJointActionError, match="same role"):
        wrapped_env.decode(
            np.asarray([destination, destination, 0, 0, 0], dtype=np.int64)
        )


def test_step_rejects_wrong_agent_order(wrapped_env) -> None:
    action = JointAction(
        actions=(
            AgentAction(agent_id="agv-1", destination_id=0),
            AgentAction(agent_id="agv-0", destination_id=0),
            AgentAction(agent_id="agv-2", destination_id=0),
            AgentAction(agent_id="picker-0", destination_id=0),
            AgentAction(agent_id="picker-1", destination_id=0),
        )
    )

    with pytest.raises(InvalidJointActionError, match="canonical order"):
        wrapped_env.step(action)


def test_seeded_no_op_steps_replay_through_wrapper(wrapped_env) -> None:
    def run_steps() -> tuple:
        records = []
        for _ in range(5):
            observation, rewards, terminated, truncated, info = wrapped_env.step(
                wrapped_env.no_op()
            )
            records.append(
                (
                    observation,
                    tuple(rewards),
                    tuple(terminated),
                    tuple(truncated),
                    tuple(sorted(info.items())),
                )
            )
        return tuple(records)

    first = run_steps()
    wrapped_env.reset(seed=0)
    second = run_steps()

    assert first == second
