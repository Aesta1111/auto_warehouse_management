import numpy as np
import pytest

from warehouse_ai.domain import ActionMask, AgentAction


def test_action_mask_is_boolean_and_read_only() -> None:
    mask = ActionMask(
        agent_ids=("agv-0", "picker-0"),
        values=np.asarray([[1, 1, 0], [1, 0, 1]]),
    )

    assert mask.values.dtype == np.bool_
    assert mask.destination_count == 3
    with pytest.raises(ValueError):
        mask.values[0, 0] = False


def test_agent_action_rejects_negative_destination() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AgentAction(agent_id="agv-0", destination_id=-1)
