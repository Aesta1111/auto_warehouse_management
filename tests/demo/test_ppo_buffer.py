from __future__ import annotations

import numpy as np
import pytest
import torch

from warehouse_ai.demo.ppo.buffer import DirectiveRolloutBuffer


def _add(buffer: DirectiveRolloutBuffer, index: int) -> None:
    mask = np.asarray([index % 2 == 0, True, index % 2 == 1], dtype=np.bool_)
    buffer.add(
        observation=np.asarray([index, index + 0.5], dtype=np.float32),
        action_mask=mask,
        action=1,
        reward=float(index),
        terminated=index == 3,
        truncated=False,
        old_value=float(index) / 10,
        next_value=float(index + 1) / 10,
        old_log_probability=-float(index),
    )


def test_buffer_lifecycle_dtypes_masks_and_seeded_minibatch_coverage() -> None:
    buffer = DirectiveRolloutBuffer(4, (2,), 3)
    assert buffer.observations.dtype == np.float32
    assert buffer.action_masks.dtype == np.bool_
    assert buffer.actions.dtype == np.int64
    with pytest.raises(RuntimeError, match="finalized"):
        list(buffer.iter_minibatches(2, generator=torch.Generator()))

    for index in range(4):
        _add(buffer, index)
    with pytest.raises(RuntimeError, match="overflow"):
        _add(buffer, 4)
    buffer.finalize(gamma=0.99, gae_lambda=0.95)
    with pytest.raises(RuntimeError, match="finalized"):
        _add(buffer, 0)

    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)
    first = list(buffer.iter_minibatches(3, generator=first_generator))
    second = list(buffer.iter_minibatches(3, generator=second_generator))
    first_order = torch.cat([batch.observations[:, 0] for batch in first])
    second_order = torch.cat([batch.observations[:, 0] for batch in second])
    torch.testing.assert_close(first_order, second_order)
    assert sorted(first_order.tolist()) == [0.0, 1.0, 2.0, 3.0]
    for batch in first:
        assert batch.action_masks.dtype is torch.bool
        assert batch.actions.dtype is torch.int64
        assert torch.all(batch.action_masks.gather(1, batch.actions[:, None]))

    buffer.reset()
    assert buffer.size == 0
    assert not buffer.finalized


def test_buffer_rejects_partial_finalization_and_wrong_mask() -> None:
    buffer = DirectiveRolloutBuffer(2, (2,), 3)
    _add(buffer, 0)
    with pytest.raises(RuntimeError, match="partially"):
        buffer.finalize(gamma=0.99, gae_lambda=0.95)
    with pytest.raises(ValueError, match="allowed"):
        buffer.add(
            observation=np.zeros(2, dtype=np.float32),
            action_mask=np.asarray([True, False, False], dtype=np.bool_),
            action=1,
            reward=0.0,
            terminated=False,
            truncated=False,
            old_value=0.0,
            next_value=0.0,
            old_log_probability=0.0,
        )


def test_simultaneous_termination_and_time_limit_uses_termination_semantics() -> None:
    buffer = DirectiveRolloutBuffer(1, (2,), 2)
    buffer.add(
        observation=np.zeros(2, dtype=np.float32),
        action_mask=np.ones(2, dtype=np.bool_),
        action=0,
        reward=1.0,
        terminated=True,
        truncated=True,
        old_value=0.5,
        next_value=99.0,
        old_log_probability=0.0,
    )
    buffer.finalize(gamma=1.0, gae_lambda=1.0)
    assert buffer.bootstrap_masks[0] == 0.0
    assert buffer.trace_masks[0] == 0.0
    assert buffer.advantages[0] == 0.5
    assert buffer.returns[0] == 1.0
