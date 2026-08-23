from __future__ import annotations

import math

import torch

from warehouse_ai.demo.ppo.buffer import PpoBatch
from warehouse_ai.demo.ppo.math import compute_gae, compute_ppo_loss


def test_gae_distinguishes_termination_truncation_and_rollout_boundary() -> None:
    advantages, returns = compute_gae(
        rewards=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        values=torch.tensor([0.5, 1.0, 1.5, 2.0]),
        next_values=torch.tensor([1.0, 99.0, 2.0, 5.0]),
        bootstrap_masks=torch.tensor([1.0, 0.0, 1.0, 1.0]),
        trace_masks=torch.tensor([1.0, 0.0, 0.0, 1.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )

    # Index 1 terminates: no bootstrap and no trace. Index 2 truncates: it
    # bootstraps from value 2 but cannot inherit index 3's advantage. Index 3
    # is an unfinished rollout boundary and therefore bootstraps normally.
    torch.testing.assert_close(advantages, torch.tensor([2.5, 1.0, 3.5, 7.0]))
    torch.testing.assert_close(returns, torch.tensor([3.0, 2.0, 5.0, 9.0]))


def _batch(advantages: torch.Tensor) -> PpoBatch:
    size = advantages.shape[0]
    return PpoBatch(
        observations=torch.zeros((size, 2)),
        action_masks=torch.ones((size, 2), dtype=torch.bool),
        actions=torch.zeros(size, dtype=torch.int64),
        old_log_probabilities=torch.full((size,), math.log(0.5)),
        old_values=torch.zeros(size),
        advantages=advantages,
        returns=torch.tensor([2.0, 0.0]),
    )


def test_ppo_loss_matches_clipping_value_entropy_kl_and_clip_fraction() -> None:
    batch = _batch(torch.tensor([1.0, -1.0]))
    terms = compute_ppo_loss(
        batch,
        new_log_probabilities=torch.log(torch.tensor([0.75, 0.25])),
        new_values=torch.tensor([1.0, 1.0]),
        entropy=torch.tensor([0.5, 0.5]),
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        normalize_advantage=False,
    )

    torch.testing.assert_close(terms.policy, torch.tensor(-0.2))
    torch.testing.assert_close(terms.value, torch.tensor(1.0))
    torch.testing.assert_close(terms.entropy, torch.tensor(0.5))
    torch.testing.assert_close(terms.total, torch.tensor(0.295))
    ratios = torch.tensor([1.5, 0.5])
    expected_kl = torch.mean((ratios - 1.0) - torch.log(ratios))
    torch.testing.assert_close(terms.approximate_kl, expected_kl)
    torch.testing.assert_close(terms.clip_fraction, torch.tensor(1.0))


def test_ppo_loss_normalizes_advantages_per_minibatch() -> None:
    batch = _batch(torch.tensor([1.0, 3.0]))
    terms = compute_ppo_loss(
        batch,
        new_log_probabilities=batch.old_log_probabilities.clone(),
        new_values=torch.tensor([2.0, 0.0]),
        entropy=torch.zeros(2),
        clip_range=0.2,
        entropy_coefficient=0.0,
        value_coefficient=0.0,
        normalize_advantage=True,
    )
    torch.testing.assert_close(terms.policy, torch.tensor(0.0), atol=1e-7, rtol=0)
