"""Pure tensor arithmetic for repository-owned PPO training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from warehouse_ai.demo.ppo_buffer import PpoBatch
    from warehouse_ai.demo.ppo_model import PpoLossTerms


_ADVANTAGE_EPSILON = 1e-8


def _require_vector(name: str, value: Tensor, length: int | None = None) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional tensor")
    if length is not None and value.shape[0] != length:
        raise ValueError(f"{name} length does not match rewards")


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    bootstrap_masks: Tensor,
    trace_masks: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """Compute GAE with distinct bootstrap and recursive-trace boundaries.

    ``bootstrap_masks`` is zero only for true termination. ``trace_masks`` is
    zero for termination and truncation, preventing a reset state's advantage
    from leaking backward while still allowing truncations to bootstrap from
    their returned terminal observations.
    """
    _require_vector("rewards", rewards)
    length = rewards.shape[0]
    for name, value in (
        ("values", values),
        ("next_values", next_values),
        ("bootstrap_masks", bootstrap_masks),
        ("trace_masks", trace_masks),
    ):
        _require_vector(name, value, length)
    if length == 0:
        raise ValueError("GAE requires at least one transition")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must be in [0, 1]")
    for name, mask in (
        ("bootstrap_masks", bootstrap_masks),
        ("trace_masks", trace_masks),
    ):
        if not bool(torch.all((mask == 0) | (mask == 1))):
            raise ValueError(f"{name} must contain only zero or one")

    advantages = torch.empty_like(rewards)
    following_advantage = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(length - 1, -1, -1):
        delta = (
            rewards[index]
            + gamma * next_values[index] * bootstrap_masks[index]
            - values[index]
        )
        following_advantage = (
            delta + gamma * gae_lambda * trace_masks[index] * following_advantage
        )
        advantages[index] = following_advantage
    return advantages, advantages + values


def compute_ppo_loss(
    batch: PpoBatch,
    new_log_probabilities: Tensor,
    new_values: Tensor,
    entropy: Tensor,
    *,
    clip_range: float,
    entropy_coefficient: float,
    value_coefficient: float,
    normalize_advantage: bool,
) -> PpoLossTerms:
    """Return clipped-surrogate PPO loss terms without side effects."""
    from warehouse_ai.demo.ppo_model import PpoLossTerms

    batch_size = batch.actions.shape[0]
    for name, value in (
        ("new_log_probabilities", new_log_probabilities),
        ("new_values", new_values),
        ("entropy", entropy),
    ):
        _require_vector(name, value, batch_size)
    if clip_range < 0.0:
        raise ValueError("clip_range must be non-negative")
    if entropy_coefficient < 0.0 or value_coefficient < 0.0:
        raise ValueError("loss coefficients must be non-negative")

    advantages = batch.advantages
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + _ADVANTAGE_EPSILON
        )

    log_ratio = new_log_probabilities - batch.old_log_probabilities
    ratio = torch.exp(log_ratio)
    unclipped = advantages * ratio
    clipped = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = torch.mean((batch.returns - new_values) ** 2)
    mean_entropy = entropy.mean()
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )

    with torch.no_grad():
        approximate_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
    return PpoLossTerms(
        total=total,
        policy=policy_loss,
        value=value_loss,
        entropy=mean_entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
    )
