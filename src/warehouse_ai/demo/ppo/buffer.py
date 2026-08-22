"""Fixed-size rollout storage for repository-owned directive PPO."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from warehouse_ai.demo.ppo.math import compute_gae


BUFFER_SCHEMA_VERSION = "directive-rollout-buffer-v1"


@dataclass(frozen=True)
class PpoBatch:
    observations: Tensor
    action_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor

    def to(self, device: torch.device | str) -> Self:
        return type(self)(
            observations=self.observations.to(device),
            action_masks=self.action_masks.to(device),
            actions=self.actions.to(device),
            old_log_probabilities=self.old_log_probabilities.to(device),
            old_values=self.old_values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )


class DirectiveRolloutBuffer:
    """Preallocated storage with an explicit collecting/finalized lifecycle."""

    def __init__(
        self, capacity: int, observation_shape: tuple[int, ...], directive_count: int
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if not observation_shape or any(size <= 0 for size in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if directive_count <= 0:
            raise ValueError("directive_count must be positive")
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.directive_count = int(directive_count)
        self.observations = np.empty((capacity, *observation_shape), dtype=np.float32)
        self.action_masks = np.empty((capacity, directive_count), dtype=np.bool_)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.old_values = np.empty(capacity, dtype=np.float32)
        self.next_values = np.empty(capacity, dtype=np.float32)
        self.old_log_probabilities = np.empty(capacity, dtype=np.float32)
        self.bootstrap_masks = np.empty(capacity, dtype=np.float32)
        self.trace_masks = np.empty(capacity, dtype=np.float32)
        self.advantages = np.empty(capacity, dtype=np.float32)
        self.returns = np.empty(capacity, dtype=np.float32)
        self._position = 0
        self._finalized = False

    @property
    def size(self) -> int:
        return self._position

    @property
    def finalized(self) -> bool:
        return self._finalized

    def add(
        self,
        *,
        observation: NDArray[np.float32],
        action_mask: NDArray[np.bool_],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        old_value: float,
        next_value: float,
        old_log_probability: float,
    ) -> None:
        if self._finalized or self._position >= self.capacity:
            raise RuntimeError("rollout buffer is finalized or full")
        observation_array = np.asarray(observation, dtype=np.float32)
        mask_array = np.asarray(action_mask, dtype=np.bool_)
        if observation_array.shape != self.observation_shape:
            raise ValueError("observation shape does not match the rollout buffer")
        if mask_array.shape != (self.directive_count,) or not bool(np.any(mask_array)):
            raise ValueError("invalid action mask shape or contents")
        if action < 0 or action >= self.directive_count or not bool(mask_array[action]):
            raise ValueError("stored action must be allowed by its exact action mask")
        index = self._position
        self.observations[index] = observation_array
        self.action_masks[index] = mask_array
        self.actions[index] = action
        self.rewards[index] = reward
        self.old_values[index] = old_value
        self.next_values[index] = next_value
        self.old_log_probabilities[index] = old_log_probability
        self.bootstrap_masks[index] = 0.0 if terminated else 1.0
        self.trace_masks[index] = 0.0 if terminated or truncated else 1.0
        self._position += 1

    def finalize(self, *, gamma: float, gae_lambda: float) -> None:
        if self._finalized or self._position != self.capacity:
            raise RuntimeError("buffer must be full and not already finalized")
        advantages, returns = compute_gae(
            torch.from_numpy(self.rewards),
            torch.from_numpy(self.old_values),
            torch.from_numpy(self.next_values),
            torch.from_numpy(self.bootstrap_masks),
            torch.from_numpy(self.trace_masks),
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.advantages[:] = advantages.numpy()
        self.returns[:] = returns.numpy()
        self._finalized = True

    def iter_minibatches(
        self, batch_size: int, *, generator: torch.Generator
    ) -> Iterator[PpoBatch]:
        if not self._finalized:
            raise RuntimeError("rollout buffer must be finalized before batching")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = torch.randperm(self.capacity, generator=generator)
        for start in range(0, self.capacity, batch_size):
            selected = indices[start : start + batch_size].numpy()
            yield PpoBatch(
                observations=torch.from_numpy(self.observations[selected].copy()),
                action_masks=torch.from_numpy(self.action_masks[selected].copy()),
                actions=torch.from_numpy(self.actions[selected].copy()),
                old_log_probabilities=torch.from_numpy(
                    self.old_log_probabilities[selected].copy()
                ),
                old_values=torch.from_numpy(self.old_values[selected].copy()),
                advantages=torch.from_numpy(self.advantages[selected].copy()),
                returns=torch.from_numpy(self.returns[selected].copy()),
            )

    def reset(self) -> None:
        self._position = 0
        self._finalized = False
