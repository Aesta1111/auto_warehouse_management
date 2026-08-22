"""Stateful rollout collection without optimizer responsibilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from warehouse_ai.demo.ppo.buffer import DirectiveRolloutBuffer
from warehouse_ai.demo.ppo.environment import DirectiveGymEnv
from warehouse_ai.demo.ppo.model import DirectivePpoModel


def _raw_module(module: Any) -> DirectivePpoModel:
    return getattr(module, "module", module)


def _value_for_observation(
    module: Any,
    observation: np.ndarray[Any, np.dtype[np.float32]],
    action_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> float:
    model = _raw_module(module)
    device = next(model.parameters()).device
    observations = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    masks = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    with torch.inference_mode():
        return float(model(observations, masks).values.item())


class DirectiveRolloutCollector:
    """Collect one exact on-policy rollout into caller-owned storage."""

    def __init__(self) -> None:
        self._observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._environment_identity: int | None = None

    def collect(
        self,
        module: Any,
        environment: DirectiveGymEnv,
        buffer: DirectiveRolloutBuffer,
        *,
        rollout_steps: int,
        generator: torch.Generator,
    ) -> None:
        if rollout_steps != buffer.capacity:
            raise ValueError("rollout_steps must equal buffer capacity")
        if buffer.size != 0 or buffer.finalized:
            raise RuntimeError("collector requires a fresh rollout buffer")
        if self._environment_identity not in (None, id(environment)):
            raise RuntimeError("one collector cannot switch training environments")
        self._environment_identity = id(environment)
        if self._observation is None:
            self._observation, _ = environment.reset(seed=environment.episode_seeds[0])

        model = _raw_module(module)
        model.eval()
        device = next(model.parameters()).device
        for _ in range(rollout_steps):
            observation = self._observation
            action_mask = environment.action_masks()
            observations = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            masks = torch.as_tensor(
                action_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            with torch.inference_mode():
                sample = model.act(
                    observations,
                    masks,
                    deterministic=False,
                    generator=generator,
                )
            action = int(sample.actions.item())
            next_observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            next_value = (
                0.0
                if terminated
                else _value_for_observation(
                    model, next_observation, environment.action_masks()
                )
            )
            buffer.add(
                observation=observation,
                action_mask=action_mask,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                old_value=float(sample.values.item()),
                next_value=next_value,
                old_log_probability=float(sample.log_probabilities.item()),
            )
            if terminated or truncated:
                next_observation, _ = environment.reset()
            self._observation = next_observation
