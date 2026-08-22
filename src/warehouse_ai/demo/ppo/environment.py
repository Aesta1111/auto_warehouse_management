"""Gym adapter exposing a scalar masked directive action."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.contracts.directives import DirectiveContext, DirectiveVocabulary
from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.env import MaskAwareDiscrete, TarwareEnvWrapper


def _coerce_scalar_action(action: int | NDArray[np.int64]) -> int:
    if isinstance(action, (bool, np.bool_)):
        raise TypeError("directive actions must be integer IDs, not booleans")
    value = np.asarray(action)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
        raise TypeError("directive actions must be scalar integer IDs")
    return int(value.item())


class DirectiveGymEnv(gym.Env[NDArray[np.float32], Any]):
    """Translate one scalar directive to a validated project joint action."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        environment_id: str,
        vocabulary: DirectiveVocabulary,
        registry: DirectiveRegistry,
        episode_seeds: Sequence[int],
        episode_steps: int = 500,
    ) -> None:
        super().__init__()
        seeds = tuple(int(seed) for seed in episode_seeds)
        if not seeds or episode_steps <= 0:
            raise ValueError("episode seeds and a positive step count are required")
        if registry.vocabulary.digest() != vocabulary.digest():
            raise ValueError("registry vocabulary does not match the vocabulary")
        self.environment_id = environment_id
        self.vocabulary = vocabulary
        self.registry = registry
        self.episode_seeds = seeds
        self.episode_steps = int(episode_steps)
        self.wrapper = TarwareEnvWrapper(environment_id)
        self._episode_index = 0
        self._step_count = 0
        initial = self.wrapper.reset(seed=seeds[0])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial.encoded_shape,
            dtype=np.float32,
        )
        self.action_space = MaskAwareDiscrete(vocabulary.size, self.action_masks)

    @property
    def context(self) -> DirectiveContext:
        return DirectiveContext.capture(self.wrapper)

    def action_masks(self) -> NDArray[np.bool_]:
        return self.registry.action_mask(self.context).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        selected_seed = (
            self.episode_seeds[self._episode_index % len(self.episode_seeds)]
            if seed is None
            else int(seed)
        )
        self._episode_index += 1
        self._step_count = 0
        return self.wrapper.reset(seed=selected_seed).encode(), {}

    def step(
        self, action: int | NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._step_count >= self.episode_steps:
            raise RuntimeError("episode is truncated; reset before stepping")
        directive_id = _coerce_scalar_action(action)
        context = self.context
        concrete = self.registry.execute(directive_id, context, self.wrapper)
        observation, rewards, terminated, truncated, simulator_info = self.wrapper.step(
            concrete
        )
        self._step_count += 1
        simulator_terminated = bool(all(terminated))
        simulator_truncated = bool(all(truncated))
        info = dict(simulator_info)
        info.update(
            {
                "directive_id": directive_id,
                "directive_command": self.vocabulary.by_id(directive_id).command,
                "executed_destinations": [
                    int(item.destination_id) for item in concrete.actions
                ],
                "executor_status": "ok",
                "simulator_terminated": simulator_terminated,
                "simulator_truncated": simulator_truncated,
            }
        )
        return (
            observation.encode(),
            float(sum(rewards)),
            simulator_terminated,
            simulator_truncated or self._step_count >= self.episode_steps,
            info,
        )

    def render(self) -> Any:
        return self.wrapper.render(mode="human")

    def close(self) -> None:
        self.wrapper.close()


DirectiveSb3Env = DirectiveGymEnv
