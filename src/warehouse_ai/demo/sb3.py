"""MaskablePPO facade for the isolated directive prototype."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.demo.models import DirectiveContext, DirectiveVocabulary
from warehouse_ai.env import MaskAwareDiscrete, TarwareEnvWrapper


def _coerce_scalar_action(action: int | NDArray[np.int64]) -> int:
    """Accept only one integer directive ID, never a batch or float."""
    if isinstance(action, (bool, np.bool_)):
        raise TypeError("directive actions must be integer IDs, not booleans")
    value = np.asarray(action)
    if value.ndim != 0 or not np.issubdtype(value.dtype, np.integer):
        raise TypeError("directive actions must be scalar integer IDs")
    return int(value.item())


class DirectiveSb3Env(gym.Env[NDArray[np.float32], Any]):
    """Expose one TA-RWARE wrapper as a masked scalar directive MDP.

    The environment owns exactly one :class:`TarwareEnvWrapper`.  Directive
    behaviors perform the only translation from the scalar PPO action to a
    validated project-owned ``JointAction``; this facade adds no simulator
    semantics of its own.
    """

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
        if not seeds:
            raise ValueError("episode_seeds must contain at least one seed")
        if episode_steps <= 0:
            raise ValueError("episode_steps must be positive")
        if registry.vocabulary.digest() != vocabulary.digest():
            raise ValueError(
                "registry vocabulary does not match the supplied vocabulary"
            )

        self.environment_id = environment_id
        self.vocabulary = vocabulary
        self.registry = registry
        self.episode_seeds = seeds
        self.episode_steps = int(episode_steps)
        self.wrapper = TarwareEnvWrapper(environment_id)
        self._episode_index = 0
        self._step_count = 0

        initial = self.wrapper.reset(seed=self.episode_seeds[0])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial.encoded_shape,
            dtype=np.float32,
        )
        self.action_space = MaskAwareDiscrete(vocabulary.size, self.action_masks)

    @property
    def context(self) -> DirectiveContext:
        """Return the current immutable controller-visible request snapshot."""
        return DirectiveContext.capture(self.wrapper)

    def action_masks(self) -> NDArray[np.bool_]:
        """Return one valid/inapplicable boolean entry per directive ID."""
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
        observation = self.wrapper.reset(seed=selected_seed)
        return observation.encode(), {}

    def step(
        self, action: int | NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Execute one directive and return the unmodified summed reward."""
        if self._step_count >= self.episode_steps:
            raise RuntimeError("episode is truncated; call reset before stepping again")

        directive_id = _coerce_scalar_action(action)
        context = self.context
        action_mask = self.registry.action_mask(context)
        if directive_id < 0 or directive_id >= self.vocabulary.size:
            # Let the registry provide the stable unknown-directive exception.
            self.registry.execute(directive_id, context, self.wrapper)
            raise AssertionError("unknown directive execution unexpectedly returned")
        if not bool(action_mask[directive_id]):
            # This is deliberately rejection-only.  Training must never learn
            # from a silently repaired transition.
            self.registry.execute(directive_id, context, self.wrapper)
            raise AssertionError(
                "inapplicable directive execution unexpectedly returned"
            )

        concrete_action = self.registry.execute(directive_id, context, self.wrapper)
        observation, rewards, terminated, truncated, simulator_info = self.wrapper.step(
            concrete_action
        )
        self._step_count += 1

        simulator_terminated = bool(all(terminated))
        simulator_truncated = bool(all(truncated))
        episode_truncated = (
            simulator_truncated or self._step_count >= self.episode_steps
        )
        info = dict(simulator_info)
        info.update(
            {
                "directive_id": directive_id,
                "directive_command": self.vocabulary.by_id(directive_id).command,
                "executed_destinations": [
                    int(item.destination_id) for item in concrete_action.actions
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
            episode_truncated,
            info,
        )

    def render(self) -> Any:
        """Render through the existing project-owned TA-RWARE wrapper."""
        return self.wrapper.render(mode="human")

    def close(self) -> None:
        """Close the one owned simulator wrapper."""
        self.wrapper.close()
