"""Single-environment Gymnasium adapter for the SB3 compatibility spike."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from warehouse_ai.domain import ActionRepair
from warehouse_ai.env.tarware import TarwareEnvWrapper


class MaskAwareDiscrete(gym.spaces.Discrete):
    """A ``Discrete`` space whose samples respect a live action mask.

    Gymnasium's explicit ``Discrete.sample(mask=...)`` contract uses an
    ``int8`` vector, while an environment's ``action_masks()`` method usually
    returns booleans.  This space accepts both representations, validates the
    mask, and samples only from currently valid IDs.  An all-zero mask is an
    invariant violation for the directive environment because the wait
    directive must always remain available.
    """

    def __init__(
        self,
        n: int,
        mask_provider: Callable[[], NDArray[np.bool_]],
    ) -> None:
        super().__init__(n)
        self._mask_provider = mask_provider

    def sample(self, mask: Any = None) -> np.int64:
        raw_mask = self._mask_provider() if mask is None else mask
        mask_array = np.asarray(raw_mask)
        if mask_array.shape != (self.n,):
            raise ValueError(
                f"discrete action mask shape {mask_array.shape} does not match {(self.n,)}"
            )
        if not np.issubdtype(mask_array.dtype, np.bool_) and not np.issubdtype(
            mask_array.dtype, np.integer
        ):
            raise ValueError("discrete action mask must contain booleans or integers")
        if not np.all(np.logical_or(mask_array == 0, mask_array == 1)):
            raise ValueError("discrete action mask values must be 0 or 1")

        valid_actions = np.flatnonzero(mask_array.astype(np.bool_, copy=False))
        if len(valid_actions) == 0:
            raise ValueError("discrete action mask must allow at least one action")
        return np.int64(self.start + int(self.np_random.choice(valid_actions)))


class MaskAwareMultiDiscrete(gym.spaces.MultiDiscrete):
    """A ``MultiDiscrete`` space whose random samples respect live masks.

    Stable-Baselines3's environment checker samples the action space without
    asking for masks. Respecting the live per-component mask here lets that
    checker exercise the real adapter contract safely.
    """

    def __init__(
        self,
        nvec: NDArray[np.int64],
        mask_provider: Callable[[], NDArray[np.bool_]],
    ) -> None:
        super().__init__(nvec, dtype=np.int64)
        self._mask_provider = mask_provider

    def sample(self, mask: Any = None) -> NDArray[np.int64]:
        if mask is not None:
            return np.asarray(super().sample(mask=mask), dtype=np.int64)
        live_mask = self._mask_provider()
        choices = [int(self.np_random.choice(np.flatnonzero(row))) for row in live_mask]
        return np.asarray(choices, dtype=np.int64)


class TarwareSb3Env(gym.Env[NDArray[np.float32], Any]):
    """Expose centralized TA-RWARE state as one masked joint-action MDP."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, environment_id: str) -> None:
        super().__init__()
        self.wrapper = TarwareEnvWrapper(environment_id)
        initial = self.wrapper.reset(seed=0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=initial.encoded_shape,
            dtype=np.float32,
        )
        destination_count = self.wrapper.action_mask.destination_count
        nvec = np.full(len(initial.agents), destination_count, dtype=np.int64)
        self.action_space = MaskAwareMultiDiscrete(nvec, self._mask_matrix)
        self.repair_log: list[ActionRepair] = []

    def action_masks(self) -> NDArray[np.bool_]:
        """Return SB3-Contrib's flattened per-component mask contract."""
        return self._mask_matrix().reshape(-1).copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        observation = self.wrapper.reset(seed=0 if seed is None else seed)
        self.repair_log.clear()
        return observation.encode(), {}

    def step(
        self, action: NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        projected = self.wrapper.project(action)
        observation, rewards, terminated, truncated, info = self.wrapper.step(
            projected.action
        )
        self.repair_log.extend(projected.repairs)
        step_info = dict(info)
        step_info["action_repairs"] = [
            repair.model_dump(mode="json") for repair in projected.repairs
        ]
        return (
            observation.encode(),
            float(sum(rewards)),
            bool(all(terminated)),
            bool(all(truncated)),
            step_info,
        )

    def render(self) -> Any:
        return self.wrapper.render(mode="human")

    def close(self) -> None:
        self.wrapper.close()

    def _mask_matrix(self) -> NDArray[np.bool_]:
        return self.wrapper.action_mask.values.copy()
