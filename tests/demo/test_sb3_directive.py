from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.directives import (
    SemanticsError,
    SemanticsFailure,
    build_registry,
)
from warehouse_ai.demo.ppo.environment import GymEnv
from warehouse_ai.env import MaskAwareDiscrete


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def catalog() -> DirectiveCatalog:
    return DirectiveCatalog.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def directive_env(catalog: DirectiveCatalog) -> Iterator[GymEnv]:
    environment = GymEnv(
        ENVIRONMENT_ID,
        catalog,
        build_registry(catalog),
        (8100, 8101),
        episode_steps=16,
    )
    try:
        yield environment
    finally:
        environment.close()


def test_mask_aware_discrete_samples_only_valid_ids() -> None:
    space = MaskAwareDiscrete(
        4, lambda: np.asarray([True, False, True, False], dtype=np.bool_)
    )
    assert all(int(space.sample()) in (0, 2) for _ in range(20))
    assert int(space.sample(mask=np.asarray([0, 1, 0, 0], dtype=np.int8))) == 1

    with pytest.raises(ValueError, match="at least one"):
        space.sample(mask=np.zeros(4, dtype=np.int8))
    with pytest.raises(ValueError, match="shape"):
        space.sample(mask=np.ones(3, dtype=np.int8))


def test_directive_facade_passes_checker_and_exposes_scalar_mask(
    directive_env: GymEnv,
) -> None:
    observation, info = directive_env.reset(seed=8100)

    assert observation.shape == (125,)
    assert observation.dtype == np.float32
    assert info == {}
    assert directive_env.action_masks().shape == (4,)
    check_env(directive_env, warn=True)


def test_directive_facade_rejects_inapplicable_and_batched_actions(
    directive_env: GymEnv,
) -> None:
    directive_env.reset(seed=8100)
    with pytest.raises(SemanticsError) as error:
        directive_env.step(2)
    assert error.value.code is SemanticsFailure.INAPPLICABLE
    with pytest.raises(TypeError, match="scalar"):
        directive_env.step(np.asarray([0], dtype=np.int64))
