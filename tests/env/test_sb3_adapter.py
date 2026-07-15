from __future__ import annotations

import numpy as np
from stable_baselines3.common.env_checker import check_env

from warehouse_ai.env import TarwareSb3Env

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def test_sb3_adapter_spaces_masks_and_checker() -> None:
    env = TarwareSb3Env(ENVIRONMENT_ID)
    try:
        observation, info = env.reset(seed=0)

        assert observation.shape == (125,)
        assert observation.dtype == np.float32
        assert info == {}
        assert env.action_masks().shape == (275,)
        check_env(env, warn=True)
    finally:
        env.close()
