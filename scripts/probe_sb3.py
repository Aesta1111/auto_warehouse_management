"""Run the MaskablePPO compatibility smoke test on centralized TA-RWARE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import sb3_contrib
import stable_baselines3
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_checker import check_env

from warehouse_ai.env import TarwareSb3Env

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = TarwareSb3Env(ENVIRONMENT_ID)
    try:
        check_env(env, warn=True)
        model = MaskablePPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            n_steps=64,
            batch_size=64,
            n_epochs=2,
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        repairs_during_training = len(env.repair_log)
        observation, _ = env.reset(seed=args.seed)
        masks = get_action_masks(env)
        action_before, _ = model.predict(
            observation, action_masks=masks, deterministic=True
        )

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "maskable_ppo"
            model.save(checkpoint)
            loaded = MaskablePPO.load(checkpoint, env=env)
            action_after, _ = loaded.predict(
                observation, action_masks=masks, deterministic=True
            )

        observation_shape = env.observation_space.shape
        if observation_shape is None:
            raise RuntimeError("SB3 adapter observation shape is undefined")
        action_nvec = cast(Any, env.action_space).nvec
        summary: dict[str, Any] = {
            "schema_version": 1,
            "environment": ENVIRONMENT_ID,
            "seed": args.seed,
            "timesteps": args.timesteps,
            "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__,
            "observation_shape": list(observation_shape),
            "action_nvec": action_nvec.tolist(),
            "mask_shape": list(env.action_masks().shape),
            "deterministic_action": np.asarray(action_before).tolist(),
            "save_reload_action_equal": bool(
                np.array_equal(action_before, action_after)
            ),
            "repairs_during_training": repairs_during_training,
        }
        payload = json.dumps(summary, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    finally:
        env.close()


if __name__ == "__main__":
    main()
