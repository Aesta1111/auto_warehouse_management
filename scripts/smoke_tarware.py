"""Run one pinned TA-RWARE heuristic episode and print a JSON summary."""

from __future__ import annotations

import argparse
import json
from typing import Any

import gymnasium as gym
import tarware  # noqa: F401 - importing registers the TA-RWARE environments
from tarware.heuristic import heuristic_episode

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open TA-RWARE's Pyglet viewer while the episode runs.",
    )
    return parser.parse_args()


def sum_metric(infos: list[dict[str, Any]], name: str) -> int:
    return sum(int(info.get(name, 0)) for info in infos)


def main() -> None:
    args = parse_args()
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    warehouse = env.unwrapped

    try:
        observations = warehouse.reset(seed=args.seed)
        observation_shapes = [list(observation.shape) for observation in observations]
        action_sizes = [int(space.n) for space in warehouse.action_space]
        infos, total_return, agent_returns = heuristic_episode(
            warehouse,
            render=args.render,
            seed=args.seed,
        )
    finally:
        env.close()

    summary = {
        "environment": ENVIRONMENT_ID,
        "seed": args.seed,
        "steps": len(infos),
        "observation_shapes": observation_shapes,
        "action_sizes": action_sizes,
        "total_return": float(total_return),
        "agent_returns": agent_returns.tolist(),
        "metrics": {
            "shelf_deliveries": sum_metric(infos, "shelf_deliveries"),
            "clashes": sum_metric(infos, "clashes"),
            "stucks": sum_metric(infos, "stucks"),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
