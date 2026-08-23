"""Run one episode with the project-owned heuristic controller.

Produces the same summary shape as ``scripts/smoke_tarware.py`` (which runs
TA-RWARE's own upstream FIFO/nearest-agent heuristic) so the two can be
compared directly for the same seed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from warehouse_ai.controllers import HeuristicController
from warehouse_ai.env import TarwareEnvWrapper

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open TA-RWARE's Pyglet viewer while the episode runs.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the JSON summary to, in addition to stdout.",
    )
    return parser.parse_args()


def sum_metric(infos: list[dict[str, Any]], name: str) -> int:
    return sum(int(info.get(name, 0)) for info in infos)


def main() -> None:
    args = parse_args()
    controller = HeuristicController()
    env = TarwareEnvWrapper(ENVIRONMENT_ID)

    try:
        env.reset(seed=args.seed)
        infos: list[dict[str, Any]] = []
        agent_returns = np.zeros(len(env.observation.agents), dtype=np.float64)
        total_return = 0.0
        step = 0
        while step < args.max_steps:
            if args.render:
                env.render(mode="human")
            destinations = controller.choose_destinations(env.observation, env.action_mask)
            action = env.decode(np.asarray(destinations, dtype=np.int64))
            _, rewards, terminated, truncated, info = env.step(action)
            agent_returns += np.asarray(rewards, dtype=np.float64)
            total_return += float(sum(rewards))
            infos.append(info)
            step += 1
            if all(terminated) or all(truncated):
                break
    finally:
        env.close()

    summary = {
        "environment": ENVIRONMENT_ID,
        "controller": "project_heuristic",
        "seed": args.seed,
        "steps": len(infos),
        "total_return": total_return,
        "agent_returns": agent_returns.tolist(),
        "metrics": {
            "shelf_deliveries": sum_metric(infos, "shelf_deliveries"),
            "clashes": sum_metric(infos, "clashes"),
            "stucks": sum_metric(infos, "stucks"),
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
