"""Measure TA-RWARE determinism, snapshots, and replay for the tiny layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import tarware  # noqa: F401 - registers TA-RWARE environments
from tarware.heuristic import heuristic_episode

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
CHECKPOINTS = (100, 250, 400)
SUFFIX_LENGTH = 50


def canonical(value: Any) -> Any:
    """Convert simulator values to stable JSON-compatible structures."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def canonical_step(action: Any, result: Any) -> dict[str, Any]:
    observations, rewards, terminated, truncated, info = result
    return {
        "action": canonical(action),
        "observations": canonical(observations),
        "rewards": canonical(rewards),
        "terminated": canonical(terminated),
        "truncated": canonical(truncated),
        "info": canonical(info),
    }


class RecordingWarehouse:
    """Transparent proxy that records actions and exact step outputs."""

    def __init__(self, warehouse: Any) -> None:
        self._warehouse = warehouse
        self.actions: list[list[int]] = []
        self.steps: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._warehouse, name)

    def step(self, action: Any) -> Any:
        result = self._warehouse.step(action)
        action_list = [int(value) for value in action]
        self.actions.append(action_list)
        self.steps.append(canonical_step(action_list, result))
        return result


def record_heuristic(
    seed: int, *, render: bool = False
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    recorder = RecordingWarehouse(env.unwrapped)
    try:
        heuristic_episode(recorder, render=render, seed=seed)
        return recorder.actions, recorder.steps
    finally:
        env.close()


def trace_hash(steps: list[dict[str, Any]]) -> str:
    payload = json.dumps(steps, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def replay_suffix(
    seed: int, actions: list[list[int]], checkpoint: int
) -> list[dict[str, Any]]:
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    warehouse = env.unwrapped
    try:
        warehouse.reset(seed=seed)
        for action in actions[:checkpoint]:
            warehouse.step(action)
        return [
            canonical_step(action, warehouse.step(action))
            for action in actions[checkpoint : checkpoint + SUFFIX_LENGTH]
        ]
    finally:
        env.close()


def snapshot_suffix(
    seed: int, actions: list[list[int]], checkpoint: int
) -> tuple[bool, bool]:
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    warehouse = env.unwrapped
    clone = None
    try:
        warehouse.reset(seed=seed)
        for action in actions[:checkpoint]:
            warehouse.step(action)
        clone = pickle.loads(pickle.dumps(warehouse, protocol=pickle.HIGHEST_PROTOCOL))
        first_action = actions[checkpoint]
        original_first = canonical_step(first_action, warehouse.step(first_action))
        independent = int(clone._cur_steps) == checkpoint
        clone_first = canonical_step(first_action, clone.step(first_action))
        original_suffix = [original_first]
        clone_suffix = [clone_first]
        for action in actions[checkpoint + 1 : checkpoint + SUFFIX_LENGTH]:
            original_suffix.append(canonical_step(action, warehouse.step(action)))
            clone_suffix.append(canonical_step(action, clone.step(action)))
        return independent, original_suffix == clone_suffix
    finally:
        if clone is not None:
            clone.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_results: list[dict[str, Any]] = []
    first_actions: dict[int, list[list[int]]] = {}
    first_steps: dict[int, list[dict[str, Any]]] = {}
    for seed in args.seeds:
        actions_a, steps_a = record_heuristic(seed, render=args.render)
        actions_b, steps_b = record_heuristic(seed)
        first_actions[seed] = actions_a
        first_steps[seed] = steps_a
        seed_results.append(
            {
                "seed": seed,
                "steps": len(steps_a),
                "trace_sha256": trace_hash(steps_a),
                "repeat_trace_sha256": trace_hash(steps_b),
                "exact_repeat": steps_a == steps_b,
                "actions_exact_repeat": actions_a == actions_b,
            }
        )

    checkpoint_seed = args.seeds[0]
    actions = first_actions[checkpoint_seed]
    steps = first_steps[checkpoint_seed]
    checkpoints = []
    for checkpoint in CHECKPOINTS:
        replayed = replay_suffix(checkpoint_seed, actions, checkpoint)
        independent, snapshot_equal = snapshot_suffix(
            checkpoint_seed, actions, checkpoint
        )
        checkpoints.append(
            {
                "step": checkpoint,
                "suffix_steps": SUFFIX_LENGTH,
                "replay_exact": replayed
                == steps[checkpoint : checkpoint + SUFFIX_LENGTH],
                "snapshot_independent": independent,
                "snapshot_exact": snapshot_equal,
            }
        )

    summary = {
        "schema_version": 1,
        "environment": ENVIRONMENT_ID,
        "seeds": seed_results,
        "snapshot_and_replay_seed": checkpoint_seed,
        "checkpoints": checkpoints,
        "render_requested": bool(args.render),
        "snapshot_supported": all(item["snapshot_exact"] for item in checkpoints),
        "replay_fallback_selected": all(item["replay_exact"] for item in checkpoints)
        and not all(item["snapshot_exact"] for item in checkpoints),
        "all_checks_passed": all(item["exact_repeat"] for item in seed_results)
        and all(item["actions_exact_repeat"] for item in seed_results)
        and all(item["replay_exact"] for item in checkpoints),
    }
    payload = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
