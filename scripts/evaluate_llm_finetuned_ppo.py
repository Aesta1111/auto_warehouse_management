"""Evaluate parent and fine-tuned directive PPO with the same frozen Llm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.contracts.configuration import load_ppo_evaluation_run
from warehouse_ai.demo.evaluation.finetune_runner import run_fine_tune_comparison


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("selection", "confirmation", "final"),
        default="selection",
        help="Gated seed partition to evaluate (default: selection).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repository_root()
    loaded = load_ppo_evaluation_run(args.config, root)
    config = loaded.comparison_runtime(args.partition)
    summary = run_fine_tune_comparison(
        config,
        fine_config=loaded.fine_tuning.runtime,
        repository_root=root,
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
