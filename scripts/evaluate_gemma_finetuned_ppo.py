"""Evaluate parent and fine-tuned directive PPO with the same frozen Gemma."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.finetuning import FineTuneEvaluationConfig
from warehouse_ai.demo.evaluation.finetune_runner import run_fine_tune_comparison


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repository_root()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_yaml_config(config_path, FineTuneEvaluationConfig)
    summary = run_fine_tune_comparison(config, repository_root=root)
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
