"""Run the matched, unscored directive PPO/Gemma demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.evaluation import run_matched_demo
from warehouse_ai.demo.models import DirectiveDemoConfig, load_yaml_config


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
    config = load_yaml_config(config_path, DirectiveDemoConfig)
    summary = run_matched_demo(config, repository_root=root)
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    print(f"output_directory: {root / summary.output_directory}")


if __name__ == "__main__":
    main()
