"""Fine-tune a pretrained directive PPO through the frozen Gemma actioner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.contracts.configuration import load_ppo_fine_tuning_run
from warehouse_ai.demo.ppo.finetune_service import run_gemma_fine_tune


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repository_root()
    loaded = load_ppo_fine_tuning_run(args.config, root)
    manifest = run_gemma_fine_tune(loaded.runtime, repository_root=root)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
