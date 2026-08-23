"""Train and freeze the configured isolated directive PPO prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.artifacts import write_json_atomic
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.contracts.configuration import load_ppo_training_run
from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.ppo.service import train_directive_ppo
from warehouse_ai.demo.ppo.training import PPO_CHECKPOINT_NAME


PPO_MANIFEST_NAME = "ppo-training-manifest.json"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repository_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for the ignored checkpoint and small manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = _repository_root()
    loaded = load_ppo_training_run(args.config, root)
    config = loaded.runtime
    catalog = DirectiveCatalog.load(_repository_path(root, config.vocabulary_path))
    registry = build_registry(catalog)
    output_directory = _repository_path(
        root, args.output or loaded.run.output_directory
    )
    manifest = train_directive_ppo(
        config,
        catalog,
        registry,
        output_directory,
    )
    manifest_path = output_directory / PPO_MANIFEST_NAME
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    print(f"checkpoint: {output_directory / PPO_CHECKPOINT_NAME}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
