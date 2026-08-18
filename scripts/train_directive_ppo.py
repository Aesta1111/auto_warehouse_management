"""Train and freeze a versioned isolated directive PPO prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_ai.demo.artifacts import write_json_atomic
from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.models import (
    DirectivePpoConfig,
    DirectiveVocabulary,
    load_yaml_config,
)
from warehouse_ai.demo.ppo import train_directive_ppo


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
    config = load_yaml_config(_repository_path(root, args.config), DirectivePpoConfig)
    vocabulary = DirectiveVocabulary.load(
        _repository_path(root, config.vocabulary_path)
    )
    registry = build_directive_registry(vocabulary)
    default_output = Path(f"results/demo/directive-ppo-v{config.schema_version}")
    output_directory = _repository_path(root, args.output or default_output)
    manifest = train_directive_ppo(
        config,
        vocabulary,
        registry,
        output_directory,
    )
    manifest_path = output_directory / "directive-ppo-manifest.json"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    checkpoint_name = (
        "directive-ppo-v1.zip" if config.schema_version == 1 else "directive-ppo-v2.pt"
    )
    print(f"checkpoint: {output_directory / checkpoint_name}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
