"""Probe directive applicability and deterministic execution."""

from __future__ import annotations

import argparse
from pathlib import Path

from warehouse_ai.demo.artifacts import write_json_atomic
from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.directives import (
    DirectiveProbeConfig,
    DirectiveVocabulary,
)
from warehouse_ai.demo.directives.probe import run_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    resolve = lambda path: path if path.is_absolute() else root / path  # noqa: E731
    config = load_yaml_config(resolve(args.config), DirectiveProbeConfig)
    vocabulary = DirectiveVocabulary.load(resolve(config.vocabulary_path))
    report = run_probe(config, vocabulary)
    output = resolve(args.output)
    write_json_atomic(output, report)
    print(report.model_dump_json(indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
