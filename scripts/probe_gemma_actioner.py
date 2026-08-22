"""Run the exact-model directive preflight service."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.evaluation import DirectiveDemoConfig
from warehouse_ai.demo.inference.preflight import run_preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    path = args.config if args.config.is_absolute() else root / args.config
    try:
        config = load_yaml_config(path, DirectiveDemoConfig)
        report = run_preflight(config, repository_root=root)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"preflight could not complete: {type(exc).__name__}: {exc}")
        return 1
    print(report.model_dump_json(indent=2))
    if report.passed:
        return 0
    if config.selected_model == "e4b" and report.health.healthy:
        print("E4B failed; use a separately configured E2B server.", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
