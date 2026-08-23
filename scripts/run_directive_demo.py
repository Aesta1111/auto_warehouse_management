"""Run the matched, unscored directive PPO/Gemma demonstration."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from warehouse_ai.demo.contracts.configuration import load_ppo_evaluation_run
from warehouse_ai.demo.evaluation.runner import run_matched_demo


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _powershell_executable() -> str:
    """Return the available PowerShell executable for the local launcher."""
    candidates = ("powershell.exe", "pwsh") if os.name == "nt" else ("pwsh",)
    for candidate in candidates:
        executable = shutil.which(candidate)
        if executable is not None:
            return executable
    raise RuntimeError(
        "PowerShell is required to start Gemma; install powershell.exe or pwsh."
    )


def _start_gemma(args: argparse.Namespace, root: Path) -> int:
    """Delegate Gemma preparation and serving to the pinned PowerShell script."""
    script_path = root / "scripts" / "start_local_gemma.ps1"
    if not script_path.is_file():
        raise FileNotFoundError(f"Gemma launcher script is missing: {script_path}")

    command = [
        _powershell_executable(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script_path),
        "-Port",
        str(args.port),
        "-ContextSize",
        str(args.context_size),
        "-GpuLayers",
        str(args.gpu_layers),
    ]
    if args.prepare:
        command.append("-Prepare")

    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(command="run")
    parser.add_argument(
        "--config",
        type=Path,
        help="Demo YAML configuration (required for the default run command).",
    )
    subparsers = parser.add_subparsers(dest="command")
    start_gemma = subparsers.add_parser(
        "start-gemma",
        help="Prepare and start the local Gemma llama.cpp server.",
    )
    start_gemma.add_argument(
        "--prepare",
        action="store_true",
        help="Download/prepare the pinned runtime and model when missing.",
    )
    start_gemma.add_argument("--port", type=int, default=8080)
    start_gemma.add_argument("--context-size", type=int, default=4096)
    start_gemma.add_argument("--gpu-layers", type=int, default=24)

    args = parser.parse_args()
    if args.command == "run" and args.config is None:
        parser.error("--config is required unless the start-gemma subcommand is used")
    return args


def main() -> int:
    args = parse_args()
    root = _repository_root()
    if args.command == "start-gemma":
        return _start_gemma(args, root)

    config = load_ppo_evaluation_run(args.config, root).demo_runtime()
    summary = run_matched_demo(config, repository_root=root)
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    print(f"output_directory: {root / summary.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
