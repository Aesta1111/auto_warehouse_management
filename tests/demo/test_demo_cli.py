from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from subprocess import CompletedProcess
from types import ModuleType
from typing import Any


ROOT = Path(__file__).parents[2]


def _cli_module() -> ModuleType:
    path = ROOT / "scripts/run_directive_demo.py"
    spec = importlib.util.spec_from_file_location("run_directive_demo", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_llm_subcommand_parses_launcher_options(monkeypatch: Any) -> None:
    module = _cli_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_directive_demo.py",
            "start-llm",
            "--prepare",
            "--port",
            "9090",
            "--context-size",
            "2048",
            "--gpu-layers",
            "12",
        ],
    )

    args = module.parse_args()

    assert args.command == "start-llm"
    assert args.prepare is True
    assert args.port == 9090
    assert args.context_size == 2048
    assert args.gpu_layers == 12


def test_start_llm_delegates_to_powershell_launcher(monkeypatch: Any) -> None:
    module = _cli_module()
    calls: list[tuple[list[str], Path, bool]] = []

    monkeypatch.setattr(module.shutil, "which", lambda _: "powershell.exe")

    def fake_run(
        command: list[str], *, cwd: Path, check: bool
    ) -> CompletedProcess[str]:
        calls.append((command, cwd, check))
        return CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["run_directive_demo.py", "start-llm", "--prepare"]
    )

    assert module.main() == 0
    assert len(calls) == 1
    command, cwd, check = calls[0]
    assert cwd == ROOT
    assert check is False
    assert command[:5] == [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
    ]
    assert command[5] == str(ROOT / "scripts/start_local_llm.ps1")
    assert command[-7:] == [
        "-Port",
        "8080",
        "-ContextSize",
        "8192",
        "-GpuLayers",
        "99",
        "-Prepare",
    ]
