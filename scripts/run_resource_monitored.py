"""Run one experiment command and record elapsed time plus NVIDIA metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def _gpu_snapshot() -> list[dict[str, str | int | float | None]]:
    query = ",".join(
        (
            "index",
            "name",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "power.draw",
        )
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return [{"error": f"nvidia-smi unavailable: {exc}"}]

    snapshots: list[dict[str, str | int | float | None]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        index, name, total, used, utilization, power = fields
        snapshots.append(
            {
                "index": int(index),
                "name": name,
                "memory_total_mib": int(float(total)),
                "memory_used_mib": int(float(used)),
                "utilization_gpu_percent": int(float(utilization)),
                "power_draw_watts": float(power),
            }
        )
    return snapshots


def _write_sample(
    handle: TextIO,
    *,
    phase: str,
    event: str,
    started: float,
) -> None:
    record = {
        "phase": phase,
        "event": event,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gpus": _gpu_snapshot(),
    }
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=150.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.interval_seconds < 120 or args.interval_seconds > 600:
        parser.error("interval must be between 120 and 600 seconds")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with args.log.open("a", encoding="utf-8") as handle:
        header = {
            "phase": args.phase,
            "command": command,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        handle.flush()
        _write_sample(handle, phase=args.phase, event="start", started=started)
        process = subprocess.Popen(command)
        while True:
            try:
                process.wait(timeout=args.interval_seconds)
                break
            except subprocess.TimeoutExpired:
                _write_sample(
                    handle, phase=args.phase, event="interval", started=started
                )
        _write_sample(handle, phase=args.phase, event="end", started=started)
        footer = {
            "phase": args.phase,
            "return_code": process.returncode,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        handle.write(json.dumps(footer, sort_keys=True) + "\n")
        handle.flush()
    return int(process.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
