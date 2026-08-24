"""Run pinned TA-RWARE heuristic episodes and emit reproducible metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO, cast

import gymnasium as gym
import numpy as np
import tarware  # noqa: F401 - importing registers the TA-RWARE environments
from numpy.typing import NDArray
from tarware.heuristic import heuristic_episode
from tarware.warehouse import Warehouse

ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
DEFAULT_EPISODE_STEPS = 500
_GPU_QUERY_FIELDS = (
    "index",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def canonical(value: Any) -> Any:
    """Convert simulator values to stable JSON-compatible structures."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonical(item) for item in value]
    raise TypeError(f"unsupported value {type(value)!r}")


def stable_sha256(value: Any) -> str:
    """Hash a canonical JSON value."""
    payload = json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RecordingWarehouse:
    """Transparent TA-RWARE proxy that records exact heuristic step outputs."""

    def __init__(self, warehouse: Warehouse) -> None:
        self._warehouse = warehouse
        self.actions: list[list[int]] = []
        self.steps: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._warehouse, name)

    def step(self, action: Any) -> Any:
        started = time.perf_counter()
        result = self._warehouse.step(action)
        observations, rewards, terminated, truncated, info = result
        action_values = [int(value) for value in action]
        self.actions.append(action_values)
        self.steps.append(
            {
                "action": action_values,
                "observations": canonical(observations),
                "rewards": canonical(rewards),
                "terminated": canonical(terminated),
                "truncated": canonical(truncated),
                "simulator_info": canonical(info),
                "step_wall_seconds": time.perf_counter() - started,
            }
        )
        return result


def _optional_float(value: str) -> float | None:
    normalized = value.strip()
    if normalized.casefold() in {"", "n/a", "[n/a]", "not supported"}:
        return None
    return float(normalized)


def parse_gpu_output(payload: str) -> list[dict[str, Any]]:
    """Parse the stable CSV emitted by the configured nvidia-smi query."""
    rows: list[dict[str, Any]] = []
    for row in csv.reader(payload.splitlines()):
        if not row:
            continue
        if len(row) != len(_GPU_QUERY_FIELDS):
            raise ValueError(
                f"nvidia-smi returned {len(row)} columns, expected "
                f"{len(_GPU_QUERY_FIELDS)}"
            )
        rows.append(
            {
                "gpu_index": int(row[0].strip()),
                "gpu_name": row[1].strip(),
                "gpu_utilization_percent": _optional_float(row[2]),
                "gpu_memory_used_mib": _optional_float(row[3]),
                "gpu_memory_total_mib": _optional_float(row[4]),
                "gpu_power_watts": _optional_float(row[5]),
            }
        )
    return rows


class GpuMonitor:
    """Sample NVIDIA GPU utilization while the short heuristic run executes."""

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("telemetry interval must be positive")
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self._executable = shutil.which("nvidia-smi")
        self._started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._started = time.perf_counter()
        self._sample("start")
        if self._executable is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))
        self._sample("end")

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample("interval")

    def _sample(self, sample_type: str) -> None:
        if self._executable is None:
            if not self.errors:
                self.errors.append("nvidia-smi was not found")
            return
        command = [
            self._executable,
            f"--query-gpu={','.join(_GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(  # noqa: S603 - fixed local executable/query
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for parsed in parse_gpu_output(completed.stdout):
                self.samples.append(
                    {
                        "sample_type": sample_type,
                        "timestamp_utc": utc_now(),
                        "elapsed_seconds": time.perf_counter() - self._started,
                        **parsed,
                    }
                )
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")


def _sum_info_metric(infos: Sequence[Mapping[str, Any]], name: str) -> int:
    return sum(int(info.get(name, 0)) for info in infos)


def _busy_agent_steps(infos: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        sum(int(bool(value)) for value in info.get("vehicles_busy", []))
        for info in infos
    )


def _metric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_gpu(
    samples: Sequence[Mapping[str, Any]], errors: Sequence[str]
) -> dict[str, Any]:
    """Aggregate samples for the first visible NVIDIA GPU."""
    gpu_zero = [sample for sample in samples if int(sample["gpu_index"]) == 0]

    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in gpu_zero if sample.get(key) is not None]

    return {
        "sample_count": len(gpu_zero),
        "interval_sample_count": sum(
            sample.get("sample_type") == "interval" for sample in gpu_zero
        ),
        "gpu_name": gpu_zero[0]["gpu_name"] if gpu_zero else None,
        "gpu_utilization_percent": _metric_summary(
            values("gpu_utilization_percent")
        ),
        "gpu_memory_used_mib": _metric_summary(values("gpu_memory_used_mib")),
        "gpu_memory_total_mib": gpu_zero[0].get("gpu_memory_total_mib")
        if gpu_zero
        else None,
        "gpu_power_watts": _metric_summary(values("gpu_power_watts")),
        "errors": list(errors),
    }


def _open_jsonl(path: Path | None, *, overwrite: bool) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w" if overwrite else "x", encoding="utf-8", newline="\n")


def _guard_output_paths(paths: Sequence[Path | None], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [str(path) for path in paths if path is not None and path.exists()]
    if existing:
        raise FileExistsError(f"refusing to replace existing outputs: {existing}")


def _write_jsonl(handle: TextIO | None, record: Mapping[str, Any]) -> None:
    if handle is None:
        return
    handle.write(json.dumps(canonical(record), sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


def run_seed(
    seed: int,
    *,
    render: bool,
    expected_steps: int,
    step_log: TextIO | None,
) -> dict[str, Any]:
    """Run one measured heuristic episode."""
    env = gym.make(ENVIRONMENT_ID, disable_env_checker=True)
    warehouse = cast(Warehouse, env.unwrapped)
    recorder = RecordingWarehouse(warehouse)
    started_at = utc_now()
    started = time.perf_counter()
    process_started = time.process_time()
    try:
        observations = cast(
            tuple[NDArray[np.floating[Any]], ...],
            warehouse.reset(seed=seed),
        )
        observation_shapes = [list(observation.shape) for observation in observations]
        action_space = cast(gym.spaces.Tuple, warehouse.action_space)
        action_sizes = [
            int(cast(gym.spaces.Discrete, space).n) for space in action_space.spaces
        ]
        infos, total_return, agent_returns = heuristic_episode(
            recorder,
            render=render,
            seed=seed,
        )
    finally:
        env.close()
    wall_seconds = time.perf_counter() - started
    process_cpu_seconds = time.process_time() - process_started
    ended_at = utc_now()

    if len(infos) != len(recorder.steps):
        raise RuntimeError("heuristic info count does not match recorded step count")
    if len(infos) != expected_steps:
        raise RuntimeError(
            f"seed {seed} ran {len(infos)} steps, expected {expected_steps}"
        )
    if not math.isclose(
        float(np.sum(agent_returns)), float(total_return), rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("agent return sum does not match global return")

    for step_index, step in enumerate(recorder.steps):
        _write_jsonl(
            step_log,
            {
                "schema_version": 1,
                "phase": "selection_heuristic_baseline",
                "environment_id": ENVIRONMENT_ID,
                "seed": seed,
                "step": step_index,
                **step,
            },
        )

    info_records = cast(list[Mapping[str, Any]], infos)
    step_latencies = [float(step["step_wall_seconds"]) for step in recorder.steps]
    return {
        "status": "completed",
        "seed": seed,
        "steps": len(infos),
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": process_cpu_seconds,
        "steps_per_second": len(infos) / wall_seconds,
        "observation_shapes": observation_shapes,
        "action_sizes": action_sizes,
        "total_return": float(total_return),
        "agent_returns": [float(value) for value in agent_returns],
        "completed_tasks": _sum_info_metric(info_records, "shelf_deliveries"),
        "clashes": _sum_info_metric(info_records, "clashes"),
        "stuck_events": _sum_info_metric(info_records, "stucks"),
        "agvs_distance_travelled_total": _sum_info_metric(
            info_records, "agvs_distance_travelled"
        ),
        "pickers_distance_travelled_total": _sum_info_metric(
            info_records, "pickers_distance_travelled"
        ),
        "agvs_idle_time_total": _sum_info_metric(info_records, "agvs_idle_time"),
        "pickers_idle_time_total": _sum_info_metric(
            info_records, "pickers_idle_time"
        ),
        "busy_agent_steps_total": _busy_agent_steps(info_records),
        "nonzero_destination_selections_total": sum(
            int(destination != 0)
            for action in recorder.actions
            for destination in action
        ),
        "terminated_step_count": sum(
            all(bool(value) for value in step["terminated"])
            for step in recorder.steps
        ),
        "truncated_step_count": sum(
            all(bool(value) for value in step["truncated"])
            for step in recorder.steps
        ),
        "simulator_step_wall_seconds": _metric_summary(step_latencies),
        "action_trace_sha256": stable_sha256(recorder.actions),
        "step_trace_sha256": stable_sha256(
            [
                {key: value for key, value in step.items() if key != "step_wall_seconds"}
                for step in recorder.steps
            ]
        ),
        "llm_calls": 0,
        "fallbacks": 0,
        "invalid_actions": 0,
        "effect_violations": 0,
        "model_output_errors": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed", type=int, default=None)
    seed_group.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--expected-steps",
        type=int,
        default=DEFAULT_EPISODE_STEPS,
        help="Fail if an episode does not match the selection step cap.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open TA-RWARE's Pyglet viewer while each episode runs.",
    )
    parser.add_argument("--output", type=Path, help="Write the aggregate JSON report.")
    parser.add_argument(
        "--step-log-output", type=Path, help="Write one raw JSONL record per step."
    )
    parser.add_argument(
        "--telemetry-output", type=Path, help="Write raw GPU samples as JSONL."
    )
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=0.25,
        help="GPU sample interval; short because heuristic episodes finish quickly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow requested output files to be replaced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed or 0]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")
    if args.expected_steps <= 0:
        raise ValueError("expected steps must be positive")
    _guard_output_paths(
        (args.output, args.step_log_output, args.telemetry_output),
        overwrite=args.overwrite,
    )

    run_started_at = utc_now()
    run_started = time.perf_counter()
    monitor = GpuMonitor(args.telemetry_interval_seconds)
    step_log = _open_jsonl(args.step_log_output, overwrite=args.overwrite)
    try:
        monitor.start()
        episodes = [
            run_seed(
                seed,
                render=args.render,
                expected_steps=args.expected_steps,
                step_log=step_log,
            )
            for seed in seeds
        ]
    finally:
        monitor.stop()
        if step_log is not None:
            step_log.close()

    for sample_index, sample in enumerate(monitor.samples):
        sample["sample_index"] = sample_index
    telemetry_log = _open_jsonl(args.telemetry_output, overwrite=args.overwrite)
    try:
        for sample in monitor.samples:
            _write_jsonl(
                telemetry_log,
                {
                    "schema_version": 1,
                    "phase": "selection_heuristic_baseline",
                    **sample,
                },
            )
    finally:
        if telemetry_log is not None:
            telemetry_log.close()

    run_ended_at = utc_now()
    summary = {
        "schema_version": 2,
        "phase": "selection_heuristic_baseline",
        "status": "completed",
        "environment_id": ENVIRONMENT_ID,
        "selection_environment_match": True,
        "selection_seeds": seeds,
        "expected_episode_steps": args.expected_steps,
        "render_requested": bool(args.render),
        "started_at_utc": run_started_at,
        "ended_at_utc": run_ended_at,
        "wall_seconds": time.perf_counter() - run_started,
        "episodes": episodes,
        "group": {
            "episode_count": len(episodes),
            "mean_completed_tasks": sum(
                int(episode["completed_tasks"]) for episode in episodes
            )
            / len(episodes),
            "mean_total_return": sum(
                float(episode["total_return"]) for episode in episodes
            )
            / len(episodes),
            "total_clashes": sum(int(episode["clashes"]) for episode in episodes),
            "total_stuck_events": sum(
                int(episode["stuck_events"]) for episode in episodes
            ),
            "mean_wall_seconds": sum(
                float(episode["wall_seconds"]) for episode in episodes
            )
            / len(episodes),
        },
        "telemetry_interval_seconds": args.telemetry_interval_seconds,
        "gpu": summarize_gpu(monitor.samples, monitor.errors),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "step_log_path": str(args.step_log_output) if args.step_log_output else None,
        "telemetry_log_path": str(args.telemetry_output)
        if args.telemetry_output
        else None,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open(
            "w" if args.overwrite else "x", encoding="utf-8", newline="\n"
        ) as output:
            output.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
