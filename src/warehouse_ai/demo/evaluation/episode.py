"""One controller episode and its append-only decision log."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from warehouse_ai.demo.artifacts import append_jsonl
from warehouse_ai.demo.contracts.evaluation import DemoStepRecord, EpisodeResult
from warehouse_ai.demo.contracts.inference import FailureCode
from warehouse_ai.demo.controllers import DemoController
from warehouse_ai.env import TarwareEnvWrapper


_INVALID_ACTION_CODES = frozenset(
    {
        FailureCode.MALFORMED_RESPONSE,
        FailureCode.INVALID_ACTION,
        FailureCode.STALE_STATE,
    }
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("simulator info contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"simulator info contains unsupported value {type(value)!r}")


def _metric_count(info: Mapping[str, Any], key: str) -> int:
    value = info.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.integer):
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"TA-RWARE info metric {key!r} must be non-negative integer")
    return value


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def run_episode(
    wrapper: TarwareEnvWrapper,
    controller: DemoController,
    *,
    seed: int,
    max_steps: int,
    render: bool,
    decision_log_path: Path,
) -> EpisodeResult:
    """Run one controller episode and flush every decision record."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    wrapper.reset(seed=int(seed))
    started = time.perf_counter()
    steps = completed_tasks = clashes = stuck_events = 0
    llm_calls = invalid_actions = effect_violations = fallbacks = 0
    total_return = 0.0
    directive_counts: dict[str, int] = {}
    latencies: list[float] = []

    while steps < max_steps:
        decision = controller.choose_action(wrapper)
        command = decision.prediction.directive.command
        directive_counts[command] = directive_counts.get(command, 0) + 1
        attempted_llm = (
            controller.controller_name == "ppo_gemma" and command.casefold() != "wait"
        )
        if attempted_llm:
            llm_calls += 1
            invalid_actions += int(decision.failure_code in _INVALID_ACTION_CODES)
            effect_violations += int(
                decision.failure_code is FailureCode.EFFECT_VIOLATION
            )
        if decision.failure_code is not None or decision.source == "automatic_no_op":
            fallbacks += 1
        if decision.llm_latency_ms is not None:
            latencies.append(float(decision.llm_latency_ms))

        _, rewards, terminated, truncated, simulator_info = wrapper.step(
            decision.action
        )
        reward_values = tuple(float(value) for value in rewards)
        terminated_values = tuple(bool(value) for value in terminated)
        truncated_values = tuple(bool(value) for value in truncated)
        if not terminated_values or not truncated_values:
            raise ValueError("TA-RWARE must return one termination flag per agent")
        if not isinstance(simulator_info, Mapping):
            raise TypeError("TA-RWARE simulator info must be a mapping")
        normalized = _json_safe(simulator_info)
        if not isinstance(normalized, dict):
            raise TypeError("normalized simulator info must be a JSON object")
        append_jsonl(
            decision_log_path,
            DemoStepRecord(
                decision=decision,
                rewards=reward_values,
                terminated=terminated_values,
                truncated=truncated_values,
                simulator_info=normalized,
            ),
        )
        if render:
            wrapper.render()
        steps += 1
        total_return += sum(reward_values)
        completed_tasks += _metric_count(simulator_info, "shelf_deliveries")
        clashes += _metric_count(simulator_info, "clashes")
        stuck_events += _metric_count(simulator_info, "stucks")
        if all(terminated_values) or all(truncated_values):
            break

    return EpisodeResult(
        controller=controller.controller_name,
        seed=int(seed),
        steps=steps,
        completed_tasks=completed_tasks,
        total_return=total_return,
        clashes=clashes,
        stuck_events=stuck_events,
        wall_seconds=time.perf_counter() - started,
        directive_counts=dict(sorted(directive_counts.items())),
        llm_calls=llm_calls,
        invalid_actions=invalid_actions,
        effect_violations=effect_violations,
        fallbacks=fallbacks,
        llm_latency_median_ms=_percentile(latencies, 0.5),
        llm_latency_p95_ms=_percentile(latencies, 0.95),
    )
