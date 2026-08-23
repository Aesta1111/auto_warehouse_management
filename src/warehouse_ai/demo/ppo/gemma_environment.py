"""Gym environment executing PPO directives through frozen Gemma."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.artifacts import append_jsonl
from warehouse_ai.demo.contracts.finetuning import (
    FineTuneStepRecord,
    FineTuneTrainingSummary,
)
from warehouse_ai.demo.contracts.inference import FailureCode
from warehouse_ai.demo.directives import Registry
from warehouse_ai.demo.execution import DirectiveExecutionService
from warehouse_ai.demo.inference.actioner import Actioner
from warehouse_ai.demo.ppo.environment import GymEnv, _coerce_scalar_action
from warehouse_ai.demo.contracts.directives import DirectiveCatalog


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


class FineTuneMetrics:
    """Mutable training-only counters with one immutable summary boundary."""

    def __init__(self) -> None:
        self.simulator_return = 0.0
        self.learning_return = 0.0
        self.gemma_calls = 0
        self.fallbacks = 0
        self.effect_violations = 0
        self.failure_counts: Counter[str] = Counter()
        self.directive_counts: Counter[str] = Counter()
        self.latencies: list[float] = []

    def summary(
        self,
        *,
        completed_environment_steps: int,
        optimizer_steps: int,
        reference_kl_max: float,
    ) -> FineTuneTrainingSummary:
        return FineTuneTrainingSummary(
            completed_environment_steps=completed_environment_steps,
            optimizer_steps=optimizer_steps,
            simulator_return=self.simulator_return,
            learning_return=self.learning_return,
            gemma_calls=self.gemma_calls,
            fallbacks=self.fallbacks,
            effect_violations=self.effect_violations,
            failure_counts=dict(sorted(self.failure_counts.items())),
            directive_counts=dict(sorted(self.directive_counts.items())),
            reference_kl_max=reference_kl_max,
            latency_median_ms=_percentile(self.latencies, 0.5),
            latency_p95_ms=_percentile(self.latencies, 0.95),
        )


class GemmaGymEnv(GymEnv):
    """Use the production Gemma validator/fallback path during PPO collection."""

    def __init__(
        self,
        environment_id: str,
        catalog: DirectiveCatalog,
        registry: Registry,
        episode_seeds: tuple[int, ...],
        actioner: Actioner,
        *,
        episode_steps: int = 500,
        fallback_penalty: float = 0.001,
        step_log_path: Path | None = None,
    ) -> None:
        if fallback_penalty < 0.0:
            raise ValueError("fallback_penalty must be non-negative")
        super().__init__(
            environment_id,
            catalog,
            registry,
            episode_seeds,
            episode_steps,
        )
        self.execution = DirectiveExecutionService(registry, actioner)
        self.fallback_penalty = float(fallback_penalty)
        self.step_log_path = step_log_path
        self.metrics = FineTuneMetrics()
        self.total_steps = 0

    def step(
        self, action: int | NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        if self._step_count >= self.episode_steps:
            raise RuntimeError("episode is truncated; reset before stepping")
        directive_id = _coerce_scalar_action(action)
        context = self.context
        directive = self.catalog.by_id(directive_id)
        outcome = self.execution.execute(context, directive, self.wrapper)
        observation, rewards, terminated, truncated, simulator_info = self.wrapper.step(
            outcome.action
        )
        simulator_reward = float(sum(rewards))
        learning_reward = simulator_reward - (
            self.fallback_penalty if outcome.failure_code is not None else 0.0
        )
        attempted_gemma = directive_id != self.catalog.wait_directive_id
        self._step_count += 1
        self.total_steps += 1
        simulator_terminated = bool(all(terminated))
        simulator_truncated = bool(all(truncated))
        self._record(
            context=context,
            directive_id=directive_id,
            command=directive.command,
            outcome=outcome,
            simulator_reward=simulator_reward,
            learning_reward=learning_reward,
            attempted_gemma=attempted_gemma,
        )
        info = dict(simulator_info)
        info.update(
            {
                "directive_id": directive_id,
                "directive_command": directive.command,
                "executed_destinations": [
                    int(item.destination_id) for item in outcome.action.actions
                ],
                "executor_status": outcome.source,
                "failure_code": (
                    None if outcome.failure_code is None else outcome.failure_code.value
                ),
                "simulator_reward": simulator_reward,
                "learning_reward": learning_reward,
                "simulator_terminated": simulator_terminated,
                "simulator_truncated": simulator_truncated,
            }
        )
        return (
            observation.encode(),
            learning_reward,
            simulator_terminated,
            simulator_truncated or self._step_count >= self.episode_steps,
            info,
        )

    def _record(
        self,
        *,
        context,
        directive_id: int,
        command: str,
        outcome,
        simulator_reward: float,
        learning_reward: float,
        attempted_gemma: bool,
    ) -> None:
        self.metrics.simulator_return += simulator_reward
        self.metrics.learning_return += learning_reward
        self.metrics.directive_counts[command] += 1
        self.metrics.gemma_calls += int(attempted_gemma)
        if outcome.failure_code is not None:
            code = outcome.failure_code.value
            self.metrics.fallbacks += 1
            self.metrics.failure_counts[code] += 1
            self.metrics.effect_violations += int(
                outcome.failure_code is FailureCode.EFFECT_VIOLATION
            )
        if outcome.result is not None:
            self.metrics.latencies.append(float(outcome.result.latency_ms))
        if self.step_log_path is not None:
            append_jsonl(
                self.step_log_path,
                FineTuneStepRecord(
                    environment_step=self.total_steps - 1,
                    episode_seed=self.current_seed,
                    episode_step=context.observation.step,
                    state_id=context.state_id,
                    directive_id=directive_id,
                    directive_command=command,
                    destinations=tuple(
                        int(item.destination_id) for item in outcome.action.actions
                    ),
                    simulator_reward=simulator_reward,
                    learning_reward=learning_reward,
                    source=outcome.source,
                    failure_code=(
                        None
                        if outcome.failure_code is None
                        else outcome.failure_code.value
                    ),
                    llm_latency_ms=(
                        None if outcome.result is None else outcome.result.latency_ms
                    ),
                ),
            )
