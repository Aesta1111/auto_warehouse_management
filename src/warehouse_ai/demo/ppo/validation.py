"""Held-out directive-policy evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from warehouse_ai.demo.contracts.directives import DirectiveVocabulary
from warehouse_ai.demo.contracts.ppo import (
    DirectivePpoConfig,
    DirectivePpoValidationSummary,
)
from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.demo.ppo.environment import DirectiveGymEnv
from warehouse_ai.demo.ppo.policy import DirectivePpoPolicy


def _completed_tasks(info: dict[str, Any]) -> int:
    return int(info.get("shelf_deliveries", 0))


def _entropy(probabilities: tuple[float, ...]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def evaluate_directive_ppo(
    policy: DirectivePpoPolicy,
    config: DirectivePpoConfig,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
) -> DirectivePpoValidationSummary:
    if policy.vocabulary.digest() != vocabulary.digest():
        raise ValueError("policy vocabulary does not match evaluation vocabulary")
    if registry.vocabulary.digest() != vocabulary.digest():
        raise ValueError("registry vocabulary does not match evaluation vocabulary")
    environment = DirectiveGymEnv(
        config.environment_id,
        vocabulary,
        registry,
        config.validation_seeds,
        config.episode_steps,
    )
    returns: list[float] = []
    completed: list[int] = []
    counts: dict[str, int] = {}
    selected_non_wait: set[int] = set()
    inapplicable = 0
    masked = 0
    entropies: list[float] = []
    try:
        for seed in config.validation_seeds:
            environment.reset(seed=seed)
            episode_return = 0.0
            episode_completed = 0
            for _ in range(config.episode_steps):
                context = environment.context
                mask = environment.action_masks()
                masked += int((~mask).sum())
                prediction = policy.predict(context, mask)
                directive_id = prediction.directive.directive_id
                if not bool(mask[directive_id]):
                    inapplicable += 1
                if directive_id != vocabulary.wait_directive_id:
                    selected_non_wait.add(directive_id)
                command = prediction.directive.command
                counts[command] = counts.get(command, 0) + 1
                entropies.append(_entropy(prediction.probabilities))
                _, reward, terminated, truncated, info = environment.step(directive_id)
                episode_return += reward
                episode_completed += _completed_tasks(info)
                if terminated or truncated:
                    break
            returns.append(episode_return)
            completed.append(episode_completed)
    finally:
        environment.close()
    non_wait_count = len(selected_non_wait)
    return DirectivePpoValidationSummary(
        seeds=config.validation_seeds,
        mean_completed_tasks=float(np.mean(completed)),
        mean_return=float(np.mean(returns)),
        directive_counts=dict(sorted(counts.items())),
        non_wait_directives_used=non_wait_count,
        inapplicable_directives=inapplicable,
        masked_action_count=masked,
        mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
        collapsed_to_one_non_wait=non_wait_count < 2,
    )


def is_better_validation(
    candidate: DirectivePpoValidationSummary,
    incumbent: DirectivePpoValidationSummary | None,
) -> bool:
    if incumbent is None:
        return True
    return (candidate.mean_completed_tasks, candidate.mean_return) > (
        incumbent.mean_completed_tasks,
        incumbent.mean_return,
    )
