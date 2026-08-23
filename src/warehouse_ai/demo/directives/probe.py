"""Real-state compatibility probe for the frozen directive catalog."""

from __future__ import annotations

import numpy as np

from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    ProbeConfig,
    ProbeReport,
    DirectiveCatalog,
)
from warehouse_ai.demo.directives.registry import (
    SemanticsError,
    SemanticsFailure,
    build_registry,
)
from warehouse_ai.env import TarwareEnvWrapper


def _probe_seed(
    config: ProbeConfig, catalog: DirectiveCatalog, seed: int
) -> tuple[int, int, int, int]:
    registry = build_registry(catalog)
    wrapper = TarwareEnvWrapper(config.environment_id)
    states = uncovered = executor_failures = effect_failures = 0
    try:
        wrapper.reset(seed=seed)
        for _ in range(config.episode_steps):
            context = WarehouseContext.capture(wrapper)
            states += 1
            try:
                applicable = np.flatnonzero(registry.action_mask(context))
            except SemanticsError:
                uncovered += 1
                break
            if len(applicable) == 0:
                uncovered += 1
                break
            for directive_id in (int(value) for value in applicable):
                try:
                    first = registry.execute(directive_id, context, wrapper)
                    second = registry.execute(directive_id, context, wrapper)
                    executor_failures += int(first != second)
                except SemanticsError as exc:
                    if exc.code is SemanticsFailure.EFFECT:
                        effect_failures += 1
                    else:
                        executor_failures += 1
            wait_id = catalog.wait_directive_id
            non_wait = [int(value) for value in applicable if int(value) != wait_id]
            selected = non_wait[0] if non_wait else wait_id
            try:
                action = registry.execute(selected, context, wrapper)
                _, _, terminated, truncated, _ = wrapper.step(action)
            except SemanticsError as exc:
                if exc.code is SemanticsFailure.EFFECT:
                    effect_failures += 1
                else:
                    executor_failures += 1
                break
            if all(terminated) or all(truncated):
                break
    finally:
        wrapper.close()
    return states, uncovered, executor_failures, effect_failures


def run_probe(config: ProbeConfig, catalog: DirectiveCatalog) -> ProbeReport:
    totals = [0, 0, 0, 0]
    for seed in config.probe_seeds:
        totals = [
            left + right
            for left, right in zip(
                totals, _probe_seed(config, catalog, seed), strict=True
            )
        ]
    states, uncovered, executor_failures, effect_failures = totals
    registry = build_registry(catalog)
    return ProbeReport(
        schema_version=config.schema_version,
        environment_id=config.environment_id,
        vocabulary_sha256=catalog.digest(),
        behavior_sha256=registry.behavior_digest(),
        behavior_engine_version=(
            registry.engine_version if config.schema_version == 2 else None
        ),
        probe_seeds=config.probe_seeds,
        states_checked=states,
        uncovered_states=uncovered,
        executor_failures=executor_failures,
        effect_failures=effect_failures,
        passed=bool(
            states > 0
            and uncovered == 0
            and executor_failures == 0
            and effect_failures == 0
        ),
    )
