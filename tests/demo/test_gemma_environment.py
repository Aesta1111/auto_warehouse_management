from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
    GemmaActionResponse,
)
from warehouse_ai.demo.directives import Registry, build_registry
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.demo.ppo.gemma_environment import GemmaGymEnv


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


class ExecutorShapedActioner:
    def __init__(self, environment: GemmaGymEnv, registry: Registry) -> None:
        self.environment = environment
        self.registry = registry
        self.calls = 0

    def act(self, request: ActionerRequest) -> ActionerResult:
        self.calls += 1
        action = self.registry.execute(
            request.directive.directive_id,
            request.context,
            self.environment.wrapper,
        )
        return ActionerResult(
            response=GemmaActionResponse(
                destinations=tuple(item.destination_id for item in action.actions),
                reason="valid fake action",
            ),
            raw_text="{}",
            model_id="fake-model",
            latency_ms=2.0,
        )


class FailingActioner:
    def act(self, request: ActionerRequest) -> ActionerResult:
        del request
        raise ActionerError(FailureCode.EFFECT_VIOLATION, "injected failure")


def _environment(
    actioner: Actioner,
) -> tuple[GemmaGymEnv, DirectiveCatalog, Registry]:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directives-v1.yaml")
    registry = build_registry(catalog)
    environment = GemmaGymEnv(
        ENVIRONMENT_ID,
        catalog,
        registry,
        (8100,),
        actioner,
        fallback_penalty=0.001,
    )
    environment.reset(seed=8100)
    return environment, catalog, registry


def _applicable_non_wait(environment: GemmaGymEnv, wait_id: int) -> int:
    return next(
        int(value)
        for value in np.flatnonzero(environment.action_masks())
        if int(value) != wait_id
    )


def test_valid_gemma_transition_preserves_simulator_reward() -> None:
    placeholder = cast(Actioner, FailingActioner())
    environment, catalog, registry = _environment(placeholder)
    actioner = ExecutorShapedActioner(environment, registry)
    environment.execution = environment.execution.__class__(
        registry, cast(Actioner, actioner)
    )
    try:
        directive_id = _applicable_non_wait(environment, catalog.wait_directive_id)
        _, reward, _, _, info = environment.step(directive_id)

        assert actioner.calls == 1
        assert reward == pytest.approx(info["simulator_reward"])
        assert info["executor_status"] == "gemma"
        assert info["failure_code"] is None
        assert environment.metrics.gemma_calls == 1
        assert environment.metrics.fallbacks == 0
    finally:
        environment.close()


def test_failure_penalty_is_visible_only_to_learning_reward() -> None:
    environment, catalog, _ = _environment(cast(Actioner, FailingActioner()))
    try:
        directive_id = _applicable_non_wait(environment, catalog.wait_directive_id)
        _, reward, _, _, info = environment.step(directive_id)

        assert reward == pytest.approx(info["simulator_reward"] - 0.001)
        assert info["learning_reward"] == pytest.approx(reward)
        assert info["failure_code"] == FailureCode.EFFECT_VIOLATION.value
        assert environment.metrics.fallbacks == 1
        assert environment.metrics.effect_violations == 1
    finally:
        environment.close()


def test_wait_bypasses_gemma_and_has_no_fallback_penalty() -> None:
    environment, catalog, _ = _environment(cast(Actioner, FailingActioner()))
    try:
        _, reward, _, _, info = environment.step(catalog.wait_directive_id)

        assert reward == pytest.approx(info["simulator_reward"])
        assert info["executor_status"] == "directive_executor"
        assert environment.metrics.gemma_calls == 0
        assert environment.metrics.fallbacks == 0
    finally:
        environment.close()
