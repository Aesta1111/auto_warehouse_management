from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    PpoPolicyOutput,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
    LlmActionResponse,
)
from warehouse_ai.demo.controllers import (
    ExecutorController,
    LlmController,
)
from warehouse_ai.demo.directives import Registry, build_registry
from warehouse_ai.demo.execution import DirectiveExecutionService
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def catalog() -> DirectiveCatalog:
    return DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")


@pytest.fixture
def registry(catalog: DirectiveCatalog) -> Registry:
    return build_registry(catalog)


class FakePolicy:
    def __init__(self, catalog: DirectiveCatalog, selected_id: int) -> None:
        self.catalog = catalog
        self.selected_id = selected_id

    def predict(self, context: WarehouseContext, mask: np.ndarray) -> PpoPolicyOutput:
        selected = self.selected_id if bool(mask[self.selected_id]) else 0
        probabilities = [0.0] * self.catalog.size
        probabilities[selected] = 1.0
        return PpoPolicyOutput(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.catalog.by_id(selected),
            probabilities=tuple(probabilities),
            value_estimate=0.0,
            policy_latency_ms=0.0,
            checkpoint_sha256="0" * 64,
        )


class FakeActioner:
    def __init__(
        self,
        wrapper: TarwareEnvWrapper,
        registry: Registry,
        mode: str = "valid",
    ) -> None:
        self.wrapper = wrapper
        self.registry = registry
        self.mode = mode
        self.called = False

    def act(self, request: ActionerRequest) -> ActionerResult:
        self.called = True
        code_by_mode = {
            "health": FailureCode.HEALTH,
            "timeout": FailureCode.TIMEOUT,
            "transport": FailureCode.TRANSPORT,
            "malformed": FailureCode.MALFORMED_RESPONSE,
        }
        if self.mode in code_by_mode:
            raise ActionerError(code_by_mode[self.mode], f"injected {self.mode}")
        action = self.registry.execute(
            request.directive.directive_id,
            request.context,
            self.wrapper,
        )
        destinations = tuple(item.destination_id for item in action.actions)
        if self.mode == "invalid_length":
            destinations = (0,)
        elif self.mode == "invalid_action":
            destinations = (999, *destinations[1:])
        elif self.mode == "effect":
            destinations = (0,) * len(destinations)
        elif self.mode == "stale":
            self.wrapper.step(self.wrapper.no_op())
        return ActionerResult(
            response=LlmActionResponse(
                destinations=destinations,
                reason="fake action",
            ),
            raw_text="{}",
            model_id="fake-model",
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=4,
        )


class BrokenRegistry(Registry):
    def execute(self, directive_id, context, wrapper):  # type: ignore[no-untyped-def,override]
        del directive_id, context, wrapper
        raise RuntimeError("injected executor failure")


def _wrapper() -> TarwareEnvWrapper:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    return wrapper


def _policy(catalog: DirectiveCatalog, selected: int) -> PpoPolicy:
    return cast(PpoPolicy, FakePolicy(catalog, selected))


def test_valid_model_output_matches_executor_and_preserves_provenance(
    catalog: DirectiveCatalog, registry: Registry
) -> None:
    executor_wrapper = _wrapper()
    llm_wrapper = _wrapper()
    try:
        policy = _policy(catalog, 1)
        expected = ExecutorController(policy, registry).choose_action(executor_wrapper)
        actual = LlmController(
            policy,
            registry,
            cast(Actioner, FakeActioner(llm_wrapper, registry)),
        ).choose_action(llm_wrapper)
        assert actual.action == expected.action
        assert actual.source == "llm"
        assert actual.failure_code is None
        assert actual.prompt_tokens == 10
        assert actual.completion_tokens == 4
    finally:
        executor_wrapper.close()
        llm_wrapper.close()


@pytest.mark.parametrize(
    ("mode", "code"),
    [
        ("health", FailureCode.HEALTH),
        ("timeout", FailureCode.TIMEOUT),
        ("transport", FailureCode.TRANSPORT),
        ("malformed", FailureCode.MALFORMED_RESPONSE),
        ("invalid_length", FailureCode.INVALID_ACTION),
        ("invalid_action", FailureCode.INVALID_ACTION),
        ("effect", FailureCode.EFFECT_VIOLATION),
        ("stale", FailureCode.STALE_STATE),
    ],
)
def test_every_structured_model_failure_uses_deterministic_fallback(
    mode: str,
    code: FailureCode,
    catalog: DirectiveCatalog,
    registry: Registry,
) -> None:
    wrapper = _wrapper()
    try:
        controller = LlmController(
            _policy(catalog, 1),
            registry,
            cast(Actioner, FakeActioner(wrapper, registry, mode)),
        )
        decision = controller.choose_action(wrapper)
        assert decision.source == "directive_executor"
        assert decision.failure_code is code
        assert decision.failure_detail
        assert not decision.directive_effect_valid
        registry.validate_effect(1, WarehouseContext.capture(wrapper), decision.action)
    finally:
        wrapper.close()


def test_wait_bypasses_model(catalog: DirectiveCatalog, registry: Registry) -> None:
    wrapper = _wrapper()
    try:
        actioner = FakeActioner(wrapper, registry, "timeout")
        decision = LlmController(
            _policy(catalog, 0), registry, cast(Actioner, actioner)
        ).choose_action(wrapper)
        assert not actioner.called
        assert decision.source == "directive_executor"
        assert decision.failure_code is None
        assert decision.action == wrapper.no_op()
    finally:
        wrapper.close()


def test_shared_execution_accepts_directive_without_frozen_policy_output(
    catalog: DirectiveCatalog,
    registry: Registry,
) -> None:
    wrapper = _wrapper()
    try:
        actioner = FakeActioner(wrapper, registry)
        context = WarehouseContext.capture(wrapper)
        outcome = DirectiveExecutionService(
            registry,
            cast(Actioner, actioner),
        ).execute(context, catalog.by_id(1), wrapper)

        assert actioner.called
        assert outcome.source == "llm"
        assert outcome.failure_code is None
        registry.validate_effect(1, context, outcome.action)
    finally:
        wrapper.close()


def test_executor_failure_has_one_code_and_automatic_noop(
    catalog: DirectiveCatalog,
) -> None:
    wrapper = _wrapper()
    broken = BrokenRegistry(catalog)
    try:
        decision = ExecutorController(_policy(catalog, 1), broken).choose_action(
            wrapper
        )
        assert decision.source == "automatic_no_op"
        assert decision.failure_code is FailureCode.EXECUTOR_FAILURE
        assert decision.action == wrapper.no_op()
    finally:
        wrapper.close()
