from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
    GemmaActionResponse,
)
from warehouse_ai.demo.controllers import (
    DirectiveExecutorController,
    DirectiveGemmaController,
)
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.directives.rules import RULES
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.demo.ppo.policy import DirectivePpoPolicy
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def registry(vocabulary: DirectiveVocabulary) -> DirectiveRegistry:
    return build_directive_registry(vocabulary)


class FakePolicy:
    def __init__(self, vocabulary: DirectiveVocabulary, selected_id: int) -> None:
        self.vocabulary = vocabulary
        self.selected_id = selected_id

    def predict(
        self, context: DirectiveContext, mask: np.ndarray
    ) -> DirectivePrediction:
        selected = self.selected_id if bool(mask[self.selected_id]) else 0
        probabilities = [0.0] * self.vocabulary.size
        probabilities[selected] = 1.0
        return DirectivePrediction(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.vocabulary.by_id(selected),
            probabilities=tuple(probabilities),
            value_estimate=0.0,
            policy_latency_ms=0.0,
            checkpoint_sha256="0" * 64,
        )


class FakeActioner:
    def __init__(
        self,
        wrapper: TarwareEnvWrapper,
        registry: DirectiveRegistry,
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
            request.prediction.directive.directive_id,
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
            response=GemmaActionResponse(
                destinations=destinations,
                reason="fake action",
            ),
            raw_text="{}",
            model_id="fake-model",
            latency_ms=1.0,
            prompt_tokens=10,
            completion_tokens=4,
        )


class BrokenRegistry(DirectiveRegistry):
    def execute(self, directive_id, context, wrapper):  # type: ignore[no-untyped-def,override]
        del directive_id, context, wrapper
        raise RuntimeError("injected executor failure")


def _wrapper() -> TarwareEnvWrapper:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    return wrapper


def _policy(vocabulary: DirectiveVocabulary, selected: int) -> DirectivePpoPolicy:
    return cast(DirectivePpoPolicy, FakePolicy(vocabulary, selected))


def test_valid_model_output_matches_executor_and_preserves_provenance(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    executor_wrapper = _wrapper()
    gemma_wrapper = _wrapper()
    try:
        policy = _policy(vocabulary, 1)
        expected = DirectiveExecutorController(policy, registry).choose_action(
            executor_wrapper
        )
        actual = DirectiveGemmaController(
            policy,
            registry,
            cast(Actioner, FakeActioner(gemma_wrapper, registry)),
        ).choose_action(gemma_wrapper)
        assert actual.action == expected.action
        assert actual.source == "gemma"
        assert actual.failure_code is None
        assert actual.prompt_tokens == 10
        assert actual.completion_tokens == 4
    finally:
        executor_wrapper.close()
        gemma_wrapper.close()


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
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
) -> None:
    wrapper = _wrapper()
    try:
        controller = DirectiveGemmaController(
            _policy(vocabulary, 1),
            registry,
            cast(Actioner, FakeActioner(wrapper, registry, mode)),
        )
        decision = controller.choose_action(wrapper)
        assert decision.source == "directive_executor"
        assert decision.failure_code is code
        assert decision.failure_detail
        assert not decision.directive_effect_valid
        registry.validate_effect(1, DirectiveContext.capture(wrapper), decision.action)
    finally:
        wrapper.close()


def test_wait_bypasses_model(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    wrapper = _wrapper()
    try:
        actioner = FakeActioner(wrapper, registry, "timeout")
        decision = DirectiveGemmaController(
            _policy(vocabulary, 0), registry, cast(Actioner, actioner)
        ).choose_action(wrapper)
        assert not actioner.called
        assert decision.source == "directive_executor"
        assert decision.failure_code is None
        assert decision.action == wrapper.no_op()
    finally:
        wrapper.close()


def test_executor_failure_has_one_code_and_automatic_noop(
    vocabulary: DirectiveVocabulary,
) -> None:
    wrapper = _wrapper()
    broken = BrokenRegistry(vocabulary, RULES)
    try:
        decision = DirectiveExecutorController(
            _policy(vocabulary, 1), broken
        ).choose_action(wrapper)
        assert decision.source == "automatic_no_op"
        assert decision.failure_code is FailureCode.EXECUTOR_FAILURE
        assert decision.action == wrapper.no_op()
    finally:
        wrapper.close()
