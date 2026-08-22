from __future__ import annotations

from pathlib import Path
from typing import cast
import numpy as np
import pytest

from warehouse_ai.demo.actioner import (
    ActionerSchemaError,
    ActionerTimeoutError,
)
from warehouse_ai.demo.controllers import (
    DirectiveExecutorController,
    DirectiveGemmaController,
)
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.models import (
    ActionerRequest,
    DemoStepRecord,
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
    ActionerResult,
    SingleJointActionPlan,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
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
    """Test-only deterministic policy implementing the public prediction seam."""

    def __init__(self, vocabulary: DirectiveVocabulary, selected_id: int) -> None:
        self.vocabulary = vocabulary
        self.selected_id = selected_id

    def predict(
        self, context: DirectiveContext, mask: np.ndarray
    ) -> DirectivePrediction:
        selected_id = self.selected_id
        if not bool(mask[selected_id]):
            selected_id = self.vocabulary.wait_directive_id
        probabilities = [0.0] * self.vocabulary.size
        probabilities[selected_id] = 1.0
        return DirectivePrediction(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.vocabulary.by_id(selected_id),
            probabilities=tuple(probabilities),
            value_estimate=0.0,
            policy_latency_ms=0.0,
            checkpoint_sha256="0" * 64,
        )


class FakeActioner:
    """Test-only actioner used to exercise the generic Phase 4 protocol."""

    def __init__(self, registry: DirectiveRegistry, mode: str = "valid") -> None:
        self.registry = registry
        self.mode = mode
        self.called = False

    def act(
        self, request: ActionerRequest, wrapper: TarwareEnvWrapper
    ) -> ActionerResult[SingleJointActionPlan]:
        self.called = True
        if self.mode == "timeout":
            raise ActionerTimeoutError("fake response timed out")
        if self.mode == "malformed":
            raise ActionerSchemaError("fake response was malformed")

        action = self.registry.execute(
            request.prediction.directive.directive_id, request.context, wrapper
        )
        if self.mode == "invalid":
            # A valid Pydantic plan containing a wrong action shape must still be
            # rejected by the controller's wrapper validation.
            action = action.model_copy(update={"actions": ()})
        elif self.mode == "effect":
            action = wrapper.no_op()

        state_id = "stale" if self.mode == "stale" else request.context.state_id
        plan = SingleJointActionPlan(
            state_id=state_id,
            directive_id=request.prediction.directive.directive_id,
            action=action,
            reason="fake action",
        )
        return ActionerResult[SingleJointActionPlan](
            plan=plan,
            raw_text='{"destinations":[1,0,0,0,0],"reason":"fake action"}',
            model_id="fake-actioner",
            latency_ms=1.0,
        )


class BrokenExecutorRegistry(DirectiveRegistry):
    def execute(
        self, directive_id: int, context: DirectiveContext, wrapper: TarwareEnvWrapper
    ):  # type: ignore[override]
        del directive_id, context, wrapper
        raise RuntimeError("injected executor failure")


def _wrapper() -> TarwareEnvWrapper:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    return wrapper


def _policy(vocabulary: DirectiveVocabulary, selected_id: int) -> DirectivePpoPolicy:
    return cast(DirectivePpoPolicy, FakePolicy(vocabulary, selected_id))


def test_executor_and_fake_actioner_choose_same_frozen_directive(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    executor_wrapper = _wrapper()
    actioner_wrapper = _wrapper()
    try:
        policy = _policy(vocabulary, selected_id=1)
        executor = DirectiveExecutorController(policy, registry)
        actioner = FakeActioner(registry)
        gemma = DirectiveGemmaController(policy, registry, actioner)

        executor_decision = executor.choose_action(executor_wrapper)
        gemma_decision = gemma.choose_action(actioner_wrapper)

        assert (
            executor_decision.prediction.directive
            == gemma_decision.prediction.directive
        )
        assert executor_decision.action == gemma_decision.action
        assert gemma_decision.source == "gemma"
        assert gemma_decision.directive_effect_valid
    finally:
        executor_wrapper.close()
        actioner_wrapper.close()


def test_valid_fake_plan_reaches_wrapper_unchanged(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    wrapper = _wrapper()
    try:
        actioner = FakeActioner(registry)
        controller = DirectiveGemmaController(
            _policy(vocabulary, selected_id=1), registry, actioner
        )
        decision = controller.choose_action(wrapper)
        _, rewards, terminated, truncated, simulator_info = wrapper.step(
            decision.action
        )
        record = DemoStepRecord(
            decision=decision,
            rewards=tuple(float(value) for value in rewards),
            terminated=tuple(bool(value) for value in terminated),
            truncated=tuple(bool(value) for value in truncated),
            simulator_info=simulator_info,
        )

        assert actioner.called
        assert decision.source == "gemma"
        assert record.decision.action == decision.action
        assert all(not value for value in terminated)
        assert all(not value for value in truncated)
    finally:
        wrapper.close()


@pytest.mark.parametrize("mode", ["malformed", "invalid", "effect", "timeout", "stale"])
def test_fake_actioner_failures_use_same_directive_executor_fallback(
    mode: str,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
) -> None:
    wrapper = _wrapper()
    try:
        controller = DirectiveGemmaController(
            _policy(vocabulary, selected_id=1),
            registry,
            FakeActioner(registry, mode=mode),
        )  # type: ignore[arg-type]
        before = DirectiveContext.capture(wrapper)
        decision = controller.choose_action(wrapper)
        expected = registry.execute(1, before, wrapper)
        _, rewards, terminated, truncated, simulator_info = wrapper.step(
            decision.action
        )
        record = DemoStepRecord(
            decision=decision,
            rewards=tuple(float(value) for value in rewards),
            terminated=tuple(bool(value) for value in terminated),
            truncated=tuple(bool(value) for value in truncated),
            simulator_info=simulator_info,
        )

        assert decision.source == "directive_executor"
        assert decision.action == expected
        assert not decision.directive_effect_valid
        assert decision.fallback_reason is not None
        assert record.decision.fallback_reason == decision.fallback_reason
    finally:
        wrapper.close()


def test_wait_bypasses_actioner(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    wrapper = _wrapper()
    try:
        actioner = FakeActioner(registry, mode="timeout")
        controller = DirectiveGemmaController(
            _policy(vocabulary, selected_id=vocabulary.wait_directive_id),
            registry,
            actioner,
        )  # type: ignore[arg-type]
        decision = controller.choose_action(wrapper)

        assert not actioner.called
        assert decision.source == "directive_executor"
        assert decision.action == wrapper.no_op()
        assert decision.directive_effect_valid
    finally:
        wrapper.close()


def test_failed_directive_fallback_uses_automatic_no_op(
    vocabulary: DirectiveVocabulary, registry: DirectiveRegistry
) -> None:
    broken_registry = BrokenExecutorRegistry(vocabulary, registry._behaviors)  # type: ignore[attr-defined]
    wrapper = _wrapper()
    try:
        controller = DirectiveGemmaController(
            _policy(vocabulary, selected_id=1),
            broken_registry,
            FakeActioner(registry, mode="timeout"),
        )  # type: ignore[arg-type]
        decision = controller.choose_action(wrapper)

        assert decision.source == "automatic_no_op"
        assert decision.action == wrapper.no_op()
        assert "executor failed" in (decision.fallback_reason or "")
    finally:
        wrapper.close()
