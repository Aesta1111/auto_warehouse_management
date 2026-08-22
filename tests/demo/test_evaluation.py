from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from warehouse_ai.demo.actioner import ActionerSchemaError
from warehouse_ai.demo.controllers import (
    DirectiveExecutorController,
    DirectiveGemmaController,
)
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.evaluation import run_episode
from warehouse_ai.demo.models import (
    ActionerRequest,
    ActionerResult,
    DemoDecision,
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
    SingleJointActionPlan,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
from warehouse_ai.domain import JointAction
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


class FixedPolicy:
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


class FailingActioner:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def act(
        self, request: ActionerRequest, wrapper: TarwareEnvWrapper
    ) -> ActionerResult[SingleJointActionPlan]:
        del request, wrapper
        raise self.error


def _vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


def _registry() -> DirectiveRegistry:
    return build_directive_registry(_vocabulary())


def test_run_episode_flushes_each_decision_and_uses_pinned_info_metrics(
    tmp_path: Path,
) -> None:
    vocabulary = _vocabulary()
    registry = build_directive_registry(vocabulary)
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    try:
        controller = DirectiveExecutorController(
            cast(DirectivePpoPolicy, FixedPolicy(vocabulary, selected_id=1)),
            registry,
        )
        log_path = tmp_path / "executor.jsonl"
        result = run_episode(
            wrapper,
            controller,
            seed=0,
            max_steps=3,
            render=False,
            decision_log_path=log_path,
        )
    finally:
        wrapper.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert result.controller == "ppo_executor"
    assert result.steps == 3
    assert result.llm_calls == 0
    assert result.fallbacks == 0
    assert len(lines) == result.steps
    assert all(line.startswith("{") for line in lines)


def test_run_episode_counts_failed_llm_response_as_fallback_and_invalid_action(
    tmp_path: Path,
) -> None:
    vocabulary = _vocabulary()
    registry = build_directive_registry(vocabulary)
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    try:
        controller = DirectiveGemmaController(
            cast(DirectivePpoPolicy, FixedPolicy(vocabulary, selected_id=1)),
            registry,
            FailingActioner(ActionerSchemaError("malformed")),  # type: ignore[arg-type]
        )
        result = run_episode(
            wrapper,
            controller,
            seed=0,
            max_steps=1,
            render=False,
            decision_log_path=tmp_path / "gemma.jsonl",
        )
    finally:
        wrapper.close()

    assert result.controller == "ppo_gemma"
    assert result.llm_calls == 1
    assert result.invalid_actions == 1
    assert result.fallbacks == 1
    assert result.effect_violations == 0


class StubController:
    controller_name = "ppo_executor"

    def __init__(self, decision: DemoDecision) -> None:
        self.decision = decision

    def choose_action(self, wrapper: Any) -> DemoDecision:
        del wrapper
        return self.decision


class TwoStepWrapper:
    def __init__(self) -> None:
        self.calls = 0
        self.render_calls = 0

    def reset(self, *, seed: int) -> None:
        assert seed == 9100

    def step(
        self, action: Any
    ) -> tuple[None, list[float], list[bool], list[bool], dict[str, int]]:
        del action
        self.calls += 1
        if self.calls == 1:
            return (
                None,
                [1.0, 2.0],
                [True, False],
                [False, False],
                {
                    "shelf_deliveries": 1,
                    "clashes": 2,
                    "stucks": 3,
                },
            )
        return (
            None,
            [3.0, 4.0],
            [True, True],
            [False, False],
            {
                "shelf_deliveries": 4,
                "clashes": 5,
                "stucks": 6,
            },
        )

    def render(self) -> None:
        self.render_calls += 1


def test_run_episode_waits_for_all_agents_to_terminate(tmp_path: Path) -> None:
    vocabulary = _vocabulary()
    prediction = DirectivePrediction(
        state_id="state",
        episode_step=0,
        directive=vocabulary.by_id(vocabulary.wait_directive_id),
        probabilities=(1.0, 0.0, 0.0, 0.0),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256="0" * 64,
    )
    decision = DemoDecision(
        prediction=prediction,
        action=JointAction(actions=()),
        reason="wait",
        source="directive_executor",
        model_id=None,
        raw_response=None,
        directive_effect_valid=True,
        llm_latency_ms=None,
        fallback_reason=None,
    )
    wrapper = TwoStepWrapper()
    result = run_episode(
        wrapper,  # type: ignore[arg-type]
        StubController(decision),  # type: ignore[arg-type]
        seed=9100,
        max_steps=3,
        render=True,
        decision_log_path=tmp_path / "stub.jsonl",
    )

    assert result.steps == 2
    assert result.completed_tasks == 5
    assert result.clashes == 7
    assert result.stuck_events == 9
    assert result.total_return == 10.0
    assert wrapper.render_calls == 2
