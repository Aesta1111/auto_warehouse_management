from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    PpoPolicyOutput,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.evaluation import DemoDecision
from warehouse_ai.demo.contracts.inference import ActionerRequest, FailureCode
from warehouse_ai.demo.controllers import (
    ExecutorController,
    GemmaController,
)
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.evaluation.episode import run_episode
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.domain import JointAction
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


class FixedPolicy:
    def __init__(self, catalog: DirectiveCatalog, selected: int) -> None:
        self.catalog = catalog
        self.selected = selected

    def predict(self, context: WarehouseContext, mask: np.ndarray) -> PpoPolicyOutput:
        selected = self.selected if bool(mask[self.selected]) else 0
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


class FailingActioner:
    def __init__(self, code: FailureCode) -> None:
        self.code = code

    def act(self, request: ActionerRequest):  # type: ignore[no-untyped-def]
        del request
        raise ActionerError(self.code, "injected failure")


def _catalog() -> DirectiveCatalog:
    return DirectiveCatalog.load(ROOT / "configs/demo/directives-v1.yaml")


def _policy(catalog: DirectiveCatalog, selected: int) -> PpoPolicy:
    return cast(PpoPolicy, FixedPolicy(catalog, selected))


def test_episode_flushes_schema_v2_records(tmp_path: Path) -> None:
    catalog = _catalog()
    registry = build_registry(catalog)
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    try:
        result = run_episode(
            wrapper,
            ExecutorController(_policy(catalog, 1), registry),
            seed=0,
            max_steps=3,
            render=False,
            decision_log_path=tmp_path / "executor-v2.jsonl",
        )
    finally:
        wrapper.close()
    lines = (tmp_path / "executor-v2.jsonl").read_text(encoding="utf-8").splitlines()
    assert result.schema_version == 2
    assert result.steps == len(lines) == 3
    assert result.llm_calls == result.fallbacks == 0
    assert all('"schema_version":2' in line for line in lines)


def test_episode_counts_structured_failures_without_parsing_exception_names(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    registry = build_registry(catalog)
    for code, invalid, effect in (
        (FailureCode.MALFORMED_RESPONSE, 1, 0),
        (FailureCode.INVALID_ACTION, 1, 0),
        (FailureCode.STALE_STATE, 1, 0),
        (FailureCode.EFFECT_VIOLATION, 0, 1),
    ):
        wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
        try:
            controller = GemmaController(
                _policy(catalog, 1),
                registry,
                cast(Actioner, FailingActioner(code)),
            )
            result = run_episode(
                wrapper,
                controller,
                seed=0,
                max_steps=1,
                render=False,
                decision_log_path=tmp_path / f"{code.value}.jsonl",
            )
        finally:
            wrapper.close()
        assert result.llm_calls == result.fallbacks == 1
        assert result.invalid_actions == invalid
        assert result.effect_violations == effect


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

    def step(self, action: Any):  # type: ignore[no-untyped-def]
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


def test_episode_waits_for_all_agents_to_terminate(tmp_path: Path) -> None:
    catalog = _catalog()
    prediction = PpoPolicyOutput(
        state_id="state",
        episode_step=0,
        directive=catalog.by_id(0),
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
        directive_effect_valid=True,
    )
    wrapper = TwoStepWrapper()
    result = run_episode(
        cast(Any, wrapper),
        cast(Any, StubController(decision)),
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
