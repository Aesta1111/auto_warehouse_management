"""PPO prediction with shared directive execution strategies."""

from __future__ import annotations

from typing import Literal, Protocol

from warehouse_ai.demo.contracts.directives import WarehouseContext, PpoPolicyOutput
from warehouse_ai.demo.contracts.evaluation import DemoDecision
from warehouse_ai.demo.directives import Registry
from warehouse_ai.demo.execution import (
    ActionValidator,
    DecisionFailure,
    DirectiveExecutionService,
    ExecutionOutcome,
)
from warehouse_ai.demo.inference.actioner import Actioner
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.env import TarwareEnvWrapper

__all__ = [
    "ActionValidator",
    "DecisionFailure",
    "DemoController",
    "ExecutorController",
    "LlmController",
]


class DemoController(Protocol):
    @property
    def controller_name(self) -> Literal["ppo_executor", "ppo_llm"]: ...

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...


class DecisionPipeline:
    """Own policy prediction and one shared action execution service."""

    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
        execution: DirectiveExecutionService,
    ) -> None:
        if policy.catalog.digest() != registry.catalog.digest():
            raise ValueError("policy catalog does not match directive registry")
        self.policy = policy
        self.registry = registry
        self.execution = execution

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision:
        context = WarehouseContext.capture(wrapper)
        mask = self.registry.action_mask(context)
        prediction = self.policy.predict(context, mask)
        self._validate_prediction(context, prediction, mask)
        outcome = self.execution.execute(
            context,
            prediction.directive,
            wrapper,
            prediction=prediction,
        )
        return self._build_decision(prediction, outcome)

    @staticmethod
    def _validate_prediction(context, prediction, mask) -> None:
        if prediction.state_id != context.state_id:
            raise RuntimeError("PPO prediction state does not match its context")
        directive_id = prediction.directive.directive_id
        if (
            directive_id < 0
            or directive_id >= len(mask)
            or not bool(mask[directive_id])
        ):
            raise RuntimeError("PPO selected an unknown or inapplicable directive")

    @staticmethod
    def _build_decision(
        prediction: PpoPolicyOutput,
        outcome: ExecutionOutcome,
    ) -> DemoDecision:
        result = outcome.result
        return DemoDecision(
            prediction=prediction,
            action=outcome.action,
            reason=outcome.reason,
            source=outcome.source,
            model_id=None if result is None else result.model_id,
            raw_response=None if result is None else result.raw_text,
            directive_effect_valid=outcome.directive_effect_valid,
            llm_latency_ms=None if result is None else result.latency_ms,
            failure_code=outcome.failure_code,
            failure_detail=outcome.failure_detail,
            prompt_tokens=None if result is None else result.prompt_tokens,
            completion_tokens=None if result is None else result.completion_tokens,
        )


class ExecutorController(DecisionPipeline):
    controller_name: Literal["ppo_executor"] = "ppo_executor"

    def __init__(self, policy: PpoPolicy, registry: Registry) -> None:
        super().__init__(policy, registry, DirectiveExecutionService(registry))


class LlmController(DecisionPipeline):
    controller_name: Literal["ppo_llm"] = "ppo_llm"

    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
        actioner: Actioner,
    ) -> None:
        super().__init__(
            policy,
            registry,
            DirectiveExecutionService(registry, actioner),
        )
