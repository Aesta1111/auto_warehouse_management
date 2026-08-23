"""One decision pipeline with replaceable action-source strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    PpoPolicyOutput,
)
from warehouse_ai.demo.contracts.evaluation import DemoDecision
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
)
from warehouse_ai.demo.directives import SemanticsError, Registry
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.domain import JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


class DemoController(Protocol):
    @property
    def controller_name(self) -> Literal["ppo_executor", "ppo_gemma"]: ...

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...


class DecisionFailure(RuntimeError):
    """A controller failure that is safe to expose in an artifact."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        result: ActionerResult | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.result = result


class ActionValidator:
    """Convert one untrusted destination vector into one trusted action."""

    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def validate(
        self,
        request: ActionerRequest,
        result: ActionerResult,
        wrapper: TarwareEnvWrapper,
    ) -> JointAction:
        current = WarehouseContext.capture(wrapper)
        if current.state_id != request.context.state_id:
            raise DecisionFailure(
                FailureCode.STALE_STATE,
                "warehouse state changed while inference was running",
                result,
            )

        destinations = result.response.destinations
        expected_count = len(current.observation.agents)
        if len(destinations) != expected_count:
            raise DecisionFailure(
                FailureCode.INVALID_ACTION,
                f"model returned {len(destinations)} destinations; expected {expected_count}",
                result,
            )
        try:
            action = wrapper.decode(destinations)
        except (InvalidJointActionError, TypeError, ValueError) as exc:
            raise DecisionFailure(
                FailureCode.INVALID_ACTION,
                "model destinations failed strict wrapper decoding",
                result,
            ) from exc

        expected_agents = tuple(agent.agent_id for agent in current.observation.agents)
        actual_agents = tuple(item.agent_id for item in action.actions)
        if actual_agents != expected_agents:
            raise DecisionFailure(
                FailureCode.INVALID_ACTION,
                "decoded action is not in canonical agent order",
                result,
            )
        try:
            self.registry.validate_effect(
                request.prediction.directive.directive_id,
                current,
                action,
            )
        except SemanticsError as exc:
            raise DecisionFailure(
                FailureCode.EFFECT_VIOLATION,
                "model action violates the selected directive effect",
                result,
            ) from exc
        return action


@dataclass(frozen=True)
class StrategyDecision:
    action: JointAction
    reason: str
    source: Literal["gemma", "directive_executor"]
    result: ActionerResult | None = None


class ActionStrategy(Protocol):
    def decide(
        self,
        context: WarehouseContext,
        prediction: PpoPolicyOutput,
        wrapper: TarwareEnvWrapper,
    ) -> StrategyDecision: ...


class ExecutorStrategy:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry

    def decide(
        self,
        context: WarehouseContext,
        prediction: PpoPolicyOutput,
        wrapper: TarwareEnvWrapper,
    ) -> StrategyDecision:
        try:
            action = self.registry.execute(
                prediction.directive.directive_id, context, wrapper
            )
        except Exception as exc:
            raise DecisionFailure(
                FailureCode.EXECUTOR_FAILURE,
                f"directive executor failed: {exc}",
            ) from exc
        return StrategyDecision(
            action=action,
            reason=prediction.directive.description,
            source="directive_executor",
        )


class GemmaStrategy:
    def __init__(self, registry: Registry, actioner: Actioner) -> None:
        self.registry = registry
        self.actioner = actioner
        self.validator = ActionValidator(registry)
        self.executor = ExecutorStrategy(registry)

    def decide(
        self,
        context: WarehouseContext,
        prediction: PpoPolicyOutput,
        wrapper: TarwareEnvWrapper,
    ) -> StrategyDecision:
        if prediction.directive.directive_id == self.registry.catalog.wait_directive_id:
            return self.executor.decide(context, prediction, wrapper)

        request = ActionerRequest(context=context, prediction=prediction)
        try:
            result = self.actioner.act(request)
        except ActionerError as exc:
            raise DecisionFailure(exc.code, str(exc)) from exc
        except Exception as exc:
            raise DecisionFailure(
                FailureCode.TRANSPORT,
                f"unexpected actioner failure: {exc}",
            ) from exc
        action = self.validator.validate(request, result, wrapper)
        return StrategyDecision(
            action=action,
            reason=result.response.reason,
            source="gemma",
            result=result,
        )


class DecisionPipeline:
    """Own policy prediction, action selection, and deterministic fallback."""

    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
        strategy: ActionStrategy,
    ) -> None:
        if policy.catalog.digest() != registry.catalog.digest():
            raise ValueError("policy catalog does not match directive registry")
        self.policy = policy
        self.registry = registry
        self.strategy = strategy

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision:
        context = WarehouseContext.capture(wrapper)
        mask = self.registry.action_mask(context)
        prediction = self.policy.predict(context, mask)
        self._validate_prediction(context, prediction, mask)
        try:
            selected = self.strategy.decide(context, prediction, wrapper)
        except DecisionFailure as failure:
            return self._fallback(prediction, wrapper, failure)
        return self._build_decision(prediction, selected)

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
        prediction: PpoPolicyOutput, selected: StrategyDecision
    ) -> DemoDecision:
        result = selected.result
        return DemoDecision(
            prediction=prediction,
            action=selected.action,
            reason=selected.reason,
            source=selected.source,
            model_id=None if result is None else result.model_id,
            raw_response=None if result is None else result.raw_text,
            directive_effect_valid=True,
            llm_latency_ms=None if result is None else result.latency_ms,
            prompt_tokens=None if result is None else result.prompt_tokens,
            completion_tokens=None if result is None else result.completion_tokens,
        )

    def _fallback(
        self,
        prediction: PpoPolicyOutput,
        wrapper: TarwareEnvWrapper,
        failure: DecisionFailure,
    ) -> DemoDecision:
        code = failure.code
        detail = str(failure)
        source: Literal["directive_executor", "automatic_no_op"]
        try:
            live_context = WarehouseContext.capture(wrapper)
            action = self.registry.execute(
                prediction.directive.directive_id, live_context, wrapper
            )
            source = "directive_executor"
        except Exception as exc:
            code = FailureCode.EXECUTOR_FAILURE
            detail = f"{detail}; deterministic executor failed: {exc}"
            try:
                action = wrapper.no_op()
            except Exception as no_op_error:
                raise RuntimeError(
                    "directive fallback and automatic no-op both failed"
                ) from no_op_error
            source = "automatic_no_op"
        result = failure.result
        return DemoDecision(
            prediction=prediction,
            action=action,
            reason=prediction.directive.description,
            source=source,
            model_id=None if result is None else result.model_id,
            raw_response=None if result is None else result.raw_text,
            directive_effect_valid=False,
            llm_latency_ms=None if result is None else result.latency_ms,
            failure_code=code,
            failure_detail=detail,
            prompt_tokens=None if result is None else result.prompt_tokens,
            completion_tokens=None if result is None else result.completion_tokens,
        )


class ExecutorController(DecisionPipeline):
    controller_name: Literal["ppo_executor"] = "ppo_executor"

    def __init__(self, policy: PpoPolicy, registry: Registry) -> None:
        super().__init__(policy, registry, ExecutorStrategy(registry))


class GemmaController(DecisionPipeline):
    controller_name: Literal["ppo_gemma"] = "ppo_gemma"

    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
        actioner: Actioner,
    ) -> None:
        super().__init__(policy, registry, GemmaStrategy(registry, actioner))
