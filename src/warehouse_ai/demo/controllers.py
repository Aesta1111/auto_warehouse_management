"""Frozen-directive controllers for the Phase 4 orchestration seam."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from warehouse_ai.demo.actioner import (
    Actioner,
    ActionerDecodeError,
    ActionerEffectError,
    ActionerError,
    ActionerSchemaError,
    StaleActionPlanError,
)
from warehouse_ai.demo.directives import (
    DirectiveError,
    DirectiveRegistry,
)
from warehouse_ai.demo.models import (
    ActionerRequest,
    ActionerResult,
    DemoDecision,
    DirectiveContext,
    DirectivePrediction,
    SingleJointActionPlan,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


class DemoController(Protocol):
    """Common action-selection operation used by the future episode runner."""

    @property
    def controller_name(self) -> Literal["ppo_executor", "ppo_gemma"]: ...

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...


def _validate_policy_registry(
    policy: DirectivePpoPolicy, registry: DirectiveRegistry
) -> None:
    if policy.vocabulary.digest() != registry.vocabulary.digest():
        raise ValueError("policy vocabulary does not match directive registry")


def _predict(
    policy: DirectivePpoPolicy,
    registry: DirectiveRegistry,
    wrapper: TarwareEnvWrapper,
) -> tuple[DirectiveContext, DirectivePrediction]:
    """Capture one context and make one deterministic masked PPO prediction."""
    context = DirectiveContext.capture(wrapper)
    directive_mask = registry.action_mask(context)
    prediction = policy.predict(context, directive_mask)
    if prediction.state_id != context.state_id:
        raise RuntimeError("PPO prediction state does not match its request context")
    directive_id = prediction.directive.directive_id
    if directive_id < 0 or directive_id >= registry.vocabulary.size:
        raise RuntimeError("PPO selected an unknown directive")
    if not bool(directive_mask[directive_id]):
        raise RuntimeError("PPO selected an inapplicable directive")
    return context, prediction


def _executor_decision(
    context: DirectiveContext,
    prediction: DirectivePrediction,
    registry: DirectiveRegistry,
    wrapper: TarwareEnvWrapper,
) -> DemoDecision:
    action = registry.execute(prediction.directive.directive_id, context, wrapper)
    return DemoDecision(
        prediction=prediction,
        action=action,
        reason=prediction.directive.description,
        source="directive_executor",
        model_id=None,
        raw_response=None,
        directive_effect_valid=True,
        llm_latency_ms=None,
        fallback_reason=None,
    )


def _error_text(error: BaseException) -> str:
    detail = str(error).strip()
    return type(error).__name__ if not detail else f"{type(error).__name__}: {detail}"


def _fallback_decision(
    *,
    prediction: DirectivePrediction,
    registry: DirectiveRegistry,
    wrapper: TarwareEnvWrapper,
    error: BaseException,
    actioner_result: ActionerResult[Any] | None,
) -> DemoDecision:
    """Use the selected directive's executor, retaining actioner failure data."""
    raw_response = None if actioner_result is None else actioner_result.raw_text
    model_id = None if actioner_result is None else actioner_result.model_id
    latency = None if actioner_result is None else actioner_result.latency_ms
    fallback_reason = _error_text(error)

    try:
        live_context = DirectiveContext.capture(wrapper)
        action = registry.execute(
            prediction.directive.directive_id, live_context, wrapper
        )
    except Exception as fallback_error:
        fallback_reason = (
            f"{fallback_reason}; deterministic executor failed: "
            f"{_error_text(fallback_error)}"
        )
        try:
            action = wrapper.no_op()
        except Exception as no_op_error:
            raise RuntimeError(
                "directive fallback and automatic no-op both failed"
            ) from no_op_error
        return DemoDecision(
            prediction=prediction,
            action=action,
            reason="Automatic no-op after actioner and directive fallback failure.",
            source="automatic_no_op",
            model_id=model_id,
            raw_response=raw_response,
            directive_effect_valid=False,
            llm_latency_ms=latency,
            fallback_reason=fallback_reason,
        )

    return DemoDecision(
        prediction=prediction,
        action=action,
        reason=prediction.directive.description,
        source="directive_executor",
        model_id=model_id,
        raw_response=raw_response,
        directive_effect_valid=False,
        llm_latency_ms=latency,
        fallback_reason=fallback_reason,
    )


class DirectiveExecutorController:
    """Apply the selected frozen directive through its deterministic executor."""

    def __init__(self, policy: DirectivePpoPolicy, registry: DirectiveRegistry) -> None:
        _validate_policy_registry(policy, registry)
        self.policy = policy
        self.registry = registry

    @property
    def controller_name(self) -> Literal["ppo_executor"]:
        return "ppo_executor"

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision:
        context, prediction = _predict(self.policy, self.registry, wrapper)
        return _executor_decision(context, prediction, self.registry, wrapper)


class DirectiveGemmaController:
    """Delegate one selected directive to an actioner with strict fallback."""

    def __init__(
        self,
        policy: DirectivePpoPolicy,
        registry: DirectiveRegistry,
        actioner: Actioner[SingleJointActionPlan],
    ) -> None:
        _validate_policy_registry(policy, registry)
        self.policy = policy
        self.registry = registry
        self.actioner = actioner

    @property
    def controller_name(self) -> Literal["ppo_gemma"]:
        return "ppo_gemma"

    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision:
        context, prediction = _predict(self.policy, self.registry, wrapper)
        directive_id = prediction.directive.directive_id

        if directive_id == self.registry.vocabulary.wait_directive_id:
            try:
                action = wrapper.no_op()
                self.registry.validate_effect(directive_id, context, action)
            except Exception as error:
                return _fallback_decision(
                    prediction=prediction,
                    registry=self.registry,
                    wrapper=wrapper,
                    error=error,
                    actioner_result=None,
                )
            return DemoDecision(
                prediction=prediction,
                action=action,
                reason=prediction.directive.description,
                source="directive_executor",
                model_id=None,
                raw_response=None,
                directive_effect_valid=True,
                llm_latency_ms=None,
                fallback_reason=None,
            )

        request = ActionerRequest(context=context, prediction=prediction)
        actioner_result: ActionerResult[Any] | None = None
        try:
            actioner_result = self.actioner.act(request, wrapper)
            if not isinstance(actioner_result, ActionerResult):
                raise ActionerSchemaError("actioner did not return an ActionerResult")
            if not isinstance(actioner_result.plan, SingleJointActionPlan):
                raise ActionerSchemaError(
                    "actioner result did not contain a SingleJointActionPlan"
                )
            plan = actioner_result.plan
            if plan.state_id != context.state_id:
                raise StaleActionPlanError(
                    "actioner plan state differs from request state"
                )
            if plan.directive_id != directive_id:
                raise ActionerEffectError(
                    "actioner plan directive differs from the selected PPO directive"
                )

            current_context = DirectiveContext.capture(wrapper)
            if current_context.state_id != context.state_id:
                raise StaleActionPlanError(
                    "warehouse state changed while the actioner was responding"
                )

            destinations = tuple(
                int(item.destination_id) for item in plan.action.actions
            )
            try:
                decoded = wrapper.decode(destinations)
            except (InvalidJointActionError, ValueError, TypeError) as error:
                raise ActionerDecodeError(
                    "actioner plan did not pass strict wrapper decoding"
                ) from error
            if decoded != plan.action:
                raise ActionerDecodeError(
                    "actioner plan action is not in canonical agent order"
                )
            try:
                self.registry.validate_effect(
                    directive_id, current_context, plan.action
                )
            except DirectiveError as error:
                raise ActionerEffectError(
                    "actioner plan violates the selected directive effect"
                ) from error

            return DemoDecision(
                prediction=prediction,
                action=plan.action,
                reason=plan.reason,
                source="gemma",
                model_id=actioner_result.model_id,
                raw_response=actioner_result.raw_text,
                directive_effect_valid=True,
                llm_latency_ms=actioner_result.latency_ms,
                fallback_reason=None,
            )
        except ActionerError as error:
            return _fallback_decision(
                prediction=prediction,
                registry=self.registry,
                wrapper=wrapper,
                error=error,
                actioner_result=actioner_result,
            )
        except Exception as error:
            # A third-party actioner is outside this module's control. Keep an
            # unexpected fake/client failure visible while preserving the live
            # demo's deterministic fallback contract.
            wrapped = ActionerError("unexpected actioner failure")
            wrapped.__cause__ = error
            return _fallback_decision(
                prediction=prediction,
                registry=self.registry,
                wrapper=wrapper,
                error=wrapped,
                actioner_result=actioner_result,
            )
