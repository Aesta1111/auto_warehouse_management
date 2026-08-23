"""Shared directive execution for evaluation and Gemma-in-the-loop training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from warehouse_ai.demo.contracts.directives import (
    Directive,
    WarehouseContext,
    PpoPolicyOutput,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
)
from warehouse_ai.demo.directives import SemanticsError, Registry
from warehouse_ai.demo.inference.actioner import Actioner, ActionerError
from warehouse_ai.domain import JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


class DecisionFailure(RuntimeError):
    """A directive execution failure that is safe to record."""

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
        expected = tuple(agent.agent_id for agent in current.observation.agents)
        actual = tuple(item.agent_id for item in action.actions)
        if actual != expected:
            raise DecisionFailure(
                FailureCode.INVALID_ACTION,
                "decoded action is not in canonical agent order",
                result,
            )
        try:
            self.registry.validate_effect(
                request.directive.directive_id, current, action
            )
        except SemanticsError as exc:
            raise DecisionFailure(
                FailureCode.EFFECT_VIOLATION,
                "model action violates the selected directive effect",
                result,
            ) from exc
        return action


@dataclass(frozen=True)
class ExecutionOutcome:
    action: JointAction
    reason: str
    source: Literal["gemma", "directive_executor", "automatic_no_op"]
    directive_effect_valid: bool
    result: ActionerResult | None = None
    failure_code: FailureCode | None = None
    failure_detail: str | None = None


class DirectiveExecutionService:
    """Resolve a directive through Gemma or the deterministic executor."""

    def __init__(self, registry: Registry, actioner: Actioner | None = None) -> None:
        self.registry = registry
        self.actioner = actioner
        self.validator = ActionValidator(registry)

    def execute(
        self,
        context: WarehouseContext,
        directive: Directive,
        wrapper: TarwareEnvWrapper,
        *,
        prediction: PpoPolicyOutput | None = None,
    ) -> ExecutionOutcome:
        if self.actioner is None or directive.directive_id == (
            self.registry.catalog.wait_directive_id
        ):
            return self._execute_deterministically(context, directive, wrapper)
        request = (
            ActionerRequest(context=context, prediction=prediction)
            if prediction is not None
            else ActionerRequest(context=context, selected_directive=directive)
        )
        try:
            result = self.actioner.act(request)
            action = self.validator.validate(request, result, wrapper)
        except ActionerError as exc:
            return self._fallback(
                directive, wrapper, DecisionFailure(exc.code, str(exc))
            )
        except DecisionFailure as exc:
            return self._fallback(directive, wrapper, exc)
        except Exception as exc:
            return self._fallback(
                directive,
                wrapper,
                DecisionFailure(
                    FailureCode.TRANSPORT,
                    f"unexpected actioner failure: {exc}",
                ),
            )
        return ExecutionOutcome(
            action=action,
            reason=result.response.reason,
            source="gemma",
            directive_effect_valid=True,
            result=result,
        )

    def _execute_deterministically(
        self,
        context: WarehouseContext,
        directive: Directive,
        wrapper: TarwareEnvWrapper,
    ) -> ExecutionOutcome:
        try:
            action = self.registry.execute(directive.directive_id, context, wrapper)
        except Exception as exc:
            return self._fallback(
                directive,
                wrapper,
                DecisionFailure(
                    FailureCode.EXECUTOR_FAILURE,
                    f"directive executor failed: {exc}",
                ),
            )
        return ExecutionOutcome(
            action=action,
            reason=directive.description,
            source="directive_executor",
            directive_effect_valid=True,
        )

    def _fallback(
        self,
        directive: Directive,
        wrapper: TarwareEnvWrapper,
        failure: DecisionFailure,
    ) -> ExecutionOutcome:
        code = failure.code
        detail = str(failure)
        try:
            live_context = WarehouseContext.capture(wrapper)
            action = self.registry.execute(
                directive.directive_id, live_context, wrapper
            )
            source: Literal["directive_executor", "automatic_no_op"] = (
                "directive_executor"
            )
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
        return ExecutionOutcome(
            action=action,
            reason=directive.description,
            source=source,
            directive_effect_valid=False,
            result=failure.result,
            failure_code=code,
            failure_detail=detail,
        )
