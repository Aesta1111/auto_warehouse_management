"""Actioner protocol and errors for the actioner-neutral directive demo.

The real ``llama.cpp`` client belongs to Phase 5.  Keeping this module free of
HTTP imports lets Phase 4 exercise the complete orchestration with a test-only
fake actioner.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from warehouse_ai.demo.models import ActionerRequest, ActionerResult
from warehouse_ai.env import TarwareEnvWrapper


PlanT = TypeVar("PlanT")


class Actioner(Protocol[PlanT]):
    """Produce a typed plan from one immutable controller request."""

    def act(
        self,
        request: ActionerRequest,
        wrapper: TarwareEnvWrapper,
    ) -> ActionerResult[PlanT]: ...


class ActionerError(RuntimeError):
    """Base error for failures crossing the actioner boundary."""


class ActionerHealthError(ActionerError):
    """The actioner endpoint is unavailable or advertises the wrong model."""


class ActionerTimeoutError(ActionerError):
    """The actioner did not produce a response within its configured budget."""


class ActionerHttpError(ActionerError):
    """The actioner transport returned an unsuccessful response."""


class EmptyActionerResponseError(ActionerError):
    """The actioner returned no usable response content."""


class ActionerSchemaError(ActionerError):
    """The actioner response did not satisfy the expected plan schema."""


class ActionerDecodeError(ActionerError):
    """The proposed destinations could not pass wrapper decoding."""


class ActionerEffectError(ActionerError):
    """The proposed action contradicted the selected directive effect."""


class StaleActionPlanError(ActionerError):
    """The visible state changed while an actioner response was in flight."""
