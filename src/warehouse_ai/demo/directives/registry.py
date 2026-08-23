"""Shared engine for directive applicability, execution, and validation."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.demo.contracts.directives import (
    Directive,
    DirectiveCatalog,
    WarehouseContext,
)
from warehouse_ai.demo.directives.rules import agent_is_eligible, eligible_destinations
from warehouse_ai.domain import AgentRole, JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


BEHAVIOR_ENGINE_VERSION = "directive-rule-engine-v2"


class SemanticsFailure(str, Enum):
    UNKNOWN = "unknown"
    INAPPLICABLE = "inapplicable"
    EXECUTOR = "executor"
    EFFECT = "effect"


class SemanticsError(RuntimeError):
    def __init__(self, code: SemanticsFailure, message: str) -> None:
        super().__init__(message)
        self.code = code


class Registry:
    """Execute the complete directive semantics owned by one catalog."""

    def __init__(self, catalog: DirectiveCatalog) -> None:
        self.catalog = catalog

    @property
    def engine_version(self) -> str:
        return BEHAVIOR_ENGINE_VERSION

    def _resolve(self, directive_id: int) -> Directive:
        try:
            return self.catalog.by_id(int(directive_id))
        except (KeyError, TypeError, ValueError) as exc:
            raise SemanticsError(
                SemanticsFailure.UNKNOWN, f"unknown directive ID {directive_id!r}"
            ) from exc

    def _legal_candidates(
        self, directive: Directive, context: WarehouseContext, row: int
    ) -> tuple[int, ...]:
        destinations = eligible_destinations(directive, context)
        return tuple(
            int(item)
            for item in np.flatnonzero(context.action_mask.values[row])
            if int(item) != 0 and int(item) in destinations
        )

    def _proposal(
        self, directive: Directive, context: WarehouseContext
    ) -> tuple[int, ...]:
        if directive.effect.maximum == 0:
            return (0,) * len(context.observation.agents)
        result = [0] * len(context.observation.agents)
        selected: set[tuple[AgentRole, int]] = set()
        for row, agent in enumerate(context.observation.agents):
            if not agent_is_eligible(directive, context, row):
                continue
            for destination in self._legal_candidates(directive, context, row):
                pair = (agent.role, destination)
                if pair not in selected:
                    result[row] = destination
                    selected.add(pair)
                    break
        return tuple(result)

    def is_applicable(self, directive: Directive, context: WarehouseContext) -> bool:
        if directive.effect.minimum == 0:
            return True
        return any(
            agent_is_eligible(directive, context, row)
            and bool(self._legal_candidates(directive, context, row))
            for row in range(len(context.observation.agents))
        )

    def action_mask(self, context: WarehouseContext) -> NDArray[np.bool_]:
        mask = np.asarray(
            [self.is_applicable(item, context) for item in self.catalog.directives],
            dtype=np.bool_,
        )
        if not bool(mask[self.catalog.wait_directive_id]) or not bool(np.any(mask)):
            raise SemanticsError(
                SemanticsFailure.INAPPLICABLE,
                "wait and at least one directive must be applicable",
            )
        mask.setflags(write=False)
        return mask

    def behavior_digest(self) -> str:
        payload = {
            "engine_version": BEHAVIOR_ENGINE_VERSION,
            "rules": [item.descriptor() for item in self.catalog.directives],
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def execute(
        self,
        directive_id: int,
        context: WarehouseContext,
        wrapper: TarwareEnvWrapper,
    ) -> JointAction:
        directive = self._resolve(directive_id)
        if not self.is_applicable(directive, context):
            raise SemanticsError(
                SemanticsFailure.INAPPLICABLE,
                f"directive {directive_id} is not applicable in state {context.state_id}",
            )
        try:
            action = wrapper.decode(self._proposal(directive, context))
        except (InvalidJointActionError, ValueError, TypeError) as exc:
            raise SemanticsError(
                SemanticsFailure.EXECUTOR,
                f"directive {directive_id} proposed an invalid joint action",
            ) from exc
        self.validate_effect(directive_id, context, action)
        return action

    def validate_effect(
        self, directive_id: int, context: WarehouseContext, action: JointAction
    ) -> None:
        directive = self._resolve(directive_id)
        if not self.is_applicable(directive, context):
            raise SemanticsError(
                SemanticsFailure.INAPPLICABLE, "directive is inapplicable"
            )
        expected = tuple(agent.agent_id for agent in context.observation.agents)
        if tuple(item.agent_id for item in action.actions) != expected:
            raise SemanticsError(
                SemanticsFailure.EFFECT, "action agents are not canonical"
            )
        eligible = eligible_destinations(directive, context)
        matches = sum(
            destination != 0
            and destination in eligible
            and agent_is_eligible(directive, context, row)
            and bool(context.action_mask.values[row, destination])
            for row, destination in enumerate(
                int(item.destination_id) for item in action.actions
            )
        )
        if not directive.effect.accepts(matches):
            raise SemanticsError(
                SemanticsFailure.EFFECT,
                f"action does not satisfy {directive.behavior_key} effect",
            )


def build_registry(catalog: DirectiveCatalog) -> Registry:
    return Registry(catalog)
