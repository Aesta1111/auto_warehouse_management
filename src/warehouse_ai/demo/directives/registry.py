"""Shared engine for directive applicability, execution, and validation."""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.demo.contracts.directives import DirectiveContext, DirectiveVocabulary
from warehouse_ai.demo.directives.rules import (
    RULES,
    DirectiveRule,
    agent_is_eligible,
    eligible_destinations,
)
from warehouse_ai.domain import AgentRole, JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


BEHAVIOR_ENGINE_VERSION = "directive-rule-engine-v2"


class DirectiveFailure(str, Enum):
    UNKNOWN = "unknown"
    INAPPLICABLE = "inapplicable"
    EXECUTOR = "executor"
    EFFECT = "effect"


class DirectiveError(RuntimeError):
    def __init__(self, code: DirectiveFailure, message: str) -> None:
        super().__init__(message)
        self.code = code


class DirectiveRegistry:
    """Bind frozen vocabulary IDs to declarative rules."""

    def __init__(
        self, vocabulary: DirectiveVocabulary, rules: tuple[DirectiveRule, ...]
    ) -> None:
        self.vocabulary = vocabulary
        self._rules = {rule.behavior_key: rule for rule in rules}
        expected = {item.behavior_key for item in vocabulary.directives}
        if set(self._rules) != expected:
            raise ValueError("rule table does not match directive vocabulary")

    @property
    def engine_version(self) -> str:
        return BEHAVIOR_ENGINE_VERSION

    def _resolve(self, directive_id: int) -> DirectiveRule:
        try:
            definition = self.vocabulary.by_id(int(directive_id))
            return self._rules[definition.behavior_key]
        except (KeyError, TypeError, ValueError) as exc:
            raise DirectiveError(
                DirectiveFailure.UNKNOWN, f"unknown directive ID {directive_id!r}"
            ) from exc

    def _legal_candidates(
        self, rule: DirectiveRule, context: DirectiveContext, row: int
    ) -> tuple[int, ...]:
        destinations = eligible_destinations(rule, context)
        return tuple(
            int(item)
            for item in np.flatnonzero(context.action_mask.values[row])
            if int(item) != 0 and int(item) in destinations
        )

    def _proposal(
        self, rule: DirectiveRule, context: DirectiveContext
    ) -> tuple[int, ...]:
        if rule.effect.maximum == 0:
            return (0,) * len(context.observation.agents)
        result = [0] * len(context.observation.agents)
        selected: set[tuple[AgentRole, int]] = set()
        for row, agent in enumerate(context.observation.agents):
            if not agent_is_eligible(rule, context, row):
                continue
            for destination in self._legal_candidates(rule, context, row):
                pair = (agent.role, destination)
                if pair not in selected:
                    result[row] = destination
                    selected.add(pair)
                    break
        return tuple(result)

    def is_applicable(self, rule: DirectiveRule, context: DirectiveContext) -> bool:
        if rule.effect.minimum == 0:
            return True
        return any(
            agent_is_eligible(rule, context, row)
            and bool(self._legal_candidates(rule, context, row))
            for row in range(len(context.observation.agents))
        )

    def action_mask(self, context: DirectiveContext) -> NDArray[np.bool_]:
        mask = np.asarray(
            [
                self.is_applicable(self._rules[item.behavior_key], context)
                for item in self.vocabulary.directives
            ],
            dtype=np.bool_,
        )
        if not bool(mask[self.vocabulary.wait_directive_id]) or not bool(np.any(mask)):
            raise DirectiveError(
                DirectiveFailure.INAPPLICABLE,
                "wait and at least one directive must be applicable",
            )
        mask.setflags(write=False)
        return mask

    def behavior_digest(self) -> str:
        payload = {
            "engine_version": BEHAVIOR_ENGINE_VERSION,
            "rules": [
                self._rules[item.behavior_key].descriptor()
                for item in self.vocabulary.directives
            ],
        }
        return sha256_bytes(canonical_json_bytes(payload))

    def execute(
        self,
        directive_id: int,
        context: DirectiveContext,
        wrapper: TarwareEnvWrapper,
    ) -> JointAction:
        rule = self._resolve(directive_id)
        if not self.is_applicable(rule, context):
            raise DirectiveError(
                DirectiveFailure.INAPPLICABLE,
                f"directive {directive_id} is not applicable in state {context.state_id}",
            )
        try:
            action = wrapper.decode(self._proposal(rule, context))
        except (InvalidJointActionError, ValueError, TypeError) as exc:
            raise DirectiveError(
                DirectiveFailure.EXECUTOR,
                f"directive {directive_id} proposed an invalid joint action",
            ) from exc
        self.validate_effect(directive_id, context, action)
        return action

    def validate_effect(
        self, directive_id: int, context: DirectiveContext, action: JointAction
    ) -> None:
        rule = self._resolve(directive_id)
        if not self.is_applicable(rule, context):
            raise DirectiveError(
                DirectiveFailure.INAPPLICABLE, "directive is inapplicable"
            )
        expected = tuple(agent.agent_id for agent in context.observation.agents)
        if tuple(item.agent_id for item in action.actions) != expected:
            raise DirectiveError(
                DirectiveFailure.EFFECT, "action agents are not canonical"
            )
        eligible = eligible_destinations(rule, context)
        matches = sum(
            destination != 0
            and destination in eligible
            and agent_is_eligible(rule, context, row)
            and bool(context.action_mask.values[row, destination])
            for row, destination in enumerate(
                int(item.destination_id) for item in action.actions
            )
        )
        if not rule.effect.accepts(matches):
            raise DirectiveError(
                DirectiveFailure.EFFECT,
                f"action does not satisfy {rule.behavior_key} effect",
            )


def build_directive_registry(vocabulary: DirectiveVocabulary) -> DirectiveRegistry:
    return DirectiveRegistry(vocabulary, RULES)
