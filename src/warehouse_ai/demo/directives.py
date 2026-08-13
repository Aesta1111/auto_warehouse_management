"""Directive semantics and deterministic concrete-action executors.

The directive policy owns only a small discrete vocabulary.  This module is the
closed, reviewed mapping from those vocabulary entries to controller-visible
rules and deterministic destination proposals.  It deliberately has no PPO,
prompt, HTTP, or controller dependencies.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.demo.models import DirectiveContext, DirectiveVocabulary
from warehouse_ai.domain import AgentRole, JointAction
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


class DirectiveError(RuntimeError):
    """Base class for closed-vocabulary semantic and executor failures."""


class UnknownDirectiveError(DirectiveError):
    """Raised when a caller selects an ID outside the frozen vocabulary."""


class InapplicableDirectiveError(DirectiveError):
    """Raised when a valid directive is masked in the current state."""


class DirectiveExecutorError(DirectiveError):
    """Raised when a deterministic executor cannot produce a valid action."""


class DirectiveEffectError(DirectiveError):
    """Raised when an action contradicts the selected directive's effect rule."""


@runtime_checkable
class DirectiveBehavior(Protocol):
    """Semantic and deterministic execution contract for one directive."""

    @property
    def behavior_key(self) -> str: ...

    def is_applicable(self, context: DirectiveContext) -> bool: ...

    def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]: ...

    def validate_effect(
        self, context: DirectiveContext, action: JointAction
    ) -> None: ...


def _requested_shelf_ids(context: DirectiveContext) -> frozenset[int]:
    return frozenset(
        int(shelf.destination_id)
        for shelf in context.observation.shelves
        if shelf.requested
    )


def _legal_candidates(
    context: DirectiveContext,
    row: int,
    eligible_destinations: frozenset[int],
) -> tuple[int, ...]:
    """Return legal nonzero destinations in deterministic numeric order."""
    return tuple(
        int(destination)
        for destination in np.flatnonzero(context.action_mask.values[row])
        if int(destination) != 0 and int(destination) in eligible_destinations
    )


def _all_rows(_row: int) -> bool:
    return True


def _propose_for(
    context: DirectiveContext,
    eligible_destinations: frozenset[int],
    row_is_eligible: Callable[[int], bool] | None = None,
) -> tuple[int, ...]:
    """Assign the lowest available destination to each eligible agent.

    Same-role duplicate destinations are avoided here so the proposal can pass
    the wrapper's strict joint-action validator without compatibility repairs.
    ``row_is_eligible`` is a small predicate supplied by the concrete behavior;
    keeping it here makes the tie-breaking rule identical across behaviors.
    """
    predicate = _all_rows if row_is_eligible is None else row_is_eligible

    destinations = [0] * len(context.observation.agents)
    selected_by_role: set[tuple[AgentRole, int]] = set()
    for row, agent in enumerate(context.observation.agents):
        if agent.busy or not predicate(row):
            continue
        for destination in _legal_candidates(context, row, eligible_destinations):
            selection = (agent.role, destination)
            if selection in selected_by_role:
                continue
            destinations[row] = destination
            selected_by_role.add(selection)
            break
    return tuple(destinations)


def _has_candidate(
    context: DirectiveContext,
    eligible_destinations: frozenset[int],
    row_is_eligible: Callable[[int], bool] | None = None,
) -> bool:
    predicate = _all_rows if row_is_eligible is None else row_is_eligible
    return any(
        not agent.busy
        and predicate(row)
        and _legal_candidates(context, row, eligible_destinations)
        for row, agent in enumerate(context.observation.agents)
    )


def _action_destinations(
    context: DirectiveContext, action: JointAction
) -> tuple[int, ...]:
    expected_ids = tuple(agent.agent_id for agent in context.observation.agents)
    actual_ids = tuple(item.agent_id for item in action.actions)
    if actual_ids != expected_ids:
        raise DirectiveEffectError(
            "action agents must match the observation's canonical order"
        )
    return tuple(int(item.destination_id) for item in action.actions)


def _validate_selected_destination(
    context: DirectiveContext,
    row: int,
    destination: int,
) -> bool:
    if destination == 0:
        return False
    agent = context.observation.agents[row]
    return not agent.busy and bool(context.action_mask.values[row, destination])


class WaitBehavior:
    """Leave every agent unassigned for this decision."""

    @property
    def behavior_key(self) -> str:
        return "wait"

    def is_applicable(self, context: DirectiveContext) -> bool:
        del context
        return True

    def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]:
        return (0,) * len(context.observation.agents)

    def validate_effect(self, context: DirectiveContext, action: JointAction) -> None:
        destinations = _action_destinations(context, action)
        if any(destination != 0 for destination in destinations):
            raise DirectiveEffectError("wait requires an all-zero joint action")


class ServiceRequestedShelfBehavior:
    """Send idle agents to currently requested shelf locations."""

    @property
    def behavior_key(self) -> str:
        return "service_requested_shelf"

    def is_applicable(self, context: DirectiveContext) -> bool:
        return _has_candidate(context, _requested_shelf_ids(context))

    def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]:
        return _propose_for(context, _requested_shelf_ids(context))

    def validate_effect(self, context: DirectiveContext, action: JointAction) -> None:
        destinations = _action_destinations(context, action)
        requested = _requested_shelf_ids(context)
        if any(
            destination in requested
            and _validate_selected_destination(context, row, destination)
            for row, destination in enumerate(destinations)
        ):
            return
        raise DirectiveEffectError(
            "service-requested-shelf requires an idle agent to select a requested shelf"
        )


class DeliverCarriedShelfBehavior:
    """Send idle AGVs carrying a shelf to a legal goal destination."""

    @property
    def behavior_key(self) -> str:
        return "deliver_carried_shelf"

    @staticmethod
    def _eligible_row(context: DirectiveContext, row: int) -> bool:
        agent = context.observation.agents[row]
        return agent.role is AgentRole.AGV and agent.carrying_shelf

    def is_applicable(self, context: DirectiveContext) -> bool:
        return _has_candidate(
            context,
            frozenset(
                int(destination)
                for destination in context.observation.goal_destination_ids
            ),
            lambda row: self._eligible_row(context, row),
        )

    def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]:
        goals = frozenset(
            int(destination) for destination in context.observation.goal_destination_ids
        )
        return _propose_for(
            context, goals, lambda row: self._eligible_row(context, row)
        )

    def validate_effect(self, context: DirectiveContext, action: JointAction) -> None:
        destinations = _action_destinations(context, action)
        goals = frozenset(
            int(destination) for destination in context.observation.goal_destination_ids
        )
        if any(
            destination in goals
            and self._eligible_row(context, row)
            and _validate_selected_destination(context, row, destination)
            for row, destination in enumerate(destinations)
        ):
            return
        raise DirectiveEffectError(
            "deliver-carried-shelf requires a carrying AGV to select a goal"
        )


class RepositionIdleAgentBehavior:
    """Move an idle, non-carrying agent to a legal non-requested destination."""

    @property
    def behavior_key(self) -> str:
        return "reposition_idle_agent"

    def _eligible_destinations(self, context: DirectiveContext) -> frozenset[int]:
        all_nonzero = frozenset(range(1, context.action_mask.destination_count))
        return all_nonzero - _requested_shelf_ids(context)

    @staticmethod
    def _eligible_row(context: DirectiveContext, row: int) -> bool:
        return not context.observation.agents[row].carrying_shelf

    def is_applicable(self, context: DirectiveContext) -> bool:
        return _has_candidate(
            context,
            self._eligible_destinations(context),
            lambda row: self._eligible_row(context, row),
        )

    def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]:
        return _propose_for(
            context,
            self._eligible_destinations(context),
            lambda row: self._eligible_row(context, row),
        )

    def validate_effect(self, context: DirectiveContext, action: JointAction) -> None:
        destinations = _action_destinations(context, action)
        eligible = self._eligible_destinations(context)
        if any(
            destination in eligible
            and self._eligible_row(context, row)
            and _validate_selected_destination(context, row, destination)
            for row, destination in enumerate(destinations)
        ):
            return
        raise DirectiveEffectError(
            "reposition-idle-agent requires an idle agent to select a non-requested shelf"
        )


_BEHAVIOR_TYPES: dict[str, type[DirectiveBehavior]] = {
    "wait": WaitBehavior,
    "service_requested_shelf": ServiceRequestedShelfBehavior,
    "deliver_carried_shelf": DeliverCarriedShelfBehavior,
    "reposition_idle_agent": RepositionIdleAgentBehavior,
}


def default_behaviors() -> dict[str, DirectiveBehavior]:
    """Construct the reviewed behavior objects for the frozen vocabulary."""
    return {key: behavior_type() for key, behavior_type in _BEHAVIOR_TYPES.items()}


class DirectiveRegistry:
    """Resolve vocabulary IDs into closed deterministic directive behaviors."""

    def __init__(
        self,
        vocabulary: DirectiveVocabulary,
        behaviors: Mapping[str, DirectiveBehavior],
    ) -> None:
        self.vocabulary = vocabulary
        self._behaviors = dict(behaviors)
        expected = {definition.behavior_key for definition in vocabulary.directives}
        actual = set(self._behaviors)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"behavior mapping does not match vocabulary; missing={missing}, "
                f"unexpected={unexpected}"
            )
        for behavior_key, behavior in self._behaviors.items():
            if behavior.behavior_key != behavior_key:
                raise ValueError(
                    f"behavior {behavior_key!r} reports key {behavior.behavior_key!r}"
                )

    def _resolve(self, directive_id: int) -> tuple[str, DirectiveBehavior]:
        try:
            definition = self.vocabulary.by_id(int(directive_id))
        except (KeyError, TypeError, ValueError) as exc:
            raise UnknownDirectiveError(
                f"unknown directive ID {directive_id!r}"
            ) from exc
        return definition.behavior_key, self._behaviors[definition.behavior_key]

    def action_mask(self, context: DirectiveContext) -> NDArray[np.bool_]:
        """Return the current one-dimensional mask over directive IDs."""
        mask = np.asarray(
            [
                self._behaviors[definition.behavior_key].is_applicable(context)
                for definition in self.vocabulary.directives
            ],
            dtype=np.bool_,
        )
        wait_index = self.vocabulary.wait_directive_id
        if not bool(mask[wait_index]):
            raise DirectiveError("the wait directive must always be applicable")
        if not bool(np.any(mask)):
            raise DirectiveError("at least one directive must be applicable")
        mask.setflags(write=False)
        return mask

    def behavior_digest(self) -> str:
        """Hash bindings and implementation bytes for manifest compatibility."""
        bindings = tuple(
            (
                definition.directive_id,
                definition.behavior_key,
                type(self._behaviors[definition.behavior_key]).__module__,
                type(self._behaviors[definition.behavior_key]).__qualname__,
            )
            for definition in self.vocabulary.directives
        )
        return sha256_bytes(
            canonical_json_bytes(bindings) + b"\x00" + Path(__file__).read_bytes()
        )

    def execute(
        self,
        directive_id: int,
        context: DirectiveContext,
        wrapper: TarwareEnvWrapper,
    ) -> JointAction:
        """Execute and strictly validate one applicable directive."""
        _, behavior = self._resolve(directive_id)
        try:
            if not behavior.is_applicable(context):
                raise InapplicableDirectiveError(
                    f"directive {directive_id} is not applicable in state {context.state_id}"
                )
            raw_destinations = behavior.propose_destinations(context)
        except DirectiveError:
            raise
        except Exception as exc:
            raise DirectiveExecutorError(
                f"directive {directive_id} executor failed while proposing destinations"
            ) from exc

        try:
            action = wrapper.decode(raw_destinations)
        except (InvalidJointActionError, ValueError, TypeError) as exc:
            raise DirectiveExecutorError(
                f"directive {directive_id} proposed an invalid joint action"
            ) from exc

        self.validate_effect(directive_id, context, action)
        return action

    def validate_effect(
        self,
        directive_id: int,
        context: DirectiveContext,
        action: JointAction,
    ) -> None:
        """Validate a concrete action against the selected directive."""
        _, behavior = self._resolve(directive_id)
        try:
            if not behavior.is_applicable(context):
                raise InapplicableDirectiveError(
                    f"directive {directive_id} is not applicable in state {context.state_id}"
                )
            behavior.validate_effect(context, action)
        except DirectiveError:
            raise
        except Exception as exc:
            raise DirectiveEffectError(
                f"directive {directive_id} effect validation failed"
            ) from exc


def build_directive_registry(vocabulary: DirectiveVocabulary) -> DirectiveRegistry:
    """Build the registry for the behavior keys in ``vocabulary``."""
    unknown = set(
        definition.behavior_key for definition in vocabulary.directives
    ) - set(_BEHAVIOR_TYPES)
    if unknown:
        raise ValueError(f"no reviewed behavior implementation for {sorted(unknown)}")
    return DirectiveRegistry(vocabulary, default_behaviors())
