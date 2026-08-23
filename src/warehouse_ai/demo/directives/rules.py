"""Directive semantic selectors and the private behavior-key factory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from warehouse_ai.domain import AgentRole

if TYPE_CHECKING:
    from warehouse_ai.demo.contracts.directives import Directive, WarehouseContext


class AgentSelector(str, Enum):
    ANY_IDLE = "any_idle"
    CARRYING_AGV = "carrying_agv"
    NON_CARRYING_IDLE = "non_carrying_idle"


class DestinationSelector(str, Enum):
    NONE = "none"
    REQUESTED_SHELVES = "requested_shelves"
    GOALS = "goals"
    NON_REQUESTED = "non_requested"


@dataclass(frozen=True)
class EffectCardinality:
    minimum: int
    maximum: int | None = None

    def accepts(self, matches: int) -> bool:
        return matches >= self.minimum and (
            self.maximum is None or matches <= self.maximum
        )


def _semantics_for_behavior_key(
    behavior_key: str,
) -> tuple[AgentSelector, DestinationSelector, EffectCardinality]:
    """Return the closed executable semantics for one YAML behavior key."""
    if behavior_key == "wait":
        return (
            AgentSelector.ANY_IDLE,
            DestinationSelector.NONE,
            EffectCardinality(0, 0),
        )
    if behavior_key == "service_requested_shelf":
        return (
            AgentSelector.ANY_IDLE,
            DestinationSelector.REQUESTED_SHELVES,
            EffectCardinality(1),
        )
    if behavior_key == "deliver_carried_shelf":
        return (
            AgentSelector.CARRYING_AGV,
            DestinationSelector.GOALS,
            EffectCardinality(1),
        )
    if behavior_key == "reposition_idle_agent":
        return (
            AgentSelector.NON_CARRYING_IDLE,
            DestinationSelector.NON_REQUESTED,
            EffectCardinality(1),
        )
    raise ValueError(f"unknown directive behavior key {behavior_key!r}")


def eligible_destinations(
    directive: Directive, context: WarehouseContext
) -> frozenset[int]:
    if directive.destinations is DestinationSelector.NONE:
        return frozenset()
    if directive.destinations is DestinationSelector.GOALS:
        return frozenset(int(item) for item in context.observation.goal_destination_ids)
    requested = frozenset(
        int(shelf.destination_id)
        for shelf in context.observation.shelves
        if shelf.requested
    )
    if directive.destinations is DestinationSelector.REQUESTED_SHELVES:
        return requested
    return frozenset(range(1, context.action_mask.destination_count)) - requested


def agent_is_eligible(
    directive: Directive, context: WarehouseContext, row: int
) -> bool:
    agent = context.observation.agents[row]
    if agent.busy:
        return False
    if directive.agents is AgentSelector.CARRYING_AGV:
        return agent.role is AgentRole.AGV and agent.carrying_shelf
    if directive.agents is AgentSelector.NON_CARRYING_IDLE:
        return not agent.carrying_shelf
    return True
