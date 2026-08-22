"""Table-driven directive specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from warehouse_ai.demo.contracts.directives import DirectiveContext
from warehouse_ai.domain import AgentRole


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


@dataclass(frozen=True)
class DirectiveRule:
    """Stable semantic description for one vocabulary behavior key."""

    behavior_key: str
    agents: AgentSelector
    destinations: DestinationSelector
    effect: EffectCardinality

    def descriptor(self) -> dict[str, Any]:
        return asdict(self)


RULES: tuple[DirectiveRule, ...] = (
    DirectiveRule(
        "wait",
        AgentSelector.ANY_IDLE,
        DestinationSelector.NONE,
        EffectCardinality(0, 0),
    ),
    DirectiveRule(
        "service_requested_shelf",
        AgentSelector.ANY_IDLE,
        DestinationSelector.REQUESTED_SHELVES,
        EffectCardinality(1),
    ),
    DirectiveRule(
        "deliver_carried_shelf",
        AgentSelector.CARRYING_AGV,
        DestinationSelector.GOALS,
        EffectCardinality(1),
    ),
    DirectiveRule(
        "reposition_idle_agent",
        AgentSelector.NON_CARRYING_IDLE,
        DestinationSelector.NON_REQUESTED,
        EffectCardinality(1),
    ),
)


def eligible_destinations(
    rule: DirectiveRule, context: DirectiveContext
) -> frozenset[int]:
    if rule.destinations is DestinationSelector.NONE:
        return frozenset()
    if rule.destinations is DestinationSelector.GOALS:
        return frozenset(int(item) for item in context.observation.goal_destination_ids)
    requested = frozenset(
        int(shelf.destination_id)
        for shelf in context.observation.shelves
        if shelf.requested
    )
    if rule.destinations is DestinationSelector.REQUESTED_SHELVES:
        return requested
    return frozenset(range(1, context.action_mask.destination_count)) - requested


def agent_is_eligible(rule: DirectiveRule, context: DirectiveContext, row: int) -> bool:
    agent = context.observation.agents[row]
    if agent.busy:
        return False
    if rule.agents is AgentSelector.CARRYING_AGV:
        return agent.role is AgentRole.AGV and agent.carrying_shelf
    if rule.agents is AgentSelector.NON_CARRYING_IDLE:
        return not agent.carrying_shelf
    return True
