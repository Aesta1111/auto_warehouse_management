from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from warehouse_ai.demo.contracts.directives import DirectiveContext, DirectiveVocabulary
from warehouse_ai.demo.directives import (
    BEHAVIOR_ENGINE_VERSION,
    DirectiveError,
    DirectiveFailure,
    DirectiveRegistry,
    build_directive_registry,
)
from warehouse_ai.demo.directives.rules import (
    RULES,
    AgentSelector,
    DestinationSelector,
    EffectCardinality,
)
from warehouse_ai.domain import ActionMask, AgentAction, JointAction
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def wrapper() -> Iterator[TarwareEnvWrapper]:
    environment = TarwareEnvWrapper(ENVIRONMENT_ID)
    environment.reset(seed=0)
    yield environment
    environment.close()


def test_rule_table_is_the_frozen_four_command_specification() -> None:
    assert BEHAVIOR_ENGINE_VERSION == "directive-rule-engine-v2"
    assert [rule.behavior_key for rule in RULES] == [
        "wait",
        "service_requested_shelf",
        "deliver_carried_shelf",
        "reposition_idle_agent",
    ]
    assert [rule.agents for rule in RULES] == [
        AgentSelector.ANY_IDLE,
        AgentSelector.ANY_IDLE,
        AgentSelector.CARRYING_AGV,
        AgentSelector.NON_CARRYING_IDLE,
    ]
    assert [rule.destinations for rule in RULES] == [
        DestinationSelector.NONE,
        DestinationSelector.REQUESTED_SHELVES,
        DestinationSelector.GOALS,
        DestinationSelector.NON_REQUESTED,
    ]
    assert [rule.effect.minimum for rule in RULES] == [0, 1, 1, 1]


def test_initial_state_parity_and_deterministic_numeric_tie_breaking(
    vocabulary: DirectiveVocabulary, wrapper: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapper)
    mask = registry.action_mask(context)
    assert tuple(bool(value) for value in mask) == (True, True, False, False)
    assert not mask.flags.writeable
    for directive_id in np.flatnonzero(mask):
        first = registry.execute(int(directive_id), context, wrapper)
        second = registry.execute(int(directive_id), context, wrapper)
        assert first == second
        registry.validate_effect(int(directive_id), context, first)
    destinations = [
        item.destination_id for item in registry.execute(1, context, wrapper).actions
    ]
    assert destinations == sorted(destinations, key=lambda value: value == 0)
    selected = [value for value in destinations if value]
    assert len(selected) == len(set(selected))


def _action(
    context: DirectiveContext, row: int | None, destination: int = 0
) -> JointAction:
    return JointAction(
        actions=tuple(
            AgentAction(
                agent_id=agent.agent_id,
                destination_id=destination if index == row else 0,
            )
            for index, agent in enumerate(context.observation.agents)
        )
    )


def _allow(context: DirectiveContext, row: int, destination: int) -> DirectiveContext:
    values = context.action_mask.values.copy()
    values[row, destination] = True
    return context.model_copy(
        update={
            "action_mask": ActionMask(
                agent_ids=context.action_mask.agent_ids,
                values=values,
            )
        }
    )


def test_effect_validation_covers_all_four_rules(
    vocabulary: DirectiveVocabulary, wrapper: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    base = DirectiveContext.capture(wrapper)
    requested = next(s.destination_id for s in base.observation.shelves if s.requested)
    registry.validate_effect(0, base, _action(base, None))
    registry.validate_effect(1, base, _action(base, 0, requested))

    goal = base.observation.goal_destination_ids[0]
    carrying_agent = base.observation.agents[0].model_copy(
        update={"busy": False, "carrying_shelf": True}
    )
    carrying = base.model_copy(
        update={
            "observation": base.observation.model_copy(
                update={"agents": (carrying_agent, *base.observation.agents[1:])}
            )
        }
    )
    carrying = _allow(carrying, 0, goal)
    assert bool(registry.action_mask(carrying)[2])
    registry.validate_effect(2, carrying, _action(carrying, 0, goal))

    non_requested = next(
        shelf.destination_id
        for shelf in base.observation.shelves
        if not shelf.requested
    )
    reposition = _allow(base, 0, non_requested)
    assert bool(registry.action_mask(reposition)[3])
    registry.validate_effect(3, reposition, _action(reposition, 0, non_requested))

    with pytest.raises(DirectiveError) as error:
        registry.validate_effect(1, base, _action(base, None))
    assert error.value.code is DirectiveFailure.EFFECT


def test_errors_are_structured_and_digest_tracks_semantics_not_file_bytes(
    vocabulary: DirectiveVocabulary, wrapper: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapper)
    with pytest.raises(DirectiveError) as unknown:
        registry.execute(99, context, wrapper)
    assert unknown.value.code is DirectiveFailure.UNKNOWN
    with pytest.raises(DirectiveError) as inapplicable:
        registry.execute(2, context, wrapper)
    assert inapplicable.value.code is DirectiveFailure.INAPPLICABLE

    changed = (*RULES[:3], replace(RULES[3], effect=EffectCardinality(2)))
    assert (
        DirectiveRegistry(vocabulary, changed).behavior_digest()
        != registry.behavior_digest()
    )
