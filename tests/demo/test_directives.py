from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from warehouse_ai.demo.directives import (
    DeliverCarriedShelfBehavior,
    DirectiveEffectError,
    DirectiveExecutorError,
    DirectiveRegistry,
    InapplicableDirectiveError,
    RepositionIdleAgentBehavior,
    ServiceRequestedShelfBehavior,
    UnknownDirectiveError,
    WaitBehavior,
    build_directive_registry,
)
from warehouse_ai.demo.models import DirectiveContext, DirectiveVocabulary
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def wrapped_env() -> Iterator[TarwareEnvWrapper]:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


def test_default_registry_binds_each_frozen_behavior(
    vocabulary: DirectiveVocabulary,
) -> None:
    registry = build_directive_registry(vocabulary)

    assert registry.behavior_digest()
    assert isinstance(WaitBehavior(), WaitBehavior)
    assert isinstance(ServiceRequestedShelfBehavior(), ServiceRequestedShelfBehavior)
    assert isinstance(DeliverCarriedShelfBehavior(), DeliverCarriedShelfBehavior)
    assert isinstance(RepositionIdleAgentBehavior(), RepositionIdleAgentBehavior)


def test_action_mask_keeps_wait_and_detects_requested_shelf(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapped_env)

    mask = registry.action_mask(context)

    assert mask.shape == (vocabulary.size,)
    assert bool(mask[vocabulary.wait_directive_id])
    assert bool(mask[1])
    assert not bool(mask[2])
    assert not bool(mask[3])
    assert not mask.flags.writeable


def test_every_applicable_executor_is_deterministic_and_strict(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapped_env)

    for directive_id in np.flatnonzero(registry.action_mask(context)):
        first = registry.execute(int(directive_id), context, wrapped_env)
        second = registry.execute(int(directive_id), context, wrapped_env)
        assert first == second
        registry.validate_effect(int(directive_id), context, first)


def test_registry_distinguishes_unknown_and_inapplicable_directives(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapped_env)

    with pytest.raises(UnknownDirectiveError):
        registry.execute(99, context, wrapped_env)
    with pytest.raises(InapplicableDirectiveError):
        registry.execute(2, context, wrapped_env)


def test_wait_effect_rejects_a_nonzero_action(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    registry = build_directive_registry(vocabulary)
    context = DirectiveContext.capture(wrapped_env)
    action = wrapped_env.decode(np.asarray([7, 0, 0, 0, 0], dtype=np.int64))

    with pytest.raises(DirectiveEffectError, match="wait"):
        registry.validate_effect(vocabulary.wait_directive_id, context, action)


def test_registry_wraps_invalid_behavior_output_as_executor_error(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    class BadService(ServiceRequestedShelfBehavior):
        def propose_destinations(self, context: DirectiveContext) -> tuple[int, ...]:
            del context
            return (99, 99, 0, 0, 0)

    behaviors = {
        "wait": WaitBehavior(),
        "service_requested_shelf": BadService(),
        "deliver_carried_shelf": DeliverCarriedShelfBehavior(),
        "reposition_idle_agent": RepositionIdleAgentBehavior(),
    }
    registry = DirectiveRegistry(vocabulary, behaviors)

    with pytest.raises(DirectiveExecutorError) as error:
        registry.execute(1, DirectiveContext.capture(wrapped_env), wrapped_env)
    assert error.value.__cause__ is not None


def test_behavior_digest_includes_behavior_binding_class(
    vocabulary: DirectiveVocabulary,
) -> None:
    first = build_directive_registry(vocabulary)

    class AlternateWait(WaitBehavior):
        pass

    alternate = DirectiveRegistry(
        vocabulary,
        {
            "wait": AlternateWait(),
            "service_requested_shelf": ServiceRequestedShelfBehavior(),
            "deliver_carried_shelf": DeliverCarriedShelfBehavior(),
            "reposition_idle_agent": RepositionIdleAgentBehavior(),
        },
    )

    assert first.behavior_digest() != alternate.behavior_digest()
