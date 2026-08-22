"""Deterministic state and request helpers for model preflight."""

from __future__ import annotations

import time

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.inference import ActionerRequest
from warehouse_ai.demo.controllers import ActionValidator, DecisionFailure
from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.demo.inference.actioner import ActionerError, GemmaActioner
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.demo.ppo.policy import DirectivePpoPolicy
from warehouse_ai.env import TarwareEnvWrapper


def forced_request(
    context: DirectiveContext,
    policy: DirectivePpoPolicy,
    vocabulary: DirectiveVocabulary,
    directive_id: int,
) -> ActionerRequest:
    probabilities = [0.0] * vocabulary.size
    probabilities[directive_id] = 1.0
    prediction = DirectivePrediction(
        state_id=context.state_id,
        episode_step=context.observation.step,
        directive=vocabulary.by_id(directive_id),
        probabilities=tuple(probabilities),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256=policy.checkpoint_sha256,
    )
    return ActionerRequest(context=context, prediction=prediction)


def policy_request(
    context: DirectiveContext,
    policy: DirectivePpoPolicy,
    registry: DirectiveRegistry,
) -> ActionerRequest:
    return ActionerRequest(
        context=context,
        prediction=policy.predict(context, registry.action_mask(context)),
    )


def applicable_directives(
    context: DirectiveContext,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
    *,
    include_wait: bool,
) -> tuple[int, ...]:
    return tuple(
        directive_id
        for directive_id, applicable in enumerate(registry.action_mask(context))
        if applicable and (include_wait or directive_id != vocabulary.wait_directive_id)
    )


def find_state(
    wrapper: TarwareEnvWrapper,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
    *,
    seed: int,
    max_steps: int,
    required: int,
    include_wait: bool = False,
) -> tuple[DirectiveContext, tuple[int, ...]] | None:
    wrapper.reset(seed=seed)
    for _ in range(max_steps):
        context = DirectiveContext.capture(wrapper)
        directives = applicable_directives(
            context, registry, vocabulary, include_wait=include_wait
        )
        if len(directives) >= required:
            return context, directives
        wrapper.step(wrapper.no_op())
    return None


def timed_action(
    actioner: GemmaActioner,
    validator: ActionValidator,
    request: ActionerRequest,
    wrapper: TarwareEnvWrapper,
) -> tuple[float, bool, str | None]:
    started = time.perf_counter()
    try:
        result = actioner.act(request)
        validator.validate(request, result, wrapper)
    except (ActionerError, DecisionFailure) as exc:
        return time.perf_counter() - started, False, f"{exc.code.value}: {exc}"
    return time.perf_counter() - started, True, None


def prompt_schema_digest(context: DirectiveContext, request: ActionerRequest) -> str:
    del context
    schema = GemmaPromptBuilder().build(request).response_schema
    return sha256_bytes(canonical_json_bytes(schema))
