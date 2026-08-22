from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    GemmaActionResponse,
)
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def wrapped_env():
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


def _request(
    wrapper: TarwareEnvWrapper, vocabulary: DirectiveVocabulary
) -> ActionerRequest:
    context = DirectiveContext.capture(wrapper)
    prediction = DirectivePrediction(
        state_id=context.state_id,
        episode_step=context.observation.step,
        directive=vocabulary.by_id(1),
        probabilities=(0.0, 1.0, 0.0, 0.0),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256="0" * 64,
    )
    return ActionerRequest(context=context, prediction=prediction)


def test_prompt_and_schema_are_deterministic_and_state_only(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, vocabulary)
    builder = GemmaPromptBuilder()

    first = builder.build(request)
    second = builder.build(request)

    assert first == second
    assert builder.serialize_state(request.context) == builder.serialize_state(
        request.context
    )
    assert len(first.messages) == 2
    assert first.messages[0].role == "system"
    user_text = first.messages[1].content
    payload = json.loads(builder.serialize_state(request.context))
    assert payload["same_role_unique_nonzero_groups"] == {
        "agv": [0, 1, 2],
        "picker": [3, 4],
    }
    assert len(payload["agents"]) == len(request.context.observation.agents)
    assert hashlib.sha256(user_text.encode()).hexdigest() == (
        "9326a44c3cf3dc86481c304ed2929263519930d237d8cff9605964bef1032443"
    )
    assert "probabilities" not in user_text
    assert "value_estimate" not in user_text
    assert "three AGVs" not in user_text
    assert "agv positions [0, 1, 2]" in user_text
    sections = (
        "OBJECTIVE",
        "DIRECTIVE",
        "AGENTS",
        "REQUESTED SHELVES",
        "GOALS",
        "LEGAL DESTINATIONS",
        "OUTPUT INSTRUCTIONS",
    )
    assert [user_text.index(section) for section in sections] == sorted(
        user_text.index(section) for section in sections
    )
    assert first.response_schema["additionalProperties"] is False
    assert first.response_schema["properties"]["destinations"]["minItems"] == 5
    assert first.response_schema["properties"]["destinations"]["maxItems"] == 5
    assert (
        first.response_schema["properties"]["destinations"]["items"]["maximum"]
        == request.context.action_mask.destination_count - 1
    )


def test_actioner_records_reject_mismatched_request_state(
    vocabulary: DirectiveVocabulary, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, vocabulary)
    with pytest.raises(ValidationError, match="states differ"):
        ActionerRequest(
            context=request.context,
            prediction=request.prediction.model_copy(update={"state_id": "stale"}),
        )


def test_gemma_response_reason_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 160"):
        GemmaActionResponse(destinations=(0,), reason="x" * 161)
