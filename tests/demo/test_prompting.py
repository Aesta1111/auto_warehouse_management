from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    PpoPolicyOutput,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    GemmaActionResponse,
)
from warehouse_ai.demo.contracts.prompting import GemmaPromptSpec
from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def catalog() -> DirectiveCatalog:
    return DirectiveCatalog.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def wrapped_env():
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


@pytest.fixture
def prompt_spec() -> GemmaPromptSpec:
    return load_yaml_config(
        ROOT / "configs/demo/gemma-prompt-few-shot-v1.yaml", GemmaPromptSpec
    )


def _request(wrapper: TarwareEnvWrapper, catalog: DirectiveCatalog) -> ActionerRequest:
    context = WarehouseContext.capture(wrapper)
    prediction = PpoPolicyOutput(
        state_id=context.state_id,
        episode_step=context.observation.step,
        directive=catalog.by_id(1),
        probabilities=(0.0, 1.0, 0.0, 0.0),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256="0" * 64,
    )
    return ActionerRequest(context=context, prediction=prediction)


def test_prompt_and_schema_are_deterministic_and_state_only(
    catalog: DirectiveCatalog, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, catalog)
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


def test_few_shot_prompt_is_deterministic_and_excludes_catalog_description(
    catalog: DirectiveCatalog,
    prompt_spec: GemmaPromptSpec,
    wrapped_env: TarwareEnvWrapper,
) -> None:
    request = _request(wrapped_env, catalog)
    builder = GemmaPromptBuilder(prompt_spec, catalog)

    first = builder.build(request)
    second = builder.build(request)

    assert first == second
    assert len(first.messages) == 10
    assert [message.role for message in first.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    live_text = first.messages[-1].content
    all_text = "\n".join(message.content for message in first.messages)
    assert "instruction:" in live_text
    assert "description:" not in all_text
    assert request.prediction.directive.description not in all_text
    assert "JOINT-ACTION SAFETY RULES" in live_text
    assert "OUTPUT INSTRUCTIONS" in live_text
    assert "probabilities" not in all_text
    assert "value_estimate" not in all_text
    assert "9100" not in all_text
    assert (
        sum(
            "SYNTHETIC FEW-SHOT EXAMPLE" in message.content
            for message in first.messages
        )
        == 4
    )
    assert builder.contract_digest == (
        "923ca9f418d9f6bb0c1f12071b8352d0e81dcebb233a9cacbb5d4fba00db04be"
    )
    messages_json = json.dumps(
        [message.model_dump(mode="json") for message in first.messages],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert hashlib.sha256(messages_json.encode()).hexdigest() == (
        "2c0a3f67ff674d7cc052320e36b77a5282182fdbc744c922bdb6b1933b5375fe"
    )


def test_prompt_spec_rejects_incomplete_or_misaligned_examples(
    prompt_spec: GemmaPromptSpec,
) -> None:
    missing_directive = prompt_spec.model_dump(mode="json")
    missing_directive["directives"] = missing_directive["directives"][:-1]
    with pytest.raises(ValidationError):
        GemmaPromptSpec.model_validate(missing_directive)

    blank_instruction = prompt_spec.model_dump(mode="json")
    blank_instruction["directives"][0]["instruction"] = " "
    with pytest.raises(ValidationError, match="blank"):
        GemmaPromptSpec.model_validate(blank_instruction)

    missing_example = prompt_spec.model_dump(mode="json")
    del missing_example["directives"][0]["example"]
    with pytest.raises(ValidationError, match="Field required"):
        GemmaPromptSpec.model_validate(missing_example)

    malformed_response = prompt_spec.model_dump(mode="json")
    malformed_response["directives"][0]["example"]["response"]["extra"] = True
    with pytest.raises(ValidationError, match="Extra inputs"):
        GemmaPromptSpec.model_validate(malformed_response)

    wrong_length = prompt_spec.model_dump(mode="json")
    wrong_length["directives"][0]["example"]["response"]["destinations"] = [
        0,
        0,
        0,
        0,
    ]
    with pytest.raises(ValidationError, match="one destination per agent"):
        GemmaPromptSpec.model_validate(wrong_length)

    illegal = prompt_spec.model_dump(mode="json")
    illegal["directives"][1]["example"]["response"]["destinations"][0] = 99
    with pytest.raises(ValidationError, match="illegal destination"):
        GemmaPromptSpec.model_validate(illegal)

    duplicate = prompt_spec.model_dump(mode="json")
    duplicate["directives"][1]["example"]["legal_destinations"][1] = [0, 11]
    duplicate["directives"][1]["example"]["response"]["destinations"][1] = 11
    with pytest.raises(ValidationError, match="same-role duplicate"):
        GemmaPromptSpec.model_validate(duplicate)


def test_actioner_records_reject_mismatched_request_state(
    catalog: DirectiveCatalog, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, catalog)
    with pytest.raises(ValidationError, match="states differ"):
        ActionerRequest(
            context=request.context,
            prediction=request.prediction.model_copy(update={"state_id": "stale"}),
        )


def test_gemma_response_reason_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 160"):
        GemmaActionResponse(destinations=(0,), reason="x" * 161)
