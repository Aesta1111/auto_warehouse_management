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
    LlmActionResponse,
)
from warehouse_ai.demo.contracts.prompting import LlmPromptSpec
from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.inference.prompting import LlmPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


@pytest.fixture
def catalog() -> DirectiveCatalog:
    return DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")


@pytest.fixture
def wrapped_env():
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


@pytest.fixture
def prompt_spec() -> LlmPromptSpec:
    return load_yaml_config(
        ROOT / "configs/demo/llm-actioner-few-shot-prompt.yaml", LlmPromptSpec
    )


def _request(
    wrapper: TarwareEnvWrapper,
    catalog: DirectiveCatalog,
    directive_id: int = 1,
) -> ActionerRequest:
    context = WarehouseContext.capture(wrapper)
    prediction = PpoPolicyOutput(
        state_id=context.state_id,
        episode_step=context.observation.step,
        directive=catalog.by_id(directive_id),
        probabilities=tuple(
            1.0 if index == directive_id else 0.0 for index in range(catalog.size)
        ),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256="0" * 64,
    )
    return ActionerRequest(context=context, prediction=prediction)


def test_prompt_and_schema_are_deterministic_and_state_only(
    catalog: DirectiveCatalog, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, catalog)
    builder = LlmPromptBuilder()

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
    schema_candidates = tuple(
        tuple(candidate)
        for candidate in first.response_schema["properties"]["destinations"]["enum"]
    )
    assert schema_candidates == builder.safe_candidate_actions(request)


def test_few_shot_prompt_is_deterministic_and_excludes_catalog_description(
    catalog: DirectiveCatalog,
    prompt_spec: LlmPromptSpec,
    wrapped_env: TarwareEnvWrapper,
) -> None:
    request = _request(wrapped_env, catalog)
    builder = LlmPromptBuilder.load(
        ROOT / "configs/demo/llm-actioner-few-shot-prompt.yaml", catalog
    )

    first = builder.build(request)
    second = builder.build(request)

    assert first == second
    assert len(first.messages) == 2
    assert [message.role for message in first.messages] == ["system", "user"]
    live_text = first.messages[1].content
    all_text = "\n".join(message.content for message in first.messages)
    assert "command: service-requested-shelf" in live_text
    assert (
        "instruction: Prefer a legal destination that advances requested shelf work."
        in live_text
    )
    assert request.directive.description in all_text
    assert "ACTION RULES" in live_text
    assert "FINAL CHECK" in live_text
    assert "SELECTED DIRECTIVE" in live_text
    assert "LIVE STATE" in live_text
    assert "action_slots" in live_text
    assert "allowed_destination_ids" in live_text
    assert "probabilities" not in all_text
    assert "value_estimate" not in all_text
    assert "9100" not in all_text
    assert live_text.count("## SERVICE REQUESTED SHELF EXAMPLE") == 3
    assert "## WAIT EXAMPLE" not in live_text
    assert "## DELIVER CARRIED SHELF EXAMPLE" not in live_text
    assert "## REPOSITION IDLE AGENT EXAMPLE" not in live_text
    assert live_text.count("EXAMPLE ") == prompt_spec.examples_per_directive
    assert prompt_spec.example_count == 9
    assert "{{SELECTED_DIRECTIVE" not in live_text
    assert "{{LIVE_STATE_JSON}}" not in live_text
    assert builder.contract_digest == (
        "9d9cda6755f8bfefef8924c09da14e11e223e788cf28492223840961139cc108"
    )
    messages_json = json.dumps(
        [message.model_dump(mode="json") for message in first.messages],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert hashlib.sha256(messages_json.encode()).hexdigest() == (
        "f3bbc375ecb9e5701540487f567edb9018471eff5e82a9424a03a4290c639e08"
    )
    assert builder.template_digest == (
        "1faa0fb7a9dad3802929838fc26570190d7ecaa62da591deeb447a6e70634ec0"
    )
    assert builder.template_digests == {
        "service_requested_shelf": (
            "b9ae66053dbc64b8e7abea19f36db0214851a7fb4508b9423fde2b7f49b287a4"
        ),
        "deliver_carried_shelf": (
            "2e6ba17ca8250d5fdeab774d4d1bf68926e5774e1106ec83cfe1922539a51239"
        ),
        "reposition_idle_agent": (
            "7a160fdf7763219641ba7ef6cd4a63de28b4792e9bd63e3b3f07ca29a3a93154"
        ),
    }


def test_safe_candidate_schema_excludes_invalid_agent_index_destinations(
    catalog: DirectiveCatalog,
) -> None:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=9300)
    try:
        request = _request(wrapper, catalog, directive_id=1)
        builder = LlmPromptBuilder.load(
            ROOT / "configs/demo/llm-actioner-few-shot-prompt.yaml", catalog
        )

        payload = builder.action_state_payload(request)
        allowed = tuple(
            tuple(slot["allowed_destination_ids"]) for slot in payload["action_slots"]
        )
        schema_candidates = tuple(
            tuple(candidate)
            for candidate in builder.build(request).response_schema["properties"][
                "destinations"
            ]["enum"]
        )

        assert allowed[3:] == ((0,), (0,))
        assert 3 not in allowed[3]
        assert 4 not in allowed[4]
        assert (0, 0, 0, 3, 4) not in schema_candidates
        assert schema_candidates == builder.safe_candidate_actions(request)
        assert all(
            any(destination != 0 for destination in action)
            for action in schema_candidates
        )
    finally:
        wrapper.close()


@pytest.mark.parametrize(
    ("directive_id", "expected_heading"),
    (
        (1, "## SERVICE REQUESTED SHELF EXAMPLE"),
        (2, "## DELIVER CARRIED SHELF EXAMPLE"),
        (3, "## REPOSITION IDLE AGENT EXAMPLE"),
    ),
)
def test_each_llm_directive_loads_only_its_own_examples(
    catalog: DirectiveCatalog,
    wrapped_env: TarwareEnvWrapper,
    monkeypatch: pytest.MonkeyPatch,
    directive_id: int,
    expected_heading: str,
) -> None:
    request = _request(wrapped_env, catalog, directive_id)
    builder = LlmPromptBuilder.load(
        ROOT / "configs/demo/llm-actioner-few-shot-prompt.yaml", catalog
    )
    monkeypatch.setattr(
        builder,
        "safe_candidate_actions",
        lambda _request: ((0,) * len(request.context.observation.agents),),
    )

    text = builder.build(request).messages[1].content

    assert text.count(expected_heading) == 3
    assert text.count("EXAMPLE ") == 3


def test_wait_bypasses_the_llm_prompt(
    catalog: DirectiveCatalog,
    wrapped_env: TarwareEnvWrapper,
) -> None:
    request = _request(wrapped_env, catalog, 0)
    builder = LlmPromptBuilder.load(
        ROOT / "configs/demo/llm-actioner-few-shot-prompt.yaml", catalog
    )

    with pytest.raises(ValueError, match="bypasses the LLM"):
        builder.build(request)


def test_prompt_spec_rejects_incomplete_or_misaligned_examples(
    catalog: DirectiveCatalog,
    prompt_spec: LlmPromptSpec,
) -> None:
    missing_directive = prompt_spec.model_dump(mode="json")
    missing_directive["required_behavior_keys"] = missing_directive[
        "required_behavior_keys"
    ][:-1]
    with pytest.raises(ValidationError):
        LlmPromptSpec.model_validate(missing_directive)

    wrong_count = prompt_spec.model_dump(mode="json")
    wrong_count["example_count"] = 8
    with pytest.raises(ValidationError, match="example_count"):
        LlmPromptSpec.model_validate(wrong_count)

    duplicate_key = prompt_spec.model_dump(mode="json")
    duplicate_key["required_behavior_keys"] = [
        "service_requested_shelf",
        "service_requested_shelf",
        "deliver_carried_shelf",
    ]
    with pytest.raises(ValidationError):
        LlmPromptSpec.model_validate(duplicate_key)

    misaligned_command = prompt_spec.model_copy(
        update={
            "command_metadata": {
                **prompt_spec.command_metadata,
                "service_requested_shelf": "not-service",
            }
        }
    )
    with pytest.raises(ValueError, match="command metadata"):
        misaligned_command.validate_catalog(catalog)

    missing_template = prompt_spec.model_dump(mode="json")
    del missing_template["prompt_template_paths"]["service_requested_shelf"]
    with pytest.raises(ValidationError, match="template paths"):
        LlmPromptSpec.model_validate(missing_template)


def test_actioner_records_reject_mismatched_request_state(
    catalog: DirectiveCatalog, wrapped_env: TarwareEnvWrapper
) -> None:
    request = _request(wrapped_env, catalog)
    prediction = request.prediction
    assert prediction is not None
    with pytest.raises(ValidationError, match="states differ"):
        ActionerRequest(
            context=request.context,
            prediction=prediction.model_copy(update={"state_id": "stale"}),
        )


def test_llm_response_reason_is_bounded() -> None:
    with pytest.raises(ValidationError, match="at most 160"):
        LlmActionResponse(destinations=(0,), reason="x" * 161)
