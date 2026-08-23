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
    assert (
        first.response_schema["properties"]["destinations"]["items"]["maximum"]
        == request.context.action_mask.destination_count - 1
    )


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
    assert "GLOBAL RULES" in live_text
    assert "FINAL CHECK" in live_text
    assert "SELECTED DIRECTIVE" in live_text
    assert "LIVE STATE" in live_text
    assert "probabilities" not in all_text
    assert "value_estimate" not in all_text
    assert "9100" not in all_text
    assert live_text.count("## WAIT EXAMPLE") == 3
    assert live_text.count("## SERVICE REQUESTED SHELF EXAMPLE") == 3
    assert live_text.count("## DELIVER CARRIED SHELF EXAMPLE") == 3
    assert live_text.count("## REPOSITION IDLE AGENT EXAMPLE") == 3
    assert live_text.count("EXAMPLE ") == prompt_spec.example_count
    assert "{{SELECTED_DIRECTIVE" not in live_text
    assert "{{LIVE_STATE_JSON}}" not in live_text
    assert builder.contract_digest == (
        "b7c1ad9a7ca22a8e658dfdeeaa1be41eec21236444d76d428881b89be16295b8"
    )
    messages_json = json.dumps(
        [message.model_dump(mode="json") for message in first.messages],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert hashlib.sha256(messages_json.encode()).hexdigest() == (
        "c5400b4ee2946df3f0175434092a9da6e934bf4b541ab430f294cf0252f1cc37"
    )
    assert builder.template_digest == (
        "9d7301b3652de4d8456234c925149cdcc1d6d69c6c96cd28b17cb2e73f0c1d38"
    )


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
    wrong_count["example_count"] = 11
    with pytest.raises(ValidationError, match="example_count"):
        LlmPromptSpec.model_validate(wrong_count)

    duplicate_key = prompt_spec.model_dump(mode="json")
    duplicate_key["required_behavior_keys"] = [
        "wait",
        "wait",
        "deliver_carried_shelf",
        "reposition_idle_agent",
    ]
    with pytest.raises(ValidationError):
        LlmPromptSpec.model_validate(duplicate_key)

    misaligned_command = prompt_spec.model_copy(
        update={
            "command_metadata": {
                **prompt_spec.command_metadata,
                "wait": "not-wait",
            }
        }
    )
    with pytest.raises(ValueError, match="command metadata"):
        misaligned_command.validate_catalog(catalog)

    missing_template = prompt_spec.model_dump(mode="json")
    missing_template["prompt_template_path"] = None
    with pytest.raises(ValidationError, match="TXT path"):
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
