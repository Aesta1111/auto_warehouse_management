from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from warehouse_ai.demo.actioner import (
    ActionerDecodeError,
    ActionerEffectError,
    ActionerHealthError,
    ActionerSchemaError,
    EmptyActionerResponseError,
    LlamaCppActioner,
)
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.models import (
    ActionerRequest,
    DirectiveContext,
    DirectivePrediction,
    DirectiveVocabulary,
    GemmaPreflightReport,
    LlamaCppConfig,
    ModelHealth,
)
from warehouse_ai.demo.prompting import GemmaPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
MODEL_ID = "google/gemma-4-E4B-it"


@pytest.fixture
def vocabulary() -> DirectiveVocabulary:
    return DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")


@pytest.fixture
def registry(vocabulary: DirectiveVocabulary) -> DirectiveRegistry:
    return build_directive_registry(vocabulary)


@pytest.fixture
def wrapped_env() -> Iterator[TarwareEnvWrapper]:
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    try:
        yield wrapper
    finally:
        wrapper.close()


def _config() -> LlamaCppConfig:
    return LlamaCppConfig(
        base_url=cast(Any, "http://testserver"),
        model_id=MODEL_ID,
        model_file_sha256="0" * 64,
        context_size=4096,
        gpu_layers=999,
        request_timeout_seconds=2.0,
        max_new_tokens=192,
    )


def _request(
    wrapped_env: TarwareEnvWrapper, vocabulary: DirectiveVocabulary
) -> ActionerRequest:
    context = DirectiveContext.capture(wrapped_env)
    probabilities = [0.0] * vocabulary.size
    probabilities[1] = 1.0
    return ActionerRequest(
        context=context,
        prediction=DirectivePrediction(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=vocabulary.by_id(1),
            probabilities=tuple(probabilities),
            value_estimate=0.0,
            policy_latency_ms=0.0,
            checkpoint_sha256="0" * 64,
        ),
    )


def _response_handler(
    mode: str,
    destinations: tuple[int, ...],
    calls: list[str],
    payloads: list[dict[str, Any]],
):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": MODEL_ID}]}, request=request
            )
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404, request=request)

        body = json.loads(request.content.decode("utf-8"))
        payloads.append(body)
        if mode == "malformed":
            content = "{not-json"
        elif mode == "wrong-length":
            content = json.dumps({"destinations": [0], "reason": "bad"})
        elif mode == "out-of-range":
            content = json.dumps({"destinations": [999, 0, 0, 0, 0], "reason": "bad"})
        elif mode == "effect":
            content = json.dumps({"destinations": [0, 0, 0, 0, 0], "reason": "bad"})
        elif mode == "empty":
            content = ""
        elif mode == "extra-key":
            content = json.dumps(
                {
                    "destinations": list(destinations),
                    "reason": "ok",
                    "extra": True,
                }
            )
        else:
            content = json.dumps(
                {"destinations": list(destinations), "reason": "valid"}
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    return handler


def test_health_and_generation_use_pinned_schema_request(
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
    wrapped_env: TarwareEnvWrapper,
) -> None:
    request = _request(wrapped_env, vocabulary)
    expected = registry.execute(1, request.context, wrapped_env)
    destinations = tuple(item.destination_id for item in expected.actions)
    calls: list[str] = []
    payloads: list[dict[str, Any]] = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            _response_handler("valid", destinations, calls, payloads)
        )
    )
    actioner = LlamaCppActioner(_config(), GemmaPromptBuilder(), registry, client)
    try:
        health = actioner.health()
        result = actioner.act(request, wrapped_env)
    finally:
        actioner.close()

    assert health.healthy
    assert health.model_id == MODEL_ID
    assert health.advertised_model_ids == (MODEL_ID,)
    assert result.plan.action == expected
    assert result.plan.directive_id == 1
    assert json.loads(result.raw_text)["destinations"] == list(destinations)
    assert calls == ["/health", "/v1/models", "/v1/chat/completions"]
    body = payloads[0]
    assert body["model"] == MODEL_ID
    assert body["temperature"] == 0
    assert body["max_tokens"] == 192
    assert body["seed"] == 0
    assert body["response_format"]["type"] == "json_schema"
    schema = body["response_format"]["json_schema"]
    assert schema["name"] == "warehouse_single_action"
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert schema["schema"]["properties"]["destinations"]["minItems"] == 5


@pytest.mark.parametrize(
    ("mode", "error_type"),
    [
        ("malformed", ActionerSchemaError),
        ("wrong-length", ActionerSchemaError),
        ("out-of-range", ActionerDecodeError),
        ("effect", ActionerEffectError),
        ("empty", EmptyActionerResponseError),
        ("extra-key", ActionerSchemaError),
    ],
)
def test_generation_failures_are_strict_and_do_not_retry(
    mode: str,
    error_type: type[Exception],
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
    wrapped_env: TarwareEnvWrapper,
) -> None:
    request = _request(wrapped_env, vocabulary)
    expected = registry.execute(1, request.context, wrapped_env)
    destinations = tuple(item.destination_id for item in expected.actions)
    calls: list[str] = []
    payloads: list[dict[str, Any]] = []
    client = httpx.Client(
        transport=httpx.MockTransport(
            _response_handler(mode, destinations, calls, payloads)
        )
    )
    actioner = LlamaCppActioner(_config(), GemmaPromptBuilder(), registry, client)
    try:
        with pytest.raises(error_type):
            actioner.act(request, wrapped_env)
    finally:
        actioner.close()

    assert calls == ["/health", "/v1/models", "/v1/chat/completions"]
    assert len(payloads) == 1


def test_health_rejects_advertised_model_mismatch(
    registry: DirectiveRegistry,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        return httpx.Response(
            200,
            json={"data": [{"id": "wrong-model"}]},
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    actioner = LlamaCppActioner(_config(), GemmaPromptBuilder(), registry, client)
    try:
        with pytest.raises(ActionerHealthError, match="was not advertised"):
            actioner.health()
    finally:
        actioner.close()
        client.close()


def test_injected_http_client_remains_owned_by_caller(
    registry: DirectiveRegistry,
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, request=request)
        )
    )
    actioner = LlamaCppActioner(_config(), GemmaPromptBuilder(), registry, client)

    actioner.close()

    assert not client.is_closed
    client.close()


def test_preflight_report_records_independent_gates() -> None:
    now = datetime.now(timezone.utc)
    health = ModelHealth(
        base_url=cast(Any, "http://testserver"),
        model_id=MODEL_ID,
        advertised_model_ids=(MODEL_ID,),
        status="ok",
        healthy=True,
        checked_at_utc=now,
    )
    report = GemmaPreflightReport(
        selected_model="e4b",
        model_id=MODEL_ID,
        model_file_sha256="0" * 64,
        base_url=cast(Any, "http://testserver"),
        context_size=4096,
        gpu_layers=999,
        prompt_schema_sha256="1" * 64,
        health=health,
        warmup_completed=True,
        measured_requests=5,
        valid_responses=5,
        invalid_responses=0,
        directive_classes=("service_requested_shelf", "reposition_idle_agent"),
        sensitivity_states=3,
        sensitivity_pairs=6,
        sensitivity_valid_pairs=6,
        latency_seconds=(0.1, 0.2, 0.3, 0.4, 0.5),
        latency_median_seconds=0.3,
        latency_p95_seconds=0.49,
        p95_limit_seconds=10.0,
        validity_passed=True,
        sensitivity_passed=True,
        latency_passed=True,
        passed=True,
        hardware={"platform": "test"},
        checked_at_utc=now,
    )

    assert report.model_dump(mode="json")["health"]["healthy"] is True
    assert report.passed

    with pytest.raises(ValidationError, match="cannot exceed"):
        GemmaPreflightReport.model_validate(
            {**report.model_dump(), "valid_responses": 6}
        )
