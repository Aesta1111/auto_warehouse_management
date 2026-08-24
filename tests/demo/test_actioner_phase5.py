from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from warehouse_ai.demo.contracts.configuration import load_ppo_evaluation_run
from warehouse_ai.demo.contracts.directives import (
    WarehouseContext,
    PpoPolicyOutput,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    FailureCode,
    LlmPreflightReport,
    GenerationResult,
    LlamaCppBackendConfig,
    ModelHealth,
    PromptEnvelope,
)
from warehouse_ai.demo.evaluation.verification import verify_prompt_contract
from warehouse_ai.demo.inference.actioner import ActionerError, LlmActioner
from warehouse_ai.demo.inference.backend import InferenceError, ModelBackend
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend
from warehouse_ai.demo.inference.prompting import LlmPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"
MODEL_ID = "google/llm-4-E4B-it"


def _config() -> LlamaCppBackendConfig:
    return LlamaCppBackendConfig(
        base_url=cast(Any, "http://testserver"),
        model_id=MODEL_ID,
        model_file_sha256="0" * 64,
        context_size=4096,
        gpu_layers=24,
        request_timeout_seconds=2.0,
    )


def _request() -> tuple[TarwareEnvWrapper, ActionerRequest]:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")
    wrapper = TarwareEnvWrapper(ENVIRONMENT_ID)
    wrapper.reset(seed=0)
    context = WarehouseContext.capture(wrapper)
    return wrapper, ActionerRequest(
        context=context,
        prediction=PpoPolicyOutput(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=catalog.by_id(1),
            probabilities=(0.0, 1.0, 0.0, 0.0),
            value_estimate=0.0,
            policy_latency_ms=0.0,
            checkpoint_sha256="0" * 64,
        ),
    )


class FakeBackend:
    def __init__(self, raw_text: str) -> None:
        self.raw_text = raw_text
        self.prompt: PromptEnvelope | None = None

    def health(self) -> ModelHealth:
        raise AssertionError("health is not used by LlmActioner")

    def generate(self, prompt: PromptEnvelope) -> GenerationResult:
        self.prompt = prompt
        return GenerationResult(
            raw_text=self.raw_text,
            model_id=MODEL_ID,
            latency_ms=2.5,
            prompt_tokens=12,
            completion_tokens=5,
        )

    def close(self) -> None:
        pass


def test_actioner_is_backend_neutral_and_only_parses_messages() -> None:
    wrapper, request = _request()
    backend = FakeBackend('{"destinations":[1,0,0,0,0],"reason":"ok"}')
    try:
        result = LlmActioner(cast(ModelBackend, backend), LlmPromptBuilder()).act(
            request
        )
        assert result.response.destinations == (1, 0, 0, 0, 0)
        assert result.prompt_tokens == 12
        assert backend.prompt is not None
        assert backend.prompt.response_schema["additionalProperties"] is False
    finally:
        wrapper.close()


@pytest.mark.parametrize(
    "content",
    [
        "{not-json",
        '{"destinations":[1,0,0,0,0],"reason":"ok","extra":true}',
        '{"destinations":[1,0,0,0,0]}',
    ],
)
def test_actioner_rejects_malformed_response_once(content: str) -> None:
    wrapper, request = _request()
    try:
        with pytest.raises(ActionerError) as error:
            LlmActioner(
                cast(ModelBackend, FakeBackend(content)), LlmPromptBuilder()
            ).act(request)
        assert error.value.code is FailureCode.MALFORMED_RESPONSE
    finally:
        wrapper.close()


def test_llama_cpp_adapter_parses_health_and_openai_envelope() -> None:
    calls: list[str] = []
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200, json={"data": [{"id": MODEL_ID}]}, request=request
            )
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"destinations":[0]}'}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
            request=request,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = LlamaCppBackend(_config(), client)
    wrapper, request = _request()
    try:
        prompt = LlmPromptBuilder().build(request)
        result = backend.generate(prompt)
        assert result.raw_text == '{"destinations":[0]}'
        assert result.prompt_tokens == 7
        assert calls == ["/health", "/v1/models", "/v1/chat/completions"]
        envelope = payloads[0]
        assert envelope["temperature"] == 0
        assert envelope["response_format"]["json_schema"]["strict"] is True
    finally:
        backend.close()
        client.close()
        wrapper.close()


def test_llama_cpp_health_rejects_model_mismatch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = (
            {"status": "ok"}
            if request.url.path == "/health"
            else {"data": [{"id": "wrong"}]}
        )
        return httpx.Response(200, json=payload, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(InferenceError) as error:
            LlamaCppBackend(_config(), client).health()
        assert error.value.code is FailureCode.HEALTH
    finally:
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
    report = LlmPreflightReport(
        schema_version=2,
        selected_model="e4b",
        model_id=MODEL_ID,
        model_file_sha256="0" * 64,
        base_url=cast(Any, "http://testserver"),
        context_size=4096,
        gpu_layers=24,
        prompt_schema_sha256="1" * 64,
        health=health,
        warmup_completed=True,
        measured_requests=5,
        valid_responses=5,
        invalid_responses=0,
        sensitivity_states=3,
        sensitivity_pairs=6,
        sensitivity_valid_pairs=6,
        latency_seconds=(0.1, 0.2, 0.3, 0.4, 0.5),
        latency_median_seconds=0.3,
        latency_p95_seconds=0.49,
        p95_limit_seconds=3.92,
        validity_passed=True,
        sensitivity_passed=True,
        latency_passed=True,
        passed=True,
        checked_at_utc=now,
    )
    assert report.passed
    with pytest.raises(ValidationError, match="cannot exceed"):
        LlmPreflightReport.model_validate({**report.model_dump(), "valid_responses": 6})


def test_preflight_fails_closed_on_prompt_contract_mismatch() -> None:
    config = load_ppo_evaluation_run(
        Path("configs/demo/ppo-llm-evaluation-run.yaml"), ROOT
    ).demo_runtime()
    catalog = DirectiveCatalog.load(ROOT / config.vocabulary_path)
    assert config.prompt_spec_path is not None
    builder = LlmPromptBuilder.load(ROOT / config.prompt_spec_path, catalog)
    now = datetime.now(timezone.utc)
    health = ModelHealth(
        base_url=cast(Any, "http://testserver"),
        model_id=MODEL_ID,
        advertised_model_ids=(MODEL_ID,),
        status="ok",
        healthy=True,
        checked_at_utc=now,
    )
    report = LlmPreflightReport(
        schema_version=5,
        selected_model="qwen3.5-2b-q8",
        model_id=MODEL_ID,
        model_file_sha256="0" * 64,
        base_url=cast(Any, "http://testserver"),
        context_size=8192,
        gpu_layers=99,
        prompt_schema_sha256="1" * 64,
        prompt_contract_sha256="f" * 64,
        prompt_template_sha256="0" * 64,
        prompt_template_sha256_by_behavior={"placeholder": "0" * 64},
        health=health,
        warmup_completed=True,
        measured_requests=5,
        valid_responses=5,
        invalid_responses=0,
        sensitivity_states=3,
        sensitivity_pairs=6,
        sensitivity_valid_pairs=6,
        latency_seconds=(0.1, 0.2, 0.3, 0.4, 0.5),
        latency_median_seconds=0.3,
        latency_p95_seconds=0.49,
        p95_limit_seconds=3.92,
        validity_passed=True,
        sensitivity_passed=True,
        latency_passed=True,
        passed=True,
        checked_at_utc=now,
    )
    with pytest.raises(ValueError, match="prompt contract"):
        verify_prompt_contract(config, report, builder)

    matching = report.model_copy(
        update={
            "prompt_contract_sha256": builder.contract_digest,
            "prompt_template_sha256": builder.template_digest,
            "prompt_template_sha256_by_behavior": builder.template_digests,
        }
    )
    verify_prompt_contract(config, matching, builder)

    incomplete = matching.model_copy(update={"passed": False})
    verify_prompt_contract(config, incomplete, builder)
    strict_config = config.model_copy(update={"require_passing_preflight": True})
    with pytest.raises(ValueError, match="did not pass"):
        verify_prompt_contract(strict_config, incomplete, builder)

    wrong_per_behavior = matching.model_copy(
        update={"prompt_template_sha256_by_behavior": {"placeholder": "0" * 64}}
    )
    with pytest.raises(ValueError, match="per-behavior"):
        verify_prompt_contract(config, wrong_per_behavior, builder)
