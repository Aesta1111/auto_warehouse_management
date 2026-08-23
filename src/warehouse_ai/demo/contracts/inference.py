"""Inference, prompting, and preflight contracts."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, Literal, Self

from pydantic import (
    AnyHttpUrl,
    Field,
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from warehouse_ai.demo.contracts.base import DemoModel, NonBlankStr, Sha256, UtcDateTime
from warehouse_ai.demo.contracts.directives import (
    Directive,
    WarehouseContext,
    PpoPolicyOutput,
)


class FailureCode(str, Enum):
    HEALTH = "health"
    TIMEOUT = "timeout"
    TRANSPORT = "transport"
    MALFORMED_RESPONSE = "malformed_response"
    INVALID_ACTION = "invalid_action"
    EFFECT_VIOLATION = "effect_violation"
    STALE_STATE = "stale_state"
    EXECUTOR_FAILURE = "executor_failure"


class ActionerRequest(DemoModel):
    context: WarehouseContext
    prediction: PpoPolicyOutput | None = None
    selected_directive: Directive | None = None

    @model_validator(mode="after")
    def validate_prediction_state(self) -> Self:
        if (self.prediction is None) == (self.selected_directive is None):
            raise ValueError("exactly one directive source is required")
        if (
            self.prediction is not None
            and self.context.state_id != self.prediction.state_id
        ):
            raise ValueError("actioner request context and prediction states differ")
        return self

    @property
    def directive(self) -> Directive:
        if self.prediction is not None:
            return self.prediction.directive
        assert self.selected_directive is not None
        return self.selected_directive


class GemmaActionResponse(DemoModel):
    destinations: tuple[NonNegativeInt, ...] = Field(min_length=1)
    reason: str = Field(max_length=160)


class ActionerResult(DemoModel):
    response: GemmaActionResponse
    raw_text: str
    model_id: NonBlankStr
    latency_ms: NonNegativeFloat
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None


class GenerationResult(DemoModel):
    raw_text: str
    model_id: NonBlankStr
    latency_ms: NonNegativeFloat
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None


class ChatMessage(DemoModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PromptEnvelope(DemoModel):
    messages: tuple[ChatMessage, ...]
    response_schema: dict[str, Any]


class LlamaCppBackendConfig(DemoModel):
    backend: Literal["llama_cpp"] = "llama_cpp"
    base_url: AnyHttpUrl
    model_id: NonBlankStr
    model_file_sha256: Sha256
    model_revision: NonBlankStr | None = None
    model_filename: NonBlankStr | None = None
    quantization: NonBlankStr | None = None
    runtime_version: NonBlankStr | None = None
    server_command: NonBlankStr | None = None
    context_size: PositiveInt
    gpu_layers: NonNegativeInt
    request_timeout_seconds: PositiveFloat = 12.0
    max_new_tokens: PositiveInt = 192
    temperature: Literal[0] = 0
    seed: int = 0


class ModelHealth(DemoModel):
    schema_version: Literal[1, 2] = 2
    base_url: AnyHttpUrl
    model_id: NonBlankStr
    advertised_model_ids: tuple[NonBlankStr, ...] = Field(min_length=1)
    status: NonBlankStr
    healthy: bool
    checked_at_utc: UtcDateTime

    @field_validator("advertised_model_ids")
    @classmethod
    def validate_models(cls, value: Sequence[str]) -> tuple[str, ...]:
        models = tuple(value)
        if len(set(models)) != len(models):
            raise ValueError("advertised_model_ids must be unique")
        return models


class GemmaPreflightReport(DemoModel):
    schema_version: Literal[1, 2, 3] = 2
    selected_model: Literal["e4b", "e2b"]
    backend: Literal["llama_cpp"] = "llama_cpp"
    model_id: NonBlankStr
    model_file_sha256: Sha256
    base_url: AnyHttpUrl
    context_size: PositiveInt
    gpu_layers: NonNegativeInt
    prompt_schema_sha256: Sha256
    prompt_contract_sha256: Sha256 | None = None
    health: ModelHealth
    warmup_completed: bool
    measured_requests: NonNegativeInt
    valid_responses: NonNegativeInt
    invalid_responses: NonNegativeInt
    directive_classes: tuple[NonBlankStr, ...] = ()
    sensitivity_states: NonNegativeInt = 0
    sensitivity_pairs: NonNegativeInt = 0
    sensitivity_valid_pairs: NonNegativeInt = 0
    sensitivity_results: tuple[dict[str, JsonValue], ...] = ()
    latency_seconds: tuple[NonNegativeFloat, ...] = ()
    latency_median_seconds: NonNegativeFloat = 0.0
    latency_p95_seconds: NonNegativeFloat = 0.0
    p95_limit_seconds: PositiveFloat
    validity_passed: bool
    sensitivity_passed: bool
    latency_passed: bool
    passed: bool
    hardware: dict[str, JsonValue] = Field(default_factory=dict)
    server_command: str | None = None
    quantization: str | None = None
    checked_at_utc: UtcDateTime
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if self.valid_responses + self.invalid_responses > self.measured_requests:
            raise ValueError("response counts cannot exceed measured_requests")
        if self.sensitivity_valid_pairs > self.sensitivity_pairs:
            raise ValueError("sensitivity_valid_pairs cannot exceed sensitivity_pairs")
        if self.schema_version == 3 and self.prompt_contract_sha256 is None:
            raise ValueError("schema-v3 preflight requires prompt_contract_sha256")
        if self.schema_version < 3 and self.prompt_contract_sha256 is not None:
            raise ValueError(
                "legacy preflight reports cannot contain prompt_contract_sha256"
            )
        return self
