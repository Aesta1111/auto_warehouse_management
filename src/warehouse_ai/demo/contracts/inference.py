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
from warehouse_ai.demo.contracts.directives import WarehouseContext, PpoPolicyOutput


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
    prediction: PpoPolicyOutput

    @model_validator(mode="after")
    def validate_prediction_state(self) -> Self:
        if self.context.state_id != self.prediction.state_id:
            raise ValueError("actioner request context and prediction states differ")
        return self


class GemmaActionResponse(DemoModel):
    destinations: tuple[NonNegativeInt, ...] = Field(min_length=1)
    reason: str = Field(max_length=160)


DirectiveBehaviorKey = Literal[
    "wait",
    "service_requested_shelf",
    "deliver_carried_shelf",
    "reposition_idle_agent",
]
ExampleAgentRole = Literal["agv", "picker"]


class FewShotExample(DemoModel):
    """One compact synthetic state and its fully validated action response."""

    behavior_key: DirectiveBehaviorKey
    agent_roles: tuple[ExampleAgentRole, ...] = Field(min_length=5, max_length=5)
    idle_agent_positions: tuple[NonNegativeInt, ...]
    carrying_shelf_positions: tuple[NonNegativeInt, ...] = ()
    requested_destination_ids: tuple[PositiveInt, ...] = ()
    goal_destination_ids: tuple[PositiveInt, ...] = ()
    legal_destinations: tuple[tuple[NonNegativeInt, ...], ...] = Field(
        min_length=5, max_length=5
    )
    response: GemmaActionResponse

    @model_validator(mode="after")
    def validate_example(self) -> Self:
        agent_count = len(self.agent_roles)
        position_fields = {
            "idle_agent_positions": self.idle_agent_positions,
            "carrying_shelf_positions": self.carrying_shelf_positions,
        }
        for name, positions in position_fields.items():
            if len(set(positions)) != len(positions):
                raise ValueError(f"{name} must not contain duplicates")
            if any(position >= agent_count for position in positions):
                raise ValueError(f"{name} contains an out-of-range position")
        for name, destinations in (
            ("requested_destination_ids", self.requested_destination_ids),
            ("goal_destination_ids", self.goal_destination_ids),
        ):
            if len(set(destinations)) != len(destinations):
                raise ValueError(f"{name} must not contain duplicates")
        if len(self.response.destinations) != agent_count:
            raise ValueError("example response must contain one destination per agent")
        for row, legal in enumerate(self.legal_destinations):
            if not legal or 0 not in legal:
                raise ValueError(
                    "every example legal-destination row must contain zero"
                )
            if len(set(legal)) != len(legal):
                raise ValueError(
                    "example legal destinations must not contain duplicates"
                )
            if self.response.destinations[row] not in legal:
                raise ValueError("example response contains an illegal destination")
        selected_by_role: set[tuple[str, int]] = set()
        for role, destination in zip(
            self.agent_roles, self.response.destinations, strict=True
        ):
            if destination == 0:
                continue
            pair = (role, destination)
            if pair in selected_by_role:
                raise ValueError(
                    "example response contains a same-role duplicate destination"
                )
            selected_by_role.add(pair)
        if not self.response.reason.strip():
            raise ValueError("example response reason must not be blank")

        idle = set(self.idle_agent_positions)
        carrying = set(self.carrying_shelf_positions)
        requested = set(self.requested_destination_ids)
        goals = set(self.goal_destination_ids)
        selected = tuple(
            (row, destination)
            for row, destination in enumerate(self.response.destinations)
            if destination != 0
        )
        if self.behavior_key == "wait":
            valid = not selected
        elif self.behavior_key == "service_requested_shelf":
            valid = bool(selected) and all(
                row in idle and destination in requested
                for row, destination in selected
            )
        elif self.behavior_key == "deliver_carried_shelf":
            valid = bool(selected) and all(
                row in idle
                and row in carrying
                and self.agent_roles[row] == "agv"
                and destination in goals
                for row, destination in selected
            )
        else:
            valid = bool(selected) and all(
                row in idle and row not in carrying and destination not in requested
                for row, destination in selected
            )
        if not valid:
            raise ValueError(
                f"example response does not satisfy {self.behavior_key} instruction"
            )
        return self


class DirectivePrompt(DemoModel):
    behavior_key: DirectiveBehaviorKey
    command: NonBlankStr
    instruction: NonBlankStr
    example: FewShotExample

    @model_validator(mode="after")
    def validate_example_behavior(self) -> Self:
        if self.example.behavior_key != self.behavior_key:
            raise ValueError("example behavior_key must match its directive prompt")
        return self


class GemmaPromptSpec(DemoModel):
    schema_version: Literal[1]
    prompt_id: NonBlankStr
    system_message: NonBlankStr
    directives: tuple[DirectivePrompt, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_directives(self) -> Self:
        expected = (
            "wait",
            "service_requested_shelf",
            "deliver_carried_shelf",
            "reposition_idle_agent",
        )
        actual = tuple(item.behavior_key for item in self.directives)
        if actual != expected:
            raise ValueError(
                "prompt directives must contain all four behavior keys in canonical order"
            )
        commands = tuple(item.command.casefold() for item in self.directives)
        if len(set(commands)) != len(commands):
            raise ValueError("prompt directive commands must be unique")
        return self

    def validate_catalog(self, catalog: Any) -> None:
        """Reject prompt text that is not aligned with the frozen catalog."""
        expected = tuple(
            (item.behavior_key, item.command) for item in catalog.directives
        )
        actual = tuple((item.behavior_key, item.command) for item in self.directives)
        if actual != expected:
            raise ValueError("prompt directives do not match the directive catalog")

    def by_behavior_key(self, behavior_key: str) -> DirectivePrompt:
        for directive in self.directives:
            if directive.behavior_key == behavior_key:
                return directive
        raise KeyError(behavior_key)


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
