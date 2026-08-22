"""Actioner protocol, failures, and the Phase 5 local Gemma client."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import ValidationError

from warehouse_ai.demo.directives import DirectiveError, DirectiveRegistry
from warehouse_ai.demo.models import (
    ActionerRequest,
    ActionerResult,
    GemmaActionResponse,
    LlamaCppConfig,
    ModelHealth,
    SingleJointActionPlan,
)
from warehouse_ai.demo.prompting import GemmaPromptBuilder
from warehouse_ai.env import InvalidJointActionError, TarwareEnvWrapper


PlanT = TypeVar("PlanT")


class Actioner(Protocol[PlanT]):
    """Produce a typed plan from one immutable controller request."""

    def act(
        self,
        request: ActionerRequest,
        wrapper: TarwareEnvWrapper,
    ) -> ActionerResult[PlanT]: ...


class ActionerError(RuntimeError):
    """Base error for failures crossing the actioner boundary."""


class ActionerHealthError(ActionerError):
    """The actioner endpoint is unavailable or advertises the wrong model."""


class ActionerTimeoutError(ActionerError):
    """The actioner did not produce a response within its configured budget."""


class ActionerHttpError(ActionerError):
    """The actioner transport returned an unsuccessful response."""


class EmptyActionerResponseError(ActionerError):
    """The actioner returned no usable response content."""


class ActionerSchemaError(ActionerError):
    """The actioner response did not satisfy the expected plan schema."""


class ActionerDecodeError(ActionerError):
    """The proposed destinations could not pass wrapper decoding."""


class ActionerEffectError(ActionerError):
    """The proposed action contradicted the selected directive effect."""


class StaleActionPlanError(ActionerError):
    """The visible state changed while an actioner response was in flight."""


class LlamaCppActioner(Actioner[SingleJointActionPlan]):
    """Call a pinned local llama.cpp OpenAI-compatible server.

    The client deliberately performs no retry. The controller owns the
    deterministic directive-executor fallback, which keeps a failed model
    request visible in the decision record instead of hiding it in transport
    behavior. An injected :class:`httpx.Client` is never closed by this class.
    """

    def __init__(
        self,
        config: LlamaCppConfig,
        prompt_builder: GemmaPromptBuilder,
        registry: DirectiveRegistry,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.prompt_builder = prompt_builder
        self.registry = registry
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.request_timeout_seconds,
            headers={
                "accept": "application/json",
                "content-type": "application/json",
            },
        )
        self._health: ModelHealth | None = None

    def health(self) -> ModelHealth:
        """Verify endpoint readiness and the advertised model identity.

        llama.cpp exposes a lightweight ``/health`` endpoint and an
        OpenAI-compatible ``/v1/models`` endpoint. Both are checked before a
        generation is accepted. The successful result is cached for the life
        of this actioner; callers can explicitly call ``health`` again to
        refresh it before a new episode.
        """
        health_response = self._get(
            "/health", error_type="health", timeout_message="health check timed out"
        )
        if not 200 <= health_response.status_code < 300:
            raise ActionerHealthError(
                f"llama.cpp health endpoint returned HTTP "
                f"{health_response.status_code}: {self._response_detail(health_response)}"
            )

        models_response = self._get(
            "/v1/models",
            error_type="health",
            timeout_message="model identity check timed out",
        )
        if not 200 <= models_response.status_code < 300:
            raise ActionerHealthError(
                f"llama.cpp model endpoint returned HTTP "
                f"{models_response.status_code}: {self._response_detail(models_response)}"
            )

        try:
            models_payload = self._json_mapping(
                models_response, "model identity response was not valid JSON"
            )
        except ActionerSchemaError as exc:
            raise ActionerHealthError(
                "llama.cpp model identity response was not valid JSON"
            ) from exc
        advertised_ids = self._advertised_model_ids(models_payload)
        if self.config.model_id not in advertised_ids:
            raise ActionerHealthError(
                f"configured model {self.config.model_id!r} was not advertised; "
                f"server models={advertised_ids!r}"
            )

        health_payload = self._try_json_mapping(health_response)
        status = "ok"
        if health_payload is not None and isinstance(health_payload.get("status"), str):
            status = health_payload["status"].strip() or "ok"
        elif health_response.text.strip():
            status = health_response.text.strip()

        result = ModelHealth(
            base_url=self.config.base_url,
            model_id=self.config.model_id,
            advertised_model_ids=advertised_ids,
            status=status,
            healthy=True,
            checked_at_utc=datetime.now(timezone.utc),
        )
        self._health = result
        return result

    def act(
        self,
        request: ActionerRequest,
        wrapper: TarwareEnvWrapper,
    ) -> ActionerResult[SingleJointActionPlan]:
        """Generate, parse, and strictly validate one concrete joint action."""
        if self._health is None:
            self.health()

        envelope = self.prompt_builder.build(request)
        payload = {
            "model": self.config.model_id,
            "messages": [
                message.model_dump(mode="json") for message in envelope.messages
            ],
            "temperature": self.config.temperature,
            "top_p": 1.0,
            "max_tokens": self.config.max_new_tokens,
            "seed": self.config.seed,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "warehouse_single_action",
                    "strict": True,
                    "schema": envelope.response_schema,
                },
            },
        }

        started = time.perf_counter()
        try:
            response = self._client.post(
                self._endpoint("/v1/chat/completions"),
                json=payload,
                timeout=self.config.request_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ActionerTimeoutError("llama.cpp generation timed out") from exc
        except httpx.RequestError as exc:
            raise ActionerHttpError("llama.cpp generation request failed") from exc
        latency_seconds = time.perf_counter() - started

        if not 200 <= response.status_code < 300:
            raise ActionerHttpError(
                f"llama.cpp generation returned HTTP {response.status_code}: "
                f"{self._response_detail(response)}"
            )

        response_payload = self._json_mapping(
            response, "llama.cpp response was not valid JSON"
        )
        raw_text = self._content(response_payload)
        try:
            decoded = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ActionerSchemaError(
                "llama.cpp content was not a JSON object"
            ) from exc

        try:
            parsed = GemmaActionResponse.model_validate(decoded)
        except ValidationError as exc:
            raise ActionerSchemaError(
                "llama.cpp content did not match the single-action schema"
            ) from exc

        expected_length = len(request.context.observation.agents)
        if len(parsed.destinations) != expected_length:
            raise ActionerSchemaError(
                f"llama.cpp returned {len(parsed.destinations)} destinations; "
                f"expected {expected_length}"
            )

        try:
            action = wrapper.decode(parsed.destinations)
        except (InvalidJointActionError, ValueError, TypeError) as exc:
            raise ActionerDecodeError(
                "llama.cpp destinations "
                f"{tuple(parsed.destinations)!r} failed strict wrapper validation"
            ) from exc

        directive_id = request.prediction.directive.directive_id
        try:
            self.registry.validate_effect(directive_id, request.context, action)
        except DirectiveError as exc:
            raise ActionerEffectError(
                "llama.cpp action violated the selected directive effect for "
                f"destinations {tuple(parsed.destinations)!r}"
            ) from exc

        plan = SingleJointActionPlan(
            state_id=request.context.state_id,
            directive_id=directive_id,
            action=action,
            reason=parsed.reason,
        )
        return ActionerResult[SingleJointActionPlan](
            plan=plan,
            raw_text=raw_text,
            model_id=self.config.model_id,
            latency_ms=latency_seconds * 1000.0,
        )

    def close(self) -> None:
        """Close only a client owned by this actioner."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LlamaCppActioner:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _endpoint(self, path: str) -> str:
        return f"{str(self.config.base_url).rstrip('/')}/{path.lstrip('/')}"

    def _get(
        self,
        path: str,
        *,
        error_type: str,
        timeout_message: str,
    ) -> httpx.Response:
        try:
            return self._client.get(
                self._endpoint(path), timeout=self.config.request_timeout_seconds
            )
        except httpx.TimeoutException as exc:
            if error_type == "health":
                raise ActionerTimeoutError(timeout_message) from exc
            raise ActionerHttpError(timeout_message) from exc
        except httpx.RequestError as exc:
            if error_type == "health":
                raise ActionerHealthError("llama.cpp health request failed") from exc
            raise ActionerHttpError("llama.cpp request failed") from exc

    @staticmethod
    def _json_mapping(response: httpx.Response, message: str) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise ActionerSchemaError(message) from exc
        if not isinstance(payload, Mapping):
            raise ActionerSchemaError(message)
        return payload

    @staticmethod
    def _try_json_mapping(response: httpx.Response) -> Mapping[str, Any] | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        return payload if isinstance(payload, Mapping) else None

    @staticmethod
    def _advertised_model_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
        records = payload.get("data", payload.get("models"))
        if not isinstance(records, list):
            raise ActionerHealthError(
                "llama.cpp model identity response has no data/models list"
            )
        model_ids: list[str] = []
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
                raise ActionerHealthError(
                    "llama.cpp model identity response contains an invalid model"
                )
            model_id = record["id"].strip()
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
        if not model_ids:
            raise ActionerHealthError(
                "llama.cpp model identity response advertised no models"
            )
        return tuple(model_ids)

    @staticmethod
    def _content(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ActionerSchemaError("llama.cpp response has no choices[0] message")
        first = choices[0]
        if not isinstance(first, Mapping):
            raise ActionerSchemaError("llama.cpp choice is not an object")
        message = first.get("message")
        if not isinstance(message, Mapping):
            raise ActionerSchemaError("llama.cpp choice has no message object")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise EmptyActionerResponseError("llama.cpp response content was empty")
        return content

    @staticmethod
    def _response_detail(response: httpx.Response) -> str:
        detail = response.text.strip().replace("\n", " ")
        return detail[:500] if detail else "no response body"
