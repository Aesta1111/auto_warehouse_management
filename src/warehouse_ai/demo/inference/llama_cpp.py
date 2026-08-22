"""Pinned llama.cpp OpenAI-compatible inference adapter."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import httpx

from warehouse_ai.demo.contracts.inference import (
    FailureCode,
    GenerationResult,
    LlamaCppBackendConfig,
    ModelHealth,
    PromptEnvelope,
)
from warehouse_ai.demo.inference.backend import InferenceError


class LlamaCppBackend:
    """Transport-only adapter for one pinned local llama.cpp server."""

    def __init__(
        self,
        config: LlamaCppBackendConfig,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=config.request_timeout_seconds,
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        self._health: ModelHealth | None = None

    def health(self) -> ModelHealth:
        health_response = self._get("/health")
        if not 200 <= health_response.status_code < 300:
            raise InferenceError(
                FailureCode.HEALTH,
                f"health endpoint returned HTTP {health_response.status_code}",
            )
        models_response = self._get("/v1/models")
        if not 200 <= models_response.status_code < 300:
            raise InferenceError(
                FailureCode.HEALTH,
                f"model endpoint returned HTTP {models_response.status_code}",
            )
        payload = self._json_mapping(models_response)
        advertised = self._advertised_model_ids(payload)
        if self.config.model_id not in advertised:
            raise InferenceError(
                FailureCode.HEALTH,
                f"configured model {self.config.model_id!r} is not advertised",
            )
        health_payload = self._try_json_mapping(health_response)
        status = "ok"
        if health_payload is not None and isinstance(health_payload.get("status"), str):
            status = health_payload["status"].strip() or "ok"
        elif health_response.text.strip():
            status = health_response.text.strip()
        self._health = ModelHealth(
            base_url=self.config.base_url,
            model_id=self.config.model_id,
            advertised_model_ids=advertised,
            status=status,
            healthy=True,
            checked_at_utc=datetime.now(timezone.utc),
        )
        return self._health

    def generate(self, prompt: PromptEnvelope) -> GenerationResult:
        if self._health is None:
            self.health()
        payload = {
            "model": self.config.model_id,
            "messages": [
                message.model_dump(mode="json") for message in prompt.messages
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
                    "schema": prompt.response_schema,
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
            raise InferenceError(FailureCode.TIMEOUT, "generation timed out") from exc
        except httpx.RequestError as exc:
            raise InferenceError(
                FailureCode.TRANSPORT, "generation request failed"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        if not 200 <= response.status_code < 300:
            raise InferenceError(
                FailureCode.TRANSPORT,
                f"generation returned HTTP {response.status_code}: {self._detail(response)}",
            )
        response_payload = self._json_mapping(response)
        raw_text = self._content(response_payload)
        prompt_tokens, completion_tokens = self._usage(response_payload)
        return GenerationResult(
            raw_text=raw_text,
            model_id=self.config.model_id,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> LlamaCppBackend:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _endpoint(self, path: str) -> str:
        return f"{str(self.config.base_url).rstrip('/')}/{path.lstrip('/')}"

    def _get(self, path: str) -> httpx.Response:
        try:
            return self._client.get(
                self._endpoint(path), timeout=self.config.request_timeout_seconds
            )
        except httpx.TimeoutException as exc:
            raise InferenceError(FailureCode.TIMEOUT, f"{path} timed out") from exc
        except httpx.RequestError as exc:
            raise InferenceError(FailureCode.HEALTH, f"{path} request failed") from exc

    @staticmethod
    def _json_mapping(response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise InferenceError(
                FailureCode.MALFORMED_RESPONSE, "response was not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise InferenceError(
                FailureCode.MALFORMED_RESPONSE, "response is not an object"
            )
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
            raise InferenceError(FailureCode.HEALTH, "model list is missing")
        model_ids: list[str] = []
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
                raise InferenceError(
                    FailureCode.HEALTH, "model list contains invalid data"
                )
            model_id = record["id"].strip()
            if model_id and model_id not in model_ids:
                model_ids.append(model_id)
        if not model_ids:
            raise InferenceError(FailureCode.HEALTH, "server advertised no models")
        return tuple(model_ids)

    @staticmethod
    def _content(payload: Mapping[str, Any]) -> str:
        choices = payload.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise InferenceError(
                FailureCode.MALFORMED_RESPONSE, "response content is empty"
            )
        return content

    @staticmethod
    def _usage(payload: Mapping[str, Any]) -> tuple[int | None, int | None]:
        usage = payload.get("usage")
        if not isinstance(usage, Mapping):
            return None, None
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        return (
            int(prompt) if isinstance(prompt, int) and prompt >= 0 else None,
            int(completion)
            if isinstance(completion, int) and completion >= 0
            else None,
        )

    @staticmethod
    def _detail(response: httpx.Response) -> str:
        detail = response.text.strip().replace("\n", " ")
        return detail[:500] if detail else "no response body"
