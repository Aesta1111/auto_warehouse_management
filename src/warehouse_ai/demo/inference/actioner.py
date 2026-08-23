"""Prompt/parse actioner independent of model transport and simulator state."""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import ValidationError

from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    FailureCode,
    LlmActionResponse,
)
from warehouse_ai.demo.inference.backend import InferenceError, ModelBackend
from warehouse_ai.demo.inference.prompting import LlmPromptBuilder


class ActionerError(RuntimeError):
    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class Actioner(Protocol):
    def act(self, request: ActionerRequest) -> ActionerResult: ...


class LlmActioner:
    def __init__(self, backend: ModelBackend, prompts: LlmPromptBuilder) -> None:
        self.backend = backend
        self.prompts = prompts

    def act(self, request: ActionerRequest) -> ActionerResult:
        try:
            generated = self.backend.generate(self.prompts.build(request))
        except InferenceError as exc:
            raise ActionerError(exc.code, str(exc)) from exc
        try:
            decoded = json.loads(generated.raw_text)
            response = LlmActionResponse.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ActionerError(
                FailureCode.MALFORMED_RESPONSE,
                "model content did not match the action schema",
            ) from exc
        return ActionerResult(
            response=response,
            raw_text=generated.raw_text,
            model_id=generated.model_id,
            latency_ms=generated.latency_ms,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
        )
