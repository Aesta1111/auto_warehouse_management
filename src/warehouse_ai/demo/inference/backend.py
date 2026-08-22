"""Inference backend boundary and structured failures."""

from __future__ import annotations

from typing import Protocol

from warehouse_ai.demo.contracts.inference import (
    FailureCode,
    GenerationResult,
    ModelHealth,
    PromptEnvelope,
)


class InferenceError(RuntimeError):
    def __init__(self, code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class ModelBackend(Protocol):
    def health(self) -> ModelHealth: ...

    def generate(self, prompt: PromptEnvelope) -> GenerationResult: ...

    def close(self) -> None: ...
