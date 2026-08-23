"""Bounded compatibility-spike helpers, not production hybrid training code."""

from warehouse_ai.compatibility.llm import (
    LLM_BASE_MODEL_ID,
    LLM_GGUF_FILENAME,
    LLM_GGUF_REPOSITORY,
    LLM_GGUF_REVISION,
    LLM_GGUF_SHA256,
    SoftTokenProjector,
    run_tiny_llm_probe,
)

__all__ = [
    "LLM_BASE_MODEL_ID",
    "LLM_GGUF_FILENAME",
    "LLM_GGUF_REPOSITORY",
    "LLM_GGUF_REVISION",
    "LLM_GGUF_SHA256",
    "SoftTokenProjector",
    "run_tiny_llm_probe",
]
