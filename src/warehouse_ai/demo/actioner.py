"""Deprecated inference imports; prefer ``warehouse_ai.demo.inference``."""

from warehouse_ai.demo.inference.actioner import Actioner, ActionerError, GemmaActioner
from warehouse_ai.demo.inference.backend import InferenceError, ModelBackend
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend

__all__ = [
    "Actioner",
    "ActionerError",
    "GemmaActioner",
    "InferenceError",
    "LlamaCppBackend",
    "ModelBackend",
]
