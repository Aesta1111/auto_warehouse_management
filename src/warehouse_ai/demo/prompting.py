"""Deprecated prompt import; prefer ``demo.inference.prompting``."""

from warehouse_ai.demo.contracts.inference import ChatMessage, PromptEnvelope
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder

__all__ = ["ChatMessage", "GemmaPromptBuilder", "PromptEnvelope"]
