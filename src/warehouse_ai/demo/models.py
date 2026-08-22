"""Deprecated compatibility imports; prefer ``warehouse_ai.demo.contracts``."""

from warehouse_ai.demo.contracts.base import DemoModel, load_yaml_config
from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectiveDefinition,
    DirectivePrediction,
    DirectiveProbeConfig,
    DirectiveProbeReport,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.evaluation import (
    DemoDecision,
    DemoRunSummary,
    DemoStepRecord,
    DirectiveDemoConfig,
    EpisodeResult,
)
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    ChatMessage,
    FailureCode,
    GemmaActionResponse,
    GemmaPreflightReport,
    LlamaCppBackendConfig,
    ModelHealth,
    PromptEnvelope,
)
from warehouse_ai.demo.contracts.ppo import (
    DirectivePpoConfig,
    DirectivePpoManifest,
    DirectivePpoValidationSummary,
)

LlamaCppConfig = LlamaCppBackendConfig

__all__ = [
    "ActionerRequest",
    "ActionerResult",
    "ChatMessage",
    "DemoDecision",
    "DemoModel",
    "DemoRunSummary",
    "DemoStepRecord",
    "DirectiveContext",
    "DirectiveDefinition",
    "DirectiveDemoConfig",
    "DirectivePpoConfig",
    "DirectivePpoManifest",
    "DirectivePpoValidationSummary",
    "DirectivePrediction",
    "DirectiveProbeConfig",
    "DirectiveProbeReport",
    "DirectiveVocabulary",
    "EpisodeResult",
    "FailureCode",
    "GemmaActionResponse",
    "GemmaPreflightReport",
    "LlamaCppBackendConfig",
    "LlamaCppConfig",
    "ModelHealth",
    "PromptEnvelope",
    "load_yaml_config",
]
