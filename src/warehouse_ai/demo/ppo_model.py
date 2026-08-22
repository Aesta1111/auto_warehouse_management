"""Deprecated PPO model imports; prefer ``demo.ppo.model``."""

from warehouse_ai.demo.ppo.model import (
    MODEL_IMPLEMENTATION_VERSION,
    DirectivePpoModel,
    DirectivePpoModule,
    architecture_digest,
    implementation_digest,
)
from warehouse_ai.demo.ppo.types import PpoActionSample, PpoForwardOutput, PpoLossTerms

__all__ = [
    "MODEL_IMPLEMENTATION_VERSION",
    "DirectivePpoModel",
    "DirectivePpoModule",
    "PpoActionSample",
    "PpoForwardOutput",
    "PpoLossTerms",
    "architecture_digest",
    "implementation_digest",
]
