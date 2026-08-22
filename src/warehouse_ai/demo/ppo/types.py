"""Small immutable tensor records shared by PPO components."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class PpoForwardOutput:
    policy_logits: Tensor
    values: Tensor
    policy_latent: Tensor
    value_latent: Tensor


@dataclass(frozen=True)
class PpoActionSample:
    actions: Tensor
    log_probabilities: Tensor
    values: Tensor


@dataclass(frozen=True)
class PpoLossTerms:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
