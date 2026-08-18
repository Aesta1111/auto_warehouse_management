"""Masked actor-critic model owned by the directive prototype."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from warehouse_ai.demo.artifacts import canonical_json_bytes
from warehouse_ai.demo.ppo_buffer import PpoBatch
from warehouse_ai.demo.ppo_math import compute_ppo_loss


MODEL_IMPLEMENTATION_VERSION = "directive-ppo-module-v1"


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


def architecture_digest(
    observation_size: int, directive_count: int, hidden_size: int
) -> str:
    payload = {
        "directive_count": directive_count,
        "hidden_activation": "tanh",
        "hidden_size": hidden_size,
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "observation_size": observation_size,
        "separate_policy_and_value_paths": True,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def implementation_digest() -> str:
    """Hash inference-affecting source rather than unrelated repository files."""
    paths = (Path(__file__), Path(__file__).with_name("ppo_math.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


class DirectivePpoModule(pl.LightningModule):
    """Separate policy/value paths with explicit valid-action masking."""

    def __init__(
        self,
        observation_size: int,
        directive_count: int,
        hidden_size: int,
        learning_rate: float,
        clip_range: float,
        entropy_coefficient: float,
        value_coefficient: float,
        normalize_advantage: bool,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or directive_count <= 0 or hidden_size <= 0:
            raise ValueError("model dimensions must be positive")
        self.observation_size = int(observation_size)
        self.directive_count = int(directive_count)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.normalize_advantage = bool(normalize_advantage)

        self.policy_path = nn.Sequential(
            nn.Linear(self.observation_size, self.hidden_size), nn.Tanh()
        )
        self.policy_head = nn.Linear(self.hidden_size, self.directive_count)
        self.value_path = nn.Sequential(
            nn.Linear(self.observation_size, self.hidden_size), nn.Tanh()
        )
        self.value_head = nn.Linear(self.hidden_size, 1)

    def _validate_inputs(self, observations: Tensor, action_masks: Tensor) -> None:
        if observations.ndim != 2 or observations.shape[1] != self.observation_size:
            raise ValueError("observations must have shape (batch, observation_size)")
        if action_masks.shape != (observations.shape[0], self.directive_count):
            raise ValueError("action_masks must have shape (batch, directive_count)")
        if action_masks.dtype is not torch.bool:
            raise TypeError("action_masks must use torch.bool")
        if not bool(torch.all(action_masks.any(dim=1))):
            raise ValueError("every observation must allow at least one directive")

    def forward(self, observations: Tensor, action_masks: Tensor) -> PpoForwardOutput:
        self._validate_inputs(observations, action_masks)
        policy_latent = self.policy_path(observations)
        value_latent = self.value_path(observations)
        unmasked_logits = self.policy_head(policy_latent)
        policy_logits = unmasked_logits.masked_fill(
            ~action_masks, torch.finfo(unmasked_logits.dtype).min
        )
        values = self.value_head(value_latent).squeeze(-1)
        return PpoForwardOutput(
            policy_logits=policy_logits,
            values=values,
            policy_latent=policy_latent,
            value_latent=value_latent,
        )

    def probabilities(self, observations: Tensor, action_masks: Tensor) -> Tensor:
        output = self(observations, action_masks)
        probabilities = torch.softmax(output.policy_logits, dim=-1)
        return torch.where(action_masks, probabilities, torch.zeros_like(probabilities))

    def act(
        self,
        observations: Tensor,
        action_masks: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PpoActionSample:
        output = self(observations, action_masks)
        probabilities = torch.softmax(output.policy_logits, dim=-1)
        probabilities = torch.where(
            action_masks, probabilities, torch.zeros_like(probabilities)
        )
        if deterministic:
            actions = torch.argmax(probabilities, dim=-1)
        else:
            actions = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            ).squeeze(-1)
        selected = probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
        return PpoActionSample(
            actions=actions,
            log_probabilities=torch.log(selected),
            values=output.values,
        )

    def evaluate_actions(self, batch: PpoBatch) -> tuple[Tensor, Tensor, Tensor]:
        output = self(batch.observations, batch.action_masks)
        distribution = torch.distributions.Categorical(logits=output.policy_logits)
        return (
            distribution.log_prob(batch.actions),
            output.values,
            distribution.entropy(),
        )

    def compute_loss(self, batch: PpoBatch) -> PpoLossTerms:
        log_probabilities, values, entropy = self.evaluate_actions(batch)
        return compute_ppo_loss(
            batch,
            log_probabilities,
            values,
            entropy,
            clip_range=self.clip_range,
            entropy_coefficient=self.entropy_coefficient,
            value_coefficient=self.value_coefficient,
            normalize_advantage=self.normalize_advantage,
        )

    def training_step(self, batch: PpoBatch, batch_idx: int) -> Tensor:
        del batch_idx
        terms = self.compute_loss(batch)
        self.log_dict(
            {
                "train/total_loss": terms.total,
                "train/policy_loss": terms.policy,
                "train/value_loss": terms.value,
                "train/entropy": terms.entropy,
                "train/approximate_kl": terms.approximate_kl,
                "train/clip_fraction": terms.clip_fraction,
            }
        )
        return terms.total

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)
