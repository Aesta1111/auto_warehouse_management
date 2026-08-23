"""Plain PyTorch masked actor-critic model."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

from warehouse_ai.demo.artifacts import canonical_json_bytes
from warehouse_ai.demo.ppo.buffer import PpoBatch
from warehouse_ai.demo.ppo.math import GAE_VERSION, LOSS_VERSION, compute_ppo_loss
from warehouse_ai.demo.ppo.types import PpoActionSample, PpoForwardOutput, PpoLossTerms


LEGACY_MODEL_IMPLEMENTATION_VERSION = "directive-ppo-module-v2"
MODEL_IMPLEMENTATION_VERSION = "directive-ppo-mlp-v1"


def architecture_digest(
    observation_size: int,
    directive_count: int,
    hidden_sizes: Sequence[int],
) -> str:
    payload = {
        "directive_count": directive_count,
        "hidden_activation": "tanh",
        "hidden_sizes": list(hidden_sizes),
        "model_implementation_version": MODEL_IMPLEMENTATION_VERSION,
        "observation_size": observation_size,
        "separate_policy_and_value_paths": True,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def legacy_architecture_digest(
    observation_size: int, directive_count: int, hidden_size: int
) -> str:
    """Reproduce the frozen schema-v3 architecture identity."""
    payload = {
        "directive_count": directive_count,
        "hidden_activation": "tanh",
        "hidden_size": hidden_size,
        "model_implementation_version": LEGACY_MODEL_IMPLEMENTATION_VERSION,
        "observation_size": observation_size,
        "separate_policy_and_value_paths": True,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def implementation_digest(
    model_implementation_version: str = MODEL_IMPLEMENTATION_VERSION,
    gae_version: str = GAE_VERSION,
    loss_version: str = LOSS_VERSION,
) -> str:
    """Digest the declared algorithm identity, independent of file layout."""
    payload = {
        "gae_version": gae_version,
        "loss_version": loss_version,
        "model_implementation_version": model_implementation_version,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


class Mlp(nn.Module):
    """Tanh MLP that exposes its final hidden representation."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
    ) -> None:
        super().__init__()
        sizes = tuple(int(size) for size in hidden_sizes)
        if input_size <= 0 or output_size <= 0 or not sizes:
            raise ValueError("MLP dimensions and hidden sizes must be positive")
        if any(size <= 0 for size in sizes):
            raise ValueError("MLP dimensions and hidden sizes must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_sizes = sizes
        dimensions = (self.input_size, *self.hidden_sizes, self.output_size)
        self.layers = nn.ModuleList(
            nn.Linear(source, destination)
            for source, destination in zip(dimensions, dimensions[1:])
        )

    def forward_with_latent(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        hidden = inputs
        for layer in self.layers[:-1]:
            hidden = torch.tanh(layer(hidden))
        return self.layers[-1](hidden), hidden

    def forward(self, inputs: Tensor) -> Tensor:
        output, _ = self.forward_with_latent(inputs)
        return output


class PpoModel(nn.Module):
    """Separate policy/value paths with explicit valid-action masking."""

    def __init__(
        self,
        observation_size: int,
        directive_count: int,
        hidden_sizes: Sequence[int],
        learning_rate: float,
        clip_range: float,
        entropy_coefficient: float,
        value_coefficient: float,
        normalize_advantage: bool,
    ) -> None:
        super().__init__()
        if observation_size <= 0 or directive_count <= 0:
            raise ValueError("model dimensions must be positive")
        self.observation_size = int(observation_size)
        self.directive_count = int(directive_count)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.entropy_coefficient = float(entropy_coefficient)
        self.value_coefficient = float(value_coefficient)
        self.normalize_advantage = bool(normalize_advantage)
        self.policy_network = Mlp(
            self.observation_size, self.directive_count, self.hidden_sizes
        )
        self.value_network = Mlp(
            self.observation_size, 1, self.hidden_sizes
        )

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
        unmasked_logits, policy_latent = self.policy_network.forward_with_latent(
            observations
        )
        value_output, value_latent = self.value_network.forward_with_latent(
            observations
        )
        policy_logits = unmasked_logits.masked_fill(
            ~action_masks, torch.finfo(unmasked_logits.dtype).min
        )
        return PpoForwardOutput(
            policy_logits=policy_logits,
            values=value_output.squeeze(-1),
            policy_latent=policy_latent,
            value_latent=value_latent,
        )

    def probabilities(self, observations: Tensor, action_masks: Tensor) -> Tensor:
        probabilities = torch.softmax(
            self(observations, action_masks).policy_logits, -1
        )
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
        actions = (
            torch.argmax(probabilities, dim=-1)
            if deterministic
            else torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        )
        selected = probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
        return PpoActionSample(actions, torch.log(selected), output.values)

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

    def make_optimizer(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.learning_rate)
