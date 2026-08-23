"""Versioned contracts for frozen-Gemma directive-PPO fine-tuning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from warehouse_ai.demo.contracts.base import (
    DemoModel,
    NonBlankStr,
    RepositoryPath,
    Sha256,
    UtcDateTime,
    validate_seed_partition,
)
from warehouse_ai.demo.contracts.inference import LlamaCppBackendConfig


class GemmaFineTuneConfig(DemoModel):
    """One immutable warm-started PPO fine-tuning configuration."""

    schema_version: Literal[1] = 1
    environment_id: NonBlankStr
    vocabulary_path: RepositoryPath
    parent_checkpoint_path: RepositoryPath
    parent_manifest_path: RepositoryPath
    prompt_spec_path: RepositoryPath
    preflight_report_path: RepositoryPath
    output_directory: RepositoryPath
    train_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    algorithm_seed: int = 8300
    episode_steps: PositiveInt = 500
    total_timesteps: PositiveInt = 1_024
    learning_rate: PositiveFloat = 3e-5
    n_steps: PositiveInt = 256
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: PositiveFloat = 0.2
    entropy_coefficient: NonNegativeFloat = 0.01
    value_coefficient: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat = 0.01
    max_reference_kl: PositiveFloat = 0.05
    fallback_penalty: NonNegativeFloat = 0.001
    reward_version: Literal["simulator-reward-minus-fallback-v1"] = (
        "simulator-reward-minus-fallback-v1"
    )
    inference: LlamaCppBackendConfig

    @field_validator("train_seeds", "selection_seeds", "confirmation_seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_partitions_and_batches(self) -> Self:
        partitions = {
            "train": set(self.train_seeds),
            "selection": set(self.selection_seeds),
            "confirmation": set(self.confirmation_seeds),
        }
        names = tuple(partitions)
        for index, name in enumerate(names):
            for other in names[index + 1 :]:
                overlap = sorted(partitions[name].intersection(partitions[other]))
                if overlap:
                    raise ValueError(f"{name} and {other} seeds overlap: {overlap}")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must evenly divide n_steps")
        if self.total_timesteps % self.n_steps:
            raise ValueError("total_timesteps must be a multiple of n_steps")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in their valid ranges")
        return self


class FineTuneTrainingSummary(DemoModel):
    completed_environment_steps: PositiveInt
    optimizer_steps: PositiveInt
    simulator_return: float
    learning_return: float
    gemma_calls: NonNegativeInt
    fallbacks: NonNegativeInt
    effect_violations: NonNegativeInt
    failure_counts: dict[str, NonNegativeInt]
    directive_counts: dict[str, NonNegativeInt]
    latency_median_ms: NonNegativeFloat | None = None
    latency_p95_ms: NonNegativeFloat | None = None


class GemmaFineTuneManifest(DemoModel):
    """Small provenance record for a fine-tuned PPO checkpoint."""

    schema_version: Literal[1] = 1
    environment_id: NonBlankStr
    vocabulary_sha256: Sha256
    behavior_sha256: Sha256
    behavior_engine_version: NonBlankStr
    parent_checkpoint_sha256: Sha256
    parent_manifest_sha256: Sha256
    checkpoint_sha256: Sha256
    prompt_contract_sha256: Sha256
    preflight_report_sha256: Sha256
    model_file_sha256: Sha256
    train_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    algorithm_seed: int
    reward_version: NonBlankStr
    fallback_penalty: NonNegativeFloat
    hyperparameters: dict[str, Any]
    training: FineTuneTrainingSummary
    package_versions: dict[str, str]
    git_commit: NonBlankStr
    dirty_worktree: bool
    created_at_utc: UtcDateTime

    @field_validator("train_seeds", "selection_seeds", "confirmation_seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)


class FineTuneStepRecord(DemoModel):
    """Append-only audit record for one expensive environment transition."""

    schema_version: Literal[1] = 1
    environment_step: NonNegativeInt
    episode_seed: int
    episode_step: NonNegativeInt
    state_id: NonBlankStr
    directive_id: NonNegativeInt
    directive_command: NonBlankStr
    destinations: tuple[NonNegativeInt, ...] = Field(min_length=1)
    simulator_reward: float
    learning_reward: float
    source: Literal["gemma", "directive_executor", "automatic_no_op"]
    failure_code: str | None = None
    llm_latency_ms: NonNegativeFloat | None = None
