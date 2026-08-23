"""PPO configuration, validation, and checkpoint provenance contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

from pydantic import (
    JsonValue,
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


class PpoValidationSummary(DemoModel):
    seeds: tuple[int, ...]
    mean_completed_tasks: NonNegativeFloat
    mean_return: float
    directive_counts: dict[str, NonNegativeInt]
    non_wait_directives_used: NonNegativeInt
    inapplicable_directives: NonNegativeInt
    masked_action_count: NonNegativeInt = 0
    mean_entropy: NonNegativeFloat = 0.0
    collapsed_to_one_non_wait: bool = False

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return validate_seed_partition(value, "seeds")


class PpoManifest(DemoModel):
    schema_version: Literal[1, 2, 3]
    environment_id: NonBlankStr
    vocabulary_sha256: Sha256
    behavior_sha256: Sha256
    behavior_engine_version: NonBlankStr | None = None
    checkpoint_sha256: Sha256
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int
    hyperparameters: dict[str, JsonValue]
    validation: PpoValidationSummary
    package_versions: dict[str, str]
    backend: Literal["sb3-maskable-ppo", "lightning-fabric"] | None = None
    model_implementation_version: NonBlankStr | None = None
    buffer_schema_version: NonBlankStr | None = None
    gae_version: NonBlankStr | None = None
    loss_version: NonBlankStr | None = None
    architecture_sha256: Sha256 | None = None
    implementation_sha256: Sha256 | None = None
    git_commit: NonBlankStr
    dirty_worktree: bool
    created_at_utc: UtcDateTime

    @field_validator("train_seeds", "validation_seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if set(self.train_seeds).intersection(self.validation_seeds):
            raise ValueError("train_seeds and validation_seeds must be disjoint")
        if self.validation.seeds != self.validation_seeds:
            raise ValueError("validation summary seeds must match validation_seeds")
        provenance = (
            self.backend,
            self.model_implementation_version,
            self.buffer_schema_version,
            self.gae_version,
            self.loss_version,
            self.architecture_sha256,
            self.implementation_sha256,
        )
        if self.schema_version >= 2 and any(value is None for value in provenance):
            raise ValueError("schema-v2+ manifests require implementation provenance")
        if self.schema_version >= 2 and self.backend != "lightning-fabric":
            raise ValueError("schema-v2+ manifests require lightning-fabric")
        if self.schema_version == 3 and self.behavior_engine_version is None:
            raise ValueError("schema-v3 manifests require behavior_engine_version")
        return self


class PpoConfig(DemoModel):
    schema_version: Literal[1, 2, 3]
    environment_id: NonBlankStr
    vocabulary_path: RepositoryPath
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int = 8100
    episode_steps: PositiveInt = 500
    total_timesteps: PositiveInt = 25_000
    learning_rate: PositiveFloat = 3e-4
    n_steps: PositiveInt = 256
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: PositiveFloat | None = None
    max_grad_norm: PositiveFloat | None = None
    normalize_advantage: bool | None = None
    target_kl: PositiveFloat | None = None
    backend: Literal["sb3-maskable-ppo", "lightning-fabric"] | None = None
    model_implementation_version: NonBlankStr | None = None
    buffer_schema_version: NonBlankStr | None = None
    gae_version: NonBlankStr | None = None
    loss_version: NonBlankStr | None = None

    @field_validator("train_seeds", "validation_seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        overlap = sorted(set(self.train_seeds).intersection(self.validation_seeds))
        if overlap:
            raise ValueError(f"train_seeds and validation_seeds overlap: {overlap}")
        required = (
            self.value_coefficient,
            self.max_grad_norm,
            self.normalize_advantage,
            self.backend,
            self.model_implementation_version,
            self.buffer_schema_version,
            self.gae_version,
            self.loss_version,
        )
        if self.schema_version >= 2 and any(value is None for value in required):
            raise ValueError(
                "schema-v2+ configs require optimizer and implementation fields"
            )
        if self.schema_version >= 2 and self.backend != "lightning-fabric":
            raise ValueError("schema-v2+ configs require lightning-fabric")
        return self
