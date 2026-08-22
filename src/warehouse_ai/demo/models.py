"""Immutable records and strict configuration for the directive prototype.

Phase 1 shared contracts, Phase 3 directive-PPO records, and the Phase 4
actioner/controller records live here. Runtime services remain in ordinary
typed modules rather than inside these immutable Pydantic models.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, Generic, Literal, Self, TypeVar

import yaml
from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.domain import ActionMask, JointAction, WarehouseObservation

if TYPE_CHECKING:
    from warehouse_ai.env import TarwareEnvWrapper


_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def _nonblank(value: str, field_name: str) -> str:
    """Normalize and validate a required human/configuration string."""
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must not be blank")
    return value


def _digest(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character hexadecimal SHA-256")
    return value.lower()


def _is_absolute_any_platform(path: Path) -> bool:
    """Recognize POSIX, Windows-drive, and Windows-UNC absolute paths."""
    return path.is_absolute() or PureWindowsPath(str(path)).is_absolute()


class DemoModel(BaseModel):
    """Strict immutable base for prototype records and configurations."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
        str_strip_whitespace=True,
    )

    @model_validator(mode="after")
    def reject_absolute_paths(self) -> Self:
        """Keep committed configuration paths repository-relative."""
        for field_name in type(self).model_fields:
            value = getattr(self, field_name, None)
            if isinstance(value, Path) and _is_absolute_any_platform(value):
                raise ValueError(
                    f"{field_name} must be repository-relative, not absolute"
                )
        return self


class DirectiveDefinition(DemoModel):
    """One stable neural directive and its closed behavior key."""

    directive_id: NonNegativeInt
    command: str
    description: str
    behavior_key: str

    @field_validator("command", "description", "behavior_key")
    @classmethod
    def validate_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)


class DirectiveVocabulary(DemoModel):
    """Versioned directive definitions used by the prototype policy."""

    schema_version: Literal[1]
    vocabulary_id: str
    behavior_version: str
    wait_directive_id: NonNegativeInt
    directives: tuple[DirectiveDefinition, ...] = Field(min_length=4, max_length=8)

    @field_validator("vocabulary_id", "behavior_version")
    @classmethod
    def validate_identity(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @model_validator(mode="after")
    def validate_directives(self) -> Self:
        """Validate IDs and normalize definitions into canonical order."""
        ordered = tuple(sorted(self.directives, key=lambda item: item.directive_id))
        if ordered != self.directives:
            object.__setattr__(self, "directives", ordered)

        ids = tuple(item.directive_id for item in self.directives)
        expected_ids = tuple(range(len(self.directives)))
        if ids != expected_ids:
            raise ValueError("directive IDs must be contiguous and begin at zero")

        commands = tuple(item.command.casefold() for item in self.directives)
        if len(set(commands)) != len(commands):
            raise ValueError("directive commands must be unique")
        behavior_keys = tuple(item.behavior_key for item in self.directives)
        if len(set(behavior_keys)) != len(behavior_keys):
            raise ValueError("directive behavior keys must be unique")
        if self.wait_directive_id not in ids:
            raise ValueError("wait_directive_id must identify one directive")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a vocabulary YAML document without constructing a simulator."""
        return cls.model_validate(_load_yaml_mapping(path))

    def by_id(self, directive_id: int) -> DirectiveDefinition:
        """Return a definition by stable ID."""
        if directive_id < 0 or directive_id >= len(self.directives):
            raise KeyError(directive_id)
        definition = self.directives[directive_id]
        if definition.directive_id != directive_id:
            raise KeyError(directive_id)
        return definition

    @property
    def size(self) -> int:
        """Number of directives in the vocabulary."""
        return len(self.directives)

    def digest(self) -> str:
        """Hash the canonical vocabulary JSON representation."""
        return sha256_bytes(canonical_json_bytes(self))


class DirectiveContext(DemoModel):
    """Immutable request snapshot shared by directive components."""

    state_id: str
    observation: WarehouseObservation
    action_mask: ActionMask

    @field_validator("state_id")
    @classmethod
    def validate_state_id(cls, value: str) -> str:
        return _nonblank(value, "state_id")

    @model_validator(mode="after")
    def validate_agent_order(self) -> Self:
        expected_agent_ids = tuple(agent.agent_id for agent in self.observation.agents)
        if self.action_mask.agent_ids != expected_agent_ids:
            raise ValueError("action mask ordering does not match observation agents")
        return self

    @classmethod
    def capture(cls, wrapper: TarwareEnvWrapper) -> Self:
        """Capture the current project-owned state and a stable state digest."""
        observation = wrapper.observation
        action_mask = wrapper.action_mask

        observation_bytes = canonical_json_bytes(observation)
        mask_metadata = canonical_json_bytes(
            {
                "agent_ids": action_mask.agent_ids,
                "shape": action_mask.values.shape,
            }
        )
        state_id = sha256_bytes(
            observation_bytes
            + b"\x00"
            + mask_metadata
            + b"\x00"
            + action_mask.values.tobytes(order="C")
        )
        return cls(
            state_id=state_id,
            observation=observation,
            action_mask=action_mask,
        )


class DirectivePrediction(DemoModel):
    """One deterministic masked-policy prediction at a captured state."""

    state_id: str
    episode_step: NonNegativeInt
    directive: DirectiveDefinition
    probabilities: tuple[float, ...] = Field(min_length=1)
    value_estimate: float
    policy_latency_ms: NonNegativeFloat
    checkpoint_sha256: str

    @field_validator("state_id")
    @classmethod
    def validate_prediction_state_id(cls, value: str) -> str:
        return _nonblank(value, "state_id")

    @field_validator("checkpoint_sha256")
    @classmethod
    def validate_prediction_digest(cls, value: str) -> str:
        return _digest(value, "checkpoint_sha256")

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: Sequence[float]) -> tuple[float, ...]:
        probabilities = tuple(float(item) for item in value)
        if any(
            not math.isfinite(item) or item < 0.0 or item > 1.0
            for item in probabilities
        ):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        return probabilities


class ActionerRequest(DemoModel):
    """One immutable request sent across the actioner boundary."""

    context: DirectiveContext
    prediction: DirectivePrediction

    @model_validator(mode="after")
    def validate_prediction_state(self) -> Self:
        """Keep the policy prediction and visible request at one state."""
        if self.context.state_id != self.prediction.state_id:
            raise ValueError("actioner request context and prediction states differ")
        return self


class GemmaActionResponse(DemoModel):
    """The model-facing single joint destination response."""

    destinations: tuple[NonNegativeInt, ...] = Field(min_length=1)
    reason: str = Field(max_length=160)


class SingleJointActionPlan(DemoModel):
    """One validated plan for exactly one TA-RWARE joint action."""

    state_id: str
    directive_id: NonNegativeInt
    action: JointAction
    reason: str = Field(max_length=160)

    @field_validator("state_id")
    @classmethod
    def validate_plan_state_id(cls, value: str) -> str:
        return _nonblank(value, "state_id")


PlanT = TypeVar("PlanT")


class ActionerResult(DemoModel, Generic[PlanT]):
    """A typed actioner response with transport provenance."""

    plan: PlanT
    raw_text: str
    model_id: str
    latency_ms: NonNegativeFloat

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _nonblank(value, "model_id")


class DemoDecision(DemoModel):
    """The action and provenance selected for one simulator step."""

    prediction: DirectivePrediction
    action: JointAction
    reason: str = Field(max_length=160)
    source: Literal["gemma", "directive_executor", "automatic_no_op"]
    model_id: str | None
    raw_response: str | None
    directive_effect_valid: bool
    llm_latency_ms: NonNegativeFloat | None
    fallback_reason: str | None


class DemoStepRecord(DemoModel):
    """Serializable decision plus the simulator result that followed it."""

    decision: DemoDecision
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    simulator_info: dict[str, JsonValue]


class DirectivePpoValidationSummary(DemoModel):
    """Validation metrics used to select and audit the prototype policy."""

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
    def validate_summary_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return _validate_seed_partition(value, "seeds")


class DirectivePpoManifest(DemoModel):
    """Small provenance record for a frozen directive PPO checkpoint."""

    schema_version: Literal[1, 2]
    environment_id: str
    vocabulary_sha256: str
    behavior_sha256: str
    checkpoint_sha256: str
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int
    hyperparameters: dict[str, JsonValue]
    validation: DirectivePpoValidationSummary
    package_versions: dict[str, str]
    backend: Literal["sb3-maskable-ppo", "lightning-fabric"] | None = None
    model_implementation_version: str | None = None
    buffer_schema_version: str | None = None
    gae_version: str | None = None
    loss_version: str | None = None
    architecture_sha256: str | None = None
    implementation_sha256: str | None = None
    git_commit: str
    dirty_worktree: bool
    created_at_utc: datetime

    @field_validator("environment_id", "git_commit")
    @classmethod
    def validate_manifest_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "vocabulary_sha256",
        "behavior_sha256",
        "checkpoint_sha256",
        "architecture_sha256",
        "implementation_sha256",
    )
    @classmethod
    def validate_manifest_digests(cls, value: str, info: Any) -> str:
        if value is None:
            return value
        return _digest(value, info.field_name)

    @field_validator("train_seeds", "validation_seeds")
    @classmethod
    def validate_manifest_seeds(
        cls, value: Sequence[int], info: Any
    ) -> tuple[int, ...]:
        return _validate_seed_partition(value, info.field_name)

    @field_validator("created_at_utc")
    @classmethod
    def validate_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at_utc must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_manifest_consistency(self) -> Self:
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
        if self.schema_version == 2 and any(value is None for value in provenance):
            raise ValueError(
                "schema-v2 PPO manifests require complete implementation provenance"
            )
        if self.schema_version == 2 and self.backend != "lightning-fabric":
            raise ValueError("schema-v2 PPO manifests require lightning-fabric")
        return self


def _validate_seed_partition(value: Sequence[int], field_name: str) -> tuple[int, ...]:
    seeds = tuple(value)
    if not seeds:
        raise ValueError(f"{field_name} must contain at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{field_name} must not contain duplicate seeds")
    return seeds


class DirectiveProbeConfig(DemoModel):
    """Configuration for the directive applicability/executor probe."""

    schema_version: Literal[1]
    environment_id: str
    vocabulary_path: Path
    probe_seeds: tuple[int, ...] = (8000, 8001, 8002, 8003, 8004)
    episode_steps: PositiveInt = 500

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _nonblank(value, "environment_id")

    @field_validator("probe_seeds")
    @classmethod
    def validate_probe_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return _validate_seed_partition(value, "probe_seeds")


class DirectivePpoConfig(DemoModel):
    """Configuration shared by the preserved v1 and repository-owned v2 PPO."""

    schema_version: Literal[1, 2]
    environment_id: str
    vocabulary_path: Path
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
    model_implementation_version: str | None = None
    buffer_schema_version: str | None = None
    gae_version: str | None = None
    loss_version: str | None = None

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _nonblank(value, "environment_id")

    @field_validator("train_seeds", "validation_seeds")
    @classmethod
    def validate_seed_sets(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return _validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_seed_disjointness(self) -> Self:
        overlap = sorted(set(self.train_seeds).intersection(self.validation_seeds))
        if overlap:
            raise ValueError(f"train_seeds and validation_seeds overlap: {overlap}")
        provenance = (
            self.value_coefficient,
            self.max_grad_norm,
            self.normalize_advantage,
            self.backend,
            self.model_implementation_version,
            self.buffer_schema_version,
            self.gae_version,
            self.loss_version,
        )
        if self.schema_version == 2 and any(value is None for value in provenance):
            raise ValueError(
                "schema-v2 PPO configs require optimizer and implementation fields"
            )
        if self.schema_version == 2 and self.backend != "lightning-fabric":
            raise ValueError("schema-v2 PPO configs require lightning-fabric")
        return self


class LlamaCppConfig(DemoModel):
    """Pinned local actioner endpoint contract for later prototype phases."""

    base_url: AnyHttpUrl
    model_id: str
    model_file_sha256: str
    context_size: PositiveInt
    gpu_layers: NonNegativeInt
    request_timeout_seconds: PositiveFloat = 12.0
    max_new_tokens: PositiveInt = 192
    temperature: Literal[0] = 0

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        return _nonblank(value, "model_id")

    @field_validator("model_file_sha256")
    @classmethod
    def validate_model_digest(cls, value: str) -> str:
        return _digest(value, "model_file_sha256")


class DirectiveDemoConfig(DemoModel):
    """Configuration for the matched prototype demonstration runner."""

    schema_version: Literal[1]
    environment_id: str
    episode_seed: int = 9100
    episode_steps: PositiveInt = 500
    render_hybrid: bool = True
    vocabulary_path: Path
    directive_probe_report_path: Path
    ppo_checkpoint_path: Path
    ppo_manifest_path: Path
    selected_model: Literal["e4b", "e2b"]
    preflight_report_path: Path
    output_directory: Path
    llama_cpp: LlamaCppConfig
    preflight_p95_limit_seconds: PositiveFloat = 10.0

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _nonblank(value, "environment_id")


class DirectiveProbeReport(DemoModel):
    """Small, committed-compatible output contract for the next phase."""

    schema_version: Literal[1]
    environment_id: str
    vocabulary_sha256: str
    behavior_sha256: str
    probe_seeds: tuple[int, ...]
    states_checked: PositiveInt
    uncovered_states: NonNegativeInt
    executor_failures: NonNegativeInt
    effect_failures: NonNegativeInt
    passed: bool

    @field_validator("environment_id")
    @classmethod
    def validate_environment_id(cls, value: str) -> str:
        return _nonblank(value, "environment_id")

    @field_validator("vocabulary_sha256", "behavior_sha256")
    @classmethod
    def validate_report_digests(cls, value: str, info: Any) -> str:
        return _digest(value, info.field_name)

    @field_validator("probe_seeds")
    @classmethod
    def validate_report_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return _validate_seed_partition(value, "probe_seeds")


ConfigT = TypeVar("ConfigT", bound=DemoModel)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not load YAML configuration {path}") from exc
    if not isinstance(data, Mapping):
        raise ValueError(f"YAML configuration {path} must contain a mapping")
    return data


def load_yaml_config(path: Path, model_type: type[ConfigT]) -> ConfigT:
    """Load and strictly validate one YAML configuration model."""
    return model_type.model_validate(_load_yaml_mapping(path))
