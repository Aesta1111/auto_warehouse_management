"""Directive vocabulary, state, prediction, and probe contracts."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from warehouse_ai.demo.artifacts import canonical_json_bytes, sha256_bytes
from warehouse_ai.demo.contracts.base import (
    DemoModel,
    NonBlankStr,
    RepositoryPath,
    Sha256,
    load_yaml_mapping,
    validate_seed_partition,
)
from warehouse_ai.domain import ActionMask, WarehouseObservation

if TYPE_CHECKING:
    from warehouse_ai.env import TarwareEnvWrapper


class DirectiveDefinition(DemoModel):
    directive_id: NonNegativeInt
    command: NonBlankStr
    description: NonBlankStr
    behavior_key: NonBlankStr


class DirectiveVocabulary(DemoModel):
    schema_version: Literal[1]
    vocabulary_id: NonBlankStr
    behavior_version: NonBlankStr
    wait_directive_id: NonNegativeInt
    directives: tuple[DirectiveDefinition, ...] = Field(min_length=4, max_length=8)

    @model_validator(mode="after")
    def validate_directives(self) -> Self:
        ordered = tuple(sorted(self.directives, key=lambda item: item.directive_id))
        if ordered != self.directives:
            object.__setattr__(self, "directives", ordered)
        ids = tuple(item.directive_id for item in self.directives)
        if ids != tuple(range(len(ids))):
            raise ValueError("directive IDs must be contiguous and begin at zero")
        commands = tuple(item.command.casefold() for item in self.directives)
        keys = tuple(item.behavior_key for item in self.directives)
        if len(set(commands)) != len(commands):
            raise ValueError("directive commands must be unique")
        if len(set(keys)) != len(keys):
            raise ValueError("directive behavior keys must be unique")
        if self.wait_directive_id not in ids:
            raise ValueError("wait_directive_id must identify one directive")
        return self

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate(load_yaml_mapping(path))

    def by_id(self, directive_id: int) -> DirectiveDefinition:
        if directive_id < 0 or directive_id >= len(self.directives):
            raise KeyError(directive_id)
        definition = self.directives[directive_id]
        if definition.directive_id != directive_id:
            raise KeyError(directive_id)
        return definition

    @property
    def size(self) -> int:
        return len(self.directives)

    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


class DirectiveContext(DemoModel):
    state_id: NonBlankStr
    observation: WarehouseObservation
    action_mask: ActionMask

    @model_validator(mode="after")
    def validate_agent_order(self) -> Self:
        expected = tuple(agent.agent_id for agent in self.observation.agents)
        if self.action_mask.agent_ids != expected:
            raise ValueError("action mask ordering does not match observation agents")
        return self

    @classmethod
    def capture(cls, wrapper: TarwareEnvWrapper) -> Self:
        observation = wrapper.observation
        action_mask = wrapper.action_mask
        metadata = canonical_json_bytes(
            {"agent_ids": action_mask.agent_ids, "shape": action_mask.values.shape}
        )
        state_id = sha256_bytes(
            canonical_json_bytes(observation)
            + b"\x00"
            + metadata
            + b"\x00"
            + action_mask.values.tobytes(order="C")
        )
        return cls(state_id=state_id, observation=observation, action_mask=action_mask)


class DirectivePrediction(DemoModel):
    state_id: NonBlankStr
    episode_step: NonNegativeInt
    directive: DirectiveDefinition
    probabilities: tuple[float, ...] = Field(min_length=1)
    value_estimate: float
    policy_latency_ms: NonNegativeFloat
    checkpoint_sha256: Sha256

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities(cls, value: Sequence[float]) -> tuple[float, ...]:
        probabilities = tuple(float(item) for item in value)
        if any(
            not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in probabilities
        ):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        return probabilities


class DirectiveProbeConfig(DemoModel):
    schema_version: Literal[1, 2]
    environment_id: NonBlankStr
    vocabulary_path: RepositoryPath
    probe_seeds: tuple[int, ...] = (8000, 8001, 8002, 8003, 8004)
    episode_steps: PositiveInt = 500

    @field_validator("probe_seeds")
    @classmethod
    def validate_probe_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return validate_seed_partition(value, "probe_seeds")


class DirectiveProbeReport(DemoModel):
    schema_version: Literal[1, 2]
    environment_id: NonBlankStr
    vocabulary_sha256: Sha256
    behavior_sha256: Sha256
    behavior_engine_version: NonBlankStr | None = None
    probe_seeds: tuple[int, ...]
    states_checked: PositiveInt
    uncovered_states: NonNegativeInt
    executor_failures: NonNegativeInt
    effect_failures: NonNegativeInt
    passed: bool

    @field_validator("probe_seeds")
    @classmethod
    def validate_report_seeds(cls, value: Sequence[int]) -> tuple[int, ...]:
        return validate_seed_partition(value, "probe_seeds")
