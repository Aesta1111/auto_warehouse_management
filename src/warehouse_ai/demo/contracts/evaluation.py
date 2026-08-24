"""Controller decisions, episode metrics, and matched-run configuration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import (
    JsonValue,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from warehouse_ai.demo.contracts.base import (
    DemoModel,
    NonBlankStr,
    RepositoryPath,
    Sha256,
)
from warehouse_ai.demo.contracts.directives import PpoPolicyOutput
from warehouse_ai.demo.contracts.inference import FailureCode, LlamaCppBackendConfig
from warehouse_ai.domain import JointAction


class DemoDecision(DemoModel):
    prediction: PpoPolicyOutput
    action: JointAction
    reason: str
    source: Literal["llm", "directive_executor", "automatic_no_op"]
    model_id: str | None = None
    raw_response: str | None = None
    directive_effect_valid: bool
    llm_latency_ms: NonNegativeFloat | None = None
    failure_code: FailureCode | None = None
    failure_detail: str | None = None
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None


class DemoStepRecord(DemoModel):
    schema_version: Literal[2] = 2
    decision: DemoDecision
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    simulator_info: dict[str, JsonValue]


class EpisodeResult(DemoModel):
    schema_version: Literal[2] = 2
    controller: Literal["ppo_executor", "ppo_llm"]
    seed: int
    steps: PositiveInt
    completed_tasks: NonNegativeInt
    total_return: float
    clashes: NonNegativeInt
    stuck_events: NonNegativeInt
    wall_seconds: NonNegativeFloat
    directive_counts: dict[str, NonNegativeInt]
    llm_calls: NonNegativeInt
    invalid_actions: NonNegativeInt
    effect_violations: NonNegativeInt
    fallbacks: NonNegativeInt
    llm_latency_median_ms: NonNegativeFloat | None
    llm_latency_p95_ms: NonNegativeFloat | None


class DemoRunSummary(DemoModel):
    schema_version: Literal[2, 3, 4, 5]
    demo_only: Literal[True]
    environment_id: NonBlankStr
    seed: int
    ppo_checkpoint_sha256: Sha256
    vocabulary_sha256: Sha256
    directive_probe_report_sha256: Sha256
    preflight_report_sha256: Sha256
    prompt_contract_sha256: Sha256 | None = None
    executor_episode: EpisodeResult
    llm_episode: EpisodeResult
    output_directory: RepositoryPath

    @model_validator(mode="after")
    def validate_episode_pair(self) -> Self:
        if (
            self.executor_episode.seed != self.seed
            or self.llm_episode.seed != self.seed
        ):
            raise ValueError("both episode seeds must match the demo seed")
        if self.executor_episode.controller != "ppo_executor":
            raise ValueError("executor_episode must use ppo_executor")
        if self.schema_version >= 5 and self.llm_episode.controller != "ppo_llm":
            raise ValueError("llm_episode must use ppo_llm")
        if self.schema_version >= 3 and self.prompt_contract_sha256 is None:
            raise ValueError("schema-v3+ summary requires prompt_contract_sha256")
        if self.schema_version == 2 and self.prompt_contract_sha256 is not None:
            raise ValueError("schema-v2 summary cannot identify a prompt contract")
        return self


class DemoConfig(DemoModel):
    schema_version: Literal[2, 3, 4, 5]
    environment_id: NonBlankStr
    episode_seed: int = 9100
    episode_steps: PositiveInt = 500
    render_hybrid: bool = True
    vocabulary_path: RepositoryPath
    prompt_spec_path: RepositoryPath | None = None
    directive_probe_report_path: RepositoryPath
    ppo_checkpoint_path: RepositoryPath
    ppo_manifest_path: RepositoryPath
    selected_model: Literal["e4b", "e2b", "qwen3.5-2b-q8"]
    preflight_report_path: RepositoryPath
    require_passing_preflight: bool = True
    output_directory: RepositoryPath
    inference: LlamaCppBackendConfig
    preflight_p95_limit_seconds: PositiveFloat = 3.92

    @model_validator(mode="after")
    def validate_prompt_version(self) -> Self:
        if self.schema_version >= 3 and self.prompt_spec_path is None:
            raise ValueError("schema-v3+ demo config requires prompt_spec_path")
        if self.schema_version == 2 and self.prompt_spec_path is not None:
            raise ValueError("schema-v2 demo config cannot select a prompt spec")
        return self
