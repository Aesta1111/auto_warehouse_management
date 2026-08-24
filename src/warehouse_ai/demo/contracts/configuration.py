"""Composable, user-facing configuration contracts for the directive demo."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from warehouse_ai.demo.contracts.base import (
    DemoModel,
    NonBlankStr,
    RepositoryPath,
    load_yaml_config,
    validate_seed_partition,
)
from warehouse_ai.demo.contracts.evaluation import DemoConfig
from warehouse_ai.demo.contracts.finetuning import (
    FineTuneEvaluationConfig,
    LlmFineTuneConfig,
)
from warehouse_ai.demo.contracts.inference import LlamaCppBackendConfig
from warehouse_ai.demo.contracts.ppo import PpoConfig


class PpoModelConfig(DemoModel):
    """Reusable PPO network and implementation identity."""

    schema_version: Literal[1, 2] = 2
    model_id: NonBlankStr
    hidden_sizes: tuple[PositiveInt, ...]
    backend: Literal["lightning-fabric"] = "lightning-fabric"
    model_implementation_version: NonBlankStr
    buffer_schema_version: NonBlankStr
    gae_version: NonBlankStr
    loss_version: NonBlankStr

    @field_validator("hidden_sizes")
    @classmethod
    def validate_hidden_sizes(cls, value: Sequence[int]) -> tuple[int, ...]:
        sizes = tuple(value)
        if not sizes:
            raise ValueError("hidden_sizes must contain at least one layer")
        return sizes


class LlmActionerConfig(DemoModel):
    """Reusable frozen-LLM actioner, prompt, and runtime contract."""

    schema_version: Literal[1, 2] = 2
    actioner_id: NonBlankStr
    selected_model: Literal["e4b", "e2b", "qwen3.5-2b-q8"]
    vocabulary_path: RepositoryPath
    prompt_spec_path: RepositoryPath
    inference: LlamaCppBackendConfig
    preflight_p95_limit_seconds: PositiveFloat = 3.92


class PpoTrainingRunConfig(DemoModel):
    """Environment, data partitions, optimization, and output for PPO training."""

    schema_version: Literal[1, 2] = 2
    run_id: NonBlankStr
    model_config_path: RepositoryPath
    environment_id: NonBlankStr
    vocabulary_path: RepositoryPath
    output_directory: RepositoryPath
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int = 8100
    episode_steps: PositiveInt = 500
    total_timesteps: PositiveInt = 25_000
    accelerator: Literal["auto", "cpu", "cuda"] = "auto"
    learning_rate: PositiveFloat = 3e-4
    n_steps: PositiveInt = 256
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: PositiveFloat = 0.2
    entropy_coefficient: NonNegativeFloat = 0.01
    value_coefficient: PositiveFloat = 0.5
    max_grad_norm: PositiveFloat = 0.5
    normalize_advantage: bool = True
    target_kl: PositiveFloat | None = None

    @field_validator("train_seeds", "validation_seeds")
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        overlap = sorted(set(self.train_seeds).intersection(self.validation_seeds))
        if overlap:
            raise ValueError(f"train_seeds and validation_seeds overlap: {overlap}")
        if self.batch_size > self.n_steps or self.n_steps % self.batch_size:
            raise ValueError("batch_size must evenly divide n_steps")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gamma and gae_lambda must be in their valid ranges")
        return self


class PpoFineTuningRunConfig(DemoModel):
    """Warm-started PPO adaptation run through one frozen LLM actioner."""

    schema_version: Literal[1, 2] = 2
    run_id: NonBlankStr
    ppo_training_run_config_path: RepositoryPath
    llm_actioner_config_path: RepositoryPath
    parent_checkpoint_path: RepositoryPath
    parent_manifest_path: RepositoryPath
    preflight_report_path: RepositoryPath
    require_passing_preflight: bool = True
    output_directory: RepositoryPath
    train_seeds: tuple[int, ...]
    selection_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    final_evaluation_seeds: tuple[int, ...]
    algorithm_seed: int = 8300
    episode_steps: PositiveInt = 500
    total_timesteps: PositiveInt = 1_024
    accelerator: Literal["auto", "cpu", "cuda"] = "auto"
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

    @field_validator(
        "train_seeds",
        "selection_seeds",
        "confirmation_seeds",
        "final_evaluation_seeds",
    )
    @classmethod
    def validate_seeds(cls, value: Sequence[int], info: Any) -> tuple[int, ...]:
        return validate_seed_partition(value, info.field_name)

    @model_validator(mode="after")
    def validate_partitions_and_batches(self) -> Self:
        partitions = {
            "train": set(self.train_seeds),
            "selection": set(self.selection_seeds),
            "confirmation": set(self.confirmation_seeds),
            "final": set(self.final_evaluation_seeds),
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


class PpoEvaluationRunConfig(DemoModel):
    """One evaluation plan for preflight, matched demo, and gated comparison."""

    schema_version: Literal[1, 2] = 2
    run_id: NonBlankStr
    fine_tuning_run_config_path: RepositoryPath
    fine_tuned_checkpoint_path: RepositoryPath
    fine_tuned_manifest_path: RepositoryPath
    directive_probe_report_path: RepositoryPath
    output_directory: RepositoryPath
    episode_seed: int | None = None
    matched_demo_seed: int = 9300
    final_evaluation_seed: int = 9200
    episode_steps: PositiveInt = 500
    render_llm: bool = True
    prerequisite_summary_paths: dict[Literal["confirmation", "final"], RepositoryPath]

    @field_validator("prerequisite_summary_paths")
    @classmethod
    def validate_prerequisite_paths(
        cls,
        value: Mapping[str, Path],
    ) -> dict[str, Path]:
        if set(value) != {"confirmation", "final"}:
            raise ValueError(
                "prerequisite_summary_paths must define confirmation and final"
            )
        return dict(value)

    @model_validator(mode="after")
    def validate_seed_roles(self) -> Self:
        if self.schema_version == 1:
            if self.episode_seed is None:
                raise ValueError("schema-v1 evaluation configs require episode_seed")
            return self
        if self.episode_seed is not None:
            raise ValueError("schema-v2 evaluation configs use matched_demo_seed")
        return self


@dataclass(frozen=True)
class LoadedPpoTrainingRun:
    run: PpoTrainingRunConfig
    model: PpoModelConfig
    runtime: PpoConfig


@dataclass(frozen=True)
class LoadedPpoFineTuningRun:
    run: PpoFineTuningRunConfig
    parent_run: LoadedPpoTrainingRun
    actioner: LlmActionerConfig
    runtime: LlmFineTuneConfig


@dataclass(frozen=True)
class LoadedPpoEvaluationRun:
    run: PpoEvaluationRunConfig
    fine_tuning: LoadedPpoFineTuningRun

    def demo_runtime(self) -> DemoConfig:
        parent = self.fine_tuning.runtime
        actioner = self.fine_tuning.actioner
        episode_seed = self.run.episode_seed
        if self.run.schema_version == 1:
            if episode_seed is None:
                raise ValueError("schema-v1 evaluation configs require episode_seed")
        else:
            episode_seed = self.run.matched_demo_seed
        return DemoConfig(
            schema_version=5,
            environment_id=parent.environment_id,
            episode_seed=episode_seed,
            episode_steps=self.run.episode_steps,
            render_hybrid=self.run.render_llm,
            vocabulary_path=parent.vocabulary_path,
            prompt_spec_path=parent.prompt_spec_path,
            directive_probe_report_path=self.run.directive_probe_report_path,
            ppo_checkpoint_path=parent.parent_checkpoint_path,
            ppo_manifest_path=parent.parent_manifest_path,
            selected_model=actioner.selected_model,
            preflight_report_path=parent.preflight_report_path,
            require_passing_preflight=parent.require_passing_preflight,
            output_directory=self.run.output_directory,
            inference=actioner.inference,
            preflight_p95_limit_seconds=actioner.preflight_p95_limit_seconds,
        )

    def comparison_runtime(
        self,
        partition: Literal["selection", "confirmation", "final"],
    ) -> FineTuneEvaluationConfig:
        prerequisite = (
            None
            if partition == "selection"
            else self.run.prerequisite_summary_paths[partition]
        )
        return FineTuneEvaluationConfig(
            schema_version=3,
            fine_tune_config_path=self.run.fine_tuning_run_config_path,
            fine_tune_checkpoint_path=self.run.fine_tuned_checkpoint_path,
            fine_tune_manifest_path=self.run.fine_tuned_manifest_path,
            partition=partition,
            prerequisite_summary_path=prerequisite,
            output_directory=self.run.output_directory,
        )


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_ppo_training_run(path: Path, repository_root: Path) -> LoadedPpoTrainingRun:
    root = repository_root.resolve()
    run = load_yaml_config(_resolve(root, path), PpoTrainingRunConfig)
    model = load_yaml_config(_resolve(root, run.model_config_path), PpoModelConfig)
    runtime = PpoConfig(
        schema_version=5,
        environment_id=run.environment_id,
        vocabulary_path=run.vocabulary_path,
        train_seeds=run.train_seeds,
        validation_seeds=run.validation_seeds,
        algorithm_seed=run.algorithm_seed,
        episode_steps=run.episode_steps,
        total_timesteps=run.total_timesteps,
        accelerator=run.accelerator,
        learning_rate=run.learning_rate,
        n_steps=run.n_steps,
        batch_size=run.batch_size,
        n_epochs=run.n_epochs,
        gamma=run.gamma,
        gae_lambda=run.gae_lambda,
        clip_range=run.clip_range,
        entropy_coefficient=run.entropy_coefficient,
        value_coefficient=run.value_coefficient,
        max_grad_norm=run.max_grad_norm,
        normalize_advantage=run.normalize_advantage,
        target_kl=run.target_kl,
        hidden_sizes=model.hidden_sizes,
        backend=model.backend,
        model_implementation_version=model.model_implementation_version,
        buffer_schema_version=model.buffer_schema_version,
        gae_version=model.gae_version,
        loss_version=model.loss_version,
    )
    return LoadedPpoTrainingRun(run=run, model=model, runtime=runtime)


def load_ppo_fine_tuning_run(
    path: Path,
    repository_root: Path,
) -> LoadedPpoFineTuningRun:
    root = repository_root.resolve()
    run = load_yaml_config(_resolve(root, path), PpoFineTuningRunConfig)
    parent_run = load_ppo_training_run(run.ppo_training_run_config_path, root)
    actioner = load_yaml_config(
        _resolve(root, run.llm_actioner_config_path), LlmActionerConfig
    )
    if actioner.vocabulary_path != parent_run.run.vocabulary_path:
        raise ValueError("PPO training and LLM actioner vocabularies differ")
    if run.episode_steps != parent_run.run.episode_steps:
        raise ValueError("fine-tuning and parent PPO episode lengths differ")
    runtime = LlmFineTuneConfig(
        schema_version=3,
        environment_id=parent_run.run.environment_id,
        vocabulary_path=parent_run.run.vocabulary_path,
        parent_checkpoint_path=run.parent_checkpoint_path,
        parent_manifest_path=run.parent_manifest_path,
        prompt_spec_path=actioner.prompt_spec_path,
        preflight_report_path=run.preflight_report_path,
        require_passing_preflight=run.require_passing_preflight,
        output_directory=run.output_directory,
        train_seeds=run.train_seeds,
        selection_seeds=run.selection_seeds,
        confirmation_seeds=run.confirmation_seeds,
        final_evaluation_seeds=run.final_evaluation_seeds,
        algorithm_seed=run.algorithm_seed,
        episode_steps=run.episode_steps,
        total_timesteps=run.total_timesteps,
        accelerator=run.accelerator,
        learning_rate=run.learning_rate,
        n_steps=run.n_steps,
        batch_size=run.batch_size,
        n_epochs=run.n_epochs,
        gamma=run.gamma,
        gae_lambda=run.gae_lambda,
        clip_range=run.clip_range,
        entropy_coefficient=run.entropy_coefficient,
        value_coefficient=run.value_coefficient,
        max_grad_norm=run.max_grad_norm,
        normalize_advantage=run.normalize_advantage,
        target_kl=run.target_kl,
        max_reference_kl=run.max_reference_kl,
        fallback_penalty=run.fallback_penalty,
        reward_version=run.reward_version,
        inference=actioner.inference,
    )
    return LoadedPpoFineTuningRun(
        run=run,
        parent_run=parent_run,
        actioner=actioner,
        runtime=runtime,
    )


def load_ppo_evaluation_run(
    path: Path,
    repository_root: Path,
) -> LoadedPpoEvaluationRun:
    root = repository_root.resolve()
    run = load_yaml_config(_resolve(root, path), PpoEvaluationRunConfig)
    fine_tuning = load_ppo_fine_tuning_run(run.fine_tuning_run_config_path, root)
    final_seed = (
        run.episode_seed if run.schema_version == 1 else run.final_evaluation_seed
    )
    if final_seed not in fine_tuning.run.final_evaluation_seeds:
        raise ValueError("evaluation final seed must be reserved for final evaluation")
    if run.episode_steps != fine_tuning.run.episode_steps:
        raise ValueError("evaluation and fine-tuning episode lengths differ")
    return LoadedPpoEvaluationRun(run=run, fine_tuning=fine_tuning)
