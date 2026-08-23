"""Explicit Fabric training orchestration, imported only for training."""

from __future__ import annotations

import importlib.metadata
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from warehouse_ai.demo.artifacts import capture_git_state, sha256_file
from warehouse_ai.demo.contracts.ppo import (
    PpoConfig,
    PpoManifest,
    PpoValidationSummary,
)
from warehouse_ai.demo.ppo.buffer import RolloutBuffer
from warehouse_ai.demo.ppo.collector import RolloutCollector
from warehouse_ai.demo.ppo.environment import GymEnv
from warehouse_ai.demo.ppo.model import (
    PpoModel,
    architecture_digest,
    implementation_digest,
    legacy_architecture_digest,
)
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.demo.ppo.validation import (
    evaluate_directive_ppo,
    is_better_validation,
)
from warehouse_ai.demo.inference.provenance import hardware_provenance


VALIDATION_INTERVAL = 5_000
HIDDEN_SIZE = 64
PPO_CHECKPOINT_NAME = "ppo-policy.pt"


def _accelerator_name(device: torch.device) -> Literal["cpu", "cuda"]:
    if device.type == "cuda":
        return "cuda"
    if device.type == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported Fabric device type: {device.type}")


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _package_versions() -> dict[str, str]:
    packages = (
        "auto-warehouse-management",
        "gymnasium",
        "lightning",
        "numpy",
        "tarware",
        "torch",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


class PpoTrainingLoop:
    """On-policy collect/update loop using public Fabric operations."""

    def fit(
        self,
        model: PpoModel,
        config: PpoConfig,
        training_environment: GymEnv,
        validation_environment: GymEnv,
        output_directory: Path,
    ) -> PpoManifest:
        if config.schema_version not in (2, 3, 4, 5):
            raise ValueError("repository-owned training requires schema v2 through v5")
        if training_environment.catalog.digest() != (
            validation_environment.catalog.digest()
        ):
            raise ValueError("training and validation vocabularies differ")
        assert config.max_grad_norm is not None

        from lightning.fabric import Fabric

        output_directory.mkdir(parents=True, exist_ok=True)
        random.seed(config.algorithm_seed)
        np.random.seed(config.algorithm_seed)
        torch.manual_seed(config.algorithm_seed)
        if config.accelerator == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PPO config requires CUDA, but torch.cuda is unavailable"
            )
        fabric = Fabric(
            accelerator=config.accelerator,
            devices=1,
            precision="32-true",
        )
        if config.accelerator == "cuda" and fabric.device.type != "cuda":
            raise RuntimeError(
                f"PPO config requires CUDA, but Fabric selected {fabric.device}"
            )
        optimizer = model.make_optimizer()
        module, optimizer = fabric.setup(model, optimizer)
        rollout_generator = torch.Generator(device=fabric.device.type)
        rollout_generator.manual_seed(config.algorithm_seed + 1)
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.algorithm_seed + 2)

        observation_shape = training_environment.observation_space.shape
        if observation_shape is None:
            raise RuntimeError("training environment has no fixed observation shape")
        buffer = RolloutBuffer(
            config.n_steps,
            observation_shape,
            training_environment.catalog.size,
        )
        collector = RolloutCollector()
        completed_steps = optimizer_steps = selected_steps = selected_updates = 0
        next_validation = VALIDATION_INTERVAL
        best: PpoValidationSummary | None = None
        checkpoint = output_directory / PPO_CHECKPOINT_NAME

        while completed_steps < config.total_timesteps:
            collector.collect(
                module,
                training_environment,
                buffer,
                rollout_steps=config.n_steps,
                generator=rollout_generator,
            )
            buffer.finalize(gamma=config.gamma, gae_lambda=config.gae_lambda)
            module.train()
            stop_update = False
            for _ in range(config.n_epochs):
                for batch in buffer.iter_minibatches(
                    config.batch_size, generator=minibatch_generator
                ):
                    optimizer.zero_grad()
                    terms = module.compute_loss(batch.to(fabric.device))
                    fabric.backward(terms.total)
                    fabric.clip_gradients(
                        module, optimizer, max_norm=config.max_grad_norm
                    )
                    optimizer.step()
                    optimizer_steps += 1
                    if (
                        config.target_kl is not None
                        and float(terms.approximate_kl.item()) > config.target_kl
                    ):
                        stop_update = True
                        break
                if stop_update:
                    break
            completed_steps += config.n_steps
            buffer.reset()

            if (
                completed_steps >= next_validation
                or completed_steps >= config.total_timesteps
            ):
                raw_model = getattr(module, "module", module)
                candidate = PpoPolicy.from_module(
                    raw_model,
                    training_environment.catalog,
                    "0" * 64,
                )
                summary = evaluate_directive_ppo(
                    candidate,
                    config,
                    validation_environment.catalog,
                    validation_environment.registry,
                )
                if is_better_validation(summary, best):
                    fabric.save(checkpoint, {"model": module})
                    best = summary
                    selected_steps = completed_steps
                    selected_updates = optimizer_steps
                while next_validation <= completed_steps:
                    next_validation += VALIDATION_INTERVAL

        if best is None or not checkpoint.is_file():
            raise RuntimeError("PPO training produced no checkpoint")
        if best.non_wait_directives_used < 2 or best.inapplicable_directives:
            raise RuntimeError("PPO validation failed directive coverage constraints")
        return self._manifest(
            config=config,
            environment=training_environment,
            checkpoint=checkpoint,
            observation_size=int(observation_shape[0]),
            completed_steps=completed_steps,
            optimizer_steps=optimizer_steps,
            selected_steps=selected_steps,
            selected_updates=selected_updates,
            validation=best,
            selected_device=str(fabric.device),
            selected_accelerator=_accelerator_name(fabric.device),
        )

    @staticmethod
    def _manifest(
        *,
        config: PpoConfig,
        environment: GymEnv,
        checkpoint: Path,
        observation_size: int,
        completed_steps: int,
        optimizer_steps: int,
        selected_steps: int,
        selected_updates: int,
        validation: PpoValidationSummary,
        selected_device: str,
        selected_accelerator: Literal["cpu", "cuda"],
    ) -> PpoManifest:
        git_commit, dirty = capture_git_state(_repository_root())
        hidden_sizes = tuple(
            model_size for model_size in environment_hidden_sizes(config)
        )
        hyperparameters: dict[str, Any] = {
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "entropy_coefficient": config.entropy_coefficient,
            "value_coefficient": config.value_coefficient,
            "max_grad_norm": config.max_grad_norm,
            "normalize_advantage": config.normalize_advantage,
            "target_kl": config.target_kl,
            "validation_interval": VALIDATION_INTERVAL,
            "declared_total_timesteps": config.total_timesteps,
            "completed_environment_steps": completed_steps,
            "optimizer_steps": optimizer_steps,
            "selected_checkpoint_environment_steps": selected_steps,
            "selected_checkpoint_optimizer_steps": selected_updates,
            "observation_size": observation_size,
            "directive_count": environment.catalog.size,
        }
        if config.schema_version >= 4:
            hyperparameters["hidden_sizes"] = list(hidden_sizes)
        else:
            hyperparameters["hidden_size"] = HIDDEN_SIZE
        hardware = hardware_provenance()
        hardware["selected_device"] = selected_device
        hardware["selected_accelerator"] = selected_accelerator
        return PpoManifest(
            schema_version=config.schema_version,
            environment_id=config.environment_id,
            vocabulary_sha256=environment.catalog.digest(),
            behavior_sha256=environment.registry.behavior_digest(),
            behavior_engine_version=(
                environment.registry.engine_version
                if config.schema_version >= 3
                else None
            ),
            checkpoint_sha256=sha256_file(checkpoint),
            train_seeds=config.train_seeds,
            validation_seeds=config.validation_seeds,
            algorithm_seed=config.algorithm_seed,
            hyperparameters=hyperparameters,
            validation=validation,
            package_versions=_package_versions(),
            backend=config.backend,
            model_implementation_version=config.model_implementation_version,
            buffer_schema_version=config.buffer_schema_version,
            gae_version=config.gae_version,
            loss_version=config.loss_version,
            architecture_sha256=(
                architecture_digest(
                    observation_size, environment.catalog.size, hidden_sizes
                )
                if config.schema_version >= 4
                else legacy_architecture_digest(
                    observation_size,
                    environment.catalog.size,
                    HIDDEN_SIZE,
                    config.model_implementation_version or "",
                )
            ),
            implementation_sha256=implementation_digest(
                config.model_implementation_version or "",
                config.gae_version or "",
                config.loss_version or "",
            ),
            accelerator=(
                selected_accelerator
                if selected_accelerator in {"cpu", "cuda"}
                else None
            ),
            selected_device=selected_device,
            hardware=hardware,
            git_commit=git_commit,
            dirty_worktree=dirty,
            created_at_utc=datetime.now(timezone.utc),
        )


def environment_hidden_sizes(config: PpoConfig) -> tuple[int, ...]:
    """Resolve the architecture without changing legacy schema semantics."""
    if config.schema_version >= 4:
        assert config.hidden_sizes is not None
        return tuple(config.hidden_sizes)
    return (HIDDEN_SIZE,)
