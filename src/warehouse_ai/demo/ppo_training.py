"""Rollout collection and explicit Lightning Fabric PPO orchestration."""

from __future__ import annotations

import importlib.metadata
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning.fabric import Fabric

from warehouse_ai.demo.artifacts import capture_git_state, sha256_file
from warehouse_ai.demo.models import (
    DirectivePpoConfig,
    DirectivePpoManifest,
    DirectivePpoValidationSummary,
)
from warehouse_ai.demo.ppo_buffer import DirectiveRolloutBuffer
from warehouse_ai.demo.ppo_model import (
    DirectivePpoModule,
    architecture_digest,
    implementation_digest,
)
from warehouse_ai.demo.sb3 import DirectiveSb3Env


_VALIDATION_INTERVAL = 5_000
_HIDDEN_SIZE = 64


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


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


def _value_for_observation(
    module: DirectivePpoModule,
    observation: np.ndarray[Any, np.dtype[np.float32]],
    action_mask: np.ndarray[Any, np.dtype[np.bool_]],
) -> float:
    device = next(module.parameters()).device
    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32, device=device
    ).unsqueeze(0)
    mask_tensor = torch.as_tensor(
        action_mask, dtype=torch.bool, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        raw_module = getattr(module, "module", module)
        return float(raw_module(observation_tensor, mask_tensor).values.item())


class DirectiveRolloutCollector:
    """Stateful environment collector that never performs optimization."""

    def __init__(self) -> None:
        self._observation: np.ndarray[Any, np.dtype[np.float32]] | None = None
        self._environment_identity: int | None = None

    def collect(
        self,
        module: DirectivePpoModule,
        environment: DirectiveSb3Env,
        buffer: DirectiveRolloutBuffer,
        *,
        rollout_steps: int,
        generator: torch.Generator,
    ) -> None:
        if rollout_steps != buffer.capacity:
            raise ValueError("rollout_steps must equal the fixed buffer capacity")
        if buffer.size != 0 or buffer.finalized:
            raise RuntimeError("collector requires a fresh rollout buffer")
        if self._environment_identity not in (None, id(environment)):
            raise RuntimeError("one collector cannot switch training environments")
        self._environment_identity = id(environment)
        if self._observation is None:
            observation, _ = environment.reset(seed=environment.episode_seeds[0])
            self._observation = observation

        module.eval()
        device = next(module.parameters()).device
        for _ in range(rollout_steps):
            observation = self._observation
            action_mask = environment.action_masks()
            observation_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask_tensor = torch.as_tensor(
                action_mask, dtype=torch.bool, device=device
            ).unsqueeze(0)
            with torch.inference_mode():
                sample = module.act(
                    observation_tensor,
                    mask_tensor,
                    deterministic=False,
                    generator=generator,
                )
            action = int(sample.actions.item())
            next_observation, reward, terminated, truncated, _ = environment.step(
                action
            )
            if terminated:
                next_value = 0.0
            else:
                next_value = _value_for_observation(
                    module, next_observation, environment.action_masks()
                )
            buffer.add(
                observation=observation,
                action_mask=action_mask,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                old_value=float(sample.values.item()),
                next_value=next_value,
                old_log_probability=float(sample.log_probabilities.item()),
            )
            if terminated or truncated:
                next_observation, _ = environment.reset()
            self._observation = next_observation


class DirectivePpoTrainingLoop:
    """Explicit on-policy collect/update loop using public Fabric operations."""

    def fit(
        self,
        module: DirectivePpoModule,
        config: DirectivePpoConfig,
        training_environment: DirectiveSb3Env,
        validation_environment: DirectiveSb3Env,
        output_directory: Path,
    ) -> DirectivePpoManifest:
        if config.schema_version != 2:
            raise ValueError("repository-owned training requires schema version 2")
        if training_environment.vocabulary.digest() != (
            validation_environment.vocabulary.digest()
        ):
            raise ValueError("training and validation vocabularies differ")
        assert config.value_coefficient is not None
        assert config.max_grad_norm is not None
        assert config.normalize_advantage is not None
        assert config.backend is not None
        assert config.model_implementation_version is not None
        assert config.buffer_schema_version is not None
        assert config.gae_version is not None
        assert config.loss_version is not None

        output_directory.mkdir(parents=True, exist_ok=True)
        random.seed(config.algorithm_seed)
        np.random.seed(config.algorithm_seed)
        torch.manual_seed(config.algorithm_seed)
        fabric = Fabric(accelerator="auto", devices=1, precision="32-true")
        optimizer = module.configure_optimizers()
        module, optimizer = fabric.setup(module, optimizer)
        rollout_generator = torch.Generator(device=fabric.device.type)
        rollout_generator.manual_seed(config.algorithm_seed + 1)
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.algorithm_seed + 2)

        observation_shape = training_environment.observation_space.shape
        if observation_shape is None:
            raise RuntimeError("training environment has no fixed observation shape")
        buffer = DirectiveRolloutBuffer(
            config.n_steps,
            observation_shape,
            training_environment.vocabulary.size,
        )
        collector = DirectiveRolloutCollector()
        completed_steps = 0
        next_validation = _VALIDATION_INTERVAL
        best_summary: DirectivePpoValidationSummary | None = None
        checkpoint_path = output_directory / "directive-ppo-v2.pt"
        optimizer_steps = 0
        selected_checkpoint_steps = 0
        selected_checkpoint_optimizer_steps = 0

        from warehouse_ai.demo.ppo import (
            DirectivePpoPolicy,
            evaluate_directive_ppo,
            is_better_validation,
        )

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
            for _epoch in range(config.n_epochs):
                for batch in buffer.iter_minibatches(
                    config.batch_size, generator=minibatch_generator
                ):
                    batch = batch.to(fabric.device)
                    optimizer.zero_grad()
                    terms = module.compute_loss(batch)
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

            should_validate = (
                completed_steps >= next_validation
                or completed_steps >= config.total_timesteps
            )
            if should_validate:
                raw_module = getattr(module, "module", module)
                candidate = DirectivePpoPolicy.from_module(
                    raw_module,
                    training_environment.vocabulary,
                    checkpoint_sha256="0" * 64,
                )
                summary = evaluate_directive_ppo(
                    candidate,
                    config,
                    validation_environment.vocabulary,
                    validation_environment.registry,
                )
                if is_better_validation(summary, best_summary):
                    fabric.save(
                        checkpoint_path,
                        {
                            "model": module,
                            "metadata": {
                                "observation_size": module.observation_size,
                                "directive_count": module.directive_count,
                                "hidden_size": module.hidden_size,
                                "optimizer_steps": optimizer_steps,
                                "completed_environment_steps": completed_steps,
                            },
                        },
                    )
                    best_summary = summary
                    selected_checkpoint_steps = completed_steps
                    selected_checkpoint_optimizer_steps = optimizer_steps
                while next_validation <= completed_steps:
                    next_validation += _VALIDATION_INTERVAL

        if best_summary is None or not checkpoint_path.is_file():
            raise RuntimeError("schema-v2 PPO training produced no checkpoint")
        if best_summary.non_wait_directives_used < 2:
            raise RuntimeError(
                "directive PPO validation selected fewer than two non-wait directives"
            )

        checkpoint_sha256 = sha256_file(checkpoint_path)
        git_commit, dirty_worktree = capture_git_state(_repository_root())
        observation_size = int(observation_shape[0])
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
            "policy_net_arch": {"pi": [_HIDDEN_SIZE], "vf": [_HIDDEN_SIZE]},
            "validation_interval": _VALIDATION_INTERVAL,
            "declared_total_timesteps": config.total_timesteps,
            "completed_environment_steps": completed_steps,
            "optimizer_steps": optimizer_steps,
            "selected_checkpoint_environment_steps": selected_checkpoint_steps,
            "selected_checkpoint_optimizer_steps": (
                selected_checkpoint_optimizer_steps
            ),
            "observation_size": observation_size,
            "directive_count": training_environment.vocabulary.size,
            "hidden_size": _HIDDEN_SIZE,
        }
        return DirectivePpoManifest(
            schema_version=2,
            environment_id=config.environment_id,
            vocabulary_sha256=training_environment.vocabulary.digest(),
            behavior_sha256=training_environment.registry.behavior_digest(),
            checkpoint_sha256=checkpoint_sha256,
            train_seeds=config.train_seeds,
            validation_seeds=config.validation_seeds,
            algorithm_seed=config.algorithm_seed,
            hyperparameters=hyperparameters,
            validation=best_summary,
            package_versions=_package_versions(),
            backend=config.backend,
            model_implementation_version=config.model_implementation_version,
            buffer_schema_version=config.buffer_schema_version,
            gae_version=config.gae_version,
            loss_version=config.loss_version,
            architecture_sha256=architecture_digest(
                observation_size,
                training_environment.vocabulary.size,
                _HIDDEN_SIZE,
            ),
            implementation_sha256=implementation_digest(),
            git_commit=git_commit,
            dirty_worktree=dirty_worktree,
            created_at_utc=datetime.now(timezone.utc),
        )
