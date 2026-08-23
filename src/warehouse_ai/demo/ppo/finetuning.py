"""Warm-started PPO updates against the frozen Gemma execution path."""

from __future__ import annotations

import importlib.metadata
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from warehouse_ai.demo.artifacts import capture_git_state, sha256_file
from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.finetuning import (
    GemmaFineTuneConfig,
    GemmaFineTuneManifest,
)
from warehouse_ai.demo.contracts.ppo import PpoManifest
from warehouse_ai.demo.ppo.buffer import RolloutBuffer
from warehouse_ai.demo.ppo.collector import RolloutCollector
from warehouse_ai.demo.ppo.gemma_environment import GemmaGymEnv
from warehouse_ai.demo.ppo.model import (
    MODEL_IMPLEMENTATION_VERSION,
    PpoModel,
    architecture_digest,
    implementation_digest,
)
from warehouse_ai.demo.ppo.policy import PpoPolicy, TorchPolicyAdapter


HIDDEN_SIZE = 64
CHECKPOINT_NAME = "directive-ppo-gemma-finetuned-v1.pt"
PROGRESS_NAME = "fine-tune-progress-v1.pt"


@dataclass(frozen=True)
class FineTuneArtifactIdentity:
    parent_manifest_sha256: str
    prompt_contract_sha256: str
    preflight_report_sha256: str


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
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def warm_start_model(
    config: GemmaFineTuneConfig,
    catalog: DirectiveCatalog,
    parent_policy: PpoPolicy,
) -> PpoModel:
    """Copy verified parent weights into a model with fine-tuning settings."""
    adapter = parent_policy.adapter
    if not isinstance(adapter, TorchPolicyAdapter):
        raise ValueError("Gemma fine-tuning requires a repository-owned PPO parent")
    parent = adapter.model
    if parent.directive_count != catalog.size:
        raise ValueError("parent PPO directive count does not match catalog")
    model = PpoModel(
        observation_size=parent.observation_size,
        directive_count=parent.directive_count,
        hidden_size=parent.hidden_size,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        entropy_coefficient=config.entropy_coefficient,
        value_coefficient=config.value_coefficient,
        normalize_advantage=config.normalize_advantage,
    )
    model.load_state_dict(parent.state_dict(), strict=True)
    return model


def load_fine_tuned_policy(
    checkpoint: Path,
    catalog: DirectiveCatalog,
    parent_manifest: PpoManifest,
    manifest: GemmaFineTuneManifest,
) -> PpoPolicy:
    """Load a fine-tuned policy while verifying its parent and architecture."""
    if manifest.parent_checkpoint_sha256 != parent_manifest.checkpoint_sha256:
        raise ValueError("fine-tuned policy parent does not match parent manifest")
    if sha256_file(checkpoint) != manifest.checkpoint_sha256:
        raise ValueError("fine-tuned PPO checkpoint digest does not match manifest")
    if catalog.digest() != manifest.vocabulary_sha256:
        raise ValueError("fine-tuned PPO catalog digest does not match")
    values = manifest.hyperparameters
    dimensions = (
        int(values["observation_size"]),
        int(values["directive_count"]),
        int(values["hidden_size"]),
    )
    if manifest.architecture_sha256 != architecture_digest(*dimensions):
        raise ValueError("fine-tuned PPO architecture digest does not match")
    if manifest.implementation_sha256 != implementation_digest():
        raise ValueError("fine-tuned PPO implementation digest does not match")
    model = PpoModel(
        observation_size=dimensions[0],
        directive_count=dimensions[1],
        hidden_size=dimensions[2],
        learning_rate=float(values["learning_rate"]),
        clip_range=float(values["clip_range"]),
        entropy_coefficient=float(values["entropy_coefficient"]),
        value_coefficient=float(values["value_coefficient"]),
        normalize_advantage=bool(values["normalize_advantage"]),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("fine-tuned PPO checkpoint has no model state dictionary")
    model.load_state_dict(state, strict=True)
    model.eval()
    return PpoPolicy.from_module(model, catalog, manifest.checkpoint_sha256)


def _reference_kl(module: Any, reference: PpoModel, buffer: RolloutBuffer) -> float:
    model = getattr(module, "module", module)
    device = next(model.parameters()).device
    observations = torch.from_numpy(buffer.observations.copy()).to(device)
    masks = torch.from_numpy(buffer.action_masks.copy()).to(device)
    reference = reference.to(device)
    reference.eval()
    model.eval()
    with torch.inference_mode():
        old = reference.probabilities(observations, masks).clamp_min(1e-12)
        current = model.probabilities(observations, masks).clamp_min(1e-12)
        kl = torch.sum(old * (torch.log(old) - torch.log(current)), dim=1)
    return float(torch.max(kl).item())


class GemmaFineTuneLoop:
    """Collect expensive Gemma rollouts and update a warm-started PPO."""

    def fit(
        self,
        model: PpoModel,
        reference_model: PpoModel,
        config: GemmaFineTuneConfig,
        environment: GemmaGymEnv,
        output_directory: Path,
        identity: FineTuneArtifactIdentity,
        parent_manifest: PpoManifest,
    ) -> GemmaFineTuneManifest:
        from lightning.fabric import Fabric

        output_directory.mkdir(parents=True, exist_ok=True)
        random.seed(config.algorithm_seed)
        np.random.seed(config.algorithm_seed)
        torch.manual_seed(config.algorithm_seed)
        fabric = Fabric(accelerator="auto", devices=1, precision="32-true")
        optimizer = model.make_optimizer()
        module, optimizer = fabric.setup(model, optimizer)
        rollout_generator = torch.Generator(device=fabric.device.type)
        rollout_generator.manual_seed(config.algorithm_seed + 1)
        minibatch_generator = torch.Generator(device="cpu")
        minibatch_generator.manual_seed(config.algorithm_seed + 2)
        shape = environment.observation_space.shape
        if shape is None:
            raise RuntimeError("fine-tuning environment has no observation shape")
        buffer = RolloutBuffer(config.n_steps, shape, environment.catalog.size)
        collector = RolloutCollector()
        completed_steps = optimizer_steps = 0
        maximum_reference_kl = 0.0
        checkpoint = output_directory / CHECKPOINT_NAME
        progress = output_directory / PROGRESS_NAME

        while completed_steps < config.total_timesteps:
            collector.collect(
                module,
                environment,
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
                    if float(terms.approximate_kl.item()) > config.target_kl:
                        stop_update = True
                        break
                if stop_update:
                    break
            completed_steps += config.n_steps
            rollout_kl = _reference_kl(module, reference_model, buffer)
            maximum_reference_kl = max(maximum_reference_kl, rollout_kl)
            if rollout_kl > config.max_reference_kl:
                raise RuntimeError("fine-tuned PPO exceeded max_reference_kl")
            fabric.save(
                progress,
                {
                    "model": module,
                    "optimizer": optimizer,
                    "completed_environment_steps": completed_steps,
                    "optimizer_steps": optimizer_steps,
                },
            )
            buffer.reset()

        fabric.save(checkpoint, {"model": module})
        return self._manifest(
            config,
            environment,
            checkpoint,
            identity,
            parent_manifest,
            completed_steps,
            optimizer_steps,
            maximum_reference_kl,
            int(shape[0]),
        )

    @staticmethod
    def _manifest(
        config: GemmaFineTuneConfig,
        environment: GemmaGymEnv,
        checkpoint: Path,
        identity: FineTuneArtifactIdentity,
        parent_manifest: PpoManifest,
        completed_steps: int,
        optimizer_steps: int,
        reference_kl_max: float,
        observation_size: int,
    ) -> GemmaFineTuneManifest:
        git_commit, dirty = capture_git_state(_repository_root())
        hyperparameters = {
            "observation_size": observation_size,
            "directive_count": environment.catalog.size,
            "hidden_size": HIDDEN_SIZE,
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
            "max_reference_kl": config.max_reference_kl,
            "declared_total_timesteps": config.total_timesteps,
        }
        return GemmaFineTuneManifest(
            environment_id=config.environment_id,
            vocabulary_sha256=environment.catalog.digest(),
            behavior_sha256=environment.registry.behavior_digest(),
            behavior_engine_version=environment.registry.engine_version,
            parent_checkpoint_sha256=parent_manifest.checkpoint_sha256,
            parent_manifest_sha256=identity.parent_manifest_sha256,
            checkpoint_sha256=sha256_file(checkpoint),
            architecture_sha256=architecture_digest(
                observation_size, environment.catalog.size, HIDDEN_SIZE
            ),
            implementation_sha256=implementation_digest(),
            model_implementation_version=MODEL_IMPLEMENTATION_VERSION,
            prompt_contract_sha256=identity.prompt_contract_sha256,
            preflight_report_sha256=identity.preflight_report_sha256,
            model_file_sha256=config.inference.model_file_sha256,
            train_seeds=config.train_seeds,
            selection_seeds=config.selection_seeds,
            confirmation_seeds=config.confirmation_seeds,
            final_evaluation_seeds=config.final_evaluation_seeds,
            algorithm_seed=config.algorithm_seed,
            reward_version=config.reward_version,
            fallback_penalty=config.fallback_penalty,
            hyperparameters=hyperparameters,
            training=environment.metrics.summary(
                completed_environment_steps=completed_steps,
                optimizer_steps=optimizer_steps,
                reference_kl_max=reference_kl_max,
            ),
            package_versions=_package_versions(),
            git_commit=git_commit,
            dirty_worktree=dirty,
            created_at_utc=datetime.now(timezone.utc),
        )
