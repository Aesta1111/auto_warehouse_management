"""MaskablePPO training and deterministic inference for directive actions."""

from __future__ import annotations

import importlib.metadata
import math
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from warehouse_ai.demo.artifacts import capture_git_state, sha256_file
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.models import (
    DirectiveContext,
    DirectivePpoConfig,
    DirectivePpoManifest,
    DirectivePpoValidationSummary,
    DirectivePrediction,
    DirectiveVocabulary,
)
from warehouse_ai.demo.sb3 import DirectiveSb3Env


_VALIDATION_INTERVAL = 5_000
_POLICY_NET_ARCH = {"pi": [64], "vf": [64]}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _validate_mask(
    directive_mask: NDArray[np.bool_], vocabulary: DirectiveVocabulary
) -> NDArray[np.bool_]:
    mask = np.asarray(directive_mask, dtype=np.bool_)
    expected_shape = (vocabulary.size,)
    if mask.shape != expected_shape:
        raise ValueError(
            f"directive mask shape {mask.shape} does not match {expected_shape}"
        )
    if not bool(np.any(mask)):
        raise ValueError("directive mask must allow at least one directive")
    return mask


class DirectivePpoPolicy:
    """Frozen deterministic view of a trained directive MaskablePPO model."""

    def __init__(
        self,
        model: MaskablePPO,
        vocabulary: DirectiveVocabulary,
        checkpoint_sha256: str,
    ) -> None:
        if not isinstance(model.action_space, gym.spaces.Discrete):
            raise TypeError("directive PPO must use a Discrete action space")
        if int(model.action_space.n) != vocabulary.size:
            raise ValueError("PPO action space size does not match the vocabulary")
        if len(checkpoint_sha256) != 64:
            raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
        self.model = model
        self.vocabulary = vocabulary
        self.checkpoint_sha256 = checkpoint_sha256.lower()

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        vocabulary: DirectiveVocabulary,
        manifest: DirectivePpoManifest,
    ) -> DirectivePpoPolicy:
        """Load only when checkpoint, vocabulary, and behavior digests agree."""
        actual_checkpoint_digest = sha256_file(checkpoint)
        if actual_checkpoint_digest != manifest.checkpoint_sha256:
            raise ValueError("PPO checkpoint digest does not match its manifest")
        if vocabulary.digest() != manifest.vocabulary_sha256:
            raise ValueError("current vocabulary digest does not match the manifest")
        current_behavior_digest = build_directive_registry(vocabulary).behavior_digest()
        if current_behavior_digest != manifest.behavior_sha256:
            raise ValueError(
                "current directive behavior digest does not match the manifest"
            )

        model = MaskablePPO.load(str(checkpoint), device="cpu")
        return cls(model, vocabulary, actual_checkpoint_digest)

    def predict(
        self,
        context: DirectiveContext,
        directive_mask: NDArray[np.bool_],
    ) -> DirectivePrediction:
        """Return the highest-probability valid directive without sampling."""
        mask = _validate_mask(directive_mask, self.vocabulary)
        start = time.perf_counter()
        observation = context.observation.encode()
        with torch.inference_mode():
            observation_tensor, _ = self.model.policy.obs_to_tensor(observation)
            distribution = self.model.policy.get_distribution(
                observation_tensor, action_masks=mask
            )
            categorical = getattr(distribution, "distribution", None)
            probabilities_tensor = getattr(categorical, "probs", None)
            if probabilities_tensor is None:
                raise RuntimeError(
                    "directive PPO did not return a categorical distribution"
                )
            probabilities = (
                probabilities_tensor[0].detach().cpu().numpy().astype(np.float64)
            )
            value_tensor = self.model.policy.predict_values(observation_tensor)
            value_estimate = float(value_tensor.reshape(-1)[0].item())

        probabilities = np.where(mask, probabilities, 0.0)
        probability_total = float(probabilities.sum())
        if not math.isfinite(probability_total) or probability_total <= 0.0:
            raise RuntimeError("masked directive probabilities have no positive mass")
        probabilities /= probability_total
        selected_id = int(np.argmax(probabilities))
        if not bool(mask[selected_id]):
            raise RuntimeError("masked PPO selected an inapplicable directive")

        return DirectivePrediction(
            state_id=context.state_id,
            episode_step=context.observation.step,
            directive=self.vocabulary.by_id(selected_id),
            probabilities=tuple(float(value) for value in probabilities),
            value_estimate=value_estimate,
            policy_latency_ms=(time.perf_counter() - start) * 1000.0,
            checkpoint_sha256=self.checkpoint_sha256,
        )


def _package_versions() -> dict[str, str]:
    """Capture exact installed distributions used by the prototype."""
    packages = (
        "auto-warehouse-management",
        "gymnasium",
        "numpy",
        "sb3-contrib",
        "stable-baselines3",
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


def _completed_tasks(info: dict[str, Any]) -> int:
    """Read the pinned TA-RWARE delivery counter verified by the spike."""
    return int(info.get("shelf_deliveries", 0))


def _entropy(probabilities: tuple[float, ...]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def evaluate_directive_ppo(
    policy: DirectivePpoPolicy,
    config: DirectivePpoConfig,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
) -> DirectivePpoValidationSummary:
    """Evaluate one frozen policy on validation seeds only."""
    if policy.vocabulary.digest() != vocabulary.digest():
        raise ValueError("policy vocabulary does not match evaluation vocabulary")
    if registry.vocabulary.digest() != vocabulary.digest():
        raise ValueError("registry vocabulary does not match evaluation vocabulary")

    environment = DirectiveSb3Env(
        config.environment_id,
        vocabulary,
        registry,
        config.validation_seeds,
        config.episode_steps,
    )
    episode_returns: list[float] = []
    episode_completed: list[int] = []
    directive_counts: dict[str, int] = {}
    selected_non_wait: set[int] = set()
    inapplicable_directives = 0
    masked_action_count = 0
    entropies: list[float] = []

    try:
        for seed in config.validation_seeds:
            _, _ = environment.reset(seed=seed)
            total_return = 0.0
            completed_tasks = 0
            for _step in range(config.episode_steps):
                context = environment.context
                mask = environment.action_masks()
                masked_action_count += int((~mask).sum())
                prediction = policy.predict(context, mask)
                selected_id = prediction.directive.directive_id
                if not bool(mask[selected_id]):
                    inapplicable_directives += 1
                if selected_id != vocabulary.wait_directive_id:
                    selected_non_wait.add(selected_id)
                command = prediction.directive.command
                directive_counts[command] = directive_counts.get(command, 0) + 1
                entropies.append(_entropy(prediction.probabilities))

                _, reward, terminated, truncated, info = environment.step(selected_id)
                total_return += reward
                completed_tasks += _completed_tasks(info)
                if terminated or truncated:
                    break
            episode_returns.append(total_return)
            episode_completed.append(completed_tasks)
    finally:
        environment.close()

    if not episode_returns:
        raise RuntimeError("validation produced no episodes")
    non_wait_count = len(selected_non_wait)
    return DirectivePpoValidationSummary(
        seeds=config.validation_seeds,
        mean_completed_tasks=float(np.mean(episode_completed)),
        mean_return=float(np.mean(episode_returns)),
        directive_counts=dict(sorted(directive_counts.items())),
        non_wait_directives_used=non_wait_count,
        inapplicable_directives=inapplicable_directives,
        masked_action_count=masked_action_count,
        mean_entropy=float(np.mean(entropies)) if entropies else 0.0,
        collapsed_to_one_non_wait=non_wait_count < 2,
    )


def _save_model(model: MaskablePPO, base_path: Path) -> Path:
    """Save an SB3 zip at a path whose suffix is controlled by this module."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(base_path))
    checkpoint = base_path.with_suffix(".zip")
    if not checkpoint.is_file():
        raise RuntimeError(f"SB3 did not create expected checkpoint {checkpoint}")
    return checkpoint


def _is_better(
    candidate: DirectivePpoValidationSummary,
    incumbent: DirectivePpoValidationSummary | None,
) -> bool:
    if incumbent is None:
        return True
    return (candidate.mean_completed_tasks, candidate.mean_return) > (
        incumbent.mean_completed_tasks,
        incumbent.mean_return,
    )


def train_directive_ppo(
    config: DirectivePpoConfig,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
    output_directory: Path,
) -> DirectivePpoManifest:
    """Train, validate, and freeze the best directive PPO checkpoint."""
    if registry.vocabulary.digest() != vocabulary.digest():
        raise ValueError("registry vocabulary does not match training vocabulary")
    output_directory.mkdir(parents=True, exist_ok=True)
    training_environment = DirectiveSb3Env(
        config.environment_id,
        vocabulary,
        registry,
        config.train_seeds,
        config.episode_steps,
    )
    best_checkpoint_base = output_directory / "directive-ppo-v1"
    best_summary: DirectivePpoValidationSummary | None = None
    best_checkpoint: Path | None = None

    policy_kwargs = {"net_arch": _POLICY_NET_ARCH}
    model = MaskablePPO(
        "MlpPolicy",
        training_environment,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.entropy_coefficient,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=config.algorithm_seed,
        verbose=0,
    )

    try:
        trained_steps = 0
        while trained_steps < config.total_timesteps:
            chunk = min(_VALIDATION_INTERVAL, config.total_timesteps - trained_steps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            trained_steps += chunk

            with tempfile.TemporaryDirectory(
                dir=output_directory, prefix=".directive-ppo-candidate-"
            ) as temporary_directory:
                candidate = _save_model(model, Path(temporary_directory) / "candidate")
                candidate_digest = sha256_file(candidate)
                candidate_policy = DirectivePpoPolicy(
                    model, vocabulary, candidate_digest
                )
                candidate_summary = evaluate_directive_ppo(
                    candidate_policy, config, vocabulary, registry
                )
                if _is_better(candidate_summary, best_summary):
                    _save_model(model, best_checkpoint_base)
                    best_checkpoint = best_checkpoint_base.with_suffix(".zip")
                    best_summary = candidate_summary
    finally:
        training_environment.close()

    if best_checkpoint is None or best_summary is None:
        raise RuntimeError("directive PPO training produced no validated checkpoint")
    if best_summary.non_wait_directives_used < 2:
        raise RuntimeError(
            "directive PPO validation selected fewer than two non-wait directives; "
            "revise the vocabulary or executor semantics before using Gemma"
        )

    checkpoint_digest = sha256_file(best_checkpoint)
    git_commit, dirty_worktree = capture_git_state(_repository_root())
    hyperparameters: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "n_steps": config.n_steps,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "gamma": config.gamma,
        "gae_lambda": config.gae_lambda,
        "clip_range": config.clip_range,
        "entropy_coefficient": config.entropy_coefficient,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_net_arch": _POLICY_NET_ARCH,
        "validation_interval": _VALIDATION_INTERVAL,
    }
    return DirectivePpoManifest(
        schema_version=1,
        environment_id=config.environment_id,
        vocabulary_sha256=vocabulary.digest(),
        behavior_sha256=registry.behavior_digest(),
        checkpoint_sha256=checkpoint_digest,
        train_seeds=config.train_seeds,
        validation_seeds=config.validation_seeds,
        algorithm_seed=config.algorithm_seed,
        hyperparameters=hyperparameters,
        validation=best_summary,
        package_versions=_package_versions(),
        git_commit=git_commit,
        dirty_worktree=dirty_worktree,
        created_at_utc=datetime.now(timezone.utc),
    )
