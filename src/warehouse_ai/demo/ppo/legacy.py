"""Lazy-loaded SB3 reference trainer retained for schema-v1 reproduction."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from warehouse_ai.demo.artifacts import capture_git_state, sha256_file
from warehouse_ai.demo.contracts.directives import DirectiveVocabulary
from warehouse_ai.demo.contracts.ppo import DirectivePpoConfig, DirectivePpoManifest
from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.demo.ppo.environment import DirectiveGymEnv
from warehouse_ai.demo.ppo.policy import DirectivePpoPolicy, Sb3PolicyAdapter
from warehouse_ai.demo.ppo.training import _package_versions, _repository_root
from warehouse_ai.demo.ppo.validation import (
    evaluate_directive_ppo,
    is_better_validation,
)


def _save_model(model: Any, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(base_path))
    checkpoint = base_path.with_suffix(".zip")
    if not checkpoint.is_file():
        raise RuntimeError(f"SB3 did not create expected checkpoint {checkpoint}")
    return checkpoint


def train_directive_ppo_v1(
    config: DirectivePpoConfig,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
    output_directory: Path,
) -> DirectivePpoManifest:
    """Reproduce the historical MaskablePPO reference only when requested."""
    from sb3_contrib import MaskablePPO

    if config.schema_version != 1:
        raise ValueError("legacy trainer accepts schema version 1 only")
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = DirectiveGymEnv(
        config.environment_id,
        vocabulary,
        registry,
        config.train_seeds,
        config.episode_steps,
    )
    model = MaskablePPO(
        "MlpPolicy",
        environment,
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
        policy_kwargs={"net_arch": {"pi": [64], "vf": [64]}},
        seed=config.algorithm_seed,
        verbose=0,
    )
    best = None
    checkpoint = output_directory / "directive-ppo-v1.zip"
    try:
        trained = 0
        while trained < config.total_timesteps:
            chunk = min(5_000, config.total_timesteps - trained)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            trained += chunk
            with tempfile.TemporaryDirectory(dir=output_directory) as temporary:
                candidate_path = _save_model(model, Path(temporary) / "candidate")
                candidate = DirectivePpoPolicy(
                    Sb3PolicyAdapter(model),
                    vocabulary,
                    sha256_file(candidate_path),
                )
                summary = evaluate_directive_ppo(
                    candidate, config, vocabulary, registry
                )
                if is_better_validation(summary, best):
                    checkpoint = _save_model(
                        model, output_directory / "directive-ppo-v1"
                    )
                    best = summary
    finally:
        environment.close()
    if best is None or best.non_wait_directives_used < 2:
        raise RuntimeError("legacy PPO failed validation coverage")
    git_commit, dirty = capture_git_state(_repository_root())
    return DirectivePpoManifest(
        schema_version=1,
        environment_id=config.environment_id,
        vocabulary_sha256=vocabulary.digest(),
        behavior_sha256=registry.behavior_digest(),
        checkpoint_sha256=sha256_file(checkpoint),
        train_seeds=config.train_seeds,
        validation_seeds=config.validation_seeds,
        algorithm_seed=config.algorithm_seed,
        hyperparameters={
            "learning_rate": config.learning_rate,
            "n_steps": config.n_steps,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "gamma": config.gamma,
            "gae_lambda": config.gae_lambda,
            "clip_range": config.clip_range,
            "entropy_coefficient": config.entropy_coefficient,
        },
        validation=best,
        package_versions=_package_versions(),
        git_commit=git_commit,
        dirty_worktree=dirty,
        created_at_utc=datetime.now(timezone.utc),
    )
