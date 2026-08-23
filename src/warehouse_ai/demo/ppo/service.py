"""PPO training composition entrypoint."""

from __future__ import annotations

from pathlib import Path

import torch

from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.ppo import PpoConfig, PpoManifest
from warehouse_ai.demo.directives import Registry
from warehouse_ai.demo.ppo.environment import GymEnv
from warehouse_ai.demo.ppo.model import PpoModel
from warehouse_ai.demo.ppo.training import PpoTrainingLoop, HIDDEN_SIZE


def train_directive_ppo(
    config: PpoConfig,
    catalog: DirectiveCatalog,
    registry: Registry,
    output_directory: Path,
) -> PpoManifest:
    if config.schema_version == 1:
        from warehouse_ai.demo.ppo.legacy import train_directive_ppo_v1

        return train_directive_ppo_v1(config, catalog, registry, output_directory)
    if registry.catalog.digest() != catalog.digest():
        raise ValueError("registry catalog does not match training catalog")
    assert config.value_coefficient is not None
    assert config.normalize_advantage is not None
    torch.manual_seed(config.algorithm_seed)
    training = GymEnv(
        config.environment_id,
        catalog,
        registry,
        config.train_seeds,
        config.episode_steps,
    )
    validation = GymEnv(
        config.environment_id,
        catalog,
        registry,
        config.validation_seeds,
        config.episode_steps,
    )
    observation_shape = training.observation_space.shape
    if observation_shape is None:
        raise RuntimeError("training environment has no fixed observation shape")
    model = PpoModel(
        observation_size=int(observation_shape[0]),
        directive_count=catalog.size,
        hidden_sizes=(HIDDEN_SIZE,),
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        entropy_coefficient=config.entropy_coefficient,
        value_coefficient=config.value_coefficient,
        normalize_advantage=config.normalize_advantage,
    )
    try:
        return PpoTrainingLoop().fit(
            model, config, training, validation, output_directory
        )
    finally:
        training.close()
        validation.close()
