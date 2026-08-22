"""PPO training composition entrypoint."""

from __future__ import annotations

from pathlib import Path

import torch

from warehouse_ai.demo.contracts.directives import DirectiveVocabulary
from warehouse_ai.demo.contracts.ppo import DirectivePpoConfig, DirectivePpoManifest
from warehouse_ai.demo.directives import DirectiveRegistry
from warehouse_ai.demo.ppo.environment import DirectiveGymEnv
from warehouse_ai.demo.ppo.model import DirectivePpoModel
from warehouse_ai.demo.ppo.training import DirectivePpoTrainingLoop, HIDDEN_SIZE


def train_directive_ppo(
    config: DirectivePpoConfig,
    vocabulary: DirectiveVocabulary,
    registry: DirectiveRegistry,
    output_directory: Path,
) -> DirectivePpoManifest:
    if config.schema_version == 1:
        from warehouse_ai.demo.ppo.legacy import train_directive_ppo_v1

        return train_directive_ppo_v1(config, vocabulary, registry, output_directory)
    if registry.vocabulary.digest() != vocabulary.digest():
        raise ValueError("registry vocabulary does not match training vocabulary")
    assert config.value_coefficient is not None
    assert config.normalize_advantage is not None
    torch.manual_seed(config.algorithm_seed)
    training = DirectiveGymEnv(
        config.environment_id,
        vocabulary,
        registry,
        config.train_seeds,
        config.episode_steps,
    )
    validation = DirectiveGymEnv(
        config.environment_id,
        vocabulary,
        registry,
        config.validation_seeds,
        config.episode_steps,
    )
    observation_shape = training.observation_space.shape
    if observation_shape is None:
        raise RuntimeError("training environment has no fixed observation shape")
    model = DirectivePpoModel(
        observation_size=int(observation_shape[0]),
        directive_count=vocabulary.size,
        hidden_size=HIDDEN_SIZE,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        entropy_coefficient=config.entropy_coefficient,
        value_coefficient=config.value_coefficient,
        normalize_advantage=config.normalize_advantage,
    )
    try:
        return DirectivePpoTrainingLoop().fit(
            model, config, training, validation, output_directory
        )
    finally:
        training.close()
        validation.close()
