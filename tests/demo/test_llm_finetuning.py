from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import torch

from warehouse_ai.demo.contracts.configuration import load_ppo_fine_tuning_run
from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.finetuning import LlmFineTuneConfig
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ActionerResult,
    LlmActionResponse,
)
from warehouse_ai.demo.contracts.ppo import PpoManifest
from warehouse_ai.demo.directives import Registry, build_registry
from warehouse_ai.demo.inference.actioner import Actioner
from warehouse_ai.demo.ppo.finetuning import (
    FineTuneArtifactIdentity,
    LlmFineTuneLoop,
    CHECKPOINT_NAME,
    load_fine_tuned_policy,
    warm_start_model,
)
from warehouse_ai.demo.ppo.llm_environment import LlmGymEnv
from warehouse_ai.demo.ppo.model import PpoModel
from warehouse_ai.demo.ppo.policy import PpoPolicy, TorchPolicyAdapter


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


class ValidActioner:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self.environment: LlmGymEnv | None = None

    def act(self, request: ActionerRequest) -> ActionerResult:
        assert self.environment is not None
        action = self.registry.execute(
            request.directive.directive_id,
            request.context,
            self.environment.wrapper,
        )
        return ActionerResult(
            response=LlmActionResponse(
                destinations=tuple(item.destination_id for item in action.actions),
                reason="valid fake action",
            ),
            raw_text="{}",
            model_id="fake-model",
            latency_ms=1.0,
        )


def _config() -> LlmFineTuneConfig:
    original = load_ppo_fine_tuning_run(
        Path("configs/demo/ppo-llm-fine-tuning-run.yaml"), ROOT
    ).runtime
    values = original.model_dump()
    values.update(
        {
            "train_seeds": (8100,),
            "selection_seeds": (8200,),
            "confirmation_seeds": (8201,),
            "total_timesteps": 2,
            "n_steps": 2,
            "batch_size": 1,
            "n_epochs": 1,
            "max_reference_kl": 1.0,
        }
    )
    return LlmFineTuneConfig.model_validate(values)


def _parent(catalog: DirectiveCatalog, observation_size: int) -> PpoPolicy:
    torch.manual_seed(11)
    model = PpoModel(
        observation_size=observation_size,
        directive_count=catalog.size,
        hidden_sizes=(32, 16),
        learning_rate=3e-4,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        normalize_advantage=True,
    )
    return PpoPolicy.from_module(model, catalog, "1" * 64)


def test_warm_start_preserves_every_parent_parameter() -> None:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")
    parent = _parent(catalog, 8)
    model = warm_start_model(_config(), catalog, parent)

    assert isinstance(parent.adapter, TorchPolicyAdapter)
    parent_state = parent.adapter.model.state_dict()
    assert all(
        torch.equal(value, model.state_dict()[name])
        for name, value in parent_state.items()
    )
    assert model.learning_rate == 3e-5
    assert model.hidden_sizes == (32, 16)


def test_two_step_fake_fine_tune_saves_reloadable_child_checkpoint() -> None:
    config = _config()
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")
    registry = build_registry(catalog)
    actioner = ValidActioner(registry)
    environment = LlmGymEnv(
        ENVIRONMENT_ID,
        catalog,
        registry,
        config.train_seeds,
        cast(Actioner, actioner),
        episode_steps=config.episode_steps,
        fallback_penalty=config.fallback_penalty,
    )
    actioner.environment = environment
    observation_shape = environment.observation_space.shape
    assert observation_shape is not None
    parent = _parent(catalog, int(observation_shape[0]))
    parent_manifest = PpoManifest.model_construct(
        checkpoint_sha256="1" * 64,
    )
    output = ROOT / "results" / f".test-finetune-{uuid4().hex}"
    output.mkdir(parents=False)
    try:
        manifest = LlmFineTuneLoop().fit(
            warm_start_model(config, catalog, parent),
            warm_start_model(config, catalog, parent),
            config,
            environment,
            output,
            FineTuneArtifactIdentity("2" * 64, "3" * 64, "4" * 64),
            parent_manifest,
        )
        loaded = load_fine_tuned_policy(
            output / CHECKPOINT_NAME,
            catalog,
            parent_manifest,
            manifest,
        )

        assert manifest.training.completed_environment_steps == 2
        assert manifest.training.llm_calls <= 2
        assert manifest.schema_version == 3
        assert manifest.hyperparameters["hidden_sizes"] == [32, 16]
        assert loaded.checkpoint_sha256 == manifest.checkpoint_sha256
        bad_manifest = manifest.model_copy(
            update={
                "hyperparameters": {
                    **manifest.hyperparameters,
                    "hidden_sizes": [64],
                }
            }
        )
        with pytest.raises(ValueError, match="architecture digest"):
            load_fine_tuned_policy(
                output / CHECKPOINT_NAME,
                catalog,
                parent_manifest,
                bad_manifest,
            )
    finally:
        environment.close()
        for artifact in output.iterdir():
            artifact.unlink()
        output.rmdir()
