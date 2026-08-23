from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from warehouse_ai.demo.artifacts import sha256_file
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.ppo import (
    PpoConfig,
    PpoManifest,
    PpoValidationSummary,
)
from warehouse_ai.demo.ppo.environment import GymEnv
from warehouse_ai.demo.ppo.model import (
    LEGACY_MODEL_IMPLEMENTATION_VERSION,
    MODEL_IMPLEMENTATION_VERSION,
    PpoModel,
    implementation_digest,
    legacy_architecture_digest,
)
from warehouse_ai.demo.ppo.policy import PpoPolicy, TorchPolicyAdapter
from warehouse_ai.demo.ppo.service import train_directive_ppo


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def test_short_fabric_collect_update_checkpoint_and_reload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")
    registry = build_registry(catalog)
    config = PpoConfig(
        schema_version=4,
        environment_id=ENVIRONMENT_ID,
        vocabulary_path=Path("configs/demo/directive-catalog.yaml"),
        train_seeds=(8100,),
        validation_seeds=(8200,),
        algorithm_seed=8100,
        episode_steps=16,
        total_timesteps=8,
        n_steps=8,
        batch_size=4,
        n_epochs=2,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        hidden_sizes=(64,),
        backend="lightning-fabric",
        model_implementation_version=MODEL_IMPLEMENTATION_VERSION,
        buffer_schema_version="directive-rollout-buffer-v1",
        gae_version="gae-terminated-bootstrap-truncated-trace-v1",
        loss_version="clipped-ppo-loss-v1",
    )
    summary = PpoValidationSummary(
        seeds=(8200,),
        mean_completed_tasks=0.0,
        mean_return=0.0,
        directive_counts={"service-requested-shelf": 1, "wait": 1},
        non_wait_directives_used=2,
        inapplicable_directives=0,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "warehouse_ai.demo.ppo.training.evaluate_directive_ppo",
        lambda *_args, **_kwargs: summary,
    )

    output_directory = tmp_path / "directive-ppo-v4-smoke"
    manifest = train_directive_ppo(config, catalog, registry, output_directory)
    checkpoint = output_directory / "ppo-policy.pt"
    assert manifest.schema_version == 4
    assert manifest.behavior_engine_version == registry.engine_version
    assert manifest.hyperparameters["hidden_sizes"] == [64]
    assert "hidden_size" not in manifest.hyperparameters
    assert manifest.hyperparameters["optimizer_steps"] == 4
    assert manifest.hyperparameters["selected_checkpoint_environment_steps"] == 8
    assert manifest.hyperparameters["selected_checkpoint_optimizer_steps"] == 4
    loaded = PpoPolicy.load(checkpoint, catalog, manifest)
    with pytest.raises(ValueError, match="implementation digest"):
        PpoPolicy.load(
            checkpoint,
            catalog,
            manifest.model_copy(update={"implementation_sha256": "0" * 64}),
        )
    with pytest.raises(ValueError, match="architecture digest"):
        PpoPolicy.load(
            checkpoint,
            catalog,
            manifest.model_copy(update={"architecture_sha256": "0" * 64}),
        )

    environment = GymEnv(ENVIRONMENT_ID, catalog, registry, (8200,), episode_steps=16)
    try:
        environment.reset(seed=8200)
        mask = environment.action_masks()
        first = loaded.predict(environment.context, mask)
        second = loaded.predict(environment.context, mask)
        assert first.directive == second.directive
        assert first.probabilities == second.probabilities
        assert first.value_estimate == second.value_estimate
        assert np.isclose(sum(first.probabilities), 1.0)
    finally:
        environment.close()


def test_schema_v3_state_keys_migrate_to_equivalent_mlp(tmp_path: Path) -> None:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directive-catalog.yaml")
    registry = build_registry(catalog)
    torch.manual_seed(19)
    source = PpoModel(
        observation_size=3,
        directive_count=catalog.size,
        hidden_sizes=(64,),
        learning_rate=3e-4,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        normalize_advantage=True,
    )
    reverse = {
        "policy_network.layers.0.weight": "policy_path.0.weight",
        "policy_network.layers.0.bias": "policy_path.0.bias",
        "policy_network.layers.1.weight": "policy_head.weight",
        "policy_network.layers.1.bias": "policy_head.bias",
        "value_network.layers.0.weight": "value_path.0.weight",
        "value_network.layers.0.bias": "value_path.0.bias",
        "value_network.layers.1.weight": "value_head.weight",
        "value_network.layers.1.bias": "value_head.bias",
    }
    legacy_state = {reverse[name]: value for name, value in source.state_dict().items()}
    checkpoint = tmp_path / "legacy-v3.pt"
    torch.save({"model": legacy_state}, checkpoint)
    summary = PpoValidationSummary(
        seeds=(2,),
        mean_completed_tasks=0.0,
        mean_return=0.0,
        directive_counts={"wait": 1},
        non_wait_directives_used=0,
        inapplicable_directives=0,
    )
    manifest = PpoManifest(
        schema_version=3,
        environment_id=ENVIRONMENT_ID,
        vocabulary_sha256=catalog.digest(),
        behavior_sha256=registry.behavior_digest(),
        behavior_engine_version=registry.engine_version,
        checkpoint_sha256=sha256_file(checkpoint),
        train_seeds=(1,),
        validation_seeds=(2,),
        algorithm_seed=1,
        hyperparameters={
            "observation_size": 3,
            "directive_count": catalog.size,
            "hidden_size": 64,
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "normalize_advantage": True,
        },
        validation=summary,
        package_versions={},
        backend="lightning-fabric",
        model_implementation_version=LEGACY_MODEL_IMPLEMENTATION_VERSION,
        buffer_schema_version="directive-rollout-buffer-v1",
        gae_version="gae-terminated-bootstrap-truncated-trace-v1",
        loss_version="clipped-ppo-loss-v1",
        architecture_sha256=legacy_architecture_digest(
            3, catalog.size, 64, LEGACY_MODEL_IMPLEMENTATION_VERSION
        ),
        implementation_sha256=implementation_digest(
            LEGACY_MODEL_IMPLEMENTATION_VERSION,
            "gae-terminated-bootstrap-truncated-trace-v1",
            "clipped-ppo-loss-v1",
        ),
        git_commit="synthetic-test",
        dirty_worktree=False,
        created_at_utc="2026-08-23T00:00:00Z",
    )

    loaded = PpoPolicy.load(checkpoint, catalog, manifest)
    assert isinstance(loaded.adapter, TorchPolicyAdapter)
    observations = torch.randn((2, 3))
    masks = torch.ones((2, catalog.size), dtype=torch.bool)
    expected = source(observations, masks)
    actual = loaded.adapter.model(observations, masks)
    torch.testing.assert_close(actual.policy_logits, expected.policy_logits)
    torch.testing.assert_close(actual.values, expected.values)
