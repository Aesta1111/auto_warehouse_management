from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.models import (
    DirectivePpoConfig,
    DirectivePpoValidationSummary,
    DirectiveVocabulary,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy, train_directive_ppo
from warehouse_ai.demo.sb3 import DirectiveSb3Env


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def test_short_fabric_collect_update_checkpoint_and_reload(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    vocabulary = DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")
    registry = build_directive_registry(vocabulary)
    config = DirectivePpoConfig(
        schema_version=2,
        environment_id=ENVIRONMENT_ID,
        vocabulary_path=Path("configs/demo/directives-v1.yaml"),
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
        backend="lightning-fabric",
        model_implementation_version="directive-ppo-module-v1",
        buffer_schema_version="directive-rollout-buffer-v1",
        gae_version="explicit-bootstrap-trace-v1",
        loss_version="clipped-surrogate-v1",
    )
    summary = DirectivePpoValidationSummary(
        seeds=(8200,),
        mean_completed_tasks=0.0,
        mean_return=0.0,
        directive_counts={"service-requested-shelf": 1, "wait": 1},
        non_wait_directives_used=2,
        inapplicable_directives=0,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "warehouse_ai.demo.ppo.evaluate_directive_ppo",
        lambda *_args, **_kwargs: summary,
    )

    output_directory = tmp_path / "directive-ppo-v2-smoke"
    manifest = train_directive_ppo(config, vocabulary, registry, output_directory)
    checkpoint = output_directory / "directive-ppo-v2.pt"
    assert manifest.schema_version == 2
    assert manifest.hyperparameters["optimizer_steps"] == 4
    assert manifest.hyperparameters["selected_checkpoint_environment_steps"] == 8
    assert manifest.hyperparameters["selected_checkpoint_optimizer_steps"] == 4
    loaded = DirectivePpoPolicy.load(checkpoint, vocabulary, manifest)
    with pytest.raises(ValueError, match="implementation digest"):
        DirectivePpoPolicy.load(
            checkpoint,
            vocabulary,
            manifest.model_copy(update={"implementation_sha256": "0" * 64}),
        )
    with pytest.raises(ValueError, match="architecture digest"):
        DirectivePpoPolicy.load(
            checkpoint,
            vocabulary,
            manifest.model_copy(update={"architecture_sha256": "0" * 64}),
        )

    environment = DirectiveSb3Env(
        ENVIRONMENT_ID, vocabulary, registry, (8200,), episode_steps=16
    )
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
