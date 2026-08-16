from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO

from warehouse_ai.demo.artifacts import sha256_file
from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.models import (
    DirectivePpoManifest,
    DirectivePpoValidationSummary,
    DirectiveVocabulary,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
from warehouse_ai.demo.sb3 import DirectiveSb3Env


ROOT = Path(__file__).parents[2]
ENVIRONMENT_ID = "tarware-tiny-3agvs-2pickers-partialobs-v1"


def test_policy_masks_probabilities_and_save_reload_is_deterministic(
    tmp_path: Path,
) -> None:
    vocabulary = DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")
    registry = build_directive_registry(vocabulary)
    environment = DirectiveSb3Env(
        ENVIRONMENT_ID,
        vocabulary,
        registry,
        (8100,),
        episode_steps=16,
    )
    try:
        model = MaskablePPO(
            "MlpPolicy",
            environment,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": {"pi": [64], "vf": [64]}},
            seed=8100,
            verbose=0,
        )
        checkpoint = tmp_path / "directive-ppo.zip"
        model.save(str(checkpoint))
        digest = sha256_file(checkpoint)
        validation = DirectivePpoValidationSummary(
            seeds=(8200,),
            mean_completed_tasks=0.0,
            mean_return=0.0,
            directive_counts={"wait": 1},
            non_wait_directives_used=0,
            inapplicable_directives=0,
        )
        manifest = DirectivePpoManifest(
            schema_version=1,
            environment_id=ENVIRONMENT_ID,
            vocabulary_sha256=vocabulary.digest(),
            behavior_sha256=registry.behavior_digest(),
            checkpoint_sha256=digest,
            train_seeds=(8100,),
            validation_seeds=(8200,),
            algorithm_seed=8100,
            hyperparameters={"test": True},
            validation=validation,
            package_versions={"test": "1"},
            git_commit="test",
            dirty_worktree=False,
            created_at_utc=datetime.now(timezone.utc),
        )
        loaded = DirectivePpoPolicy.load(checkpoint, vocabulary, manifest)
        environment.reset(seed=8100)
        mask = np.asarray([False, True, False, False], dtype=np.bool_)
        first = loaded.predict(environment.context, mask)
        second = loaded.predict(environment.context, mask)

        assert first.directive.directive_id == 1
        assert first.probabilities[0] == 0.0
        assert first.probabilities[2] == 0.0
        assert first.probabilities[3] == 0.0
        assert np.isclose(sum(first.probabilities), 1.0)
        assert first.directive == second.directive
        assert first.probabilities == second.probabilities
        assert first.value_estimate == second.value_estimate
    finally:
        environment.close()


def test_policy_load_rejects_behavior_or_checkpoint_digest_mismatch(
    tmp_path: Path,
) -> None:
    vocabulary = DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")
    registry = build_directive_registry(vocabulary)
    environment = DirectiveSb3Env(
        ENVIRONMENT_ID,
        vocabulary,
        registry,
        (8100,),
        episode_steps=8,
    )
    try:
        model = MaskablePPO("MlpPolicy", environment, n_steps=8, batch_size=4, seed=1)
        checkpoint = tmp_path / "directive-ppo.zip"
        model.save(str(checkpoint))
        validation = DirectivePpoValidationSummary(
            seeds=(8200,),
            mean_completed_tasks=0.0,
            mean_return=0.0,
            directive_counts={},
            non_wait_directives_used=0,
            inapplicable_directives=0,
        )
        manifest = DirectivePpoManifest(
            schema_version=1,
            environment_id=ENVIRONMENT_ID,
            vocabulary_sha256=vocabulary.digest(),
            behavior_sha256=registry.behavior_digest(),
            checkpoint_sha256=sha256_file(checkpoint),
            train_seeds=(8100,),
            validation_seeds=(8200,),
            algorithm_seed=1,
            hyperparameters={},
            validation=validation,
            package_versions={},
            git_commit="test",
            dirty_worktree=False,
            created_at_utc=datetime.now(timezone.utc),
        )
        bad_manifest = manifest.model_copy(update={"behavior_sha256": "0" * 64})
        try:
            DirectivePpoPolicy.load(checkpoint, vocabulary, bad_manifest)
        except ValueError as error:
            assert "behavior digest" in str(error)
        else:
            raise AssertionError("behavior digest mismatch was accepted")
    finally:
        environment.close()
