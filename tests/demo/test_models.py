from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.directives import (
    DirectiveContext,
    DirectiveDefinition,
    DirectiveProbeConfig,
    DirectiveProbeReport,
    DirectiveVocabulary,
)
from warehouse_ai.demo.contracts.evaluation import DirectiveDemoConfig
from warehouse_ai.demo.contracts.inference import LlamaCppBackendConfig
from warehouse_ai.demo.contracts.ppo import DirectivePpoConfig
from warehouse_ai.domain import ActionMask, AgentObservation, AgentRole, Position
from warehouse_ai.domain import ShelfObservation, WarehouseObservation


ROOT = Path(__file__).parents[2]


def test_committed_versioned_configs_load() -> None:
    vocabulary = DirectiveVocabulary.load(ROOT / "configs/demo/directives-v1.yaml")
    probe = load_yaml_config(
        ROOT / "configs/demo/directive-probe-v1.yaml", DirectiveProbeConfig
    )
    ppo = load_yaml_config(
        ROOT / "configs/demo/directive-ppo-v1.yaml", DirectivePpoConfig
    )
    ppo_v2 = load_yaml_config(
        ROOT / "configs/demo/directive-ppo-v2.yaml", DirectivePpoConfig
    )
    probe_v2 = load_yaml_config(
        ROOT / "configs/demo/directive-probe-v2.yaml", DirectiveProbeConfig
    )
    ppo_v3 = load_yaml_config(
        ROOT / "configs/demo/directive-ppo-v3.yaml", DirectivePpoConfig
    )
    demo_v2 = load_yaml_config(
        ROOT / "configs/demo/directive-demo-v2.yaml", DirectiveDemoConfig
    )

    assert vocabulary.size == 4
    assert vocabulary.by_id(vocabulary.wait_directive_id).command == "wait"
    assert probe.vocabulary_path.as_posix() == "configs/demo/directives-v1.yaml"
    assert ppo.train_seeds[-1] == 8115
    assert ppo_v2.backend == "lightning-fabric"
    assert ppo_v2.value_coefficient == 0.5
    assert probe_v2.schema_version == 2
    assert ppo_v3.schema_version == 3
    assert demo_v2.inference.backend == "llama_cpp"
    assert demo_v2.preflight_p95_limit_seconds == 3.92


def test_contract_import_does_not_load_ml_or_transport_frameworks() -> None:
    code = """
import sys
from warehouse_ai.demo.contracts.ppo import DirectivePpoConfig
from warehouse_ai.demo.contracts.evaluation import DirectiveDemoConfig
forbidden = {'torch', 'lightning', 'stable_baselines3', 'sb3_contrib', 'httpx'}
assert not forbidden.intersection(sys.modules), sorted(forbidden.intersection(sys.modules))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_vocabulary_digest_is_stable_and_ids_are_canonical() -> None:
    first = DirectiveVocabulary(
        schema_version=1,
        vocabulary_id="v1",
        behavior_version="b1",
        wait_directive_id=0,
        directives=tuple(
            DirectiveDefinition.model_validate(item)
            for item in (
                {
                    "directive_id": 2,
                    "command": "two",
                    "description": "Two",
                    "behavior_key": "two",
                },
                {
                    "directive_id": 0,
                    "command": "wait",
                    "description": "Wait",
                    "behavior_key": "wait",
                },
                {
                    "directive_id": 3,
                    "command": "three",
                    "description": "Three",
                    "behavior_key": "three",
                },
                {
                    "directive_id": 1,
                    "command": "one",
                    "description": "One",
                    "behavior_key": "one",
                },
            )
        ),
    )
    second = first.model_copy(deep=True)

    assert tuple(item.directive_id for item in first.directives) == (0, 1, 2, 3)
    assert first.digest() == second.digest()


def test_vocabulary_rejects_duplicate_behavior_keys_and_invalid_wait_id() -> None:
    definitions: list[dict[str, str | int]] = [
        {
            "directive_id": 0,
            "command": "wait",
            "description": "Wait",
            "behavior_key": "wait",
        },
        {
            "directive_id": 1,
            "command": "one",
            "description": "One",
            "behavior_key": "one",
        },
        {
            "directive_id": 2,
            "command": "two",
            "description": "Two",
            "behavior_key": "one",
        },
        {
            "directive_id": 3,
            "command": "three",
            "description": "Three",
            "behavior_key": "three",
        },
    ]
    with pytest.raises(ValidationError, match="behavior keys"):
        DirectiveVocabulary(
            schema_version=1,
            vocabulary_id="v1",
            behavior_version="b1",
            wait_directive_id=0,
            directives=tuple(
                DirectiveDefinition.model_validate(item) for item in definitions
            ),
        )

    definitions[2]["behavior_key"] = "two"
    with pytest.raises(ValidationError, match="wait_directive_id"):
        DirectiveVocabulary(
            schema_version=1,
            vocabulary_id="v1",
            behavior_version="b1",
            wait_directive_id=9,
            directives=tuple(
                DirectiveDefinition.model_validate(item) for item in definitions
            ),
        )


def test_configs_reject_absolute_paths_overlapping_seeds_and_bad_urls() -> None:
    with pytest.raises(ValidationError, match="repository-relative"):
        DirectiveProbeConfig(
            schema_version=1,
            environment_id="tiny",
            vocabulary_path=Path.cwd() / "directives.yaml",
        )

    with pytest.raises(ValidationError, match="overlap"):
        DirectivePpoConfig(
            schema_version=1,
            environment_id="tiny",
            vocabulary_path=Path("configs/demo/directives-v1.yaml"),
            train_seeds=(1, 2),
            validation_seeds=(2, 3),
        )

    with pytest.raises(ValidationError, match="base_url"):
        LlamaCppBackendConfig(
            base_url=cast(Any, "not-a-url"),
            model_id="gemma",
            model_file_sha256="0" * 64,
            context_size=1,
            gpu_layers=0,
        )


def _observation() -> WarehouseObservation:
    return WarehouseObservation(
        step=0,
        grid_rows=5,
        grid_columns=6,
        agents=(
            AgentObservation(
                agent_id="agv-0",
                simulator_id=1,
                role=AgentRole.AGV,
                position=Position(row=0, column=0),
                target=None,
                target_destination_id=None,
                busy=False,
            ),
        ),
        shelves=(
            ShelfObservation(
                destination_id=3,
                position=Position(row=1, column=1),
                occupied=True,
                requested=True,
            ),
        ),
        goal_destination_ids=(1, 2),
    )


class _ContextWrapper:
    observation = _observation()
    action_mask = ActionMask(
        agent_ids=("agv-0",), values=np.asarray([[True, False, True]])
    )


def test_context_capture_hashes_observation_and_mask_bytes() -> None:
    first = DirectiveContext.capture(cast(Any, _ContextWrapper()))
    second = DirectiveContext.capture(cast(Any, _ContextWrapper()))

    assert len(first.state_id) == 64
    assert first.state_id == second.state_id
    assert first.observation == _observation()


def test_probe_report_requires_sha256_digests() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        DirectiveProbeReport(
            schema_version=1,
            environment_id="tiny",
            vocabulary_sha256="bad",
            behavior_sha256="0" * 64,
            probe_seeds=(1,),
            states_checked=1,
            uncovered_states=0,
            executor_failures=0,
            effect_failures=0,
            passed=False,
        )
