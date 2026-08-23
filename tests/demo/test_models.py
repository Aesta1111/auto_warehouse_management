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
    WarehouseContext,
    ProbeConfig,
    ProbeReport,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.evaluation import DemoConfig
from warehouse_ai.demo.contracts.finetuning import (
    FineTuneEvaluationConfig,
    GemmaFineTuneConfig,
)
from warehouse_ai.demo.contracts.inference import LlamaCppBackendConfig
from warehouse_ai.demo.contracts.ppo import PpoConfig
from warehouse_ai.domain import ActionMask, AgentObservation, AgentRole, Position
from warehouse_ai.domain import ShelfObservation, WarehouseObservation


ROOT = Path(__file__).parents[2]


def test_committed_versioned_configs_load() -> None:
    catalog = DirectiveCatalog.load(ROOT / "configs/demo/directives-v1.yaml")
    probe = load_yaml_config(ROOT / "configs/demo/directive-probe-v1.yaml", ProbeConfig)
    ppo = load_yaml_config(ROOT / "configs/demo/directive-ppo-v1.yaml", PpoConfig)
    ppo_v2 = load_yaml_config(ROOT / "configs/demo/directive-ppo-v2.yaml", PpoConfig)
    probe_v2 = load_yaml_config(
        ROOT / "configs/demo/directive-probe-v2.yaml", ProbeConfig
    )
    ppo_v3 = load_yaml_config(ROOT / "configs/demo/directive-ppo-v3.yaml", PpoConfig)
    ppo_v4 = load_yaml_config(ROOT / "configs/demo/directive-ppo-v4.yaml", PpoConfig)
    demo_v2 = load_yaml_config(ROOT / "configs/demo/directive-demo-v2.yaml", DemoConfig)
    demo_v3 = load_yaml_config(ROOT / "configs/demo/directive-demo-v3.yaml", DemoConfig)
    fine_tune = load_yaml_config(
        ROOT / "configs/demo/directive-ppo-gemma-finetune-v1.yaml",
        GemmaFineTuneConfig,
    )
    fine_tune_evaluations = tuple(
        load_yaml_config(
            ROOT / f"configs/demo/directive-ppo-gemma-eval-{partition}-v1.yaml",
            FineTuneEvaluationConfig,
        )
        for partition in ("selection", "confirmation", "final")
    )

    assert catalog.size == 4
    assert catalog.by_id(catalog.wait_directive_id).command == "wait"
    assert [item.behavior_key for item in catalog.directives] == [
        "wait",
        "service_requested_shelf",
        "deliver_carried_shelf",
        "reposition_idle_agent",
    ]
    assert all(
        item.agents is not None
        and item.destinations is not None
        and item.effect is not None
        for item in catalog.directives
    )
    assert set(catalog.model_dump().keys()) == {
        "schema_version",
        "vocabulary_id",
        "behavior_version",
        "wait_directive_id",
        "directives",
    }
    assert set(catalog.model_dump()["directives"][0].keys()) == {
        "directive_id",
        "command",
        "description",
        "behavior_key",
    }
    assert probe.vocabulary_path.as_posix() == "configs/demo/directives-v1.yaml"
    assert ppo.train_seeds[-1] == 8115
    assert ppo_v2.backend == "lightning-fabric"
    assert ppo_v2.value_coefficient == 0.5
    assert probe_v2.schema_version == 2
    assert ppo_v3.schema_version == 3
    assert ppo_v4.schema_version == 4
    assert ppo_v4.hidden_sizes == (64,)
    assert demo_v2.inference.backend == "llama_cpp"
    assert demo_v2.preflight_p95_limit_seconds == 3.92
    assert demo_v2.prompt_spec_path is None
    assert demo_v3.schema_version == 3
    assert demo_v3.prompt_spec_path == Path(
        "configs/demo/gemma-prompt-few-shot-v1.yaml"
    )
    assert fine_tune.total_timesteps == 1_024
    assert fine_tune.n_steps == 256
    assert fine_tune.fallback_penalty == pytest.approx(0.001)
    assert fine_tune.selection_seeds == (8200, 8201)
    assert fine_tune.confirmation_seeds == (8202, 8203)
    assert fine_tune.final_evaluation_seeds == (9100,)
    assert tuple(item.partition for item in fine_tune_evaluations) == (
        "selection",
        "confirmation",
        "final",
    )

    confirmation = fine_tune_evaluations[1].model_dump()
    confirmation["prerequisite_summary_path"] = None
    with pytest.raises(ValueError, match="require a prerequisite summary"):
        FineTuneEvaluationConfig.model_validate(confirmation)


def test_gemma_fine_tune_config_rejects_overlapping_partitions() -> None:
    config = load_yaml_config(
        ROOT / "configs/demo/directive-ppo-gemma-finetune-v1.yaml",
        GemmaFineTuneConfig,
    )
    values = config.model_dump()
    values["confirmation_seeds"] = (8201, 8203)

    with pytest.raises(ValueError, match="selection and confirmation seeds overlap"):
        GemmaFineTuneConfig.model_validate(values)


def test_contract_import_does_not_load_ml_or_transport_frameworks() -> None:
    code = """
import sys
from warehouse_ai.demo.contracts.ppo import PpoConfig
from warehouse_ai.demo.contracts.evaluation import DemoConfig
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


def test_catalog_digest_is_stable_and_ids_are_canonical() -> None:
    first = DirectiveCatalog.model_validate(
        {
            "schema_version": 1,
            "vocabulary_id": "v1",
            "behavior_version": "b1",
            "wait_directive_id": 0,
            "directives": (
                {
                    "directive_id": 2,
                    "command": "two",
                    "description": "Two",
                    "behavior_key": "deliver_carried_shelf",
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
                    "behavior_key": "reposition_idle_agent",
                },
                {
                    "directive_id": 1,
                    "command": "one",
                    "description": "One",
                    "behavior_key": "service_requested_shelf",
                },
            ),
        }
    )
    second = first.model_copy(deep=True)

    assert tuple(item.directive_id for item in first.directives) == (0, 1, 2, 3)
    assert all(
        item.agents is not None
        and item.destinations is not None
        and item.effect is not None
        for item in first.directives
    )
    assert first.digest() == second.digest()
    assert "agents" not in first.model_dump_json()
    assert "destinations" not in first.model_dump_json()
    assert "effect" not in first.model_dump_json()


def test_catalog_rejects_duplicate_behavior_keys_and_invalid_wait_id() -> None:
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
            "behavior_key": "service_requested_shelf",
        },
        {
            "directive_id": 2,
            "command": "two",
            "description": "Two",
            "behavior_key": "service_requested_shelf",
        },
        {
            "directive_id": 3,
            "command": "three",
            "description": "Three",
            "behavior_key": "reposition_idle_agent",
        },
    ]
    with pytest.raises(ValidationError, match="behavior keys"):
        DirectiveCatalog.model_validate(
            {
                "schema_version": 1,
                "vocabulary_id": "v1",
                "behavior_version": "b1",
                "wait_directive_id": 0,
                "directives": definitions,
            }
        )

    definitions[2]["behavior_key"] = "deliver_carried_shelf"
    with pytest.raises(ValidationError, match="wait_directive_id"):
        DirectiveCatalog.model_validate(
            {
                "schema_version": 1,
                "vocabulary_id": "v1",
                "behavior_version": "b1",
                "wait_directive_id": 9,
                "directives": definitions,
            }
        )


def test_configs_reject_absolute_paths_overlapping_seeds_and_bad_urls() -> None:
    with pytest.raises(ValidationError, match="repository-relative"):
        ProbeConfig(
            schema_version=1,
            environment_id="tiny",
            vocabulary_path=Path.cwd() / "directives.yaml",
        )

    with pytest.raises(ValidationError, match="overlap"):
        PpoConfig(
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


def test_ppo_hidden_sizes_are_required_only_for_schema_v4() -> None:
    common = {
        "environment_id": "tiny",
        "vocabulary_path": Path("configs/demo/directives-v1.yaml"),
        "train_seeds": (1,),
        "validation_seeds": (2,),
        "value_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "backend": "lightning-fabric",
        "model_implementation_version": "directive-ppo-mlp-v1",
        "buffer_schema_version": "buffer-v1",
        "gae_version": "gae-v1",
        "loss_version": "loss-v1",
    }
    with pytest.raises(ValidationError, match="require non-empty hidden_sizes"):
        PpoConfig(schema_version=4, **common)
    with pytest.raises(ValidationError, match="legacy PPO configs"):
        PpoConfig(schema_version=3, hidden_sizes=(64,), **common)
    with pytest.raises(ValidationError, match="greater than 0"):
        PpoConfig(schema_version=4, hidden_sizes=(64, 0), **common)


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
    first = WarehouseContext.capture(cast(Any, _ContextWrapper()))
    second = WarehouseContext.capture(cast(Any, _ContextWrapper()))

    assert len(first.state_id) == 64
    assert first.state_id == second.state_id
    assert first.observation == _observation()


def test_probe_report_requires_sha256_digests() -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        ProbeReport(
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
