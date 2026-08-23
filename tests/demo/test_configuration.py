from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.configuration import (
    LlmActionerConfig,
    PpoEvaluationRunConfig,
    PpoFineTuningRunConfig,
    PpoModelConfig,
    PpoTrainingRunConfig,
    load_ppo_evaluation_run,
    load_ppo_fine_tuning_run,
    load_ppo_training_run,
)


ROOT = Path(__file__).parents[2]


def test_five_composed_configs_load_with_descriptive_names() -> None:
    model = load_yaml_config(ROOT / "configs/demo/ppo-model.yaml", PpoModelConfig)
    actioner = load_yaml_config(
        ROOT / "configs/demo/llm-actioner.yaml", LlmActionerConfig
    )
    training = load_ppo_training_run(Path("configs/demo/ppo-training-run.yaml"), ROOT)
    fine_tuning = load_ppo_fine_tuning_run(
        Path("configs/demo/ppo-gemma-fine-tuning-run.yaml"), ROOT
    )
    evaluation = load_ppo_evaluation_run(
        Path("configs/demo/ppo-gemma-evaluation-run.yaml"), ROOT
    )

    assert model.hidden_sizes == (64,)
    assert actioner.actioner_id == "gemma-e4b-few-shot-actioner"
    assert training.model == model
    assert training.runtime.hidden_sizes == model.hidden_sizes
    assert fine_tuning.actioner == actioner
    assert fine_tuning.runtime.inference == actioner.inference
    assert evaluation.demo_runtime().prompt_spec_path == actioner.prompt_spec_path
    assert evaluation.comparison_runtime("selection").prerequisite_summary_path is None
    assert evaluation.comparison_runtime("confirmation").prerequisite_summary_path == (
        Path("results/demo/ppo-gemma-fine-tuning/selection-gate.json")
    )


def test_top_level_config_contracts_are_strict() -> None:
    model = load_yaml_config(ROOT / "configs/demo/ppo-model.yaml", PpoModelConfig)
    with pytest.raises(ValidationError, match="Extra inputs"):
        PpoModelConfig.model_validate({**model.model_dump(), "hidden_size": 64})
    with pytest.raises(ValidationError, match="at least one layer"):
        PpoModelConfig.model_validate({**model.model_dump(), "hidden_sizes": []})

    training = load_yaml_config(
        ROOT / "configs/demo/ppo-training-run.yaml", PpoTrainingRunConfig
    )
    with pytest.raises(ValidationError, match="overlap"):
        PpoTrainingRunConfig.model_validate(
            {**training.model_dump(), "validation_seeds": (8100,)}
        )

    fine_tuning = load_yaml_config(
        ROOT / "configs/demo/ppo-gemma-fine-tuning-run.yaml",
        PpoFineTuningRunConfig,
    )
    with pytest.raises(ValidationError, match="overlap"):
        PpoFineTuningRunConfig.model_validate(
            {**fine_tuning.model_dump(), "confirmation_seeds": (8201,)}
        )

    evaluation = load_yaml_config(
        ROOT / "configs/demo/ppo-gemma-evaluation-run.yaml",
        PpoEvaluationRunConfig,
    )
    with pytest.raises(ValidationError, match="confirmation and final"):
        PpoEvaluationRunConfig.model_validate(
            {**evaluation.model_dump(), "prerequisite_summary_paths": {}}
        )
