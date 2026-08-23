"""Fail-closed verification of versioned demo evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from warehouse_ai.demo.artifacts import sha256_file
from warehouse_ai.demo.contracts.directives import (
    ProbeReport,
    DirectiveCatalog,
)
from warehouse_ai.demo.contracts.evaluation import DemoConfig
from warehouse_ai.demo.contracts.inference import GemmaPreflightReport
from warehouse_ai.demo.contracts.ppo import PpoManifest
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.demo.ppo.policy import PpoPolicy


@dataclass(frozen=True)
class VerifiedDemoArtifacts:
    catalog: DirectiveCatalog
    manifest: PpoManifest
    policy: PpoPolicy
    prompt_builder: GemmaPromptBuilder
    probe_digest: str
    preflight_digest: str


def resolve_path(repository_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _load_json_model(path: Path, model_type: type[Any]) -> Any:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"could not load {model_type.__name__} artifact {path}"
        ) from exc


def verify_prompt_contract(
    config: DemoConfig,
    preflight: GemmaPreflightReport,
    prompt_builder: GemmaPromptBuilder,
) -> None:
    """Require preflight evidence for the exact configured prompt contract."""
    if not preflight.passed or preflight.schema_version != config.schema_version:
        raise ValueError("Gemma preflight did not pass for this config version")
    if preflight.prompt_contract_sha256 != prompt_builder.contract_digest:
        raise ValueError("Gemma preflight prompt contract does not match config")


def verify_demo_artifacts(
    config: DemoConfig, repository_root: Path
) -> VerifiedDemoArtifacts:
    vocabulary_path = resolve_path(repository_root, config.vocabulary_path)
    probe_path = resolve_path(repository_root, config.directive_probe_report_path)
    checkpoint_path = resolve_path(repository_root, config.ppo_checkpoint_path)
    manifest_path = resolve_path(repository_root, config.ppo_manifest_path)
    preflight_path = resolve_path(repository_root, config.preflight_report_path)
    catalog = DirectiveCatalog.load(vocabulary_path)
    registry = build_registry(catalog)
    manifest = _load_json_model(manifest_path, PpoManifest)
    probe = _load_json_model(probe_path, ProbeReport)
    preflight = _load_json_model(preflight_path, GemmaPreflightReport)
    prompt_builder = (
        GemmaPromptBuilder()
        if config.prompt_spec_path is None
        else GemmaPromptBuilder.load(
            resolve_path(repository_root, config.prompt_spec_path), catalog
        )
    )
    expected_ppo_schema = 4 if config.schema_version == 4 else 3
    if manifest.schema_version != expected_ppo_schema or probe.schema_version != 2:
        raise ValueError(
            "demo config requires its versioned PPO schema and probe schema v2"
        )
    if manifest.environment_id != config.environment_id:
        raise ValueError("PPO environment does not match demo config")
    behavior_digest = registry.behavior_digest()
    expected = (catalog.digest(), behavior_digest, registry.engine_version)
    actual_manifest = (
        manifest.vocabulary_sha256,
        manifest.behavior_sha256,
        manifest.behavior_engine_version,
    )
    actual_probe = (
        probe.vocabulary_sha256,
        probe.behavior_sha256,
        probe.behavior_engine_version,
    )
    if actual_manifest != expected or actual_probe != expected:
        raise ValueError("directive artifact identities do not match current semantics")
    if not probe.passed or probe.environment_id != config.environment_id:
        raise ValueError("directive probe is failed or from another environment")
    verify_prompt_contract(config, preflight, prompt_builder)
    inference = config.inference
    if preflight.selected_model != config.selected_model:
        raise ValueError("preflight model selection does not match demo config")
    if (
        preflight.model_id != inference.model_id
        or preflight.model_file_sha256 != inference.model_file_sha256
    ):
        raise ValueError("preflight model identity does not match demo config")
    if str(preflight.base_url).rstrip("/") != str(inference.base_url).rstrip("/"):
        raise ValueError("preflight endpoint does not match demo config")
    if (
        preflight.context_size != inference.context_size
        or preflight.gpu_layers != inference.gpu_layers
    ):
        raise ValueError("preflight runtime shape does not match demo config")
    if (
        not preflight.health.healthy
        or preflight.model_id not in preflight.health.advertised_model_ids
    ):
        raise ValueError("preflight health record does not advertise a healthy model")
    if preflight.latency_p95_seconds > config.preflight_p95_limit_seconds:
        raise ValueError("preflight p95 latency exceeds the configured limit")
    excluded_seeds = (
        set(probe.probe_seeds)
        | set(manifest.train_seeds)
        | set(manifest.validation_seeds)
    )
    if config.episode_seed in excluded_seeds:
        raise ValueError("demo seed overlaps a prototype seed partition")
    policy = PpoPolicy.load(checkpoint_path, catalog, manifest)
    return VerifiedDemoArtifacts(
        catalog=catalog,
        manifest=manifest,
        policy=policy,
        prompt_builder=prompt_builder,
        probe_digest=sha256_file(probe_path),
        preflight_digest=sha256_file(preflight_path),
    )
