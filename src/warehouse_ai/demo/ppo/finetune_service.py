"""Fail-closed composition for one live Gemma PPO fine-tuning run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from warehouse_ai.demo.artifacts import sha256_file, write_json_atomic
from warehouse_ai.demo.contracts.directives import DirectiveCatalog
from warehouse_ai.demo.contracts.finetuning import (
    GemmaFineTuneConfig,
    GemmaFineTuneManifest,
)
from warehouse_ai.demo.contracts.inference import GemmaPreflightReport
from warehouse_ai.demo.contracts.ppo import PpoManifest
from warehouse_ai.demo.directives import Registry, build_registry
from warehouse_ai.demo.inference.actioner import GemmaActioner
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.demo.ppo.finetuning import (
    FineTuneArtifactIdentity,
    GemmaFineTuneLoop,
    warm_start_model,
)
from warehouse_ai.demo.ppo.gemma_environment import GemmaGymEnv
from warehouse_ai.demo.ppo.policy import PpoPolicy


@dataclass(frozen=True)
class VerifiedFineTuneArtifacts:
    catalog: DirectiveCatalog
    registry: Registry
    parent_manifest: PpoManifest
    parent_policy: PpoPolicy
    prompt_builder: GemmaPromptBuilder
    identity: FineTuneArtifactIdentity


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_json(path: Path, model_type):  # type: ignore[no-untyped-def]
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load {model_type.__name__} {path}") from exc


def verify_fine_tune_artifacts(
    config: GemmaFineTuneConfig,
    repository_root: Path,
) -> VerifiedFineTuneArtifacts:
    """Verify the parent PPO and exact frozen Gemma contract before training."""
    catalog = DirectiveCatalog.load(_resolve(repository_root, config.vocabulary_path))
    registry = build_registry(catalog)
    parent_manifest_path = _resolve(repository_root, config.parent_manifest_path)
    parent_checkpoint_path = _resolve(repository_root, config.parent_checkpoint_path)
    preflight_path = _resolve(repository_root, config.preflight_report_path)
    parent_manifest = _load_json(parent_manifest_path, PpoManifest)
    preflight = _load_json(preflight_path, GemmaPreflightReport)
    prompt_builder = GemmaPromptBuilder.load(
        _resolve(repository_root, config.prompt_spec_path), catalog
    )
    prompt_digest = prompt_builder.contract_digest
    if prompt_digest is None:
        raise ValueError("Gemma fine-tuning requires a versioned prompt contract")
    if parent_manifest.schema_version != 3:
        raise ValueError("Gemma fine-tuning requires a schema-v3 PPO parent")
    if parent_manifest.environment_id != config.environment_id:
        raise ValueError("parent PPO environment does not match fine-tuning config")
    expected = (catalog.digest(), registry.behavior_digest(), registry.engine_version)
    actual = (
        parent_manifest.vocabulary_sha256,
        parent_manifest.behavior_sha256,
        parent_manifest.behavior_engine_version,
    )
    if actual != expected:
        raise ValueError("parent PPO does not match current directive semantics")
    if not set(config.train_seeds).issubset(parent_manifest.train_seeds):
        raise ValueError("fine-tuning seeds must come from parent training seeds")
    declared_validation = set(config.selection_seeds) | set(config.confirmation_seeds)
    if declared_validation != set(parent_manifest.validation_seeds):
        raise ValueError(
            "selection and confirmation must partition parent validation seeds"
        )
    if (
        preflight.schema_version != 3
        or not preflight.passed
        or preflight.prompt_contract_sha256 != prompt_digest
    ):
        raise ValueError("fine-tuning requires a passing exact schema-v3 preflight")
    inference = config.inference
    if (
        preflight.model_id != inference.model_id
        or preflight.model_file_sha256 != inference.model_file_sha256
        or str(preflight.base_url).rstrip("/") != str(inference.base_url).rstrip("/")
        or preflight.context_size != inference.context_size
        or preflight.gpu_layers != inference.gpu_layers
    ):
        raise ValueError("fine-tuning inference runtime does not match preflight")
    parent_policy = PpoPolicy.load(parent_checkpoint_path, catalog, parent_manifest)
    return VerifiedFineTuneArtifacts(
        catalog=catalog,
        registry=registry,
        parent_manifest=parent_manifest,
        parent_policy=parent_policy,
        prompt_builder=prompt_builder,
        identity=FineTuneArtifactIdentity(
            parent_manifest_sha256=sha256_file(parent_manifest_path),
            prompt_contract_sha256=prompt_digest,
            preflight_report_sha256=sha256_file(preflight_path),
        ),
    )


def run_gemma_fine_tune(
    config: GemmaFineTuneConfig,
    *,
    repository_root: Path,
) -> GemmaFineTuneManifest:
    """Run one intentionally explicit, expensive frozen-Gemma fine-tune."""
    root = repository_root.resolve()
    verified = verify_fine_tune_artifacts(config, root)
    output = _resolve(root, config.output_directory)
    final_checkpoint = output / "directive-ppo-gemma-finetuned-v1.pt"
    manifest_path = output / "directive-ppo-gemma-finetune-manifest-v1.json"
    step_log = output / "fine-tune-steps-v1.jsonl"
    if any(path.exists() for path in (final_checkpoint, manifest_path, step_log)):
        raise FileExistsError(
            "fine-tuning output already exists; choose a new directory"
        )
    backend = LlamaCppBackend(config.inference)
    environment: GemmaGymEnv | None = None
    try:
        health = backend.health()
        if not health.healthy or config.inference.model_id not in (
            health.advertised_model_ids
        ):
            raise RuntimeError("configured Gemma server is not healthy")
        actioner = GemmaActioner(backend, verified.prompt_builder)
        environment = GemmaGymEnv(
            config.environment_id,
            verified.catalog,
            verified.registry,
            config.train_seeds,
            actioner,
            episode_steps=config.episode_steps,
            fallback_penalty=config.fallback_penalty,
            step_log_path=step_log,
        )
        model = warm_start_model(config, verified.catalog, verified.parent_policy)
        reference = warm_start_model(config, verified.catalog, verified.parent_policy)
        manifest = GemmaFineTuneLoop().fit(
            model,
            reference,
            config,
            environment,
            output,
            verified.identity,
            verified.parent_manifest,
        )
        write_json_atomic(manifest_path, manifest)
        return manifest
    finally:
        if environment is not None:
            environment.close()
        backend.close()
