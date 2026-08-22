"""Exact-model validity, sensitivity, and latency preflight service."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from warehouse_ai.demo.artifacts import write_json_atomic
from warehouse_ai.demo.contracts.directives import DirectiveVocabulary
from warehouse_ai.demo.contracts.evaluation import DirectiveDemoConfig
from warehouse_ai.demo.contracts.inference import GemmaPreflightReport, ModelHealth
from warehouse_ai.demo.contracts.ppo import DirectivePpoManifest
from warehouse_ai.demo.controllers import ActionValidator
from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.inference.actioner import GemmaActioner
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend
from warehouse_ai.demo.inference.preflight_cases import (
    find_state,
    forced_request,
    policy_request,
    prompt_schema_digest,
    timed_action,
)
from warehouse_ai.demo.inference.prompting import GemmaPromptBuilder
from warehouse_ai.demo.inference.provenance import hardware_provenance, percentile
from warehouse_ai.demo.ppo.policy import DirectivePpoPolicy
from warehouse_ai.env import TarwareEnvWrapper


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_policy(
    root: Path,
    config: DirectiveDemoConfig,
    vocabulary: DirectiveVocabulary,
) -> DirectivePpoPolicy:
    manifest_path = _resolve(root, config.ppo_manifest_path)
    manifest = DirectivePpoManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    return DirectivePpoPolicy.load(
        _resolve(root, config.ppo_checkpoint_path), vocabulary, manifest
    )


def _measured_request(context, directives, policy, registry, vocabulary, covered):
    request = policy_request(context, policy, registry)
    selected = request.prediction.directive.directive_id
    non_wait = tuple(
        directive_id
        for directive_id in directives
        if directive_id != vocabulary.wait_directive_id
    )
    if selected == vocabulary.wait_directive_id and non_wait:
        selected = non_wait[0]
    elif len(covered) < 2:
        alternate = next(
            (
                directive_id
                for directive_id in directives
                if directive_id != selected
                and vocabulary.by_id(directive_id).behavior_key not in covered
            ),
            None,
        )
        if alternate is not None:
            selected = alternate
    if selected != request.prediction.directive.directive_id:
        request = forced_request(context, policy, vocabulary, selected)
    return request, selected


def run_preflight(
    config: DirectiveDemoConfig, *, repository_root: Path | None = None
) -> GemmaPreflightReport:
    """Run the configured server gates and write a schema-v2 report."""
    root = (repository_root or _repository_root()).resolve()
    vocabulary = DirectiveVocabulary.load(_resolve(root, config.vocabulary_path))
    registry = build_directive_registry(vocabulary)
    policy = _load_policy(root, config, vocabulary)
    backend = LlamaCppBackend(config.inference)
    actioner = GemmaActioner(backend, GemmaPromptBuilder())
    validator = ActionValidator(registry)
    wrapper = TarwareEnvWrapper(config.environment_id)
    latencies: list[float] = []
    classes: list[str] = []
    sensitivity_results: list[dict[str, Any]] = []
    failures: list[str] = []
    valid_responses = invalid_responses = sensitivity_states = 0
    sensitivity_pairs = sensitivity_valid_pairs = 0
    warmup_completed = False
    prompt_digest = "0" * 64
    health: ModelHealth | None = None
    try:
        health = backend.health()
        warmup = find_state(
            wrapper,
            registry,
            vocabulary,
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            required=1,
        )
        if warmup is None:
            failures.append("no applicable non-wait directive for warm-up")
        else:
            context, directives = warmup
            request = forced_request(context, policy, vocabulary, directives[0])
            _, warmup_completed, error = timed_action(
                actioner, validator, request, wrapper
            )
            if error:
                failures.append(f"warm-up failed: {error}")
            prompt_digest = prompt_schema_digest(context, request)

        for offset in range(20):
            if len(latencies) == 5:
                break
            state = find_state(
                wrapper,
                registry,
                vocabulary,
                seed=config.episode_seed + offset,
                max_steps=config.episode_steps,
                required=2,
                include_wait=True,
            )
            if state is None:
                continue
            context, directives = state
            request, selected = _measured_request(
                context, directives, policy, registry, vocabulary, set(classes)
            )
            classes.append(vocabulary.by_id(selected).behavior_key)
            elapsed, valid, error = timed_action(actioner, validator, request, wrapper)
            latencies.append(elapsed)
            valid_responses += int(valid)
            invalid_responses += int(not valid)
            if error:
                failures.append(f"measured request failed: {error}")

        for offset in range(20, 40):
            if sensitivity_states == 3:
                break
            state = find_state(
                wrapper,
                registry,
                vocabulary,
                seed=config.episode_seed + offset,
                max_steps=config.episode_steps,
                required=2,
                include_wait=True,
            )
            if state is None:
                continue
            context, directives = state
            sensitivity_states += 1
            validities: list[bool] = []
            for directive_id in directives[:2]:
                sensitivity_pairs += 1
                request = forced_request(context, policy, vocabulary, directive_id)
                _, valid, error = timed_action(actioner, validator, request, wrapper)
                validities.append(valid)
                sensitivity_valid_pairs += int(valid)
                if error:
                    failures.append(f"sensitivity request failed: {error}")
            sensitivity_results.append(
                {
                    "state_id": context.state_id,
                    "directive_ids": list(directives[:2]),
                    "valid": validities,
                }
            )
    except Exception as exc:
        failures.append(f"preflight error: {type(exc).__name__}: {exc}")
    finally:
        wrapper.close()
        backend.close()

    if health is None:
        health = ModelHealth(
            base_url=config.inference.base_url,
            model_id=config.inference.model_id,
            advertised_model_ids=(config.inference.model_id,),
            status="unverified",
            healthy=False,
            checked_at_utc=datetime.now(timezone.utc),
        )
        failures.append("health and model identity were not verified")
    validity_passed = (
        len(latencies) == 5 and valid_responses == 5 and len(set(classes)) >= 2
    )
    sensitivity_passed = (
        sensitivity_states == 3
        and sensitivity_pairs == 6
        and sensitivity_valid_pairs == 6
    )
    latency_p95 = percentile(latencies, 0.95)
    latency_passed = (
        bool(latencies) and latency_p95 <= config.preflight_p95_limit_seconds
    )
    if not validity_passed:
        failures.append("5/5 validity and two-class coverage gate failed")
    if not sensitivity_passed:
        failures.append("6/6 directive sensitivity gate failed")
    if not latency_passed:
        failures.append("p95 latency gate failed")
    report = GemmaPreflightReport(
        schema_version=2,
        selected_model=config.selected_model,
        backend=config.inference.backend,
        model_id=config.inference.model_id,
        model_file_sha256=config.inference.model_file_sha256,
        base_url=config.inference.base_url,
        context_size=config.inference.context_size,
        gpu_layers=config.inference.gpu_layers,
        prompt_schema_sha256=prompt_digest,
        health=health,
        warmup_completed=warmup_completed,
        measured_requests=len(latencies),
        valid_responses=valid_responses,
        invalid_responses=invalid_responses,
        directive_classes=tuple(dict.fromkeys(classes)),
        sensitivity_states=sensitivity_states,
        sensitivity_pairs=sensitivity_pairs,
        sensitivity_valid_pairs=sensitivity_valid_pairs,
        sensitivity_results=tuple(sensitivity_results),
        latency_seconds=tuple(latencies),
        latency_median_seconds=statistics.median(latencies) if latencies else 0.0,
        latency_p95_seconds=latency_p95,
        p95_limit_seconds=config.preflight_p95_limit_seconds,
        validity_passed=validity_passed,
        sensitivity_passed=sensitivity_passed,
        latency_passed=latency_passed,
        passed=bool(
            health.healthy
            and warmup_completed
            and validity_passed
            and sensitivity_passed
            and latency_passed
        ),
        hardware=hardware_provenance(),
        server_command=config.inference.server_command,
        quantization=config.inference.quantization,
        checked_at_utc=datetime.now(timezone.utc),
        failure_reason="; ".join(dict.fromkeys(failures)) or None,
    )
    write_json_atomic(_resolve(root, config.preflight_report_path), report)
    return report
