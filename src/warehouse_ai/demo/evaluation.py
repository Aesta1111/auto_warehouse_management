"""Episode evaluation and matched execution for the isolated directive demo."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from warehouse_ai.demo.actioner import LlamaCppActioner
from warehouse_ai.demo.artifacts import (
    append_jsonl,
    capture_git_state,
    create_run_directory,
    sha256_file,
    write_json_atomic,
)
from warehouse_ai.demo.controllers import (
    DemoController,
    DirectiveExecutorController,
    DirectiveGemmaController,
)
from warehouse_ai.demo.directives import build_directive_registry
from warehouse_ai.demo.models import (
    DemoRunSummary,
    DemoStepRecord,
    DirectiveDemoConfig,
    DirectivePpoManifest,
    DirectiveProbeReport,
    DirectiveVocabulary,
    EpisodeResult,
    GemmaPreflightReport,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
from warehouse_ai.demo.prompting import GemmaPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


_INVALID_ACTION_ERRORS = frozenset(
    {"ActionerSchemaError", "ActionerDecodeError", "StaleActionPlanError"}
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(repository_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def _json_safe(value: Any) -> Any:
    """Convert TA-RWARE's small info payload into strict JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("simulator info contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"simulator info contains unsupported value {type(value)!r}")


def _metric_count(info: Mapping[str, Any], key: str) -> int:
    """Read one non-negative integer metric from the pinned info contract."""
    value = info.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, np.integer):
        value = int(value)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"TA-RWARE info metric {key!r} must be non-negative integer")
    return value


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _fallback_error_name(reason: str | None) -> str | None:
    if not reason:
        return None
    first_clause = reason.split(";", 1)[0]
    return first_clause.split(":", 1)[0].strip() or None


def _llm_attempted(controller: DemoController, command: str) -> bool:
    """Count one attempted call for every non-wait Gemma decision."""
    return controller.controller_name == "ppo_gemma" and command.casefold() != "wait"


def run_episode(
    wrapper: TarwareEnvWrapper,
    controller: DemoController,
    *,
    seed: int,
    max_steps: int,
    render: bool,
    decision_log_path: Path,
) -> EpisodeResult:
    """Run one controller episode and append one flushed record per decision."""
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if controller.controller_name not in {"ppo_executor", "ppo_gemma"}:
        raise ValueError(f"unsupported demo controller {controller.controller_name!r}")

    wrapper.reset(seed=int(seed))
    started = time.perf_counter()
    steps = 0
    total_return = 0.0
    completed_tasks = 0
    clashes = 0
    stuck_events = 0
    directive_counts: dict[str, int] = {}
    llm_calls = 0
    invalid_actions = 0
    effect_violations = 0
    fallbacks = 0
    latencies: list[float] = []

    while steps < max_steps:
        decision = controller.choose_action(wrapper)
        command = decision.prediction.directive.command
        directive_counts[command] = directive_counts.get(command, 0) + 1

        attempted_llm = _llm_attempted(controller, command)
        if attempted_llm:
            llm_calls += 1
            error_name = _fallback_error_name(decision.fallback_reason)
            if error_name in _INVALID_ACTION_ERRORS:
                invalid_actions += 1
            if error_name == "ActionerEffectError":
                effect_violations += 1
        if decision.fallback_reason is not None or decision.source == "automatic_no_op":
            fallbacks += 1
        if decision.llm_latency_ms is not None:
            latencies.append(float(decision.llm_latency_ms))

        _, rewards, terminated, truncated, simulator_info = wrapper.step(
            decision.action
        )
        reward_values = tuple(float(value) for value in rewards)
        terminated_values = tuple(bool(value) for value in terminated)
        truncated_values = tuple(bool(value) for value in truncated)
        if not terminated_values or not truncated_values:
            raise ValueError("TA-RWARE must return one termination flag per agent")
        if not isinstance(simulator_info, Mapping):
            raise TypeError("TA-RWARE simulator info must be a mapping")

        info = _json_safe(simulator_info)
        if not isinstance(info, dict):
            raise TypeError("normalized simulator info must be a JSON object")
        record = DemoStepRecord(
            decision=decision,
            rewards=reward_values,
            terminated=terminated_values,
            truncated=truncated_values,
            simulator_info=info,
        )
        append_jsonl(decision_log_path, record)
        if render:
            wrapper.render()

        steps += 1
        total_return += sum(reward_values)
        completed_tasks += _metric_count(simulator_info, "shelf_deliveries")
        clashes += _metric_count(simulator_info, "clashes")
        stuck_events += _metric_count(simulator_info, "stucks")

        if all(terminated_values) or all(truncated_values):
            break

    median_latency = _percentile(latencies, 0.5)
    p95_latency = _percentile(latencies, 0.95)
    return EpisodeResult(
        controller=controller.controller_name,
        seed=int(seed),
        steps=steps,
        completed_tasks=completed_tasks,
        total_return=total_return,
        clashes=clashes,
        stuck_events=stuck_events,
        wall_seconds=time.perf_counter() - started,
        directive_counts=dict(sorted(directive_counts.items())),
        llm_calls=llm_calls,
        invalid_actions=invalid_actions,
        effect_violations=effect_violations,
        fallbacks=fallbacks,
        llm_latency_median_ms=median_latency,
        llm_latency_p95_ms=p95_latency,
    )


def _load_json_model(path: Path, model_type: type[Any]) -> Any:
    try:
        return model_type.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"could not load {model_type.__name__} artifact {path}"
        ) from exc


def _verify_demo_artifacts(
    config: DirectiveDemoConfig,
    repository_root: Path,
) -> tuple[
    DirectiveVocabulary,
    DirectivePpoManifest,
    DirectiveProbeReport,
    GemmaPreflightReport,
    DirectivePpoPolicy,
    str,
    str,
]:
    """Verify every frozen artifact before constructing or resetting TA-RWARE."""
    vocabulary_path = _resolve_path(repository_root, config.vocabulary_path)
    probe_path = _resolve_path(repository_root, config.directive_probe_report_path)
    checkpoint_path = _resolve_path(repository_root, config.ppo_checkpoint_path)
    manifest_path = _resolve_path(repository_root, config.ppo_manifest_path)
    preflight_path = _resolve_path(repository_root, config.preflight_report_path)

    vocabulary = DirectiveVocabulary.load(vocabulary_path)
    registry = build_directive_registry(vocabulary)
    manifest = _load_json_model(manifest_path, DirectivePpoManifest)
    probe_report = _load_json_model(probe_path, DirectiveProbeReport)
    preflight_report = _load_json_model(preflight_path, GemmaPreflightReport)

    if manifest.environment_id != config.environment_id:
        raise ValueError("PPO manifest environment does not match demo config")
    if manifest.vocabulary_sha256 != vocabulary.digest():
        raise ValueError("PPO manifest vocabulary digest does not match vocabulary")
    behavior_digest = registry.behavior_digest()
    if manifest.behavior_sha256 != behavior_digest:
        raise ValueError("PPO manifest behavior digest does not match registry")

    if not probe_report.passed:
        raise ValueError("directive probe report did not pass")
    if probe_report.environment_id != config.environment_id:
        raise ValueError("directive probe environment does not match demo config")
    if probe_report.vocabulary_sha256 != vocabulary.digest():
        raise ValueError("directive probe vocabulary digest does not match vocabulary")
    if probe_report.behavior_sha256 != behavior_digest:
        raise ValueError("directive probe behavior digest does not match registry")

    llama_config = config.llama_cpp
    if not preflight_report.passed:
        raise ValueError("Gemma preflight report did not pass")
    if preflight_report.selected_model != config.selected_model:
        raise ValueError("preflight model selection does not match demo config")
    if preflight_report.model_id != llama_config.model_id:
        raise ValueError("preflight model ID does not match demo config")
    if preflight_report.model_file_sha256 != llama_config.model_file_sha256:
        raise ValueError("preflight model digest does not match demo config")
    if str(preflight_report.base_url).rstrip("/") != str(llama_config.base_url).rstrip(
        "/"
    ):
        raise ValueError("preflight endpoint does not match demo config")
    if preflight_report.context_size != llama_config.context_size:
        raise ValueError("preflight context size does not match demo config")
    if preflight_report.gpu_layers != llama_config.gpu_layers:
        raise ValueError("preflight GPU layer count does not match demo config")
    if not preflight_report.health.healthy:
        raise ValueError("preflight health record is not healthy")
    if preflight_report.health.model_id != preflight_report.model_id:
        raise ValueError("preflight health model ID disagrees with its report")
    if preflight_report.model_id not in preflight_report.health.advertised_model_ids:
        raise ValueError(
            "preflight health record does not advertise the selected model"
        )
    if preflight_report.latency_p95_seconds > config.preflight_p95_limit_seconds:
        raise ValueError("preflight p95 latency exceeds the configured limit")

    excluded_seeds = (
        set(probe_report.probe_seeds)
        | set(manifest.train_seeds)
        | set(manifest.validation_seeds)
    )
    if config.episode_seed in excluded_seeds:
        raise ValueError(
            f"demo seed {config.episode_seed} overlaps a prototype seed partition"
        )

    policy = DirectivePpoPolicy.load(checkpoint_path, vocabulary, manifest)
    return (
        vocabulary,
        manifest,
        probe_report,
        preflight_report,
        policy,
        sha256_file(probe_path),
        sha256_file(preflight_path),
    )


def run_matched_demo(
    config: DirectiveDemoConfig,
    *,
    repository_root: Path | None = None,
) -> DemoRunSummary:
    """Run the frozen executor and Gemma actioner on matched fresh episodes."""
    root = (repository_root or _repository_root()).resolve()
    (
        vocabulary,
        manifest,
        _probe_report,
        _preflight_report,
        policy,
        probe_digest,
        preflight_digest,
    ) = _verify_demo_artifacts(config, root)
    registry = build_directive_registry(vocabulary)

    git_commit, _dirty = capture_git_state(root)
    run_directory = create_run_directory(
        _resolve_path(root, config.output_directory),
        controller_name="ppo_executor-ppo_gemma",
        created_at_utc=datetime.now(timezone.utc),
        git_commit=git_commit,
    )
    relative_output_directory = run_directory.relative_to(root)

    actioner = LlamaCppActioner(
        config.llama_cpp,
        GemmaPromptBuilder(),
        registry,
    )
    executor_wrapper: TarwareEnvWrapper | None = None
    gemma_wrapper: TarwareEnvWrapper | None = None
    try:
        # Recheck the live endpoint before the first reset. The preflight report
        # is provenance, while this health check prevents a stale server from
        # starting a partial matched run.
        actioner.health()
        write_json_atomic(run_directory / "resolved-config.json", config)

        executor_wrapper = TarwareEnvWrapper(config.environment_id)
        executor_controller = DirectiveExecutorController(policy, registry)
        executor_episode = run_episode(
            executor_wrapper,
            executor_controller,
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            render=False,
            decision_log_path=run_directory / "ppo_executor.jsonl",
        )
        write_json_atomic(run_directory / "ppo_executor-episode.json", executor_episode)

        gemma_wrapper = TarwareEnvWrapper(config.environment_id)
        gemma_controller = DirectiveGemmaController(policy, registry, actioner)
        gemma_episode = run_episode(
            gemma_wrapper,
            gemma_controller,
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            render=config.render_hybrid,
            decision_log_path=run_directory / "ppo_gemma.jsonl",
        )
        write_json_atomic(run_directory / "ppo_gemma-episode.json", gemma_episode)
    finally:
        if executor_wrapper is not None:
            executor_wrapper.close()
        if gemma_wrapper is not None:
            gemma_wrapper.close()
        actioner.close()

    summary = DemoRunSummary(
        schema_version=1,
        demo_only=True,
        environment_id=config.environment_id,
        seed=config.episode_seed,
        ppo_checkpoint_sha256=manifest.checkpoint_sha256,
        vocabulary_sha256=vocabulary.digest(),
        directive_probe_report_sha256=probe_digest,
        preflight_report_sha256=preflight_digest,
        executor_episode=executor_episode,
        gemma_episode=gemma_episode,
        output_directory=relative_output_directory,
    )
    write_json_atomic(run_directory / "demo-summary.json", summary)
    return summary
