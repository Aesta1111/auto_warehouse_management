"""Run the real local Gemma actioner preflight.

This is intentionally the only command that decides whether a configured
llama.cpp server is ready for the directive demo. It performs transport,
schema, directive-effect, sensitivity, and latency checks, then writes one
small :class:`GemmaPreflightReport` artifact. It never starts a model server
and never changes a PPO checkpoint.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from warehouse_ai.demo.actioner import ActionerError, LlamaCppActioner
from warehouse_ai.demo.artifacts import (
    canonical_json_bytes,
    sha256_bytes,
    write_json_atomic,
)
from warehouse_ai.demo.directives import DirectiveRegistry, build_directive_registry
from warehouse_ai.demo.models import (
    ActionerRequest,
    DirectiveContext,
    DirectiveDemoConfig,
    DirectivePpoManifest,
    DirectivePrediction,
    DirectiveVocabulary,
    GemmaPreflightReport,
    load_yaml_config,
)
from warehouse_ai.demo.ppo import DirectivePpoPolicy
from warehouse_ai.demo.prompting import GemmaPromptBuilder
from warehouse_ai.env import TarwareEnvWrapper


PREFLIGHT_OK = 0
PREFLIGHT_ERROR = 1
E2B_RECOMMENDED = 2


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Return a deterministic linearly interpolated percentile."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _hardware_provenance() -> dict[str, Any]:
    """Collect small, non-secret hardware/runtime facts for the report."""
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["cuda_device"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError) as exc:
        facts["torch"] = f"unavailable: {type(exc).__name__}"
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is not None:
        try:
            result = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.used,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                facts["nvidia_smi"] = result.stdout.strip()
            else:
                facts["nvidia_smi_error"] = result.stderr.strip() or "non-zero exit"
        except (OSError, subprocess.TimeoutExpired) as exc:
            facts["nvidia_smi_error"] = type(exc).__name__
    return facts


def _load_policy(
    config: DirectiveDemoConfig,
    vocabulary: DirectiveVocabulary,
) -> tuple[DirectivePpoPolicy, DirectivePpoManifest]:
    manifest = DirectivePpoManifest.model_validate_json(
        config.ppo_manifest_path.read_text(encoding="utf-8")
    )
    policy = DirectivePpoPolicy.load(
        config.ppo_checkpoint_path,
        vocabulary,
        manifest,
    )
    return policy, manifest


def _forced_prediction(
    context: DirectiveContext,
    vocabulary: DirectiveVocabulary,
    directive_id: int,
    checkpoint_sha256: str,
) -> DirectivePrediction:
    """Create a prompt stimulus for sensitivity without changing PPO."""
    probabilities = [0.0] * vocabulary.size
    probabilities[directive_id] = 1.0
    return DirectivePrediction(
        state_id=context.state_id,
        episode_step=context.observation.step,
        directive=vocabulary.by_id(directive_id),
        probabilities=tuple(probabilities),
        value_estimate=0.0,
        policy_latency_ms=0.0,
        checkpoint_sha256=checkpoint_sha256,
    )


def _applicable_non_wait(
    context: DirectiveContext,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
) -> tuple[int, ...]:
    return _applicable_directives(context, registry, vocabulary, include_wait=False)


def _applicable_directives(
    context: DirectiveContext,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
    *,
    include_wait: bool,
) -> tuple[int, ...]:
    mask = registry.action_mask(context)
    return tuple(
        int(directive_id)
        for directive_id, applicable in enumerate(mask)
        if applicable and (include_wait or directive_id != vocabulary.wait_directive_id)
    )


def _request_for_policy(
    context: DirectiveContext,
    policy: DirectivePpoPolicy,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
) -> ActionerRequest:
    mask = registry.action_mask(context)
    prediction = policy.predict(context, mask)
    return ActionerRequest(context=context, prediction=prediction)


def _request_for_directive(
    context: DirectiveContext,
    policy: DirectivePpoPolicy,
    directive_id: int,
    vocabulary: DirectiveVocabulary,
) -> ActionerRequest:
    return ActionerRequest(
        context=context,
        prediction=_forced_prediction(
            context,
            vocabulary,
            directive_id,
            policy.checkpoint_sha256,
        ),
    )


def _timed_action(
    actioner: LlamaCppActioner,
    request: ActionerRequest,
    wrapper: TarwareEnvWrapper,
) -> tuple[float, bool, str | None, str | None]:
    started = time.perf_counter()
    try:
        result = actioner.act(request, wrapper)
    except ActionerError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        cause = exc.__cause__
        if cause is not None:
            detail += f" ({type(cause).__name__}: {cause})"
        return time.perf_counter() - started, False, None, detail
    return time.perf_counter() - started, True, result.plan.reason, None


def _prompt_schema_digest(
    context: DirectiveContext,
    vocabulary: DirectiveVocabulary,
    directive_id: int,
    checkpoint_sha256: str,
) -> str:
    request = ActionerRequest(
        context=context,
        prediction=_forced_prediction(
            context, vocabulary, directive_id, checkpoint_sha256
        ),
    )
    envelope = GemmaPromptBuilder().build(request)
    return sha256_bytes(canonical_json_bytes(envelope.response_schema))


def _find_state_with_directives(
    wrapper: TarwareEnvWrapper,
    registry: DirectiveRegistry,
    vocabulary: DirectiveVocabulary,
    *,
    seed: int,
    max_steps: int,
    required: int,
    include_wait: bool = False,
) -> tuple[DirectiveContext, tuple[int, ...]] | None:
    wrapper.reset(seed=seed)
    for _ in range(max_steps):
        context = DirectiveContext.capture(wrapper)
        directives = _applicable_directives(
            context,
            registry,
            vocabulary,
            include_wait=include_wait,
        )
        if len(directives) >= required:
            return context, directives
        wrapper.step(wrapper.no_op())
    return None


def run_preflight(
    config: DirectiveDemoConfig,
    *,
    repository_root: Path | None = None,
) -> GemmaPreflightReport:
    """Run and persist the configured server preflight."""
    del repository_root  # Config paths are intentionally repository-relative.
    vocabulary = DirectiveVocabulary.load(config.vocabulary_path)
    registry = build_directive_registry(vocabulary)
    policy, _ = _load_policy(config, vocabulary)
    actioner = LlamaCppActioner(config.llama_cpp, GemmaPromptBuilder(), registry)
    wrapper = TarwareEnvWrapper(config.environment_id)

    latencies: list[float] = []
    measured_classes: list[str] = []
    valid_responses = 0
    invalid_responses = 0
    warmup_completed = False
    sensitivity_pairs = 0
    sensitivity_valid_pairs = 0
    sensitivity_states = 0
    sensitivity_results: list[dict[str, Any]] = []
    health = None
    prompt_schema_sha256 = "0" * 64
    failure_reasons: list[str] = []

    try:
        health = actioner.health()

        warmup_state = _find_state_with_directives(
            wrapper,
            registry,
            vocabulary,
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            required=1,
        )
        if warmup_state is None:
            failure_reasons.append("no applicable non-wait directive for warm-up")
        else:
            warmup_context, warmup_directives = warmup_state
            warmup_request = _request_for_directive(
                warmup_context,
                policy,
                warmup_directives[0],
                vocabulary,
            )
            _, warmup_ok, _, warmup_error = _timed_action(
                actioner, warmup_request, wrapper
            )
            warmup_completed = warmup_ok
            if warmup_error is not None:
                failure_reasons.append(f"warm-up failed: {warmup_error}")
            prompt_schema_sha256 = _prompt_schema_digest(
                warmup_context,
                vocabulary,
                warmup_directives[0],
                policy.checkpoint_sha256,
            )

        # Measure five requests on fresh, deterministic states. Directives are
        # selected from the current applicability mask; PPO remains untouched.
        for offset in range(20):
            if len(latencies) >= 5:
                break
            state = _find_state_with_directives(
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
            request = _request_for_policy(context, policy, registry, vocabulary)
            selected = request.prediction.directive.directive_id
            if selected == vocabulary.wait_directive_id:
                non_wait = tuple(
                    directive_id
                    for directive_id in directives
                    if directive_id != vocabulary.wait_directive_id
                )
                if non_wait:
                    selected = non_wait[0]
                    request = _request_for_directive(
                        context, policy, selected, vocabulary
                    )
            elif len(set(measured_classes)) < 2:
                covered = set(measured_classes)
                alternate = next(
                    (
                        directive_id
                        for directive_id in directives
                        if directive_id != selected
                        if vocabulary.by_id(directive_id).behavior_key not in covered
                    ),
                    None,
                )
                if alternate is not None:
                    selected = alternate
                    request = _request_for_directive(
                        context, policy, selected, vocabulary
                    )
            measured_classes.append(vocabulary.by_id(selected).behavior_key)
            elapsed, valid, _, error = _timed_action(actioner, request, wrapper)
            latencies.append(elapsed)
            if valid:
                valid_responses += 1
            else:
                invalid_responses += 1
                if error is not None:
                    failure_reasons.append(f"measured request failed: {error}")

        if len(latencies) < 5:
            failure_reasons.append(
                f"only {len(latencies)} of 5 measured requests could run"
            )
        if len(set(measured_classes)) < 2:
            failure_reasons.append(
                "measured requests covered fewer than two directives"
            )

        # For sensitivity, each pair is sent while the wrapper remains at the
        # captured state. The actioner validates each response against the
        # corresponding effect rule before it is counted as valid.
        for offset in range(20, 40):
            if sensitivity_states >= 3:
                break
            state = _find_state_with_directives(
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
            pair_record: dict[str, Any] = {
                "state_id": context.state_id,
                "directive_ids": list(directives[:2]),
                "valid": [],
            }
            for directive_id in directives[:2]:
                sensitivity_pairs += 1
                request = _request_for_directive(
                    context, policy, directive_id, vocabulary
                )
                _, valid, _, error = _timed_action(actioner, request, wrapper)
                pair_record["valid"].append(valid)
                if valid:
                    sensitivity_valid_pairs += 1
                elif error is not None:
                    failure_reasons.append(f"sensitivity request failed: {error}")
            sensitivity_results.append(pair_record)

    except ActionerError as exc:
        failure_reasons.append(f"actioner preflight error: {type(exc).__name__}")
    finally:
        wrapper.close()
        actioner.close()

    if health is None:
        # Keep an operational failure serializable without claiming that the
        # endpoint was actually verified. The configured ID is recorded only
        # as an unverified placeholder and ``healthy`` remains false.
        from warehouse_ai.demo.models import ModelHealth

        health = ModelHealth(
            base_url=config.llama_cpp.base_url,
            model_id=config.llama_cpp.model_id,
            advertised_model_ids=(config.llama_cpp.model_id,),
            status="unverified",
            healthy=False,
            checked_at_utc=datetime.now(timezone.utc),
        )
        failure_reasons.append("health and model identity were not verified")

    latency_median = statistics.median(latencies) if latencies else 0.0
    latency_p95 = _percentile(latencies, 0.95)
    validity_passed = (
        len(latencies) == 5 and valid_responses == 5 and len(set(measured_classes)) >= 2
    )
    sensitivity_passed = (
        sensitivity_states >= 3
        and sensitivity_pairs >= 6
        and sensitivity_valid_pairs == sensitivity_pairs
    )
    latency_passed = (
        bool(latencies) and latency_p95 <= config.preflight_p95_limit_seconds
    )
    if sensitivity_states < 3:
        failure_reasons.append(
            f"only {sensitivity_states} of 3 sensitivity states were found"
        )
    if sensitivity_pairs and not sensitivity_passed:
        failure_reasons.append("directive sensitivity gate failed")
    if not latency_passed:
        failure_reasons.append("p95 latency gate failed")

    report = GemmaPreflightReport(
        selected_model=config.selected_model,
        model_id=config.llama_cpp.model_id,
        model_file_sha256=config.llama_cpp.model_file_sha256,
        base_url=config.llama_cpp.base_url,
        context_size=config.llama_cpp.context_size,
        gpu_layers=config.llama_cpp.gpu_layers,
        prompt_schema_sha256=prompt_schema_sha256,
        health=health,
        warmup_completed=warmup_completed,
        measured_requests=len(latencies),
        valid_responses=valid_responses,
        invalid_responses=invalid_responses,
        directive_classes=tuple(dict.fromkeys(measured_classes)),
        sensitivity_states=sensitivity_states,
        sensitivity_pairs=sensitivity_pairs,
        sensitivity_valid_pairs=sensitivity_valid_pairs,
        sensitivity_results=tuple(sensitivity_results),
        latency_seconds=tuple(latencies),
        latency_median_seconds=float(latency_median),
        latency_p95_seconds=float(latency_p95),
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
        hardware=_hardware_provenance(),
        server_command=config.llama_cpp.server_command,
        quantization=config.llama_cpp.quantization or "GGUF Q4_0",
        checked_at_utc=datetime.now(timezone.utc),
        failure_reason="; ".join(dict.fromkeys(failure_reasons)) or None,
    )
    report_path = config.preflight_report_path
    write_json_atomic(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="repository-relative directive-demo YAML configuration",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_yaml_config(args.config, DirectiveDemoConfig)
    try:
        report = run_preflight(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"preflight could not complete: {type(exc).__name__}: {exc}")
        return PREFLIGHT_ERROR

    print(report.model_dump_json(indent=2))
    if report.passed:
        return PREFLIGHT_OK
    if config.selected_model == "e4b" and report.health.healthy:
        print(
            "E4B preflight failed; start the explicitly configured E2B server "
            "and write a separate E2B config/report.",
            file=sys.stderr,
        )
        return E2B_RECOMMENDED
    return PREFLIGHT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
