"""Paired parent/fine-tuned PPO evaluation with one frozen Llm contract."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from warehouse_ai.demo.artifacts import (
    capture_git_state,
    create_run_directory,
    write_json_atomic,
)
from warehouse_ai.demo.contracts.evaluation import EpisodeResult
from warehouse_ai.demo.contracts.finetuning import (
    FineTuneComparisonSummary,
    FineTuneEvaluationConfig,
    FineTunePolicyEvaluation,
    LlmFineTuneConfig,
    LlmFineTuneManifest,
)
from warehouse_ai.demo.controllers import ExecutorController, LlmController
from warehouse_ai.demo.evaluation.episode import run_episode
from warehouse_ai.demo.inference.actioner import LlmActioner
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend
from warehouse_ai.demo.ppo.finetune_service import (
    VerifiedFineTuneArtifacts,
    verify_fine_tune_artifacts,
)
from warehouse_ai.demo.ppo.finetuning import load_fine_tuned_policy
from warehouse_ai.demo.ppo.policy import PpoPolicy
from warehouse_ai.env import TarwareEnvWrapper


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _load_manifest(path: Path) -> LlmFineTuneManifest:
    try:
        return LlmFineTuneManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load fine-tuning manifest {path}") from exc


def _load_comparison(path: Path) -> FineTuneComparisonSummary:
    try:
        return FineTuneComparisonSummary.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load prerequisite comparison {path}") from exc


def _partition_seeds(
    config: LlmFineTuneConfig,
    partition: str,
) -> tuple[int, ...]:
    if partition == "selection":
        return config.selection_seeds
    if partition == "confirmation":
        return config.confirmation_seeds
    if partition == "final":
        return config.final_evaluation_seeds
    raise ValueError(f"unknown evaluation partition {partition!r}")


def _verify_child(
    root: Path,
    evaluation: FineTuneEvaluationConfig,
    fine_config: LlmFineTuneConfig,
    verified: VerifiedFineTuneArtifacts,
) -> tuple[LlmFineTuneManifest, PpoPolicy]:
    manifest_path = _resolve(root, evaluation.fine_tune_manifest_path)
    checkpoint_path = _resolve(root, evaluation.fine_tune_checkpoint_path)
    manifest = _load_manifest(manifest_path)
    if (
        evaluation.schema_version != fine_config.schema_version
        or manifest.schema_version != fine_config.schema_version
    ):
        raise ValueError("fine-tuning config, manifest, and evaluation versions differ")
    expected = (
        verified.parent_manifest.checkpoint_sha256,
        verified.identity.parent_manifest_sha256,
        verified.identity.prompt_contract_sha256,
        verified.identity.preflight_report_sha256,
        fine_config.inference.model_file_sha256,
        fine_config.train_seeds,
        fine_config.selection_seeds,
        fine_config.confirmation_seeds,
        fine_config.final_evaluation_seeds,
    )
    actual = (
        manifest.parent_checkpoint_sha256,
        manifest.parent_manifest_sha256,
        manifest.prompt_contract_sha256,
        manifest.preflight_report_sha256,
        manifest.model_file_sha256,
        manifest.train_seeds,
        manifest.selection_seeds,
        manifest.confirmation_seeds,
        manifest.final_evaluation_seeds,
    )
    if actual != expected:
        raise ValueError("fine-tuned checkpoint provenance does not match config")
    return manifest, load_fine_tuned_policy(
        checkpoint_path,
        verified.catalog,
        verified.parent_manifest,
        manifest,
    )


def verify_evaluation_prerequisite(
    root: Path,
    evaluation: FineTuneEvaluationConfig,
    child_manifest: LlmFineTuneManifest,
) -> None:
    """Require a passing exact prior partition before exposing later seeds."""
    if evaluation.partition == "selection":
        return
    assert evaluation.prerequisite_summary_path is not None
    prerequisite = _load_comparison(
        _resolve(root, evaluation.prerequisite_summary_path)
    )
    expected_partition = (
        "selection" if evaluation.partition == "confirmation" else "confirmation"
    )
    if prerequisite.partition != expected_partition:
        raise ValueError("evaluation prerequisite is from the wrong partition")
    if prerequisite.fine_tuned_checkpoint_sha256 != child_manifest.checkpoint_sha256:
        raise ValueError("evaluation prerequisite is for another child checkpoint")
    if not prerequisite.candidate_better:
        raise ValueError("evaluation prerequisite did not pass its candidate gate")


def _run_policy(
    *,
    label: str,
    policy: PpoPolicy,
    seeds: Sequence[int],
    config: LlmFineTuneConfig,
    verified: VerifiedFineTuneArtifacts,
    actioner: LlmActioner,
    run_directory: Path,
) -> FineTunePolicyEvaluation:
    executor_results: list[EpisodeResult] = []
    llm_results: list[EpisodeResult] = []
    for seed in seeds:
        for mode in ("executor", "llm"):
            wrapper = TarwareEnvWrapper(config.environment_id)
            controller = (
                ExecutorController(policy, verified.registry)
                if mode == "executor"
                else LlmController(policy, verified.registry, actioner)
            )
            try:
                result = run_episode(
                    wrapper,
                    controller,
                    seed=seed,
                    max_steps=config.episode_steps,
                    render=False,
                    decision_log_path=(
                        run_directory
                        / f"{label}-{mode}-seed-{seed}-v{config.schema_version}.jsonl"
                    ),
                )
            finally:
                wrapper.close()
            write_json_atomic(
                run_directory
                / f"{label}-{mode}-seed-{seed}-summary-v{config.schema_version}.json",
                result,
            )
            if mode == "executor":
                executor_results.append(result)
            else:
                llm_results.append(result)
    return FineTunePolicyEvaluation(
        policy="parent" if label == "parent" else "fine_tuned",
        executor_episodes=tuple(executor_results),
        llm_episodes=tuple(llm_results),
    )


def is_better_llm_candidate(
    parent: FineTunePolicyEvaluation,
    candidate: FineTunePolicyEvaluation,
) -> bool:
    def score(evaluation: FineTunePolicyEvaluation) -> tuple[float, float, float]:
        episodes = evaluation.llm_episodes
        calls = sum(item.llm_calls for item in episodes)
        fallback_rate = sum(item.fallbacks for item in episodes) / max(calls, 1)
        return (
            mean(item.completed_tasks for item in episodes),
            mean(item.total_return for item in episodes),
            -fallback_rate,
        )

    commands = {
        command
        for episode in candidate.llm_episodes
        for command, count in episode.directive_counts.items()
        if command != "wait" and count > 0
    }
    return len(commands) >= 2 and score(candidate) > score(parent)


def run_fine_tune_comparison(
    evaluation: FineTuneEvaluationConfig,
    *,
    fine_config: LlmFineTuneConfig,
    repository_root: Path,
) -> FineTuneComparisonSummary:
    """Run one explicitly selected seed partition; never train or select here."""
    root = repository_root.resolve()
    verified = verify_fine_tune_artifacts(fine_config, root)
    child_manifest, child_policy = _verify_child(
        root, evaluation, fine_config, verified
    )
    verify_evaluation_prerequisite(root, evaluation, child_manifest)
    seeds = _partition_seeds(fine_config, evaluation.partition)
    version = evaluation.schema_version
    gate_path = (
        _resolve(root, fine_config.output_directory)
        / f"{evaluation.partition}-gate.json"
    )
    if gate_path.exists():
        raise FileExistsError("evaluation gate already exists for this child policy")
    git_commit, _ = capture_git_state(root)
    run_directory = create_run_directory(
        _resolve(root, evaluation.output_directory),
        controller_name=f"llm-finetune-{evaluation.partition}",
        created_at_utc=datetime.now(timezone.utc),
        git_commit=git_commit,
    )
    backend = LlamaCppBackend(fine_config.inference)
    try:
        health = backend.health()
        if not health.healthy:
            raise RuntimeError("configured Llm server is not healthy")
        actioner = LlmActioner(backend, verified.prompt_builder)
        parent = _run_policy(
            label="parent",
            policy=verified.parent_policy,
            seeds=seeds,
            config=fine_config,
            verified=verified,
            actioner=actioner,
            run_directory=run_directory,
        )
        child = _run_policy(
            label="fine-tuned",
            policy=child_policy,
            seeds=seeds,
            config=fine_config,
            verified=verified,
            actioner=actioner,
            run_directory=run_directory,
        )
    finally:
        backend.close()
    summary = FineTuneComparisonSummary(
        schema_version=version,
        environment_id=fine_config.environment_id,
        partition=evaluation.partition,
        seeds=seeds,
        parent_checkpoint_sha256=verified.parent_manifest.checkpoint_sha256,
        fine_tuned_checkpoint_sha256=child_manifest.checkpoint_sha256,
        prompt_contract_sha256=verified.identity.prompt_contract_sha256,
        preflight_report_sha256=verified.identity.preflight_report_sha256,
        parent=parent,
        fine_tuned=child,
        candidate_better=is_better_llm_candidate(parent, child),
        output_directory=run_directory.relative_to(root),
    )
    write_json_atomic(run_directory / "fine-tuning-comparison.json", summary)
    write_json_atomic(gate_path, summary)
    return summary
