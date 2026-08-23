"""Composition root for the matched executor/Gemma demonstration."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from warehouse_ai.demo.artifacts import (
    capture_git_state,
    create_run_directory,
    write_json_atomic,
)
from warehouse_ai.demo.contracts.evaluation import DemoRunSummary, DemoConfig
from warehouse_ai.demo.controllers import (
    ExecutorController,
    GemmaController,
)
from warehouse_ai.demo.directives import build_registry
from warehouse_ai.demo.evaluation.episode import run_episode
from warehouse_ai.demo.evaluation.verification import (
    resolve_path,
    verify_demo_artifacts,
)
from warehouse_ai.demo.inference.actioner import GemmaActioner
from warehouse_ai.demo.inference.llama_cpp import LlamaCppBackend
from warehouse_ai.env import TarwareEnvWrapper


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def run_matched_demo(
    config: DemoConfig, *, repository_root: Path | None = None
) -> DemoRunSummary:
    """Run frozen executor and Gemma strategies on matched fresh episodes."""
    root = (repository_root or _repository_root()).resolve()
    verified = verify_demo_artifacts(config, root)
    registry = build_registry(verified.catalog)
    git_commit, _ = capture_git_state(root)
    run_directory = create_run_directory(
        resolve_path(root, config.output_directory),
        controller_name=f"ppo_executor-ppo_gemma-v{config.schema_version}",
        created_at_utc=datetime.now(timezone.utc),
        git_commit=git_commit,
    )
    relative_output = run_directory.relative_to(root)
    backend = LlamaCppBackend(config.inference)
    actioner = GemmaActioner(backend, verified.prompt_builder)
    executor_wrapper: TarwareEnvWrapper | None = None
    gemma_wrapper: TarwareEnvWrapper | None = None
    try:
        backend.health()
        version = config.schema_version
        write_json_atomic(run_directory / f"resolved-config-v{version}.json", config)
        executor_wrapper = TarwareEnvWrapper(config.environment_id)
        executor_episode = run_episode(
            executor_wrapper,
            ExecutorController(verified.policy, registry),
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            render=False,
            decision_log_path=run_directory / f"ppo_executor-v{version}.jsonl",
        )
        write_json_atomic(
            run_directory / f"ppo_executor-episode-v{version}.json", executor_episode
        )
        gemma_wrapper = TarwareEnvWrapper(config.environment_id)
        gemma_episode = run_episode(
            gemma_wrapper,
            GemmaController(verified.policy, registry, actioner),
            seed=config.episode_seed,
            max_steps=config.episode_steps,
            render=config.render_hybrid,
            decision_log_path=run_directory / f"ppo_gemma-v{version}.jsonl",
        )
        write_json_atomic(
            run_directory / f"ppo_gemma-episode-v{version}.json", gemma_episode
        )
    finally:
        if executor_wrapper is not None:
            executor_wrapper.close()
        if gemma_wrapper is not None:
            gemma_wrapper.close()
        backend.close()
    summary = DemoRunSummary(
        schema_version=config.schema_version,
        demo_only=True,
        environment_id=config.environment_id,
        seed=config.episode_seed,
        ppo_checkpoint_sha256=verified.manifest.checkpoint_sha256,
        vocabulary_sha256=verified.catalog.digest(),
        directive_probe_report_sha256=verified.probe_digest,
        preflight_report_sha256=verified.preflight_digest,
        prompt_contract_sha256=verified.prompt_builder.contract_digest,
        executor_episode=executor_episode,
        gemma_episode=gemma_episode,
        output_directory=relative_output,
    )
    write_json_atomic(
        run_directory / f"demo-summary-v{config.schema_version}.json", summary
    )
    return summary
