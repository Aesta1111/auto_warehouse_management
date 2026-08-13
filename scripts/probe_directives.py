"""Probe directive applicability and deterministic executors on real TA-RWARE states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from warehouse_ai.demo.artifacts import write_json_atomic
from warehouse_ai.demo.directives import (
    DirectiveEffectError,
    DirectiveExecutorError,
    build_directive_registry,
)
from warehouse_ai.demo.models import (
    DirectiveContext,
    DirectiveProbeConfig,
    DirectiveProbeReport,
    DirectiveVocabulary,
    load_yaml_config,
)
from warehouse_ai.env import TarwareEnvWrapper


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repository_path(root: Path, path: Path) -> Path:
    """Resolve a repository-relative committed path without changing the model."""
    return path if path.is_absolute() else root / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/demo/directive-probe-report.json"),
    )
    return parser.parse_args()


def _probe_seed(
    config: DirectiveProbeConfig,
    vocabulary: DirectiveVocabulary,
    seed: int,
) -> tuple[int, int, int, int]:
    """Return ``(states, uncovered, executor_failures, effect_failures)``."""
    registry = build_directive_registry(vocabulary)
    wrapper = TarwareEnvWrapper(config.environment_id)
    states_checked = 0
    uncovered_states = 0
    executor_failures = 0
    effect_failures = 0

    try:
        wrapper.reset(seed=seed)
        for _ in range(config.episode_steps):
            context = DirectiveContext.capture(wrapper)
            states_checked += 1
            try:
                directive_mask = registry.action_mask(context)
            except Exception:
                uncovered_states += 1
                break

            applicable = np.flatnonzero(directive_mask)
            if len(applicable) == 0:
                uncovered_states += 1
                break

            # Validate every currently applicable behavior.  Calling each one
            # twice before stepping makes nondeterminism observable without
            # changing the simulator state.
            for directive_id in (int(value) for value in applicable):
                try:
                    first = registry.execute(directive_id, context, wrapper)
                    second = registry.execute(directive_id, context, wrapper)
                    if first != second:
                        executor_failures += 1
                except DirectiveEffectError:
                    effect_failures += 1
                except DirectiveExecutorError:
                    executor_failures += 1

            # Advance with the lowest applicable non-wait directive where
            # possible.  This reaches carried/goal and reposition states while
            # keeping the probe deterministic.  Wait remains the fallback.
            wait_id = vocabulary.wait_directive_id
            non_wait = [int(value) for value in applicable if int(value) != wait_id]
            selected_id = non_wait[0] if non_wait else wait_id
            try:
                action = registry.execute(selected_id, context, wrapper)
                _, _, terminated, truncated, _ = wrapper.step(action)
            except DirectiveEffectError:
                effect_failures += 1
                break
            except DirectiveExecutorError:
                executor_failures += 1
                break

            if all(terminated) or all(truncated):
                break
    finally:
        wrapper.close()

    return states_checked, uncovered_states, executor_failures, effect_failures


def run_probe(
    config: DirectiveProbeConfig, vocabulary: DirectiveVocabulary
) -> DirectiveProbeReport:
    """Run the committed probe seed set and construct its small report."""
    totals = [0, 0, 0, 0]
    for seed in config.probe_seeds:
        result = _probe_seed(config, vocabulary, seed)
        totals = [left + right for left, right in zip(totals, result, strict=True)]

    states_checked, uncovered_states, executor_failures, effect_failures = totals
    return DirectiveProbeReport(
        schema_version=1,
        environment_id=config.environment_id,
        vocabulary_sha256=vocabulary.digest(),
        behavior_sha256=build_directive_registry(vocabulary).behavior_digest(),
        probe_seeds=config.probe_seeds,
        states_checked=states_checked,
        uncovered_states=uncovered_states,
        executor_failures=executor_failures,
        effect_failures=effect_failures,
        passed=bool(
            states_checked > 0
            and uncovered_states == 0
            and executor_failures == 0
            and effect_failures == 0
        ),
    )


def main() -> None:
    args = parse_args()
    root = _repository_root()
    config = load_yaml_config(args.config, DirectiveProbeConfig)
    vocabulary = DirectiveVocabulary.load(
        _repository_path(root, config.vocabulary_path)
    )
    report = run_probe(config, vocabulary)
    output = _repository_path(root, args.output)
    write_json_atomic(output, report)
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
