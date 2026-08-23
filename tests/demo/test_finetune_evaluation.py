from __future__ import annotations

from typing import Literal

from warehouse_ai.demo.contracts.evaluation import EpisodeResult
from warehouse_ai.demo.contracts.finetuning import FineTunePolicyEvaluation
from warehouse_ai.demo.evaluation.finetune_runner import is_better_llm_candidate


def _episode(
    *,
    controller: Literal["ppo_executor", "ppo_llm"],
    tasks: int,
    total_return: float,
    fallbacks: int = 0,
    directives: dict[str, int] | None = None,
) -> EpisodeResult:
    return EpisodeResult(
        controller=controller,
        seed=8200,
        steps=10,
        completed_tasks=tasks,
        total_return=total_return,
        clashes=0,
        stuck_events=0,
        wall_seconds=1.0,
        directive_counts=directives
        or {"service-requested-shelf": 5, "deliver-carried-shelf": 5},
        llm_calls=10 if controller == "ppo_llm" else 0,
        invalid_actions=0,
        effect_violations=fallbacks,
        fallbacks=fallbacks,
        llm_latency_median_ms=1.0 if controller == "ppo_llm" else None,
        llm_latency_p95_ms=1.0 if controller == "ppo_llm" else None,
    )


def _evaluation(
    policy: Literal["parent", "fine_tuned"],
    *,
    tasks: int,
    total_return: float,
    fallbacks: int = 0,
    directives: dict[str, int] | None = None,
) -> FineTunePolicyEvaluation:
    return FineTunePolicyEvaluation(
        policy=policy,
        executor_episodes=(
            _episode(controller="ppo_executor", tasks=tasks, total_return=total_return),
        ),
        llm_episodes=(
            _episode(
                controller="ppo_llm",
                tasks=tasks,
                total_return=total_return,
                fallbacks=fallbacks,
                directives=directives,
            ),
        ),
    )


def test_candidate_selection_prioritizes_tasks_then_raw_return() -> None:
    parent = _evaluation("parent", tasks=2, total_return=0.0)
    candidate = _evaluation("fine_tuned", tasks=3, total_return=-1.0)

    assert is_better_llm_candidate(parent, candidate)


def test_candidate_selection_uses_fallback_rate_only_as_tie_breaker() -> None:
    parent = _evaluation("parent", tasks=2, total_return=0.0, fallbacks=5)
    candidate = _evaluation("fine_tuned", tasks=2, total_return=0.0, fallbacks=2)

    assert is_better_llm_candidate(parent, candidate)


def test_candidate_selection_rejects_wait_or_one_directive_collapse() -> None:
    parent = _evaluation("parent", tasks=2, total_return=0.0)
    candidate = _evaluation(
        "fine_tuned",
        tasks=3,
        total_return=1.0,
        directives={"deliver-carried-shelf": 10},
    )

    assert not is_better_llm_candidate(parent, candidate)
