# 2026-08-22: Directive-demo Phase 6 evaluation runner

## Objective

Implement the isolated directive-demo Phase 6 composition layer and verify
that it can preserve one flushed decision record per simulator step, aggregate
the pinned TA-RWARE metrics, and reject stale or mismatched artifacts before an
episode reset.

## Hypothesis

Not applicable for this implementation and artifact-boundary investigation.

## Provenance

- Git commit: `b31a420148ac524f3420c409712b284e10d335e7` plus the dirty Phase 6 changes
- Dirty worktree: yes; source, tests, and documentation changes are the subject of this entry
- Commands: `uv run ruff check src/warehouse_ai/demo scripts/run_directive_demo.py tests/demo/test_evaluation.py`; `uv run python -m pytest -q tests/demo/test_evaluation.py`
- Config path and digest: `configs/demo/directive-demo-v1.yaml`; no live matched run was performed in this entry
- Run IDs: none; no 500-step simulator/actioner run was started
- Seed set and partition: implementation test uses prototype seed `0`; the configured demo seed remains `9100`
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- PPO checkpoint digest: the runner reads and verifies the configured schema-v2 manifest at runtime
- Hardware and driver: local Windows development environment; no Gemma server request was made
- Start/end time in UTC: 2026-08-22; exact wall-clock times are not material because no live run occurred
- Estimated and actual cost: USD 0

## Method

Phase 6 adds immutable `EpisodeResult` and `DemoRunSummary` records, an
`evaluation.py` episode loop, append-and-flush JSONL writing, and a
composition-only `run_directive_demo.py` command. The runner verifies the
catalog, directive behavior, probe report, preflight report, PPO manifest,
checkpoint digest, model identity, and seed exclusions before constructing a
TA-RWARE wrapper. It then creates fresh wrappers for the deterministic
executor and Gemma episodes while holding the seed, environment, catalog,
and PPO policy fixed.

The metrics use the compatibility-spike keys `shelf_deliveries`, `clashes`,
and `stucks`. LLM attempts count non-wait Gemma decisions, while fallbacks,
schema/decode/stale failures, and directive-effect violations remain separate
fields. No significance test or improvement metric was added.

## Results

- `uv run ruff check src/warehouse_ai/demo scripts/run_directive_demo.py tests/demo/test_evaluation.py`: passed.
- `uv run python -m pytest -q tests/demo/test_evaluation.py`: 3 tests passed.
- The tests covered flushed per-step JSONL, pinned info-key aggregation,
  malformed-action fallback accounting, and stopping only after all agents
  reported termination.
- No live 500-step output, Gemma request, or comparison summary was produced.

## Observations

- The existing Phase 4 controller records expose enough information to count
  attempted non-wait Gemma calls and visible fallback causes without changing
  the frozen controller API.
- TA-RWARE's shared info dictionary provides the three counters required by the
  prototype summary.
- Atomic JSON summaries can be written only after each episode completes;
  append-only JSONL remains available if the later episode fails.

## Interpretation

The implementation satisfies the local composition and artifact-boundary
checks, but these tests do not establish that the local Gemma server can finish
the two required 500-step episodes. They also do not constitute a scientific
comparison or an improvement claim.

## Decision

Keep the pinned E4B configuration and proceed to the live matched demonstration
only after starting the already-preflighted local server. The Phase 6 roadmap
implementation is complete; the live 500-step prototype gate remains pending.

## Problems and follow-up

- Start the pinned local llama.cpp server and run
  `uv run python scripts/run_directive_demo.py --config configs/demo/directive-demo-v1.yaml`.
- Record the resulting run directory, episode summaries, wall times, latency,
  invalid actions, effect violations, and fallbacks in a new dated result entry.
- If the run fails, preserve its JSONL and document the failure rather than
  changing the primary research metric or silently substituting a model.
