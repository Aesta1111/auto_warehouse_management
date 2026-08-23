# 2026-08-22: Directive-demo Phase 4 fake actioner

## Objective

Close the Phase 4 actioner-neutral orchestration gate from `PLAN.md` by running
the test-only fake actioner through the live TA-RWARE wrapper. This validates
the success path and the deterministic fallback behavior before introducing a
real local Gemma client. The fake actioner is not production code and does not
represent a Gemma result.

## Provenance

- Base commit: `864ec64e2ab8fb458b21b326f0bfaec6a05b5748`
- Worktree: dirty; this entry and the short-episode coverage were added after
  the base commit
- Environment: Windows, Python 3.12.4, local CPU
- Environment ID: `tarware-tiny-3agvs-2pickers-partialobs-v1`
- Seed: `0`
- Episode horizon: 6 steps per scenario
- Cost: USD 0

## Method

Added a test-only short-episode harness that captures and JSON-serializes every
`DemoStepRecord`. It runs six scenarios through `GemmaController`:

1. valid fake action
2. malformed response
3. invalid action shape
4. action-effect violation
5. timeout
6. stale response

The fake actioner returns the frozen registry executor's action for the valid
case and injects one failure mode for each remaining case. The controller must
continue stepping TA-RWARE and use the selected directive's deterministic
executor whenever the fake response cannot be accepted.

## Results

- `tests/demo/test_actioner.py`: **15 passed**.
- Six scenarios completed six wrapper steps each, producing 36 serializable
  decision records.
- The valid scenario reached the actioner path and accepted a Gemma-shaped
  plan.
- Every failure scenario produced at least one directive-executor fallback
  record with a non-empty reason.
- The existing automatic-no-op test also passed when the deterministic
  directive executor itself was forced to fail.

A repository-wide run reached 66 passing tests, but four unrelated tests that
use pytest's `tmp_path` fixture could not initialize because the managed
Windows environment denies access to the pytest temporary-directory root. A
filtered run excluding only those four tests passed 66 tests with four
deselected. This is an environment limitation, not a fake-actioner failure.

## Decision

The Phase 4 fake-actioner gate passes functionally. Keep the fake actioner in
tests only, preserve the generic `Actioner[SingleJointActionPlan]` boundary,
and proceed to Phase 5: implement and preflight the pinned local `llama.cpp`
 Gemma actioner.
