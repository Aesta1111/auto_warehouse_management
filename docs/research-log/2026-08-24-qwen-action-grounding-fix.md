# 2026-08-24: Qwen action-grounding fix

## Objective

Remove the agent-index/destination-ID confusion observed in the refreshed Qwen
preflight without changing the frozen model or simulator semantics.
This follows the failed gate recorded in the
[refreshed full-demo entry](2026-08-24-qwen3.5-2b-full-demo-run.md).

## Hypothesis

Moving directive eligibility into deterministic code and constraining Qwen to a
bounded set of complete safe joint actions would eliminate strict action and
directive-effect violations.

## Provenance

- Git commit: pending at run time; this entry is committed with the fix
- Dirty worktree: yes; the refreshed demo run log and requested cleanup were
  already present
- Commands:
  - `.venv\\Scripts\\python.exe -m pytest tests/demo/test_prompting.py tests/demo/test_configuration.py`
  - `.venv\\Scripts\\ruff.exe format --check ...`
  - `.venv\\Scripts\\ruff.exe check ...`
  - `.venv\\Scripts\\python.exe scripts/probe_llm_actioner.py --config configs/demo/ppo-llm-evaluation-run.yaml`
- Prompt contract SHA-256:
  `9d9cda6755f8bfefef8924c09da14e11e223e788cf28492223840961139cc108`
- Aggregate template SHA-256:
  `1faa0fb7a9dad3802929838fc26570190d7ecaa62da591deeb447a6e70634ec0`
- Per-template SHA-256:
  - service: `b9ae66053dbc64b8e7abea19f36db0214851a7fb4508b9423fde2b7f49b287a4`
  - delivery: `2e6ba17ca8250d5fdeab774d4d1bf68926e5774e1106ec83cfe1922539a51239`
  - reposition: `7a160fdf7763219641ba7ef6cd4a63de28b4792e9bd63e3b3f07ca29a3a93154`
- Parent PPO checkpoint SHA-256:
  `f920189ffbfe80e31759443b33088a99a9710b788d49cf2f67d042f99bee08666`
- Qwen model and revision: unchanged pinned Qwen3.5-2B Q8 artifact
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB, driver 580.97
- Cost: USD 0 local compute

## Method

The prompt builder now intersects the simulator action mask with the selected
directive's executable agent and destination selectors. It emits compact
per-agent allowed destinations and a bounded set of complete candidate joint
actions. Single-assignment candidates preserve every valid agent/destination
choice, and one deterministic greedy multi-assignment candidate preserves a
coordinated option. The JSON schema requires `destinations` to equal one entire
candidate vector, preventing all-zero effect failures, illegal per-agent IDs,
and same-role duplicate destinations.

The prompt and actioner IDs, output paths, and prompt/template digests were
versioned so the failed preflight artifacts remain distinct.

## Results

- The regression test recreates seed `9300` and proves picker output positions
  3 and 4 permit only destination `0`; the prior invalid vector
  `[0, 0, 0, 3, 4]` is absent from the schema candidates.
- Twelve targeted prompt/config tests passed, and Ruff formatting/linting
  passed.
- An intermediate per-slot-enum preflight improved measured validity from
  `0/8` to `1/8` but still allowed cross-slot and all-zero effect failures.
- The final complete-candidate preflight completed warm-up and produced `8/8`
  valid measured responses with `0` invalid responses. Median latency was
  `0.9152 s`; p95 was `0.9782 s` against the `3.92 s` limit.
- The overall preflight still failed because state discovery covered only
  `service_requested_shelf`, found no sensitivity states, and therefore did not
  satisfy three-class coverage or sensitivity gates.

## Observations

The result separates model action validity from preflight state coverage. Once
the schema encoded complete safe actions, the pinned Qwen model stopped
producing invalid-action and effect-violation responses in the measured set.
The remaining failure occurred before delivery/reposition prompts could be
meaningfully exercised.

## Interpretation

The hypothesis was supported for semantic action validity. Deterministic
candidate construction is a more reliable boundary than asking a 2B model to
derive simulator eligibility from overloaded numeric fields. This does not
establish directive sensitivity or justify starting adaptation.

## Decision

Keep the safe-candidate action contract and keep the adaptation gate closed.
Treat three-class state generation and sensitivity coverage as a separate
preflight-harness problem.

## Problems and follow-up

1. Build deterministic preflight scenarios for each LLM-actioned directive.
2. Build three states with at least two applicable LLM directives for the
   sensitivity gate.
3. Re-run under a new preflight artifact path; start PPO adaptation only after
   the complete gate passes.
