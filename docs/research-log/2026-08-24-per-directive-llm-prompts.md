# 2026-08-24: Per-directive LLM prompt contract

## Objective

Split the active Qwen few-shot prompt so each LLM-actioned directive receives
only its own instructions and examples. Keep `wait` outside the LLM path because
the shared execution service already handles it deterministically.

## Hypothesis

Not applicable for this contract refactor. No model or PPO experiment was
authorized or interpreted for this change.

## Provenance

- Git commit: `b339bcd` for the implementation checkpoint; documentation was
  updated afterward in the same worktree.
- Dirty worktree: yes; this research-log entry and owning documentation were
  pending when recorded.
- Commands: automated pytest, Ruff, and basedpyright checks only after the user
  instructed that no experiment runs should be performed.
- Config path: `configs/demo/llm-actioner-few-shot-prompt.yaml`, schema v3.
- Prompt templates:
  - `configs/demo/llm-actioner-service-requested-shelf-prompt.txt`
  - `configs/demo/llm-actioner-deliver-carried-shelf-prompt.txt`
  - `configs/demo/llm-actioner-reposition-idle-agent-prompt.txt`
- Per-template SHA-256:
  - service: `8ff7003d3c2b43f9a0da0d80167aed3c0967db84305c47b47dc53686d8aad706`
  - delivery: `cfdb60eb534a327f7ea87a38aa307830534e38f0815d05605c684d87eb1af40e`
  - reposition: `cb0544e1b9a9fb9455cd81f35c3d07400f94b58387f7b40b6467928f8a603f7f`
- Aggregate template SHA-256:
  `03ce2b4e72d07f77e4d1a7967859c7d3201fcbd39133b49b1d5746dbb58135ec`.
- Prompt contract SHA-256:
  `52126c0cf1681428f962272566cae6ecf8bba1b61083d673827ed55166dc3bd8`.
- Run IDs: none.
- Seed set and partition: none used by this refactor.
- TA-RWARE commit: unchanged from the pinned project record.
- PPO checkpoint digest: none.
- Hugging Face model: unchanged pinned Qwen3.5-2B Q8 artifact; no inference was
  performed for this change.
- Hardware and driver: unchanged local RTX 4050 setup; no measured result.
- Start/end time in UTC: not recorded for the source-only refactor.
- Estimated cost before work: $0.
- Actual cost: $0 local development.

## Method

Schema v3 replaces the active shared TXT path with an ordered behavior-key to
TXT-path mapping. Each of the three LLM-actioned templates contains directive-
specific rules, exactly three varied examples, the common strict-output rules,
and the live-state markers. The builder loads only the template matching the
selected directive and hashes both each exact file and the ordered template
bundle. The contract digest covers the YAML metadata and exact bundle bytes.

`wait` is declared in `deterministic_behavior_keys` and has no prompt file. A
direct builder call for `wait` fails clearly, while normal execution continues
to bypass the actioner and use the deterministic all-no-op directive executor.
The live preflight gate now covers the three directive classes that actually
call the LLM; ordinary execution tests cover the deterministic wait path.

## Results

- Service, delivery, and reposition requests each render only their own three
  examples.
- No wait examples or unrelated directive examples appear in a rendered prompt.
- Prompt markers, command/catalog alignment, example counts, template mapping,
  aggregate digest, and per-template digests are validated fail-closed.
- The full automated suite passed: `99 passed`.
- Ruff passed and basedpyright reported `0 errors, 0 warnings, 0 notes`.
- No PPO, Qwen preflight, matched demo, adaptation, or evaluation result was
  produced.

Before the no-run instruction, the configured 100,000-step parent command had
been started briefly. It was stopped, and the exact new directory
`results/demo/ppo-training-100k-4x256-cuda` was verified and removed. No
manifest survived, no result was interpreted, and historical result directories
were untouched.

## Observations

- The prior shared prompt sent twelve examples even though only one non-wait
  directive was selected.
- The production execution service already bypassed the LLM for `wait`, so a
  wait-specific example file would have described an unreachable model call.

## Interpretation

The new contract is smaller and more directive-focused while retaining strict
state serialization and validation. Automated checks establish contract and
rendering behavior only; they do not show that Qwen action quality or latency
improved.

## Decision

Use three directive-specific prompt files and keep `wait` deterministic. Treat
the schema-v2 shared prompt and its digests as historical evidence only. Do not
perform experiment runs until the user explicitly requests them.

## Problems and follow-up

- The next live Qwen preflight must generate a new schema-v5 report because the
  prompt contract and aggregate template digest changed.
- The 100,000-step parent, matched demo, 4,096-step adaptation, and gated
  evaluations remain unrun.
- If a later design routes `wait` through the LLM, add a separately versioned
  wait template and restore wait-specific preflight coverage deliberately.
