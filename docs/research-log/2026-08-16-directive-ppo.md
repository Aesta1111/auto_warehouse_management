# 2026-08-16: Directive-demo Phase 3 PPO

## Objective

Implement and verify the isolated directive-PPO phase from `PLAN.md`: expose
the four frozen directives as a one-dimensional masked SB3 action space, train
with deterministic directive executors, and preserve validation behavior after
checkpoint reload. This investigation does not advance the main reproducible-
baselines phase or test the Gemma actioner.

## Hypothesis

The frozen directive catalog and strict executor registry will support a
small MaskablePPO policy that selects at least two non-wait directives on the
prototype validation seeds without submitting an inapplicable directive.

## Provenance

- Git commit: `d8c67a82921153173b303d8d9946c30dfc7caa83`
- Dirty worktree: yes; the Phase 3 implementation was the relevant change
- Command: `uv run python scripts/train_directive_ppo.py --config configs/demo/directive-ppo-v1.yaml`
- Config path: `configs/demo/directive-ppo-v1.yaml`
- Config SHA-256: `d56baf3091c61a8d3c0ef0a9dab89ccf8c3528a7e65352710d013356f77439c4`
- Vocabulary SHA-256: `b1182beaebab4bb1f7e43673a04b856ea487b4457d47bf973e0b421bf6fdb0f8`
- Behavior SHA-256: `ab33d4de834a76a49962231f8b50643d3d66c081e56cdbfd09ad0192276b64db`
- Run ID: `directive-ppo-v1` under `results/demo/directive-ppo-v1/`
- Training seeds: `8100`-`8115` (prototype-only)
- Validation seeds: `8200`-`8203` (prototype-only)
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- PPO checkpoint SHA-256: `2a0ff875fb1b0f584b72cf5b274df8354f3d62f6720dbe9350890037377a266f`
- Python 3.12, Windows local environment, Torch 2.8.0 CPU build; no GPU job
- Start/end time in UTC: approximately 2026-08-16T12:56Z-12:58Z
- Estimated cost before run: USD 0
- Actual cost: USD 0

## Method

Added `MaskAwareDiscrete`, `Sb3Env`, deterministic masked
`PpoPolicy` inference, validation summaries, digest-checked manifests,
and the thin `train_directive_ppo.py` command. MaskablePPO used one 64-unit
policy/value hidden layer, the committed 25,000-timestep budget, and validation
selection by mean completed tasks followed by mean return. The environment
delegated all concrete-action validation to `TarwareEnvWrapper.decode()` and
failed on masked or executor-invalid directives rather than repairing them.

## Results

The [checkpoint manifest](../../results/demo/directive-ppo-v1/directive-ppo-manifest.json)
records the complete run provenance and the checkpoint digest.

- Training completed for 25,000 timesteps.
- Mean completed tasks: `3.0`.
- Mean return: `0.8000000000000443`.
- Directive counts: deliver-carried-shelf `1752`, service-requested-shelf `12`,
  wait `236`.
- Distinct non-wait directives used: `2`.
- Inapplicable selected directives: `0`.
- Masked directive entries observed: `4109`.
- Mean policy entropy: `0.6299958944219024`.
- Checkpoint reload reproduced the same selected directives, probabilities,
  validation counts, completed-task mean, and return on the same validation
  seeds.

Focused Phase 3 tests and the existing Phase 1-2 tests passed: `19 passed`.
Ruff check and format checks passed. The repository-wide basedpyright command
still reports six pre-existing test annotation errors in
`tests/demo/test_models.py`; no Phase 3 source error was reported.

## Observations

The selected policy used deliver-carried-shelf most often, with a small number
of service-requested-shelf decisions and wait decisions. The validation mask
prevented all inapplicable directive selections. Reloading the saved checkpoint
on CPU preserved the exact probability tuple and deterministic argmax for a
fixed observation-mask pair.

## Interpretation

The evidence supports the narrow prototype hypothesis: the directive PPO
contract is trainable and reloadable under the declared executor and masking
rules. It does not show that the directive demo improves warehouse throughput,
that PPO is a scientific baseline, or that Gemma can preserve directive effects.
The policy was trained and evaluated on prototype-only seeds and must remain
excluded from the main experiment's seed partitions.

## Decision

The isolated directive-demo Phase 3 gate passes. Freeze the Phase 3 public
interfaces, checkpoint manifest, and seed exclusions. Do not mark the main
roadmap Phase 3 complete. The next prototype phase, if explicitly authorized,
is the actioner-neutral fake-client orchestration phase; no Gemma work was
performed here.

## Problems and follow-up

The repository-wide basedpyright baseline remains noisy because of six existing
test annotations; clean those separately before relying on the type-check gate.
The checkpoint is intentionally ignored by Git, while the small manifest is
kept as the reproducibility record. A later actioner phase must verify its own
preflight and must not alter this PPO checkpoint or directive catalog.
