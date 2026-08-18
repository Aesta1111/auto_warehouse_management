# 2026-08-18: Directive-demo Phase 3.5 repository-owned PPO

## Objective

Complete the isolated directive-demo Phase 3.5 gate from `PLAN.md`: replace
SB3's internally owned policy, rollout buffer, GAE, PPO loss, collection, and
optimizer loop with separately tested project modules and an explicit Lightning
Fabric loop. Preserve the Phase 3 directive MDP, seed partitions, validation
rule, and checkpoint as an immutable reference. This does not advance the main
research phases or test the Gemma actioner.

## Hypothesis

The repository-owned PPO implementation will produce a digest-verified
schema-v2 checkpoint that selects at least two non-wait directives with zero
inapplicable selections and reproduces its deterministic validation behavior
after reload. Exact equality with or improvement over SB3 is not expected.

## Provenance

- Git commit: `b31a420148ac524f3420c409712b284e10d335e7`
- Dirty worktree: yes; the Phase 3.5 implementation and documentation are the
  relevant changes
- Command: `.venv\\Scripts\\python.exe scripts\\train_directive_ppo.py --config configs\\demo\\directive-ppo-v2.yaml`
- Config path: `configs/demo/directive-ppo-v2.yaml`
- Config SHA-256: `72dc18fcdefdf5fbee2f2ec3cbf8ee04bdbc55ca1c4641d7fe823ff1ac0efd6e`
- Vocabulary SHA-256: `b1182beaebab4bb1f7e43673a04b856ea487b4457d47bf973e0b421bf6fdb0f8`
- Behavior SHA-256: `ab33d4de834a76a49962231f8b50643d3d66c081e56cdbfd09ad0192276b64db`
- Run ID: `directive-ppo-v2` under `results/demo/directive-ppo-v2/`
- Training seeds: `8100`-`8115` (prototype-only)
- Validation seeds: `8200`-`8203` (prototype-only)
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- PPO checkpoint SHA-256: `3dfa1c413b4e6ac0d1869dd55caafd16e715fdec3d6092ae41ab68e02a623069`
- Model architecture SHA-256: `770a9896a77afedb354b7e50ac50c2f17212bc3bdaf78c492d1fe1e58e84d975`
- Implementation SHA-256: `749bf2885a43be22f6339c7ef559e43ef86cb7d02ee1576f5c6aba38fe1f293d`
- Python 3.12, Windows local environment, Torch 2.8.0 CPU build,
  Lightning 2.5.6; no GPU job
- Final provenance-corrected run start/end: approximately
  2026-08-18T03:14Z-03:16Z
- Estimated cost before run: USD 0
- Actual cost: USD 0

## Method

Added four dependency-bounded modules. `ppo_math.py` contains pure GAE and
clipped-surrogate loss arithmetic. `ppo_buffer.py` owns preallocated storage and
deterministic minibatches. `ppo_model.py` owns separate 64-unit tanh policy and
value paths with exact masked probabilities. `ppo_training.py` owns stateful
collection and one long-lived optimizer wrapped by public Lightning Fabric
device, backward, clipping, and checkpoint operations. The stable `ppo.py`
facade dispatches schema v1 to the preserved SB3 path and schema v2 to the new
implementation.

The config retained the 25,000 declared timestep budget, 256-step rollouts,
ten epochs, batch size 64, episode length, seeds, and validation ranking from
Phase 3. Completed-update checkpointing required 98 full rollouts, so the
manifest records 25,088 actual environment steps and 3,920 optimizer steps.
The fixed validation rule selected the checkpoint produced at 5,120 environment
steps and 800 optimizer steps; the manifest records both selection-time and
completed-run totals.
GAE uses separate bootstrap and trace masks: termination disables both,
truncation-only bootstraps but stops its trace, and an unfinished rollout
boundary bootstraps normally.

## Results

The [checkpoint manifest](../../results/demo/directive-ppo-v2/directive-ppo-manifest.json)
records the complete run provenance and checkpoint digest.

- Mean completed tasks: `2.75`.
- Mean return: `0.5500000000000389`.
- Directive counts: deliver-carried-shelf `1437`,
  service-requested-shelf `12`, wait `551`.
- Distinct non-wait directives used: `2`.
- Inapplicable selected directives: `0`.
- Masked directive entries observed: `4547`.
- Mean policy entropy: `0.4688378464338788`.
- Two independent checkpoint loads reproduced the same directive,
  probabilities, value estimate, and exact zero masked probabilities for a
  fixed seed-8200 state.
- Re-evaluation reproduced the stored summary byte-for-value on all four
  validation seeds, and a second evaluation matched the first.
- A second full-budget run reproduced the exact checkpoint SHA-256 and
  validation summary while adding unambiguous checkpoint-selection provenance.
- Final repository gates passed: Ruff format, Ruff check, basedpyright with
  zero errors, and `52` tests.

For descriptive comparison, the Phase 3 SB3 reference completed `3.0` tasks on
average with mean return `0.8000000000000443`, directive counts `1752/12/236`
for deliver/service/wait, and mean entropy `0.6299958944219024`. The v2 policy
was slightly worse on throughput and return and waited more often. Both used two
non-wait directives and made zero inapplicable selections. This comparison is
not a statistical claim and did not influence the fixed Phase 3.5 gate.

## Observations

The first full-budget attempt stopped before its first optimizer update because
TA-RWARE can report simulator termination on the same step that the project
episode limit reports truncation. The initial buffer incorrectly rejected that
legal pair of independent Gymnasium flags. The correction gives termination
precedence: no bootstrap and no recursive trace. A regression test covers the
combined boundary. The successful run then completed without manual
intervention.

Fabric's module wrapper also attempts to instrument tensors inside forward
outputs and cannot mutate the planned frozen output dataclass. The implementation
kept the immutable record contract and invokes custom model methods through
Fabric's public wrapped-module handle; Fabric still owns device setup, backward,
gradient clipping, and checkpoint serialization. No private Lightning loop API
is used.

## Interpretation

The evidence supports the narrow Phase 3.5 hypothesis. The project now owns and
tests the algorithmic pieces that Phase 4 will depend on, and the v2 checkpoint
is reproducible under the fixed validation protocol. Its lower descriptive
score is acceptable and is reported rather than tuned away. These prototype
seeds and results remain excluded from the main scientific comparison.

## Decision

The isolated directive-demo Phase 3.5 gate passes. Phase 4 must consume only the
schema-v2 checkpoint and manifest. Preserve the Phase 3 SB3 artifacts as the
reference and do not convert or overwrite them. Do not mark the main roadmap
Phase 3 complete.

## Problems and follow-up

The ignored checkpoint file must accompany its committed manifest when running
later local demo phases. The prepared main-project Linux/A40 compatibility run
remains independently deferred.
