# 2026-08-13: Directive-demo Phase 2 semantics

## Objective

Freeze the isolated directive-demo catalog and verify that its deterministic
applicability, executors, and action-effect validators produce strict TA-RWARE
joint actions. This investigation can change the prototype catalog before
its later PPO phase; it does not change the main research phase gates.

## Hypothesis

The four retained directive concepts can be expressed using only the shared
controller-visible observation and action mask, with wait always applicable and
no executor requiring hidden simulator state.

## Provenance

- Git commit: `94193e788fd23870d40eb15a61f25a5681680715`
- Dirty worktree: yes; this entry records the Phase 2 implementation
- Command: `python scripts/probe_directives.py --config configs/demo/directive-probe-v1.yaml`
- Config path: `configs/demo/directive-probe-v1.yaml`
- Vocabulary: `configs/demo/directives-v1.yaml`, behavior version `directives-v1`
- Seed set: `8000, 8001, 8002, 8003, 8004` (prototype-only)
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- Hardware: Windows local development environment, Python 3.12
- Cost: USD 0

## Method

Implemented `DirectiveBehavior`, `Registry`, explicit semantic error
types, and one reviewed behavior class per frozen YAML command. The executors
use canonical agent order, lowest legal destination ID, same-role duplicate
avoidance, and the existing `TarwareEnvWrapper.decode()` validator. Requested
shelves, carried AGVs plus legal goals, and non-requested legal destinations
define the three non-wait effects. The probe checks every currently applicable
directive twice at each state before advancing the real wrapper.

## Results

The report is [directive-probe-report.json](../../results/summaries/directive-probe-report.json).

- States checked: 2,500 across five 500-step seeds.
- Uncovered states: 0.
- Executor failures: 0.
- Effect failures: 0.
- Report status: passed.
- Vocabulary SHA-256: `b1182beaebab4bb1f7e43673a04b856ea487b4457d47bf973e0b421bf6fdb0f8`.
- Behavior SHA-256: `ab33d4de834a76a49962231f8b50643d3d66c081e56cdbfd09ad0192276b64db`.

Unit tests cover masking, strict decoding, deterministic repeated execution,
unknown/inapplicable IDs, effect rejection, executor wrapping, and behavior
digest sensitivity.

## Observations

Wait was applicable at all sampled states. Requested-shelf applicability was
observed on the committed trajectories. Carried-shelf and reposition behavior
implementations are present and are validated whenever their state predicates
are satisfied; the probe's pass criterion is zero failures and does not treat
directive frequency as a performance result.

## Interpretation

The four-command contract is implementable without duplicating the simulator
action validator or exposing `wrapper.unwrapped` to behaviors. The behavior
digest conservatively changes when bindings, concrete behavior classes, or the
directive module implementation changes, so an old prototype checkpoint cannot
silently be paired with changed semantics.

## Decision

The prototype Phase 2 gate passes. Freeze `directives-v1.yaml` and behavior
version `directives-v1` before any prototype PPO work. Keep the prototype
unscored and keep its seeds permanently excluded from the main experiment's
training, validation, and final evaluation partitions.

## Problems and follow-up

The real Gemma actioner, directive PPO, and end-to-end demonstration remain
unimplemented. Do not start those phases until this registry/API is reviewed as
the frozen prototype contract.
