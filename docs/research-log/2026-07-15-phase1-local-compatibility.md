# 2026-07-15: Phase 1 local compatibility handoff

## Objective

Complete every locally testable Phase 1 check and prepare, but do not claim to
execute, the authoritative Linux and A40 checks.

## Hypothesis

TA-RWARE would replay exactly from fixed seeds, MaskablePPO would accept a thin
centralized adapter, and a tiny Gemma 4 model would propagate gradients through
projected tokens and LoRA. Snapshot support was uncertain.

## Provenance

- Base project Git commit: `c3fea49`
- Dirty worktree: yes; this entry describes the uncommitted compatibility slice
- TA-RWARE: `6109c33f7228cace1e463c81f0568b754ff5f198`
- Local pathfinder: `pyastar2d==1.1.4`
- Python: 3.12
- Host: Windows; NVIDIA RTX 4050 Laptop GPU, 6 GB; driver 580.97
- Hugging Face primary: `google/gemma-4-12B-it` at
  `0e2b1058541244490925fbacf8972041435691ac`
- Hugging Face fallback: `google/gemma-4-E4B-it` at
  `a4c2d58be94dda072b918d9db64ee85c8ed34e3f`
- Commands: the three `scripts/probe_*` local commands documented in `README.md`
- Cost: USD 0

## Method

The upstream heuristic was run twice for each seed 0 through 4. Every action,
observation, reward, termination flag, truncation flag, and info value was
canonicalized and hashed. At steps 100, 250, and 400, a fresh reset plus action
prefix and a pickled clone each received the same 50-action suffix.

A centralized Gymnasium adapter exposed one 125-value observation,
`MultiDiscrete([55] * 5)`, and a flattened 275-value action mask. MaskablePPO
trained for 512 timesteps, then saved and reloaded. Since component-wise masks
cannot express same-role conflicts, later duplicates were changed to no-op and
recorded.

The Gemma test used PyTorch 2.8.0, Transformers 5.13.1, PEFT 0.19.1,
bitsandbytes 0.49.2, and Accelerate 1.14.0. A random two-layer Gemma 4 model
received four projected soft tokens and rank-8 LoRA on the attention projections.
Gemma 4's per-layer token embeddings were explicitly supplied and zeroed at the
soft-token positions.

## Results

- Simulator summary: `results/summaries/phase1-tarware-local.json`
- SB3 summary: `results/summaries/phase1-sb3-local.json`
- Tiny Gemma summary: `results/summaries/phase1-gemma-tiny-local.json`
- All five 500-step traces repeated exactly, with distinct stable hashes per
  seed. Reset-plus-prefix replay matched all three 50-step suffixes.
- Pickled clones were independent but all three suffixes diverged. At checkpoint
  100 the first divergence appeared 20 actions after restore, in observations.
- The rendered seed-0 episode opened, advanced for 500 steps, and closed with 23
  deliveries, 9 clashes, 0 stuck events, and return 25.
- MaskablePPO completed 512 steps. One same-role conflict was logged. Save/reload
  preserved deterministic action `[27, 24, 53, 0, 0]`.
- The tiny Gemma loss was finite (4.2173). The base stayed frozen; projector and
  LoRA gradients were finite and nonzero, and both trainable components updated.

## Observations

Successful pickle/deepcopy construction did not imply a behaviorally valid
snapshot. SB3's per-component mask also did not encode the complete joint-action
contract. Gemma 4 accepted arbitrary embeddings only when explicit per-layer
inputs were supplied.

## Interpretation

Deterministic replay is suitable as the provisional rollout-state reconstruction
method. The SB3 adapter is feasible if repairs remain visible in results. The
architecture-level Gemma gradient route is feasible locally, but this result says
nothing yet about 12B quantized memory, speed, output quality, or cost.

## Decision

Select seed-plus-action-prefix replay and reject simulator snapshots. Lock the
two Gemma 4 revisions and the ML stack. Keep Phase 1 open until the prepared A40
and upstream-Linux pathfinder runs succeed. Preserve at least USD 15 of the USD
50 total budget for evaluation and reruns; keep 12B only if projected total
hybrid work remains at or below USD 35.

## Problems and follow-up

1. Run `scripts/run_phase1_cloud.sh` in the pinned RunPod A40 image.
2. Compare the Linux upstream-pathfinder hashes with the committed Windows hashes.
3. Record real 4-bit memory, latency, JSON validity, wall time, and pod cost.
4. Use E4B only if the 12B fit or budget rule fails.
