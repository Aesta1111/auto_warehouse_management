# Project Roadmap

## How to use this roadmap

Work from top to bottom. A phase is complete only when its exit gate passes and
the evidence is linked from a research-log entry. Checkboxes describe outcomes,
not merely code written. Failed experiments remain documented.

## Phase 0: documentation foundation

- [x] Record the research specification and scientific controls.
- [x] Define the proposed architecture and dependency boundaries.
- [x] Define local, Git, RunPod, artifact, and experiment workflows.
- [x] Add a research-log format.
- [x] Create the public GitHub repository and configure the default branch.
- [ ] Complete dependency and model license review before adding MIT license.

Exit gate: a new contributor can locate the project intent, architecture,
workflow, roadmap, and experiment-recording rules from `README.md`.

## Phase 1: compatibility spike

- [x] Pin an immutable TA-RWARE commit and record its license.
- [x] Determine a working local Python and dependency set.
- [x] Run and render the smallest meaningful TA-RWARE episode.
- [x] Document observations, task-assignment actions, rewards, episode ending,
  valid-action handling, and multi-agent behavior.
- [x] Test deterministic reset and replay with fixed seeds.
- [x] Test simulator snapshot/restore; document the replay fallback if needed.
- [x] Verify Stable-Baselines3 and action-masking compatibility.
- [x] Lock the exact Hugging Face Gemma repository ID and revision.
- [ ] Test quantized inference, embedding injection, and one LoRA forward/backward
  step on target GPU hardware.
- [ ] Measure GPU memory, latency, and estimated training cost.
- [ ] Invoke and document the 3B-4B fallback if 12B violates the budget.

Exit gate: a dated compatibility report contains pinned dependencies, measured
constraints, a working simulator smoke command, and a working tiny model step.

Stop condition: do not scaffold the full hybrid around guessed APIs if this gate
has not passed.

## Phase 2: runnable project foundation

- [x] Add `pyproject.toml`, the `uv` lockfile, package metadata, and dependency
  groups.
- [ ] Add Typer command composition and YAML/Pydantic config loading.
- [ ] Add framework-independent domain records and controller protocols.
- [ ] Add the TA-RWARE adapter behind the project-owned interface.
- [ ] Add run manifests, seed handling, structured logging, and result schemas.
- [ ] Configure Ruff, basedpyright, pytest markers, and CPU GitHub Actions.
- [ ] Add `.gitignore`, `.env.example`, and artifact manifest conventions.
- [ ] Verify setup from a fresh clone on Windows and Linux/RunPod.

Exit gate: one documented CPU smoke command creates a valid run manifest and
result file from a fresh locked environment.

## Phase 3: reproducible baselines

- [ ] Implement and document the deterministic assignment heuristic.
- [ ] Add unit and integration coverage for eligibility and tie-breaking.
- [ ] Define disjoint training, validation, and final evaluation seed partitions.
- [ ] Train PPO and select a checkpoint using validation seeds only.
- [ ] Expose PPO latents, masked action scores, preferred action, and value.
- [ ] Freeze one PPO checkpoint for baseline, label generation, and hybrid use.
- [ ] Run heuristic and PPO evaluation under identical episode settings.

Exit gate: the heuristic and PPO reproduce per-seed result tables from clean CLI
commands, and checkpoint reload preserves deterministic evaluation behavior.

## Phase 4: rollout refinement dataset

- [ ] Predeclare candidate count, horizon, continuation controller, label score,
  and tie-breaking.
- [ ] Generate PPO top candidates plus the heuristic candidate.
- [ ] Implement state cloning or verified deterministic replay.
- [ ] Generate separate training and validation datasets.
- [ ] Store dataset manifests and large data in a private Hugging Face repository.
- [ ] Audit candidate disagreement, label balance, ties, failures, and measured
  short-horizon improvement over PPO's first choice.

Exit gate: dataset provenance is complete and rollout labels sometimes differ
from and outperform PPO's immediate choice under the declared proxy.

Stop condition: if labels never provide a better target than PPO, revise the
improvement mechanism before spending money on LLM training.

## Phase 5: hybrid training

- [ ] Implement and test the PPO-latent-to-Gemma projector.
- [ ] Implement compact state and eligible-assignment serialization.
- [ ] Configure quantized Gemma with trainable LoRA modules.
- [ ] Train only the projector and LoRA adapters.
- [ ] Implement structured decoding, validation, one retry, and PPO fallback.
- [ ] Track loss, assignment accuracy, JSON validity, rollout score, latency,
  memory, and cost.
- [ ] Select the checkpoint using validation data only.

Exit gate: the frozen hybrid completes full validation episodes without manual
repair, and every invalid decision or fallback is logged.

## Phase 6: locked final evaluation

- [ ] Freeze code, lockfile, configs, prompts, checkpoints, and seed lists.
- [ ] Evaluate heuristic, PPO, and hybrid on the same ten held-out seeds.
- [ ] Preserve all per-seed metrics and failed-run records.
- [ ] Generate paired statistical summaries, tables, and plots through code.
- [ ] Report throughput, efficiency, reliability, validity, and latency.
- [ ] Write a conclusion that preserves the declared primary metric.

Exit gate: one analysis command regenerates the science-fair tables and figures
from preserved results, whether the hybrid wins, loses, or is inconclusive.

## Phase 7: presentation

- [ ] Write plain-language explanations of PPO, latent projection, LoRA, and the
  controlled experiment.
- [ ] Prepare architecture and experiment-flow figures.
- [ ] Record a simulation video as a demonstration fallback.
- [ ] Prepare the report, poster, limitations, and AI-assistance disclosure.
- [ ] Rehearse a live or cloud-hosted simulation run.

Exit gate: the presentation explains the hypothesis, controls, evidence,
limitations, and reproducibility without claiming unsupported results.

## Stretch goals

Begin only after Phase 6 passes:

1. Compare small and medium warehouse scales.
2. Compare event-driven, fixed-interval, and every-step LLM decisions.
3. Compare additional LLM sizes or model families.
4. Add a playback dashboard for saved and live runs.
