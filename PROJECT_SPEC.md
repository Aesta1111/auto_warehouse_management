# Auto Warehouse Management: Project Specification

## 1. Document status

- Status: active specification based on the project discovery interview and
  workflow-design decisions.
- Audience: the student author, science-fair judges, and coding agents working in later sessions.
- Repository state when written: empty except for Git metadata.
- Simulator: Task Assignment Robotic Warehouse (TA-RWARE), from
  <https://github.com/uoe-agents/task-assignment-robotic-warehouse>.
- Compatibility status: TA-RWARE is pinned and its local Windows interface has
  been verified. The Gemma 4 model identities are pinned, and a random tiny
  Gemma 4 model has verified the embedding/LoRA gradient path. Linux pathfinder
  equivalence and the real quantized model run remain deferred cloud checks.

This document is the source of truth for project intent. Future changes should update this file rather than relying on conversation history.

Supporting documents define implementation boundaries and working procedure:

- [Architecture](docs/ARCHITECTURE.md)
- [Development and experiment workflow](docs/WORKFLOW.md)
- [Project roadmap](ROADMAP.md)
- [Research log](docs/research-log/README.md)

## 2. Project summary

The project investigates whether a hybrid controller can improve warehouse task assignment over a PPO policy. PPO learns a compact representation of the simulated warehouse state. A learned projector maps the PPO latent representation into soft prompt tokens for an instruction-tuned Gemma model. The LLM also receives a compact human-readable state and explicit validity constraints, then returns structured robot-to-task assignments.

The main scientific comparison is:

1. A deterministic rule-based assignment heuristic.
2. PPO without an LLM.
3. The proposed PPO-latent-plus-LLM hybrid.

The primary outcome is task throughput: tasks completed during equal-length episodes. The project must report negative results honestly. The hybrid succeeding is a hypothesis, not a required outcome.

## 3. Research question and hypothesis

### Research question

Under held-out task sequences in TA-RWARE, does a Gemma task-assignment controller conditioned on projected PPO latent features outperform the original PPO controller in completed tasks per fixed-length episode?

### Hypothesis

The PPO representation provides a useful learned prior about warehouse dynamics, while the LLM can combine that prior with explicit global state and assignment constraints. Training the connector on assignments selected through short simulator lookahead may produce better task assignments than PPO's immediate preference.

### Null hypothesis

The hybrid does not improve mean task throughput over PPO on held-out task sequences. It may match PPO, perform worse, fail more often, or add unacceptable latency.

### Scientific caution

Training the LLM only to imitate PPO cannot justify an improvement claim. The improvement signal must come from short-horizon simulator rollouts that compare multiple candidate assignments. Published results from other projects are background context, not direct baselines unless their environment, workload, episode definition, and evaluation protocol are reproduced locally.

## 4. Scope and priorities

### Core scope

- Simulation only; no physical robots.
- Task assignment is learned; low-level navigation remains TA-RWARE's responsibility.
- One small, standard TA-RWARE configuration for the primary experiment.
- Held-out task sequences/random seeds; unseen layouts are not part of the core claim.
- Three controllers: heuristic, PPO, and hybrid.
- Event-driven LLM decisions when a task assignment is required.
- A fixed Gemma-family model, projected latent tokens, and LoRA adapters.
- CLI commands plus configuration files for reproducible runs.
- Local CSV/JSON results, plots, configs, and small summaries stored in Git.
- Ten independent held-out evaluation seeds per core controller.

### Stretch goals, in priority order

1. Evaluate small and medium warehouse scales.
2. Compare event-driven decisions with fixed-interval and every-step LLM calls.
3. Compare additional LLM sizes or model families.
4. Add a playback dashboard with animation, controller selection, metrics, assignments, and saved rationales.

Stretch work must not begin until the core experiment runs end to end, produces reproducible result files, and can be rerun from documented commands.

### Out of scope for the first version

- Physical warehouse hardware.
- Low-level LLM robot movement.
- Full-parameter LLM fine-tuning.
- End-to-end reinforcement learning of the LLM from episode reward.
- Unseen-layout generalization as a primary claim.
- Treating metrics chosen after seeing results as primary outcomes.

## 5. System design

### 5.1 Environment adapter

Create a thin adapter around TA-RWARE that exposes a stable project-owned interface:

- Reset with an explicit seed and configuration.
- Read the current task-assignment state.
- Enumerate eligible robot-task assignments.
- Apply a validated assignment.
- Advance the simulator using the configured navigation behavior.
- Detect assignment events, episode termination, deadlocks, and simulator errors.
- Emit step-level and episode-level metrics.

Do not spread TA-RWARE-specific data structures throughout the project. The initial compatibility spike must determine its actual observation space, action space, reward semantics, rendering support, and package installation method.

### 5.2 Rule-based controller

Implement one deterministic heuristic using information available to every controller. The specific scoring formula should be chosen only after inspecting TA-RWARE, but it must be simple, documented, deterministic under a seed, and free of privileged future information. Likely inputs are robot availability, task eligibility, and path or Manhattan distance.

The heuristic serves two roles:

- A scientific baseline.
- One alternative candidate during short-horizon label generation.

### 5.3 PPO controller

Train PPO for task assignment using a fixed training-seed set. PPO is an optimization algorithm, not itself an encoder. The encoder is the policy network component that converts the environment observation into a latent vector.

The trained checkpoint must expose:

- The frozen latent representation at each assignment event.
- Valid-action-masked assignment scores or probabilities.
- The preferred assignment.
- A value estimate when available.

Use the same frozen PPO checkpoint for the PPO baseline, hybrid conditioning, and candidate generation. This prevents separate PPO training variation from contaminating the comparison.

### 5.4 Rollout label generator

At selected training states:

1. Obtain PPO's top valid assignment candidates.
2. Add the deterministic heuristic candidate if it is not already present.
3. Clone or restore the simulator state for each candidate.
4. Execute that candidate.
5. Continue for the same fixed short horizon with the same deterministic continuation policy.
6. Score each candidate using a predefined label-generation score.
7. Store the highest-scoring valid candidate as the supervised target.

The lookahead horizon, number of PPO candidates, continuation policy, tie-breaking rule, and scoring formula must be fixed in configuration before generating the final dataset. Label generation may use throughput-related proxy terms, but final scientific evaluation uses the separately declared metrics below.

Training, validation, and final evaluation seeds must be disjoint. No final evaluation state may be used for PPO training, projector/LoRA training, prompt selection, checkpoint selection, or hyperparameter tuning.

### 5.5 Hybrid controller

The hybrid contains:

- A frozen PPO policy encoder.
- A trainable projector that maps one PPO latent vector into a fixed number of Gemma embedding vectors (soft prompt tokens).
- A frozen quantized Gemma instruction model.
- Trainable LoRA adapters on a limited, explicitly configured set of Gemma modules.
- A compact textual state summary and an explicit list or mask of eligible assignments.
- A structured output validator and deterministic fallback.

The model output should be JSON equivalent to:

```json
{
  "assignments": [
    {"robot_id": "robot-0", "task_id": "task-3"}
  ],
  "reason": "Short explanation"
}
```

The exact identifiers and whether multiple assignments can occur in one event depend on TA-RWARE's API. The validator must reject unknown robots, unknown tasks, duplicate assignments, unavailable robots, ineligible pairs, malformed output, and assignments that are no longer valid.

On invalid output, retry at most once with a compact validation error. If the retry fails, use PPO's highest-ranked currently valid assignment. Log the failure and fallback; never silently repair an output in a way that could bias results.

### 5.6 Model dependency

The primary model is `google/gemma-4-12B-it` at immutable revision
`0e2b1058541244490925fbacf8972041435691ac` (Apache-2.0). The smaller fallback
is `google/gemma-4-E4B-it` at revision
`a4c2d58be94dda072b918d9db64ee85c8ed34e3f`. The primary remains conditional
on the deferred target-GPU memory, latency, and cost measurements.

Gemma 4 uses additional token-identity embeddings at every transformer layer.
Arbitrary projected `inputs_embeds` therefore must be paired with explicit
`per_layer_inputs`; the soft-token positions use zero per-layer identity inputs.
This route produced projector and LoRA gradients in a tiny random model. The
authoritative run will use NF4 double quantization, BF16 compute, and rank-8
LoRA on the attention `q`, `k`, `v`, and `o` projections.

Before dataset generation, run a compatibility spike that locks:

- Exact Hugging Face repository ID and immutable revision.
- License acceptance and science-fair redistribution/display constraints.
- Quantization format and inference library.
- Availability of embedding-level soft prompt injection.
- LoRA compatibility with quantized weights.
- GPU memory required for training and inference.
- Structured-output validity and decision latency.

If 12B training cannot fit the time or budget, keep the same architecture and fine-tune a 3B-4B model. Prompt-only inference is not the selected fallback. Record any model substitution before final evaluation.

## 6. Metrics and evaluation protocol

### Primary metric

- Tasks completed per fixed-length episode.

### Secondary metrics

- Travel distance per completed task.
- Robot idle time.
- Collisions, deadlocks, or the closest TA-RWARE equivalents.
- Invalid LLM output count and rate.
- PPO fallback count and rate.
- Assignment-decision latency.
- End-to-end episode wall-clock time.

### Protocol

- Use one frozen configuration and ten held-out task-sequence seeds per core controller.
- Give all controllers identical environment configuration, episode length, and evaluation seeds.
- Run evaluation without training or manual intervention.
- Store per-seed results, not only aggregate values.
- Report mean, median, standard deviation, and a confidence interval for the primary metric.
- Prefer paired comparisons because each controller sees the same seeds.
- Show every seed in plots when practical; do not hide outliers without a documented simulator failure.
- Report LLM latency and invalid-output behavior even if the hybrid has higher throughput.
- Separate exploratory results from the locked final evaluation.

Do not combine metrics after observing results to manufacture a favorable score. A composite metric may be explored only if its formula and weights are declared before final evaluation, and it remains secondary.

## 7. Workflow

### Phase 1: compatibility spike

1. Pin a TA-RWARE commit and document installation.
2. Run its smallest environment and render one episode.
3. Record observations, actions, assignment events, rewards, and state-cloning support.
4. Verify cloud training and local development environments.
5. Lock the Gemma checkpoint or invoke the smaller-model fallback.
6. Prove that projected embeddings and quantized LoRA can complete one forward/backward step.

Exit criterion: a written compatibility report and a tiny smoke run. Do not build the full hybrid before this passes.

### Phase 2: reproducible baselines

1. Implement configuration loading, seed handling, environment adapter, metrics, and CLI.
2. Implement and evaluate the deterministic heuristic.
3. Train PPO, select a checkpoint using validation seeds, and freeze it.
4. Re-run both controllers from clean commands and confirm deterministic evaluation under fixed seeds.

Exit criterion: heuristic and PPO result CSVs for a small configuration.

### Phase 3: refinement dataset

1. Lock candidate count, rollout horizon, continuation policy, and label score.
2. Generate training and validation states from their assigned seed sets.
3. Produce PPO top candidates plus the heuristic candidate.
4. Run short lookaheads, select labels, and save dataset metadata.
5. Audit class balance, invalid states, ties, and whether labels actually differ from PPO.

Exit criterion: a versioned dataset manifest and evidence that rollout labels sometimes improve on PPO's first choice. If they never differ, stop and revise the scientific mechanism before LLM training.

### Phase 4: hybrid training

1. Train the projector and LoRA adapters while freezing PPO and base Gemma weights.
2. Select checkpoints only on validation data.
3. Track training loss, assignment accuracy, JSON validity, and validation rollout score.
4. Test deterministic decoding settings and fallback behavior.

Exit criterion: the hybrid completes full validation episodes without manual repair.

### Phase 5: locked evaluation and analysis

1. Freeze code, configs, checkpoints, prompts, and evaluation seeds.
2. Run heuristic, PPO, and hybrid on the same ten held-out seeds.
3. Generate tables and plots from raw result files with one analysis command.
4. Interpret both positive and negative results, including latency and reliability tradeoffs.

Exit criterion: raw results, generated figures, a statistical summary, and a reproducible command log.

### Phase 6: science-fair presentation

1. Write a plain-language explanation of PPO, latent projection, LoRA, and controlled evaluation.
2. Prepare an architecture diagram and experiment-flow diagram.
3. Record a simulation video as presentation backup.
4. Add the playback dashboard only if all core exit criteria are already satisfied.
5. Maintain an AI-assistance and external-service disclosure log.

## 8. Proposed repository structure

```text
auto_warehouse_management/
|-- AGENTS.md
|-- PROJECT_SPEC.md
|-- README.md
|-- pyproject.toml
|-- configs/
|   |-- env/
|   |-- ppo/
|   |-- hybrid/
|   `-- experiments/
|-- src/warehouse_ai/
|   |-- domain/
|   |-- env/
|   |-- controllers/
|   |-- ppo/
|   |-- hybrid/
|   |-- data/
|   |-- evaluation/
|   |-- config/
|   `-- cli.py
|-- tests/
|-- notebooks/
|-- scripts/
|-- results/
|   |-- summaries/
|   `-- figures/
`-- docs/
```

Notebooks are for exploration and presentation, not the authoritative training pipeline. Commands and config files must reproduce important results.

Large raw datasets, downloaded base models, checkpoints, caches, and full trajectory logs should be excluded from Git. Git should contain code, configurations, dataset manifests, small aggregate results, figures, and documentation.

The selected foundation stack is:

- `uv`, `pyproject.toml`, and a committed lockfile for Python environments.
- Typer for the command-line interface.
- YAML inputs validated by strict Pydantic schemas.
- Stable-Baselines3 for PPO, subject to the compatibility spike.
- Ruff, standard-mode basedpyright, `jaxtyping` at tensor boundaries, and pytest.
- CPU quality and smoke checks in GitHub Actions; GPU smoke checks run locally
  or on RunPod and are recorded in pull requests and research-log entries.

Conda is a fallback only if TA-RWARE has native dependencies that cannot be
managed reliably through `uv`.

## 9. CLI and artifact conventions

The precise CLI framework can follow the eventual Python stack, but the behavioral interface should cover:

```text
warehouse-ai smoke-test --config <path>
warehouse-ai train-ppo --config <path>
warehouse-ai generate-labels --config <path>
warehouse-ai train-hybrid --config <path>
warehouse-ai evaluate --config <path> --controller <heuristic|ppo|hybrid>
warehouse-ai analyze --experiment <id>
warehouse-ai replay --run <id>
```

Every run should write:

- Unique run ID and timestamp.
- Git commit and dirty-worktree flag.
- Full resolved configuration.
- Dependency and hardware summary.
- Seed list.
- Checkpoint/model revisions.
- Per-episode CSV metrics.
- Structured error/fallback log.
- Optional trajectories needed for replay.

Large datasets and model artifacts should be stored in private Hugging Face
repositories and referenced by immutable revision. RunPod jobs must use the same
locked package, CLI, and configuration system as local development.

## 10. Testing and acceptance criteria

### Unit tests

- Configuration parsing and validation.
- Seed propagation.
- Valid-action masking.
- Candidate deduplication and deterministic tie-breaking.
- State-to-text serialization.
- Latent projector input/output shapes and gradients.
- JSON parsing and assignment validation.
- Retry and PPO fallback behavior.
- Metric calculations.

### Integration tests

- Small TA-RWARE episode with the heuristic.
- PPO checkpoint save/load and identical deterministic action scores.
- Simulator state clone/restore or the chosen reproducible rollout substitute.
- Tiny label-generation job.
- One hybrid forward/backward step using the locked quantization setup.
- One full hybrid episode with forced malformed output to verify fallback.
- Evaluation of all controllers producing schema-compatible CSV files.

### Core acceptance criteria

- A fresh environment can run the documented smoke test.
- All three controllers complete the same ten held-out seeds.
- The final result table is generated from committed analysis code and raw per-seed results.
- The hybrid never sends an unvalidated assignment to the simulator.
- Failures, retries, fallbacks, and latency are visible in results.
- The report states whether the hypothesis was supported without changing the primary metric.
- A negative or inconclusive result is considered a completed experiment, not a failed project.

## 11. Constraints and risk register

- Schedule: less than three months; roughly 5-10 hours per week.
- Compute: cloud resources are available, with a budget below USD 50.
- Demonstration: cloud-hosted simulation is acceptable.
- Fair rules: AI assistance and cloud services are permitted.
- Experience: some machine-learning experience; documentation must remain understandable to a student and judges.

Major risks:

1. TA-RWARE may not support efficient state cloning. Mitigation: restore from serialized state if supported, or deterministically replay from an episode seed and decision index; measure generation cost before scaling.
2. A quantized 12B model plus LoRA and soft prompts may exceed budget or memory. Mitigation: run the compatibility spike first and fall back to a 3B-4B model.
3. Rollout labels may simply reproduce PPO. Mitigation: audit disagreement and measured lookahead advantage before hybrid training.
4. The LLM may add latency without throughput gain. Mitigation: act only at assignment events and report latency explicitly.
5. Candidate lookahead may optimize a myopic proxy. Mitigation: fix the horizon and scoring rule, then evaluate only on complete held-out episodes.
6. Too many stretch experiments may prevent a finished core study. Mitigation: enforce phase exit criteria and stretch-goal ordering.
7. LLM output may be invalid or nondeterministic. Mitigation: constrained schema, low-temperature deterministic decoding where supported, validation, one retry, and logged PPO fallback.

## 12. Decisions already made

- Main goal: scientific comparison rather than demo-first development.
- Deadline category: under three months.
- Control level: task assignment, not low-level movement.
- Simulator: TA-RWARE repository linked above.
- Demonstration: simulation only; dashboard optional.
- Compute: cloud; under USD 50.
- LLM: quantized Gemma-family 12B candidate from Hugging Face, with a smaller fine-tuned model as fallback.
- Architecture: projected PPO latent tokens plus readable state and constraints.
- Controller input: one centralized, typed global observation shared by every
  controller; PPO produces one joint destination action for all agents.
- Representation boundary: immutable domain records are separate from the
  deterministic numeric encoder and the future learned PPO feature extractor.
- Simulator boundary: one project-owned TA-RWARE wrapper returns typed
  observations, decodes model destination vectors, validates joint actions,
  and accepts typed actions for stepping.
- Rollout restoration: pickled simulator snapshots are rejected because their
  action suffixes diverged. Reconstruct states from an explicit seed and
  recorded action prefix, verifying the observable suffix before label use.
- PPO mask compatibility: SB3-Contrib 2.9.0 supports the five-component
  `MultiDiscrete` action, but its masks cannot express cross-agent conflicts.
  The experimental PPO adapter keeps the first canonical same-role destination
  and turns later duplicates into logged no-ops. Strict and hybrid actions are
  never silently repaired.
- Training: freeze PPO and base LLM; train projector and LoRA adapters.
- Improvement labels: short fixed-horizon rollouts over PPO top candidates plus a heuristic candidate.
- Output: structured assignment JSON plus a short reason.
- Decision timing: assignment events for the core; other timings optional.
- Baselines: deterministic heuristic, PPO, and hybrid.
- Generalization: unseen task sequences, not unseen layouts.
- Primary metric: tasks completed per fixed-length episode.
- Secondary metrics: efficiency, reliability, validity, and latency.
- Evaluation: ten paired held-out seeds.
- Negative results: report honestly.
- Workflow: CLI plus config files; local CSV/JSON tracking.
- Git artifacts: configs, code, small results, figures, and docs; no large model artifacts.
- Documentation audience: student author and science-fair judges.
- Dependencies: `uv` with `pyproject.toml`; Conda only as a compatibility
  fallback.
- Application interfaces: Typer CLI and YAML/Pydantic configuration.
- PPO implementation: Stable-Baselines3 and SB3-Contrib 2.9.0, with the
  compatibility adapter and joint-conflict projection described above.
- Gemma dependency: `google/gemma-4-12B-it@0e2b105...`, with
  `google/gemma-4-E4B-it@a4c2d58...` as the budget fallback.
- Code quality: Ruff, standard basedpyright, `jaxtyping`, and pytest.
- Cloud workflow: repository CLI on RunPod, not notebook-only training.
- Source hosting: public GitHub repository using short branches and pull
  requests with Conventional Commits.
- Continuous integration: CPU checks on GitHub; separately recorded local or
  RunPod GPU smoke tests.
- Large artifacts: private Hugging Face model and dataset repositories.
- Project tracking: versioned Markdown roadmap and one dated Markdown file per
  research-log entry.
- Diagrams: Mermaid embedded in Markdown.
- License: add MIT only after auditing TA-RWARE, model, dataset, and dependency
  terms.

## 13. Mandatory unresolved checks

These are implementation discoveries, not open product preferences:

- Pin and inspect the TA-RWARE commit, dependency versions, and license.
- Confirm the environment's exact task-assignment action model.
- Confirm whether simulator state can be cloned for candidate rollouts.
- Confirm standard layout names and choose the smallest meaningful primary configuration.
- Verify the exact Hugging Face Gemma model ID; do not assume "Gemma 4 12B" exists under that name.
- Verify quantized LoRA and soft-token injection with the selected stack.
- Benchmark memory, latency, and estimated cloud cost before generating the full dataset.
- Define numeric candidate count, rollout horizon, score weights, PPO training budget, and seed partitions after the compatibility measurements; commit them before final data generation or evaluation.

## 14. Isolated directive-demo exception

`plan.md` defines a separate neural-directive plus frozen-Gemma actioner demo.
The prototype is deliberately unscored and does not fine-tune Gemma, inject soft
tokens, change the main primary metric, consume held-out research seeds, or
complete any main-project phase gate. Its Phase 2 vocabulary is frozen as four
controller-visible commands: `wait`, `service-requested-shelf`,
`deliver-carried-shelf`, and `reposition-idle-agent`. Deterministic executors
and effect validators are implemented in `warehouse_ai.demo.directives`; the
live probe evidence is recorded in the research log. The prototype's Phase 3
directive PPO now uses a one-dimensional masked `Discrete(4)` action space and
the existing strict `TarwareEnvWrapper` for deterministic training execution.
Its checkpoint and manifest are prototype-only artifacts and cannot become the
PPO checkpoint for the main scientific experiment. Future action sequences
remain an unresolved main-project extension.
