# Architecture

## Purpose

This document defines the intended system boundaries before implementation. It
does not claim that TA-RWARE exposes methods with these names. The compatibility
spike must map its real interfaces into the project-owned concepts below.

The central rule is that TA-RWARE-specific objects remain inside the environment
adapter. Controllers, training code, dataset generation, and evaluation operate
on project-owned domain models.

## System context

```mermaid
flowchart LR
    C[Versioned experiment config] --> R[Experiment runner]
    R --> E[TA-RWARE adapter]
    E <--> S[TA-RWARE simulator]
    E --> X[Assignment context]
    X --> H[Heuristic controller]
    X --> P[PPO controller]
    X --> L[Hybrid controller]
    H --> V[Decision validator]
    P --> V
    L --> V
    V --> E
    E --> M[Metrics and trajectory recorder]
    M --> A[Run manifest and result files]
```

Every evaluated controller receives equivalent state information, eligible
assignments, episode settings, and random seeds. The validator is mandatory for
LLM output and should also assert decisions from deterministic controllers.

## Training and evaluation flow

```mermaid
flowchart TD
    T[Training seeds] --> PT[Train PPO]
    PT --> PC[Frozen PPO checkpoint]
    PC --> PG[Generate PPO top-k candidates]
    T --> ES[Restorable simulator states]
    ES --> RG[Short-horizon candidate rollouts]
    HG[Heuristic candidate] --> RG
    PG --> RG
    RG --> DS[Refinement dataset]
    DS --> HT[Train latent projector and LoRA]
    PC --> HT
    BM[Frozen quantized Gemma] --> HT
    HT --> HC[Frozen hybrid checkpoint]
    EV[Held-out evaluation seeds] --> CE[Paired controller evaluation]
    PC --> CE
    HC --> CE
    HG --> CE
    CE --> RR[Per-seed results and statistical report]
```

Training, validation, and final evaluation seed sets are disjoint. Final seeds
must not influence prompts, hyperparameters, checkpoints, label generation, or
fallback rules.

## Layering and dependency direction

```mermaid
flowchart BT
    D[Domain models and protocols]
    EA[Environment adapter] --> D
    CT[Controllers] --> D
    TR[Training and label generation] --> D
    TR --> EA
    TR --> CT
    EV[Evaluation and analysis] --> D
    EV --> EA
    EV --> CT
    CLI[CLI and configuration] --> EA
    CLI --> CT
    CLI --> TR
    CLI --> EV
```

Lower layers never import the CLI, training orchestration, or analysis code.
Domain models do not depend on TA-RWARE, Stable-Baselines3, Transformers, Typer,
or filesystem formats.

## Conceptual interfaces

These are behavioral contracts. Exact Python signatures and fields are finalized
after TA-RWARE inspection.

### Environment adapter

`EnvironmentAdapter` owns simulator translation and must support:

- `reset(seed, environment_config)` returning an initial project-owned state.
- Reading an `AssignmentContext` at each assignment event.
- Enumerating all currently eligible robot-task pairs.
- Validating and applying an `AssignmentDecision`.
- Advancing low-level navigation until the next decision event or termination.
- Reporting termination, truncation, deadlock, and simulator errors distinctly.
- Producing step and episode metrics.
- Snapshot/restore for candidate rollouts, or a deterministic replay substitute
  if TA-RWARE cannot clone state.

No controller may access hidden future orders or internal simulator state that is
unavailable to the other controllers.

### Controller

All controllers implement one conceptual operation:

```text
choose_assignment(context: AssignmentContext) -> AssignmentDecision
```

The heuristic and PPO controllers should be deterministic during evaluation.
The hybrid uses deterministic decoding where the selected model supports it.

### Domain records

- `WarehouseState`: controller-visible state at a decision event.
- `EligibleAssignment`: one currently legal robot-task pairing.
- `AssignmentContext`: state, eligible assignments, episode step, and identifiers
  required for logging.
- `AssignmentDecision`: selected pairs, optional reason, latency, validation
  status, retry count, and fallback status.
- `StepMetrics`: distance, idle time, task progress, failures, and timing for one
  simulator interval.
- `EpisodeMetrics`: aggregate primary and secondary outcomes for one seed.
- `RunManifest`: provenance needed to reproduce an entire command.

Identifiers must remain stable within an episode and must serialize without
Python-specific object representations.

### Run manifest

Every authoritative run records:

- Run ID, command, UTC creation time, and experiment name.
- Git commit and whether the worktree was dirty.
- Full resolved configuration and schema version.
- Training, validation, or evaluation seed set.
- Python and dependency lock information.
- Operating system, CPU, GPU, driver, and accelerator information.
- TA-RWARE commit and environment configuration.
- PPO checkpoint digest.
- Hugging Face repository IDs and immutable artifact revisions.
- Output file paths, status, duration, and failure summary.

## Controller behavior

### Heuristic

The deterministic heuristic provides a baseline and one candidate for rollout
label generation. Its scoring rule is chosen after inspecting TA-RWARE and then
locked before final experiments. It cannot use privileged state.

### PPO

Stable-Baselines3 is the default implementation. The policy integration must
expose the frozen encoder latent, valid-action-masked scores, preferred action,
and value estimate without maintaining a second independently trained PPO model.
Action masking support must be tested before committing to a policy class.

### Rollout label generator

For each selected training state, compare PPO's top valid candidates with the
heuristic candidate using identical short-horizon conditions. Candidate count,
horizon, continuation controller, score, and tie-breaking are configuration
values fixed before dataset generation.

If cloning is unavailable, reconstruct a state by replaying a seed and recorded
decision prefix. Verify that replay reaches the same observable state before
using its label.

### Hybrid

The hybrid freezes PPO and the quantized base Gemma model. It trains:

- A projector from one PPO latent vector to a fixed sequence of Gemma embedding
  vectors.
- LoRA adapters on an explicitly configured subset of Gemma modules.

The LLM also receives a compact textual state and explicit eligible assignments.
It returns structured robot-task pairs and a short reason. Invalid output gets
one corrective retry; a second failure falls back to PPO's highest-ranked valid
choice. All retries and fallbacks are measured.

## Future package structure

```text
src/warehouse_ai/
|-- domain/          # Framework-independent records, protocols, and validation
|-- env/             # TA-RWARE adapter and state serialization
|-- controllers/     # Shared controller protocol and heuristic implementation
|-- ppo/             # PPO policy integration, training, and checkpoints
|-- hybrid/          # Projector, prompting, LoRA, decoding, and fallback
|-- data/             # Rollout label generation and dataset manifests
|-- evaluation/       # Metrics, paired runs, statistics, plots, and replay
|-- config/           # Pydantic schemas and YAML loading
`-- cli.py            # Typer application composition only
```

Tests should mirror these boundaries. Notebooks may import the package for
analysis but cannot contain the only implementation of training or evaluation.

## Unresolved compatibility outputs

The first technical milestone must replace these unknowns with pinned facts:

- Supported Python and dependency versions.
- TA-RWARE observation, assignment action, reward, and rendering behavior.
- Multi-agent representation and valid-action masking requirements.
- Snapshot/restore feasibility and determinism.
- Smallest meaningful standard environment configuration.
- Exact Gemma Hugging Face ID, revision, license, quantization, and memory use.
- Embedding injection and quantized LoRA compatibility.

