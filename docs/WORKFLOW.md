# Development and Experiment Workflow

## Principles

- A scientific result must be reproducible from a commit, resolved config,
  immutable artifact revisions, and seeds.
- The hybrid is allowed to lose. Never change the primary metric or hide failed
  seeds after observing results.
- Make one small change at a time and keep the repository runnable.
- Use the same package, CLI, and configs locally and on RunPod.
- Keep exploratory work separate from locked final evaluation.

## Environments

### Local Windows development

The planned dependency manager is `uv` with a committed `pyproject.toml` and lock
file. `uv` can install CUDA-enabled PyTorch wheels; the NVIDIA driver remains a
machine-level prerequisite.

The future standard setup flow is:

```powershell
uv sync --all-groups
uv run warehouse-ai smoke-test --config <config>
```

Do not add Conda simultaneously. Use it only if the compatibility spike finds a
native TA-RWARE dependency that cannot be installed reliably with `uv`.

### RunPod GPU jobs

The Phase 1 handoff is pinned to
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` on one A40 48 GB pod. Keep the
compatibility job below USD 5. After cloning and checking out an exact project
commit, run:

```bash
bash scripts/run_phase1_cloud.sh
```

The script installs the exact upstream Linux pathfinder commit for the simulator
comparison, then uses `--no-sync` so the Windows compatibility override does not
replace it. It records Linux trace hashes and the quantized LLM A40 summary in
`results/summaries/`. Copy those small summaries and the actual pod cost into a
new research-log entry before stopping the pod.

1. Start from a documented image with a compatible NVIDIA driver.
2. Clone the public GitHub repository at an exact commit.
3. Supply Hugging Face credentials through RunPod secrets, never source files.
4. Run `uv sync --frozen` so the lockfile cannot change silently.
5. Execute the same Typer CLI and committed YAML config used locally.
6. Upload large artifacts to private Hugging Face repositories.
7. Download or commit only manifests, summaries, and approved figures.
8. Stop the pod after verifying uploads and recording cost in the research log.

Do not rely on an unversioned RunPod volume as the only copy of a checkpoint or
dataset.

## Git workflow

The repository will be public on GitHub. Until a remote is created, local Git is
not a sufficient backup.

### Branches

Create a short-lived branch for one roadmap item:

- `docs/<topic>` for documentation.
- `feat/<topic>` for behavior.
- `fix/<topic>` for defects.
- `test/<topic>` for test infrastructure.
- `experiment/<topic>` for an exploratory experiment definition.

Avoid long phase branches. Rebase or merge the current main branch before a PR
when needed, without rewriting published shared history.

### Commits

Use Conventional Commit prefixes:

```text
docs: define experiment workflow
feat: add deterministic assignment validator
fix: preserve evaluation seed during reset
test: cover invalid hybrid output fallback
experiment: compare rollout horizons
chore: update locked development tools
```

Commits should be focused and should not mix generated artifacts with unrelated
source changes.

### Pull requests

Every PR should state:

- Roadmap item and intended behavior.
- Acceptance criteria.
- Commands and tests run.
- Local or RunPod GPU smoke result when ML behavior changes.
- Effects on experiment comparability or stored artifact schemas.
- Documentation and research-log entries added or updated.

Prefer reviewable PRs that complete one vertical behavior over large directory
scaffolds containing unused abstractions.

## Quality gates

The planned baseline is:

```text
uv run ruff format --check .
uv run ruff check .
uv run basedpyright
uv run pytest
```

- Ruff owns formatting and linting.
- basedpyright runs in standard mode. Public APIs must still be annotated.
- `jaxtyping` documents important tensor axes and dtypes at ML boundaries.
- Targeted runtime shape checking and tests verify dimensions; static typing alone
  cannot prove tensor shapes.
- pytest markers separate ordinary tests from network, slow, simulator, and GPU
  tests.

GitHub Actions runs formatting, linting, type checking, unit tests, and tiny CPU
simulator smoke tests. It does not use the student's laptop as a self-hosted
runner. ML-related PRs run a small documented GPU smoke command locally or on
RunPod, and the result is recorded in the PR and research log.

## Configuration workflow

- Human-authored experiment inputs are YAML.
- Pydantic validates them before environment or model initialization.
- Reusable component configs own model/actioner identity; runnable configs
  reference them by repository-relative path instead of copying their fields.
- The directive prototype uses `ppo-model.yaml`, `llm-actioner.yaml`,
  `ppo-training-run.yaml`, `ppo-llm-fine-tuning-run.yaml`, and
  `ppo-llm-evaluation-run.yaml`. One evaluation config is selected with an
  explicit `selection`, `confirmation`, or `final` CLI partition.
- Each run writes the fully resolved configuration; the original YAML alone is
  insufficient provenance.
- Unknown fields are errors rather than silently ignored typos.
- Paths in committed configs are repository-relative or artifact identifiers,
  not personal absolute paths.
- Secrets and access tokens never appear in YAML.
- Component/run config schemas and generated artifact schemas are versioned
  independently. Changing a reusable component changes every run that
  references it and therefore requires new artifact provenance.

Core final-evaluation configs are frozen before held-out seeds are run. Any later
change creates a new experiment ID rather than overwriting old results.

## Artifact policy

### Stored in Git

- Source and tests.
- YAML configs and resolved example configs without secrets.
- Dataset and checkpoint manifests.
- Per-seed result tables when reasonably small.
- Aggregate summaries and generated figures.
- Documentation and research-log entries.

### Stored in private Hugging Face repositories

- Downloaded or trained model adapters and projector checkpoints.
- Generated refinement datasets.
- Large trajectory or replay files.
- Large raw result collections.

### Never committed

- Hugging Face, GitHub, or RunPod credentials.
- Base-model downloads and caches.
- Virtual environments.
- Temporary RunPod files.
- Personal machine paths.

Every external artifact reference records its repository, immutable revision or
digest, file name, producer commit, config digest, and applicable license.

## Experiment lifecycle

### 1. Define

- Create a roadmap item or research question.
- State the hypothesis and decision it can change.
- Select training, validation, or evaluation seeds from the predefined partition.
- Create a versioned config and expected acceptance criteria.
- Estimate GPU time and cost.

### 2. Validate cheaply

- Run configuration validation.
- Run unit and CPU smoke tests.
- Use a tiny local GPU test when tensor/model behavior changes.
- Confirm output schemas before spending on a full RunPod job.

### 3. Execute

- Use a clean commit where practical.
- Run with an explicit experiment ID.
- Preserve the resolved config and run manifest immediately.
- Treat interruptions and out-of-memory failures as recorded outcomes, not files
  to delete silently.

### 4. Preserve

- Upload large artifacts before stopping RunPod.
- Verify remote files and revisions can be read back.
- Store small result tables and the manifest in Git.
- Record actual cost and hardware.

### 5. Analyze

- Generate tables and plots from saved raw results through code.
- Keep exploratory and final-evaluation outputs visibly separate.
- Record failures, surprising observations, and alternative explanations.
- Link the research-log entry to the exact run IDs and commit.

### 6. Decide

- State whether the hypothesis was supported, rejected, or remained inconclusive.
- Identify the next action without retroactively modifying the experiment.
- If a bug invalidated a run, document the reason and rerun under a new run ID.

## Run and result naming

Use machine-sortable identifiers:

```text
<experiment>-<controller>-<YYYYMMDDTHHMMSSZ>-<short-commit>
```

Use UTC in manifests and file names. Human-facing research-log entries may also
show local time. Never overwrite an authoritative run directory.

## Secrets and licenses

- Read tokens from environment variables or the platform secret manager.
- Provide `.env.example` names only; never commit `.env`.
- Scrub tokens from command logs and exception output.
- Review TA-RWARE, Qwen/LLM, dataset, and transitive license terms before adding the
  MIT project license.
- Record third-party attribution and immutable source revisions.

## Documentation maintenance

- `PROJECT_SPEC.md` owns scientific intent and agreed scope.
- `docs/ARCHITECTURE.md` owns boundaries and data flow.
- This file owns development and experiment procedures.
- `ROADMAP.md` owns execution order and completion state.
- Research-log entries own dated observations and decisions.
- `AGENTS.md` points agents to these sources and must remain concise.

Update the owning document when a decision changes. Avoid duplicating large
sections across documents.

