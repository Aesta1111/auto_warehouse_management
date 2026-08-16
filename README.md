# Auto Warehouse Management

Science-fair research project investigating whether a task-assignment controller
that combines PPO latent features with an instruction-tuned LLM can outperform a
standalone PPO controller in the TA-RWARE simulator.

## Current status

The main research project remains in the compatibility-check phase. Local
simulator replay, rendering, MaskablePPO, and a tiny Gemma 4 projector/LoRA
optimizer step are verified. Pickled simulator snapshots were rejected in favor
of exact seed-plus-action replay. A separate, unscored directive-demo prototype
has completed its directive-semantics probe and Phase 3 directive PPO gate; it
does not advance the main research phases or validate the hybrid hypothesis.
The remaining Phase 1 gate is the prepared Linux/A40 cloud run for pathfinder
equivalence, real 4-bit weights, memory, latency, and cost. The main training
pipeline has not been implemented.

The hybrid improving on PPO is a hypothesis, not a required result. Negative and
inconclusive findings must be reported without changing the primary metric after
evaluation.

## Documentation map

- [Project specification](PROJECT_SPEC.md): research question, scope, metrics,
  system design, risks, and decisions.
- [Architecture](docs/ARCHITECTURE.md): proposed components, data flow,
  boundaries, and future package structure.
- [Development workflow](docs/WORKFLOW.md): local and cloud workflows, Git,
  quality checks, artifacts, secrets, and experiment reproducibility.
- [Roadmap](ROADMAP.md): ordered phases, checklists, exit gates, and stretch
  goals.
- [Research log](docs/research-log/README.md): experiment-recording rules,
  entry index, and template.
- [Agent guide](AGENTS.md): required context and rules for coding agents.

## Planned technology stack

- Python managed by `uv` and `pyproject.toml`
- Typer CLI
- YAML configuration validated with Pydantic
- Stable-Baselines3/SB3-Contrib 2.9.0
- PyTorch, Transformers, PEFT/LoRA, and `jaxtyping`
- Ruff, basedpyright, and pytest
- GitHub for public source code and CPU continuous integration
- RunPod for GPU jobs
- Private Hugging Face repositories for large model and dataset artifacts

The lockfile pins the local dependency set. The primary model is
`google/gemma-4-12B-it@0e2b105...`; the smaller fallback is
`google/gemma-4-E4B-it@a4c2d58...`.

## TA-RWARE smoke test

Install the locked environment and run one 500-step heuristic episode:

```powershell
uv sync
uv run python scripts/smoke_tarware.py --seed 0
```

Add `--render` to open the simulator viewer. On Windows, the lockfile replaces
TA-RWARE's unbuildable pathfinder fork with the API-compatible `pyastar2d` 1.1.4
wheel. Linux remains the authoritative experiment platform until identical
seeded trajectories are verified across both pathfinder builds.

## Local compatibility probes

```powershell
uv run python scripts/probe_tarware_compatibility.py --seeds 0 1 2 3 4
uv run python scripts/probe_sb3.py --timesteps 512
uv run python scripts/probe_gemma_tiny.py
```

## Directive prototype probe

The isolated prototype's Phase 2 registry and live-seed probe can be rerun with:

```powershell
uv run python scripts/probe_directives.py --config configs/demo/directive-probe-v1.yaml
```

This checks the frozen four-directive vocabulary and deterministic executor
effects on the five committed probe seeds. Its result is a prototype artifact,
not a scientific baseline or a main-project phase completion.

Train the isolated directive PPO policy with the committed prototype seeds:

```powershell
uv run python scripts/train_directive_ppo.py --config configs/demo/directive-ppo-v1.yaml
```

The command writes a small manifest and an ignored checkpoint under
`results/demo/directive-ppo-v1/`. The manifest records vocabulary and behavior
digests, seed partitions, package versions, validation metrics, and the
checkpoint SHA-256. Gemma/actioner phases are intentionally not included in
this Phase 3 command.

The deferred RunPod command is `bash scripts/run_phase1_cloud.sh` from an exact
checked-out project commit.

