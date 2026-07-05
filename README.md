# Auto Warehouse Management

Science-fair research project investigating whether a task-assignment controller
that combines PPO latent features with an instruction-tuned LLM can outperform a
standalone PPO controller in the TA-RWARE simulator.

## Current status

The project is in the compatibility-check phase. The pinned TA-RWARE tiny
environment and upstream heuristic run locally; the training pipeline has not
been implemented yet. The next technical milestone is to verify deterministic
replay and the proposed quantized Gemma and LoRA setup.

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
- Stable-Baselines3 PPO, subject to TA-RWARE compatibility
- PyTorch, Transformers, PEFT/LoRA, and `jaxtyping`
- Ruff, basedpyright, and pytest
- GitHub for public source code and CPU continuous integration
- RunPod for GPU jobs
- Private Hugging Face repositories for large model and dataset artifacts

Exact dependency versions, Python version, TA-RWARE revision, and Gemma model ID
will be locked by the compatibility spike. Do not install guessed versions and
treat them as authoritative.

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

