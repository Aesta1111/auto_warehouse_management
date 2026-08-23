# Auto Warehouse Management

Science-fair research project investigating whether a task-assignment controller
that combines PPO latent features with an instruction-tuned LLM can outperform a
standalone PPO controller in the TA-RWARE simulator.

## Current status

The main research project remains in the compatibility-check phase. Local
simulator replay, rendering, MaskablePPO, and the bounded model probes are
verified. Pickled simulator snapshots were rejected in favor of exact
seed-plus-action replay. A separate, unscored directive-demo refresh now has a
versioned Qwen3.5-2B Q8 runtime contract, an external 12-example prompt, and
CUDA-enabled 4x256 PPO settings; its full 100k-step parent/adaptation/evaluation
run remains pending. The previous failed model run and its artifacts remain
historical evidence and are not overwritten.

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

The lockfile pins the local dependency set. The active directive prototype uses
the Apache-labeled `bartowski/Qwen_Qwen3.5-2B-GGUF` repository at revision
`7d26695454df6de5fbcce2e58681e62dae06ce43`, file
`Qwen_Qwen3.5-2B-Q8_0.gguf`, SHA-256
`be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2`.

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
uv run python scripts/probe_llm_tiny.py
```

## Directive prototype probe

The active prototype configuration is composed from five descriptive files:

- `ppo-model.yaml`: PPO network shape and implementation identity.
- `llm-actioner.yaml`: frozen Qwen model, prompt, decoding, and runtime limits.
- `ppo-training-run.yaml`: baseline PPO seeds, optimization, and outputs.
- `ppo-llm-fine-tuning-run.yaml`: warm start, adaptation budget, and seed gates.
- `ppo-llm-evaluation-run.yaml`: preflight, matched demo, and gated evaluation.

The active parent is exactly `[256, 256, 256, 256]` for 100,000 configured
timesteps on CUDA. The frozen-LLM PPO adaptation is 4,096 environment steps.
The matched demonstration uses seed `9300`; confirmation remains `8202-8203`,
and final seed `9200` is exposed only after confirmation passes.

The catalog, few-shot prompt, and semantics-probe files are immutable supporting
contracts rather than runnable experiment configurations.

The isolated prototype's Phase 2 registry and live-seed probe can be rerun with:

```powershell
uv run python scripts/probe_directives.py --config configs/demo/directive-semantics-probe.yaml --output results/summaries/directive-semantics-probe-report.json
```

This checks the frozen four-directive catalog and deterministic executor
effects on the five committed probe seeds. Its result is a prototype artifact,
not a scientific baseline or a main-project phase completion.

Train the isolated directive PPO policy with the committed prototype seeds:

```powershell
uv run python scripts/train_directive_ppo.py --config configs/demo/ppo-training-run.yaml
```

`ppo-training-run.yaml` owns seeds, optimizer settings, environment, and output;
it references `ppo-model.yaml`, which owns the independent policy/value MLP
architecture and implementation identity. Running it is a full PPO training
job, not an ordinary quality check, and writes `ppo-policy.pt` plus
`ppo-training-manifest.json` under the configured output directory.
Each manifest records catalog and behavior digests, seed partitions, package
versions, validation metrics, and the checkpoint SHA-256. LLM/actioner phases
are intentionally not included in this command.

Prepare and start the real local Qwen/llama.cpp actioner on a Windows NVIDIA
machine from two terminals. The `start-llm` subcommand downloads the pinned
2.08 GB GGUF and the pinned CUDA 12.4 llama.cpp runtime into the ignored
`.local_runtime/` directory, then serves it with partial GPU offload:

```powershell
uv run python scripts/run_directive_demo.py start-llm --prepare
```

Omit `--prepare` when the pinned runtime and model are already present.

Keep that server terminal open. In a second terminal, run the real local
LLM preflight separately before the demo:

```powershell
.\.venv\Scripts\python.exe scripts\probe_llm_actioner.py --config configs\demo\ppo-llm-evaluation-run.yaml
```

The evaluation run resolves the reusable `llm-actioner.yaml` and
`llm-actioner-few-shot-prompt.yaml` plus its external TXT template. It remains blocked until the new PPO
checkpoint and manifest exist; this live preflight has not run.

The preflight writes the configured report only after checking health, model
identity, strict responses across all four directive classes, directive
sensitivity, prompt/template digests, and p95 latency. Reasoning is disabled
with `--reasoning off --reasoning-budget 0`; it never silently substitutes a model.

Run the matched, unscored demonstration only after the preflight report passes
and the local server is still running:

```powershell
uv run python scripts/run_directive_demo.py --config configs/demo/ppo-llm-evaluation-run.yaml
```

The runner uses the same seed and frozen directive PPO checkpoint for the
headless deterministic executor and the rendered Qwen actioner. It writes
flushed decision JSONL plus small atomic summaries under a unique
`results/demo/<run-id>/` directory and never computes a statistical or
improvement claim.

After training the configured PPO and producing a passing preflight, the
optional PPO adaptation can be started explicitly while the same pinned server
remains running:

```powershell
uv run python scripts/finetune_directive_ppo_with_llm.py --config configs/demo/ppo-llm-fine-tuning-run.yaml
```

This is an estimated one-hour live run and is never part of tests or ordinary
quality checks. It updates a child PPO only; Qwen weights and the parent checkpoint
remain frozen. It fails closed when the parent, prompt, preflight, model, or
seed partitions do not match.

Run paired selection evaluation only after the child manifest exists:

```powershell
uv run python scripts/evaluate_llm_finetuned_ppo.py --config configs/demo/ppo-llm-evaluation-run.yaml --partition selection
```

Change `--partition` to `confirmation` only if selection passes, and to `final`
only if confirmation passes. The single evaluation config declares both
prerequisite summaries, and the runner fails closed if a gate is missing or
failed. Each paired evaluation is a live LLM job and is intentionally
excluded from automated tests.

The deferred RunPod command is `bash scripts/run_phase1_cloud.sh` from an exact
checked-out project commit.

