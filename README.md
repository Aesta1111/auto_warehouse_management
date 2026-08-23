# Auto Warehouse Management

Science-fair research project investigating whether a task-assignment controller
that combines PPO latent features with an instruction-tuned LLM can outperform a
standalone PPO controller in the TA-RWARE simulator.

## Current status

The main research project remains in the compatibility-check phase. Local
simulator replay, rendering, MaskablePPO, and a tiny Gemma 4 projector/LoRA
optimizer step are verified. Pickled simulator snapshots were rejected in favor
of exact seed-plus-action replay. A separate, unscored directive-demo prototype
has completed its directive-semantics probe and Phase 3/3.5 directive PPO gates; it
does not advance the main research phases or validate the hybrid hypothesis.
The directive-demo Phase 5 local actioner gate now passes with the pinned E4B
Q4_0 GGUF on the local RTX 4050: five measured responses were valid, all six
directive-sensitivity responses were valid, and p95 latency was 3.56 seconds
against the 10-second limit. The remaining Phase 1 gate is the prepared
Linux/A40 cloud run for pathfinder equivalence, real 4-bit weights, memory,
latency, and cost. The main training pipeline has not been implemented.
The directive-demo Phase 6 matched runner and its live 500-step gate completed;
the negative 3/2-task executor/Gemma result remains a separate prototype, not a
main-project evaluation. Schema-v3 few-shot prompting and frozen-Gemma PPO
adaptation infrastructure are implemented offline, but neither has a live run.
The new schema-v4 PPO uses config-controlled MLP hidden sizes and initially
declares `[64]`; it has not been trained, so schema v3 remains the latest
validated directive-PPO evidence.

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

The active prototype configuration is composed from five descriptive files:

- `ppo-model.yaml`: PPO network shape and implementation identity.
- `llm-actioner.yaml`: frozen Gemma model, prompt, decoding, and runtime limits.
- `ppo-training-run.yaml`: baseline PPO seeds, optimization, and outputs.
- `ppo-gemma-fine-tuning-run.yaml`: warm start, adaptation budget, and seed gates.
- `ppo-gemma-evaluation-run.yaml`: preflight, matched demo, and gated evaluation.

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
versions, validation metrics, and the checkpoint SHA-256. Gemma/actioner phases
are intentionally not included in this command.

Prepare and start the real local Gemma/llama.cpp actioner on a Windows NVIDIA
machine from two terminals. The `start-gemma` subcommand downloads the pinned
5.15 GB GGUF and the pinned CUDA 12.4 llama.cpp runtime into the ignored
`.local_runtime/` directory, then serves it with partial GPU offload:

```powershell
uv run python scripts/run_directive_demo.py start-gemma --prepare
```

Omit `--prepare` when the pinned runtime and model are already present.

Keep that server terminal open. In a second terminal, run the real local
Gemma preflight separately before the demo:

```powershell
.\.venv\Scripts\python.exe scripts\probe_gemma_actioner.py --config configs\demo\ppo-gemma-evaluation-run.yaml
```

The evaluation run resolves the reusable `llm-actioner.yaml` and
`gemma-actioner-few-shot-prompt.yaml`. It remains blocked until the new PPO
checkpoint and manifest exist; this live preflight has not run.

The preflight writes the configured report only after checking health, model
identity, strict responses, directive sensitivity, and p95 latency. An E4B
failure returns a distinct code recommending an explicitly configured E2B run;
it never silently substitutes a model.

Run the matched, unscored demonstration only after the preflight report passes
and the local server is still running:

```powershell
uv run python scripts/run_directive_demo.py --config configs/demo/ppo-gemma-evaluation-run.yaml
```

The runner uses the same seed and frozen directive PPO checkpoint for the
headless deterministic executor and the rendered Gemma actioner. It writes
flushed decision JSONL plus small atomic summaries under a unique
`results/demo/<run-id>/` directory and never computes a statistical or
improvement claim.

After training the configured PPO and producing a passing preflight, the
optional PPO adaptation can be started explicitly while the same pinned server
remains running:

```powershell
uv run python scripts/finetune_directive_ppo_with_gemma.py --config configs/demo/ppo-gemma-fine-tuning-run.yaml
```

This is an estimated one-hour live run and is never part of tests or ordinary
quality checks. It updates a child PPO only; Gemma and the parent checkpoint
remain frozen. It fails closed when the parent, prompt, preflight, model, or
seed partitions do not match.

Run paired selection evaluation only after the child manifest exists:

```powershell
uv run python scripts/evaluate_gemma_finetuned_ppo.py --config configs/demo/ppo-gemma-evaluation-run.yaml --partition selection
```

Change `--partition` to `confirmation` only if selection passes, and to `final`
only if confirmation passes. The single evaluation config declares both
prerequisite summaries, and the runner fails closed if a gate is missing or
failed. Each paired evaluation is a live Gemma job and is intentionally
excluded from automated tests.

The deferred RunPod command is `bash scripts/run_phase1_cloud.sh` from an exact
checked-out project commit.

