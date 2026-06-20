# Agent Guide

## Required context

Read [PROJECT_SPEC.md](PROJECT_SPEC.md) completely before planning or changing
this repository. It is the cross-session source of truth for the research
question, experimental protocol, risks, and decisions made with the student.

Then read the documents relevant to the task:

- [Architecture](docs/ARCHITECTURE.md) for boundaries, data flow, and the future
  package structure.
- [Workflow](docs/WORKFLOW.md) for Git, quality gates, RunPod, artifacts,
  configuration, and experiment procedure.
- [Roadmap](ROADMAP.md) for ordering, phase gates, and current status.
- [Research log](docs/research-log/README.md) before running or interpreting an
  experiment.

## Working rules

- Treat this as a science-fair experiment, not a requirement to make the hybrid win.
- Preserve the declared primary metric and held-out evaluation protocol.
- Keep heuristic, PPO, and hybrid controllers comparable under identical seeds and episode settings.
- Do not start stretch goals until the core end-to-end experiment passes its phase exit criteria.
- Challenge technically weak assumptions and explain tradeoffs in student-accessible language.
- Prefer small, testable milestones, reproducible CLI commands, and versioned configuration files.
- Never commit downloaded base models, large checkpoints, raw generated datasets, caches, or secrets.
- Update `PROJECT_SPEC.md` whenever a later decision changes the agreed design.
- Update the owning document instead of duplicating architecture, workflow, or
  roadmap content elsewhere.
- Record meaningful investigations, failed experiments, and GPU jobs using the
  research-log template.
- Do not invent TA-RWARE or Gemma interfaces that the compatibility spike has
  not verified.

## Immediate next task

Perform the compatibility spike in `PROJECT_SPEC.md`: inspect and pin TA-RWARE, run its smallest example, document its interfaces, and verify an exact Hugging Face Gemma checkpoint plus quantized LoRA/soft-token feasibility before scaffolding the full system.
