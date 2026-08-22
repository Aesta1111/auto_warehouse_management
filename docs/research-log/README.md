# Research Log

## Purpose

The research log records why an experiment was run, exactly what produced it,
what happened, and what decision followed. It includes failed and inconclusive
runs. Machine-generated manifests contain exact provenance; these entries add
the human reasoning needed for a science-fair report.

## Entry rules

- Create one Markdown file per meaningful experiment or technical investigation.
- Name it `YYYY-MM-DD-short-topic.md`; add `-2`, `-3`, and so on only when needed.
- Start from [TEMPLATE.md](TEMPLATE.md) and remove instructional placeholders.
- Link exact run IDs, Git commits, configs, artifact revisions, and result files.
- Distinguish observations from interpretations.
- Record failed setup attempts, invalid runs, out-of-memory events, and cost.
- Do not edit an old conclusion to match later results. Add a dated correction or
  a new entry and cross-link both.
- Never include credentials, private tokens, or signed artifact URLs.

## Index

Add completed entries below in chronological order:

| Date | Topic | Outcome | Entry |
| --- | --- | --- | --- |
| 2026-06-20 | TA-RWARE source and installation audit | Native Windows blocked by upstream pathfinder build; Linux smoke test required | [Entry](2026-06-20-ta-rware-source-audit.md) |
| 2026-07-15 | Phase 1 local compatibility handoff | Replay, SB3, and tiny Gemma pass locally; cloud proof remains | [Entry](2026-07-15-phase1-local-compatibility.md) |
| 2026-08-13 | Directive-demo Phase 2 semantics | Four directive behaviors and live five-seed probe pass; prototype remains unscored | [Entry](2026-08-13-directive-semantics.md) |
| 2026-08-16 | Directive-demo Phase 3 PPO | 25,000-step masked PPO checkpoint and deterministic reload validation pass; prototype remains unscored | [Entry](2026-08-16-directive-ppo.md) |
| 2026-08-18 | Directive-demo Phase 3.5 PPO | Repository-owned PPO/Fabric checkpoint and deterministic reload gate pass; SB3 remains the descriptive reference | [Entry](2026-08-18-directive-ppo-v2.md) |
| 2026-08-22 | Directive-demo Phase 4 fake actioner | Six live-wrapper success/fallback scenarios pass; prototype remains unscored | [Entry](2026-08-22-directive-phase4-fake-actioner.md) |
| 2026-08-22 | Directive-demo Phase 5 local actioner | Pinned local E4B preflight passes: 5/5 validity, 6/6 sensitivity, p95 3.56 s | [Entry](2026-08-22-directive-phase5-actioner.md) |
