# 2026-08-24: Heuristic baseline on Qwen selection seeds

## Objective

Measure the pinned TA-RWARE heuristic on the refreshed directive-demo selection
environment and seeds so its throughput, reliability, timing, and resource use
can be compared with the existing parent/adapted executor and Qwen rows.

This is a post-selection descriptive baseline. It does not reopen or alter the
failed `candidate_better` gate, and it does not expose the final seed.

## Hypothesis

Not applicable. The goal was to collect a matched baseline without requiring
the heuristic to win.

## Provenance

- Git state during the run: dirty; `scripts/smoke_tarware.py` had the metric
  recorder and telemetry additions documented here.
- Command: `uv run python scripts/smoke_tarware.py --seeds 8200 8201
  --expected-steps 500` with JSON, step-JSONL, and telemetry-JSONL outputs.
- Environment: `tarware-tiny-3agvs-2pickers-partialobs-v1`.
- Seed set and partition: selection seeds `8200` and `8201`.
- Episode cap: 500 steps, matching the selection evaluation config.
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM;
  Windows 11; Python 3.12.4.
- Start/end: `2026-08-24T06:59:51.733254Z` to
  `2026-08-24T06:59:52.484331Z`.
- Estimated and actual cost: USD 0; local CPU execution.

## Method

The pinned `tarware.heuristic.heuristic_episode` implementation was called
without changing its assignment behavior. A transparent proxy recorded all
1,000 submitted joint actions and exact simulator step results. The runner
guarded the environment ID and expected 500-step length, reconciled global and
per-agent returns, hashed action and simulator traces, and sampled
`nvidia-smi` every 0.25 seconds because the complete phase was shorter than one
second.

The raw summary is
`results/demo/heuristic-selection-seeds-8200-8201-20260824/heuristic-summary.json`.
The step and telemetry records are beside it as `heuristic-steps.jsonl` and
`heuristic-telemetry.jsonl`. Report CSVs are under
`results/final-demo-report/`.

## Results

| Seed | Steps | Tasks | Return | Clashes | Stucks | Wall seconds | Steps/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8200 | 500 | 20 | 21.4 | 15 | 0 | 0.237079 | 2,109.0 |
| 8201 | 500 | 25 | 27.4 | 26 | 1 | 0.213928 | 2,337.2 |
| Mean / total | 500 | 22.5 mean | 24.4 mean | 41 total | 1 total | 0.225504 mean | — |

Additional totals were:

| Seed | AGV distance | Picker distance | AGV idle | Picker idle | Busy-agent steps | Nonzero destinations |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8200 | 678 | 530 | 825 | 489 | 1,986 | 626 |
| 8201 | 821 | 491 | 696 | 560 | 1,934 | 610 |

All 1,000 raw step records reconciled with the two episode summaries. There
were zero LLM calls, fallbacks, invalid actions, effect violations, malformed
responses, or other model-output errors because this controller does not use a
model.

Four GPU samples (start, two interval, end) all reported 0% utilization and
8 MiB used VRAM. Power samples ranged from 1.58 to 1.60 W. The complete
two-seed monitored phase took 0.751389 seconds.

## Observations

- The heuristic completed many more tasks on both selection seeds than any
  executor or Qwen-backed directive-policy row in this exploratory demo.
- Its 41 total clashes were lower than the parent/adapted executor total of 53
  and far below the Qwen-backed total of 918, while stuck events totaled 1.
- The heuristic was CPU-bound; the GPU remained idle apart from the driver's
  8 MiB baseline allocation.

## Interpretation

The pinned TA-RWARE heuristic is a strong descriptive baseline on these two
selection seeds. This does not establish general superiority: the sample has
only two already-used prototype seeds, and the heuristic operates directly on
simulator state while the directive policies use a constrained four-command
interface. The comparison is useful for exposing the size of the current
prototype performance gap, not for claiming a final scientific result.

## Decision

Keep the heuristic rows in the final demo report as a matched selection
baseline. Preserve `candidate_better=false`; confirmation and final evaluation
remain blocked and were not run.

## Problems and follow-up

No run error occurred. CSV rendering cannot preserve column widths, so the
machine-readable report should be inspected by headers/ranges rather than as a
presentation workbook. A future locked evaluation should compare heuristic,
PPO, and hybrid controllers on the same larger held-out seed set.
