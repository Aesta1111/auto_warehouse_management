# 2026-08-23: Directive PPO and frozen-Gemma adaptation full run

## Objective

Run the active isolated directive-demo pipeline through parent PPO training,
live E4B preflight, Gemma-in-the-loop PPO adaptation, the matched demonstration,
and the gated selection/confirmation comparisons. The configured final
partition was to run only if confirmation passed.

This is prototype evidence for the directive demo. It is not evidence for the
main PPO-latent/projector/LoRA warehouse research system.

## Hypothesis

The adapted directive PPO would preserve or improve Gemma-backed task
throughput relative to the parent while reducing executor-transfer failures.

## Provenance and partition status

- Run code commit: `4339e25bb6471f0b49b04031e429b40b1f3879bf`.
- Documentation was subsequently recorded in commit `4f5ff83`, which is being
  amended by this update.
- The run itself was dirty because the resource monitor was added before
  execution; the monitor and this expanded record are now committed separately
  from the run code.
- Configs: `configs/demo/ppo-training-run.yaml`,
  `configs/demo/ppo-gemma-fine-tuning-run.yaml`, and
  `configs/demo/ppo-gemma-evaluation-run.yaml`.
- Environment: `tarware-tiny-3agvs-2pickers-partialobs-v1`, 500 steps per
  evaluation episode, schema-v4 directive policy with four directives.
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`.
- Seeds: semantics probe `8000-8004`; PPO training `8100-8115`; parent
  validation `8200-8203`; adaptation selection `8200-8201`; confirmation
  `8202-8203`; configured final `9100`.
- Parent checkpoint SHA-256:
  `ad4d3164a39970269657ed72338df089eaef13eb33c6c019af716962efd27b60`.
- Adapted checkpoint SHA-256:
  `5c0453c04badd87dcab7f05a97c17ac6e8f5b6eb5e19052c028d6bc8fab46342`.
- Gemma model: pinned local E4B Q4_0 GGUF, file SHA-256
  `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB VRAM, driver
  580.97; installed PyTorch was `2.8.0+cpu`; llama.cpp used CUDA.
- Run envelope: 2026-08-23 13:08:09Z to 15:18:09Z, approximately 2 h 10
  min wall-clock including gaps between commands.
- Cloud cost: $0; all work ran locally, with no cloud billing measurement.

### Important seed correction

The matched demonstration used seed `9100` before the selection and
confirmation gates. Therefore, `9100` is not an untouched final held-out seed,
even though the configured final partition command was correctly not run after
confirmation failed. A future final evaluation must use a new untouched seed
set or explicitly classify `9100` as exploratory matched-demo evidence.

## Method and run order

The six substantive commands ran in this order:

1. Probe directive semantics on seeds `8000-8004`.
2. Train the parent repository-owned PPO for the configured 25,000 steps.
3. Run the live E4B preflight.
4. Run 1,024 steps of frozen-Gemma PPO adaptation on the training partition.
5. Run the matched parent executor/Gemma demonstration on seed `9100`.
6. Run the paired selection comparison, then the paired confirmation
   comparison.

Each long-running command was wrapped by
`scripts/run_resource_monitored.py`. It records GPU and VRAM at process start,
every 150 seconds, and process end in
`results/demo/resource-logs/*.jsonl`. The raw JSONL and gate manifests are the
authoritative machine-readable record; this entry preserves the scores and
interpretation.

The selection gate passed. The confirmation gate failed. The final partition
was not run.

## Resource and elapsed-time record

The GPU range is the minimum-to-maximum utilization among recorded samples;
the VRAM range is the minimum-to-maximum used memory among recorded samples.
The full sample sequence remains in the JSONL logs.

| Stage | Monitor elapsed | GPU utilization samples | VRAM samples | Outcome |
| --- | ---: | ---: | ---: | --- |
| Directive semantics probe | 150.094 s | 0-0% | 8-8 MiB | Pass; first monitor waited for its 150-second interval before exiting |
| Parent PPO training | 116.812 s | 0-0% | 8-8 MiB | Pass; CPU-only PyTorch |
| E4B preflight | 35.594 s | 0-33% | 1,924-1,968 MiB | Pass |
| PPO/Gemma adaptation | 1,266.875 s | 0-95% | 1,968-1,970 MiB | Completed |
| Matched parent demo | 714.219 s | 0-34% | 1,970-1,970 MiB | Completed |
| Selection comparison | 2,281.735 s | 0-88% | 1,970-1,982 MiB | Gate passed |
| Confirmation comparison | 3,018.594 s | 0-97% | 1,982-1,982 MiB | Gate failed |
| **Measured stage total** | **7,583.923 s** | — | — | **About 2 h 6 min 24 s** |

The resource logs contain the 150-second samples, including power draw. Peak
observed VRAM was 1,982 MiB of 6,141 MiB. The long Gemma stages were slow even
though memory fit: individual Gemma decisions were generally about 2.5-3.3 s.

## Setup, training, and preflight results

### Directive semantics

The report contains 2,500 total checked states across seeds `8000-8004`:

| Metric | Result |
| --- | ---: |
| States checked | 2,500 |
| Uncovered states | 0 |
| Executor failures | 0 |
| Effect failures | 0 |

### Parent PPO

| Metric | Result |
| --- | ---: |
| Declared training steps | 25,000 |
| Completed environment steps | 25,088 |
| Training seeds | 16 (`8100-8115`) |
| Validation seeds | 4 (`8200-8203`) |
| Validation mean completed tasks | 2.75 |
| Validation mean return | 0.55 |
| Validation mean entropy | 0.468838 |
| Masked action count | 4,547 |
| Inapplicable selections | 0 |
| Non-wait directives used | 2 |
| Validation directive counts | deliver 1,437; service 12; wait 551 |
| PPO optimizer steps | 3,920 |
| Checkpoint SHA-256 | `ad4d3164...efd27b60` |

The parent checkpoint reload reproduced the configured validation summary.

### E4B preflight

| Metric | Result |
| --- | ---: |
| Health/model identity | Pass |
| Valid structured responses | 5/5 |
| Invalid responses | 0 |
| Sensitivity pairs | 6/6 valid |
| Sensitivity classes actually checked | `wait`, `service_requested_shelf` |
| Sensitivity classes not established by this gate | The other two directive classes |
| Median latency | 2.559 s |
| p95 latency | 2.862 s |
| p95 limit | 3.920 s |
| Preflight result | Pass |

This passed the configured gate, but it was not an all-four-directive
sensitivity test.

### PPO adaptation training

| Metric | Result |
| --- | ---: |
| Adaptation environment steps | 1,024 |
| Optimizer steps | 64 |
| Gemma calls | 446 |
| Effect violations | 76 |
| Invalid actions | 3 |
| Total fallbacks | 79 |
| Simulator return | 1.580000 |
| Learning return (fallback penalty applied) | 1.501000 |
| Median Gemma latency | 2,934.415 ms |
| p95 Gemma latency | 3,315.890 ms |
| Maximum reference KL | 0.000774 |
| Configured KL ceiling | 0.050000 |
| Directive counts | deliver 435; service 11; wait 578 |

The KL guard was not close to binding. The adaptation nevertheless generated
many effect-invalid actions, which were deterministically replaced by the
fallback policy.

## Per-episode evaluation scores

All evaluation episodes were 500 steps. `Effect` means effect-violating
Gemma responses, `Invalid` means invalid action responses, and `Fallback` is
the total deterministic fallback count. Directive counts are shown as
`deliver / reposition / service / wait`; zero-valued classes are retained for
comparability.

### Executor and Gemma-backed episodes

The executor-only rows are identical for the parent and adapted checkpoints on
the selection and confirmation partitions, so duplicate adapted-executor rows
are not repeated.

| Partition | Policy | Seed | Tasks | Return | Stuck | Clashes | Effect | Invalid | Fallback | Calls | Median / p95 ms | Wall s | Directive counts D/R/S/W |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Matched | Parent executor | 9100 | 3 | 0.8 | 3 | 56 | 0 | 0 | 0 | 0 | — | 1.323 | 459/0/3/38 |
| Matched | Parent + Gemma | 9100 | 3 | 0.8 | 21 | 295 | 59 | 0 | 59 | 273 | 2554.812 / 3061.770 | 709.384 | 269/0/4/227 |
| Selection | Parent executor | 8200 | 2 | -0.2 | 84 | 457 | 0 | 0 | 0 | 0 | — | 1.338 | 61/0/3/436 |
| Selection | Parent executor | 8201 | 3 | 0.8 | 7 | 62 | 0 | 0 | 0 | 0 | — | 1.327 | 453/0/3/44 |
| Selection | Parent + Gemma | 8200 | 3 | 0.8 | 59 | 679 | 2 | 0 | 2 | 171 | 2756.126 / 3185.134 | 474.443 | 168/0/3/329 |
| Selection | Parent + Gemma | 8201 | 4 | 2.0 | 14 | 245 | 24 | 0 | 24 | 260 | 2605.054 / 3267.979 | 682.627 | 244/10/6/240 |
| Selection | Adapted executor | 8200 | 2 | -0.2 | 84 | 457 | 0 | 0 | 0 | 0 | — | 1.311 | 61/0/3/436 |
| Selection | Adapted executor | 8201 | 3 | 0.8 | 7 | 62 | 0 | 0 | 0 | 0 | — | 1.845 | 453/0/3/44 |
| Selection | Adapted + Gemma | 8200 | 3 | 0.8 | 59 | 679 | 2 | 0 | 2 | 172 | 2762.190 / 3784.164 | 487.128 | 169/0/3/328 |
| Selection | Adapted + Gemma | 8201 | 4 | 2.0 | 30 | 298 | 23 | 0 | 23 | 233 | 2608.900 / 3325.001 | 628.347 | 215/12/6/267 |
| Confirmation | Parent executor | 8202 | 3 | 0.8 | 4 | 59 | 0 | 0 | 0 | 0 | — | 1.385 | 454/0/3/43 |
| Confirmation | Parent executor | 8203 | 3 | 0.8 | 4 | 66 | 0 | 0 | 0 | 0 | — | 1.354 | 469/0/3/28 |
| Confirmation | Parent + Gemma | 8202 | 3 | 1.0 | 70 | 415 | 1 | 0 | 1 | 130 | 2632.403 / 3109.217 | 345.270 | 114/10/6/370 |
| Confirmation | Parent + Gemma | 8203 | 8 | 6.8 | 7 | 158 | 82 | 2 | 84 | 273 | 2497.619 / 3232.016 | 696.187 | 256/3/14/227 |
| Confirmation | Adapted executor | 8202 | 3 | 0.8 | 4 | 59 | 0 | 0 | 0 | 0 | — | 1.356 | 454/0/3/43 |
| Confirmation | Adapted executor | 8203 | 3 | 0.8 | 4 | 66 | 0 | 0 | 0 | 0 | — | 1.392 | 469/0/3/28 |
| Confirmation | Adapted + Gemma | 8202 | 3 | 0.8 | 5 | 172 | 133 | 0 | 133 | 458 | 2653.258 / 3128.603 | 1219.360 | 454/0/4/42 |
| Confirmation | Adapted + Gemma | 8203 | 3 | 0.8 | 7 | 215 | 32 | 0 | 32 | 290 | 2486.605 / 3077.068 | 748.571 | 286/0/4/210 |

The parent and adapted executor results were the same in task and return
outcomes on all paired seeds. Their small wall-time differences are process
runtime variation, not policy differences.

### Partition aggregates and gate scores

| Partition / policy | Episodes | Tasks total / mean | Return total / mean | Stuck total | Clashes total | Effect total | Invalid total | Fallback total | Gemma calls | Gemma wall s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched parent executor | 1 | 3 / 3.0 | 0.8 / 0.8 | 3 | 56 | 0 | 0 | 0 | 0 | 1.323 |
| Matched parent + Gemma | 1 | 3 / 3.0 | 0.8 / 0.8 | 21 | 295 | 59 | 0 | 59 | 273 | 709.384 |
| Selection parent executor | 2 | 5 / 2.5 | 0.6 / 0.3 | 91 | 519 | 0 | 0 | 0 | 0 | 2.665 |
| Selection parent + Gemma | 2 | 7 / 3.5 | 2.8 / 1.4 | 73 | 924 | 26 | 0 | 26 | 431 | 1,157.070 |
| Selection adapted + Gemma | 2 | 7 / 3.5 | 2.8 / 1.4 | 89 | 977 | 25 | 0 | 25 | 405 | 1,115.475 |
| Confirmation parent executor | 2 | 6 / 3.0 | 1.6 / 0.8 | 8 | 125 | 0 | 0 | 0 | 0 | 2.746 |
| Confirmation parent + Gemma | 2 | 11 / 5.5 | 7.8 / 3.9 | 77 | 573 | 83 | 2 | 85 | 403 | 1,041.456 |
| Confirmation adapted + Gemma | 2 | 6 / 3.0 | 1.6 / 0.8 | 12 | 387 | 165 | 0 | 165 | 748 | 1,967.930 |

The selection gate passed because parent and adapted Gemma policies tied on
tasks and return, while the adapted policy used 25 fallbacks versus 26 and
retained at least two non-wait directives. This was a one-fallback tie-break,
not a throughput improvement.

The confirmation gate failed because the adapted policy completed 6 tasks and
returned 1.6 total, versus the parent's 11 tasks and 7.8 total. Adapted
fallbacks doubled from 85 to 165, despite fewer invalid-action records.

## Total Gemma evaluation count

There were **9 PPO + Gemma evaluation episodes**:

| Policy | Matched | Selection | Confirmation | Total |
| --- | ---: | ---: | ---: | ---: |
| Parent PPO + Gemma | 1 | 2 | 2 | **5** |
| Adapted PPO + Gemma | 0 | 2 | 2 | **4** |
| **Total** | **1** | **4** | **4** | **9** |

This excludes parent-PPO training episodes, Gemma preflight calls, semantics
probe states, and executor-only comparator episodes.

## Errors, invalidations, and fallbacks

| Location | Effect violations | Invalid actions | Other errors/invalidations | Fallbacks |
| --- | ---: | ---: | --- | ---: |
| Semantics probe | 0 | — | 0 uncovered; 0 executor failures | — |
| Parent PPO validation | — | — | 0 inapplicable selections | — |
| E4B preflight | — | 0 invalid responses | 6/6 sensitivity pairs valid; only 2 directive classes checked | 0 |
| Adaptation training | 76 | 3 | KL max 0.000774, below 0.05 ceiling | 79 |
| Matched parent + Gemma | 59 | 0 | No schema-invalid action record | 59 |
| Selection parent + Gemma | 26 | 0 | No invalid actions | 26 |
| Selection adapted + Gemma | 25 | 0 | No invalid actions | 25 |
| Confirmation parent + Gemma | 83 | 2 | Two invalid actions on seed 8203 | 85 |
| Confirmation adapted + Gemma | 165 | 0 | No invalid-action records, but many effect-invalid actions | 165 |

Fallbacks were deterministic and prevented invalid actions from being sent to
the simulator. They still count as reliability failures and add latency; they
are not equivalent to successful Gemma decisions.

## Observations and analysis

1. The E4B model fit the local GPU memory budget and passed its configured
   five-response validity, six-pair sensitivity, and 2.862-second p95
   preflight gate. The gate covered only `wait` and
   `service_requested_shelf`, so it did not validate all four directive
   classes.
2. The parent PPO completed the configured training run and had a reproducible
   four-seed validation summary, but its validation mean was only 2.75 tasks
   and 0.55 return. This is descriptive prototype behavior, not a main-study
   baseline.
3. Gemma increased throughput on some exploratory paired seeds. Against the
   executor-only parent, parent + Gemma gained 1 task and 1.0 return on 8200,
   gained 1 task and 1.2 return on 8201, gained 5 tasks and 6.0 return on
   8203, and tied on tasks and return on 8202 and matched 9100. It also caused
   more clashes and fallbacks, so this is a throughput signal with a severe
   reliability and latency tradeoff, not evidence of a general improvement.
4. The adapted PPO did not improve the selection primary outcome: it tied the
   parent at 7 total tasks and 2.8 total return. Its apparent selection win
   came only from one fewer fallback.
5. The adapted PPO failed confirmation. On seeds 8202-8203 it completed 6
   tasks with 1.6 total return, while the parent completed 11 tasks with 7.8.
   The child produced 165 fallbacks versus 85, including 165 effect
   violations versus 83. The parent also had two invalid actions, but its
   higher task throughput and return dominate that secondary advantage.
6. The child had fewer stuck events and clashes in the confirmation aggregate
   than the parent (12 vs 77 stucks; 387 vs 573 clashes), but it achieved this
   while doing substantially less useful work. These secondary metrics cannot
   override the declared primary metric of completed tasks.
7. The adaptation KL guard was not close to binding, so the observed
   confirmation regression is not explained by crossing the configured KL
   ceiling. The likely explanation is narrow adaptation to the tiny training
   and selection conditions combined with a high rate of effect-invalid Gemma
   actions. This is an interpretation, not a causal proof.

## Decision

Do not promote the adapted PPO checkpoint. The confirmation gate correctly
rejected it. Keep both checkpoints and all raw manifests immutable, and do not
run the configured final partition from this run. The result is a negative
prototype result: Gemma-backed PPO adaptation was not shown to generalize, and
the matched Gemma controller added roughly twelve minutes of episode wall time
for the same task/return outcome on seed 9100.

The final scientific conclusion remains limited: this directive prototype does
not establish that PPO + Gemma, adapted or otherwise, improves the project's
declared warehouse metric. The nine Gemma-backed episodes are exploratory and
partitioned gate evidence, not the planned ten-seed locked evaluation.

## Follow-up configuration changes

For a new experiment ID, predeclare and test changes only on training and
selection partitions before rerunning an unchanged confirmation set:

1. Increase adaptation beyond 1,024 steps only with a larger, genuinely
   disjoint selection protocol; do not infer a benefit from the one-fallback
   selection tie.
2. Increase the fallback penalty above `0.001` so effect-invalid Gemma actions
   are more costly to PPO. Recheck whether this reduces fallbacks without
   collapsing task throughput.
3. Expand preflight sensitivity to all four directive classes.
4. Consider lowering `max_new_tokens` below 192 or increasing GPU offload only
   as separately versioned runtime experiments after validity is re-established.
   These changes target latency and must not alter this result.
5. Use a new untouched final seed set; do not treat seed 9100 as held out
   because it was already used for the matched demo.

Do not modify the existing checkpoint artifacts or retroactively change the
gate outcome.

## Artifact references

- Parent training manifest: `results/demo/ppo-training/ppo-training-manifest.json`.
- Adaptation manifest: `results/demo/ppo-gemma-fine-tuning/fine-tuning-manifest.json`.
- Selection gate: `results/demo/ppo-gemma-fine-tuning/selection-gate.json`.
- Confirmation gate: `results/demo/ppo-gemma-fine-tuning/confirmation-gate.json`.
- Current preflight: `results/demo/gemma-actioner-preflight.json`.
- Matched demo summary: `results/demo/ppo_executor-ppo_gemma-v4-20260823T133737600946Z-4339e25b/demo-summary-v4.json`.
- Resource samples: `results/demo/resource-logs/`.
