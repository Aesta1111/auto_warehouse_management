# 2026-08-22: Directive-demo architecture refactor and matched run

## Objective

Refactor the completed directive demo without changing the frozen four-command
catalog, seed partitions, simulator boundary, or historical artifacts. The
work also had to regenerate versioned compatibility evidence and finish the
pending matched 500-step executor/Gemma demonstration.

## Hypothesis

Not applicable to the refactor. The matched demonstration remains unscored and
may show that the Gemma executor is worse than the deterministic executor.

## Provenance

- Branch: `codex/refactor-directive-demo`
- Implementation checkpoints: `1173193`, `ddb2be0`, `ea67a12`, `d2d33d0`
- Evidence checkpoints: `c9be0b1`, `9e74848`, `f0a052e`
- Environment: `tarware-tiny-3agvs-2pickers-partialobs-v1`
- Probe seeds: 8000-8004
- PPO training seeds: 8100-8115
- PPO validation seeds: 8200-8203
- Matched demo seed: 9100
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- Model: `google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0`
- Model SHA-256: `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`
- Runtime: llama.cpp b10569, CUDA 12.4, 24 GPU layers, context 4096
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB; driver 580.97
- Cost: USD 0

## Method

The refactor introduced subsystem contracts, a table-driven directive rule
engine, policy adapters, a transport-only inference backend, one shared action
validator/controller pipeline, and separate PPO and evaluation packages. All
production modules were kept below 300 lines. Contract/config imports were
checked in a fresh interpreter to ensure they did not import Torch, Lightning,
SB3, or httpx.

The directive digest now hashes the ordered semantic rule descriptions plus
`directive-rule-engine-v2`; it no longer hashes an entire Python source file.
The active PPO is a plain `torch.nn.Module`. Lightning Fabric is imported only
inside training, while the historical SB3 reference loads only through its lazy
adapter. No Unsloth, Transformers inference, vLLM, SGLang, TensorRT-LLM, or
Ollama dependency was added. The already working llama.cpp GGUF runtime was
retained because it fits the 6 GB Windows laptop and had an actual 3.56-second
p95 baseline.

Commands:

```text
.venv\Scripts\python.exe scripts/probe_directives.py --config configs/demo/directive-probe-v2.yaml --output results/summaries/directive-probe-report-v2.json
.venv\Scripts\python.exe scripts/train_directive_ppo.py --config configs/demo/directive-ppo-v3.yaml --output results/demo/directive-ppo-v3
.venv\Scripts\python.exe scripts/probe_gemma_actioner.py --config configs/demo/directive-demo-v2.yaml
.venv\Scripts\python.exe scripts/run_directive_demo.py --config configs/demo/directive-demo-v2.yaml
```

## Results

### Static and test gates

- Ruff: passed for `src`, `scripts`, and `tests`.
- basedpyright: 0 errors and 0 warnings.
- Test collection: 51 tests.
- Rule/controller/backend/prompt/contract subset: 32 passed.
- Full demo suite: 44 assertions passed; seven `tmp_path` tests errored during
  setup/cleanup because the managed Windows session denied access to the new
  pytest base directory. This reproduces the earlier documented ACL problem.
- Full repository suite: 67 assertions passed and the same seven temp-dependent
  tests errored; pytest then raised the same ACL error while cleaning its fresh
  OS-temp base directory.

### Directive probe v2

- States: 2,500
- Uncovered states: 0
- Executor failures: 0
- Effect failures: 0
- Report SHA-256: `7f8b0bdd1fbbf88251cc032aae94acffa3a7773d600e3c16c168cac9753830e8`
- Behavior SHA-256: `8235531267098fd36affb7deac93e87bccabbfa267f7f3e218caa13031e4d291`

### PPO schema v3

- Checkpoint SHA-256: `d3df46cc325bd0baee04f46af2bb44d5a0b8dbef9d7a3340d6472f300eb01245`
- Manifest SHA-256: `ab8c9808fc0c819594fb9b6f546c048705ff0b24c56a815a25c73eac570eea5a`
- Completed environment steps: 25,088 for the declared 25,000-step budget
- Held-out mean completed tasks: 2.75
- Non-wait directives used: 2
- Inapplicable selections: 0
- Selected checkpoint point: 5,120 environment steps

### Exact E4B preflight

The first compact-JSON wording run failed: 1/5 measured responses were valid,
4/6 sensitivity responses were valid, and p95 was 4.219 seconds. Following the
predeclared fallback, the architecture remained unchanged while the proven
legacy wording was restored. Same-role groups are now derived dynamically
rather than hard-coded to three AGVs and two pickers.

The rerun passed:

- Validity: 5/5
- Sensitivity: 6/6
- Median: 3.489 seconds
- p95: 3.538 seconds, below the 3.92-second limit
- Report SHA-256: `95031d40d0fcb9d1dcf375635d4a32076cdf2519f4ac7d182ead034a1380a858`
- Deterministic initial prompt SHA-256: `9326a44c3cf3dc86481c304ed2929263519930d237d8cff9605964bef1032443`

### Matched 500-step demonstration

Run directory:
`results/demo/ppo_executor-ppo_gemma-v2-20260822T133517500260Z-f0a052e0`

| Metric | Executor | Gemma |
| --- | ---: | ---: |
| Steps | 500 | 500 |
| Completed tasks | 3 | 2 |
| Total return | 0.8 | -0.2 |
| Clashes | 56 | 457 |
| Stuck events | 3 | 28 |
| LLM calls | 0 | 462 |
| Effect violations | 0 | 361 |
| Deterministic fallbacks | 0 | 361 |
| LLM median latency | n/a | 3,177.8 ms |
| LLM p95 latency | n/a | 3,752.6 ms |

- Summary SHA-256: `cadb939b134015fb50dabbfd1839ef3649fe365adf4841ee153d94a26e092aab`
- Executor JSONL SHA-256: `f52a86559cef4f5a994be8a9ee02974dc691cb5fd93b687d32dc964279ed94b2`
- Gemma JSONL SHA-256: `07089a8aa6b3b2200c30609aa79ef3e8d4f3c674d1c146d3c5d6c4af2e41159f`

## Observations

- The Specification-style rule table preserved all probed observable
  directive semantics with much less directive-specific code.
- The compact wording was materially worse despite using the same state facts
  and schema. Prompt brevity was not automatically better for this model.
- Preflight sensitivity covered wait/service states but did not predict the
  live episode's frequent deliver-directive effect violations.
- Structured failure codes made the 361 live effect failures countable without
  parsing exception class names from text.

## Interpretation

The architectural refactor and reproducibility gates passed. The matched run
does not show an advantage for Gemma: it completed fewer tasks and required
fallback on 361 of 462 model calls. This is useful negative demo evidence and
does not alter the main experiment's metric, protocol, or phase status.

## Decision

Keep llama.cpp as the replaceable local backend and keep the legacy prompt
wording because it meets the declared latency/validity gate. Do not claim that
the directive demo validates the main hybrid hypothesis. A future investigation
may expand preflight coverage to deliver/reposition states, but that is not part
of this semantic-preserving refactor.

## Problems and follow-up

- The repository PowerShell launcher could not resolve `Get-FileHash` in its
  hidden child shell; the already pinned executable was started directly with
  the exact committed arguments.
- Run the full test suite in a normal writable temp environment to close the
  seven pytest ACL errors.
- Preserve the ignored checkpoint and step logs with the run artifacts; do not
  commit the 5.15 GB model or generated checkpoint.
