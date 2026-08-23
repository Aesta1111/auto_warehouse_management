# Neural Directive + Frozen LLM Actioner Demo

> **Current active refresh (2026-08-24):** This section supersedes the deferred
> model/runtime values below for the next prototype run. The historical full run
> and its conclusion remain unchanged in the research log.
>
> - Model: Apache-labeled `bartowski/Qwen_Qwen3.5-2B-GGUF`, revision
>   `7d26695454df6de5fbcce2e58681e62dae06ce43`, file
>   `Qwen_Qwen3.5-2B-Q8_0.gguf`, SHA-256
>   `be647507ce6cde229b838924d47bfff9763171105563f7f908670dae57c4dbe2`.
> - Prompt: `configs/demo/llm-actioner-few-shot-prompt.yaml` metadata plus
>   `llm-actioner-few-shot-prompt.txt`, three examples per directive (12 total),
>   exact template/contract digests, and strict JSON-schema output.
> - PPO: exactly `[256, 256, 256, 256]`, 100,000 parent timesteps, then 4,096
>   frozen-LLM adaptation steps. Both runs require `accelerator: cuda` and record
>   CUDA/Torch/GPU provenance.
> - Runtime: 8,192 context, full permitted GPU offload, deterministic decoding,
>   and `--reasoning off --reasoning-budget 0`.
> - Seeds: matched demo `9300`, confirmation `8202-8203`, final `9200` reserved
>   until confirmation passes.
> - Active names are neutral (`LlmActioner`, `LlmPromptBuilder`, `LlmController`,
>   and `ppo-llm-*` files). Historical logs/results retain their original names.

## 1. Purpose and boundaries

Build a separate, unscored demonstration in which a small neural network reads
the warehouse observation and emits one discrete directive token. A frozen local
LLM interprets that directive together with the visible warehouse state
and produces the concrete simulator action.

For this demo, one LLM response contains exactly one TA-RWARE `JointAction`.
The boundary must be designed so the main research project can later investigate
an ordered sequence of actions, but this prototype must not implement, simulate,
or claim support for action sequences.

## 0. 2026-08-22 architecture refactor status

The demo now uses subsystem packages under `contracts/`, `directives/`, `ppo/`,
`inference/`, and `evaluation/`; every production module is below the 300-line
soft limit. Package initializers are intentionally minimal, and loading a
contract or YAML config does not import Torch, Lightning, SB3, or httpx.

The four directive behavior classes are owned by one `DirectiveCatalog` whose
complete `Directive` records contain both YAML metadata and executable
semantics. One `Registry` implements the shared engine. The behavior digest
covers the canonical ordered semantic descriptors plus
`directive-rule-engine-v2`, rather than whole source-file bytes. The active PPO
is a plain PyTorch model with policy adapters; SB3 remains a lazy historical
reference and Fabric is training-only.

Inference uses `ModelBackend.health()` and `generate()`. `LlamaCppBackend` is
the retained implementation; `LlmActioner` only builds/parses messages, while
the shared `ActionValidator` and controller pipeline own one decode/effect check
and deterministic fallback. Failures are recorded as stable codes rather than
exception class names.

The historical versioned gates passed: the directive probe checked 2,500 states
with zero failures, PPO schema v3 used two non-wait directives with zero
inapplicable selections, and the prior quantized-LLM preflight passed 5/5
validity, 6/6 sensitivity, and 3.538 s p95. A shorter compact prompt failed,
so the proven wording was retained while role groups became dynamic. The
matched 500-step run completed 3 executor tasks and 2 quantized-LLM tasks; the
old model required 361 effect-violation fallbacks. This is negative, unscored
historical prototype evidence.

Schema-v3 few-shot prompt infrastructure was added on 2026-08-23 without a live
model run. It keeps the schema-v2 prompt byte-stable and the directive catalog
frozen, while a separate prompt spec supplies digest-verified instructions and
four synthetic user/assistant examples. It has no validity, latency, or episode
evidence yet.

A separately versioned frozen-LLM PPO adaptation path now exists as offline
infrastructure. It never updates LLM weights: it warm-starts a child PPO from a
separately trained historical checkpoint, executes sampled non-wait directives
through a frozen few-shot LLM, and updates only the child PPO. No active
100,000-step parent training, live adaptation, or evaluation has run yet.
Schema v3 remains the latest measured directive-PPO evidence.

This branch remains isolated from the authoritative projector-and-LoRA
experiment:

- Do not train or fine-tune LLM weights, add LoRA adapters, or inject soft
  tokens in this prototype.
- Do not claim that the demo improves on PPO or validates the main hypothesis.
- Do not count prototype work as completing a research roadmap phase or gate.
- Do not use prototype training or demonstration seeds in later validation or
  final evaluation.
- Keep the declared primary metric and held-out protocol unchanged.

The important demo behavior is visible end-to-end control:

```text
warehouse state -> small neural network -> directive token
                 -> frozen LLM actioner -> validated joint action -> TA-RWARE
```

## 2. Directive contract

### 2.1 Representation

Represent the neural output as a typed `DirectiveToken`, not as free-form text:

- `directive_id`: stable integer action used by the PPO policy.
- `command`: short English command shown to the LLM and in the demo UI.
- `description`: frozen one-sentence operational meaning used by legacy prompts,
  artifact identity, logs, and deterministic fallback reasons.
- `instruction`: schema-v3 prompt-only operational guidance stored outside the
  directive catalog and never used to change executable semantics.
- `applicability_rule`: deterministic rule identifying states where the
  directive is an available PPO action.
- `action_effect_rule`: testable predicate describing what a compliant concrete
  action must do when the directive is applicable.

The neural network predicts only `directive_id`. Application code resolves the
ID through a versioned directive catalog before constructing the LLM prompt.
This prevents spelling differences or invented words from changing behavior.

The exact catalog is intentionally not selected in this plan. Complete a
small directive-catalog spike before training:

1. Inspect representative tiny-layout trajectories and list the distinct
   high-level intents that precede legal assignments.
2. Propose between four and eight mutually distinguishable commands.
3. Define which commands are applicable from each controller-visible state and
   ensure every reachable state has at least one applicable command.
4. Merge or remove commands whose deterministic training executors cannot
   produce meaningfully different valid behavior.
5. Freeze the chosen IDs, English commands, descriptions, and catalog
   version in YAML before final PPO training. Freeze the
   applicability and action-effect rules in code and reference their versions
   from the YAML.

Likely concepts include servicing a requested shelf, delivering a carried
requested shelf, returning a shelf, repositioning an idle agent, and waiting,
but these are candidates rather than decisions. The spike must not invent
semantics that TA-RWARE trajectories do not support.

### 2.2 PPO training executor

Train a masked PPO policy whose discrete action space is the frozen directive
catalog. PPO selects `directive_id`; it does not directly select TA-RWARE
destination IDs. Phase 3 establishes this behavior with SB3-Contrib
`MaskablePPO`; Phase 3.5 preserves the MDP while moving the PPO implementation
into project-owned PyTorch and Lightning components.

PPO training must not call the LLM at every environment step. Instead, each
directive has a deterministic training executor that converts the directive and
current visible state into one strictly valid `JointAction`. The executor may
use the same distance and eligibility facts as the deterministic heuristic, but
its directive-specific action-effect rule must constrain the result. If a
directive is not applicable, mask it before PPO samples an action.

The training environment therefore behaves as:

```text
observation -> masked PPO directive -> deterministic directive executor
            -> validated JointAction -> TA-RWARE reward and next observation
```

At demo time, the LLM replaces only the deterministic directive executor. The PPO
policy, observation encoding, directive catalog, and action mask remain
frozen. This creates an intentional executor-transfer gap: PPO is optimized for
the deterministic executor, so the demo must measure whether the LLM preserves
each directive's action effect. Results cannot attribute differences solely to
PPO policy quality.

### 2.3 Frozen-LLM PPO adaptation

The original deterministic PPO training and failed full run remain historical
control evidence and must not be overwritten. The active refresh first trains
the separately versioned schema-v5 `[256, 256, 256, 256]` parent for 100,000
configured timesteps. After its checkpoint reload, architecture, prediction,
and non-wait coverage gates pass, the adaptation configuration warm-starts a
copy for 4,096 environment steps. Here, “fine-tuning” means PPO adaptation
with frozen Qwen inference in the environment; Qwen parameters, prompt,
quantization, and decoding stay frozen.

```text
observation -> sampled masked PPO directive -> frozen LLM actioner
            -> strict shared validation -> deterministic fallback if needed
            -> TA-RWARE transition -> PPO update
```

The shared `DirectiveExecutionService` owns validation and fallback for both
training and evaluation. Wait bypasses the LLM. Each step records the unmodified
simulator reward and a separate learning reward. The learning reward subtracts
the predeclared `0.001` penalty for any structured LLM failure because the
deterministic fallback would otherwise hide that failure from PPO. Evaluation
always reports raw simulator reward.

The v2 adaptation uses four 256-step rollouts, learning rate `3e-5`, four PPO
epochs, within-update target KL `0.01`, and maximum divergence `0.05` from the
parent over collected states. It uses a fresh optimizer. The child checkpoint
and manifest record parent, prompt, preflight, model, catalog, behavior, seed,
reward, call, failure, latency, and Git identities. Parent artifacts remain
immutable.

Seed use is fixed before live work:

- adaptation: parent training seeds `8100`-`8115`;
- selection: `8200`-`8201`;
- confirmation: `8202`-`8203`;
- final descriptive comparison: `9200` (reserved until confirmation passes).

Parent and child use identical frozen LLM contracts in paired evaluation.
Selection ranks completed tasks first, then raw return, then lower fallback
rate, and rejects collapse to fewer than two non-wait directives. Deterministic
executor episodes remain a cheap catastrophic-forgetting diagnostic. A negative
result keeps the parent and is reported without retuning on later partitions.

## 3. Demo architecture

### 3.1 Runtime data flow

At each decision opportunity:

1. Read the canonical `WarehouseObservation` and current `ActionMask` from the
   existing TA-RWARE wrapper.
2. Encode the observation using the existing deterministic numeric contract.
3. Run the frozen PPO directive policy with the current directive mask
   and choose its deterministic `directive_id`.
4. Resolve the ID to its English command and description.
5. Serialize the directive, compact visible state, action-mask facts, and exact
   output contract into the external deterministic LLM prompt template.
6. Ask the LLM to produce one complete joint destination action using
   schema-constrained JSON.
7. Parse the output and pass it through the existing strict
   `TarwareEnvWrapper.decode` validation path immediately before stepping.
8. On any timeout, server error, malformed output, masked destination,
   cross-agent conflict, action-effect violation, or stale action, execute the
   selected directive's deterministic training executor and log the fallback.
9. Record the PPO directive output, LLM action, validation result, fallback, and
   latency, then advance TA-RWARE.

Call the LLM only when at least one idle agent has a legal nonzero destination.
When every agent is busy or only no-op is legal, apply the validated all-no-op
action without an LLM call.

### 3.2 Prototype interfaces

Add the implementation under an isolated `warehouse_ai.demo` package rather
than scaffolding the unfinished production hybrid package.

The concrete files, classes, and signatures are specified in Section 7. Keep
prototype records strict, immutable, serializable Pydantic models. Keep runtime
services as ordinary typed Python classes or protocols rather than placing live
clients or simulator objects inside Pydantic models.

Use the demo-specific `Actioner.act(ActionerRequest) -> ActionerResult`
boundary. `ActionerResult` contains untrusted parsed destinations plus model,
latency, and token provenance; it does not contain a trusted `JointAction` and
does not receive the simulator wrapper. Sequence planning remains a separate
future design and is not implied by this single-step contract.

### 3.3 LLM output

The demo response is one destination vector in canonical agent order, matching
the existing `TarwareEnvWrapper.decode()` input contract:

```json
{
  "destinations": [12, 0, 0, 12, 0],
  "reason": "Act on the directive using one AGV and one picker."
}
```

Generate with a strict JSON schema that:

- Requires exactly one integer per current agent in canonical order.
- Restricts every destination to the numeric TA-RWARE action range.
- Limits `reason` to 160 characters.
- Forbids additional properties.

JSON-schema generation is only the first guard. Project-owned validation must
still reject the wrong vector length, masked destinations, busy-agent
assignments, and same-role destination conflicts.

### 3.4 LLM prompt

The active byte-stable schema-v2 prompt is described by YAML metadata and an
external TXT template containing only controller-visible information:

- The predicted English command and its frozen description.
- Episode step and the objective of completing requested shelf work.
- Each agent's ID, role, position, busy state, target, and AGV carrying state.
- Requested shelf locations and goal destinations.
- Per-agent legal destination IDs, with zero explained as no-op.
- The single-action JSON contract and a warning that every choice is validated.

The active template supplies three labeled synthetic user/assistant examples
for each of the four directive classes, twelve examples total. Examples vary
agent roles, legal destinations, carrying/requested/goal state, selected
positions, and no-op choices without exposing experiment seeds. The exact TXT
bytes, marker set, template version, and YAML metadata are covered by the
prompt-contract digest; the frozen catalog digest remains unchanged.

Use deterministic decoding, at most 192 generated tokens, and no corrective
retry. The English command guides the action, but it never overrides the action
mask or validator. When the selected directive has a machine-checkable
action-effect rule, reject an otherwise legal action that contradicts that rule
and use the selected directive's deterministic executor fallback.

## 4. Model runtime and fallback

Use the pinned Apache-labeled Qwen3.5-2B Q8 GGUF through the local `llama.cpp`
server and its OpenAI-compatible endpoint. The active runtime uses an 8,192-
token context, full permitted GPU offload, deterministic decoding, and explicit
reasoning-off flags with a zero reasoning budget.

Before running TA-RWARE, perform a Qwen/LLM preflight:

1. Verify server health and requested model identity.
2. Send one warm-up request followed by five measured action requests covering
   at least two directive classes.
3. Confirm that all five responses parse and pass strict action validation.
4. On at least three fixed states where two directives are applicable, send the
   same state with each directive and record whether the actions satisfy their
   different action-effect rules. This is a directive-sensitivity probe, not a
   performance evaluation.
5. Record median and p95 decision latency.

Proceed only when preflight completes without an out-of-memory error and the
existing p95 latency ceiling passes. If the pinned Apache-labeled artifact
fails compatibility, stop for review; do not silently substitute another model
or quantization.

Pin the exact model repository, immutable revision or file digest, GGUF
filename, quantization, `llama.cpp` version, GPU-offload settings, context size,
and server command. Model files remain outside Git.

## 5. Training and demonstration workflow

### 5.1 Directive PPO environment and checkpoint

Add a prototype Gymnasium adapter whose observation is the existing 125-value
tiny-layout encoding and whose action space is `Discrete(vocabulary_size)`.
Expose one boolean mask over directive IDs from the frozen applicability rules.
Its `step(directive_id)` operation must:

1. Reject a masked or unknown directive.
2. Resolve the directive through its deterministic training executor.
3. Strictly validate the resulting joint action without silent repair.
4. Step TA-RWARE and return the unmodified summed simulator reward.
5. Log the directive, executor output, and any executor failure.

The measured historical policies use separate one-64-unit policy and value
paths. The active schema-v5 implementation expresses both as independent `Mlp`
modules sharing one config-owned `hidden_sizes` sequence, fixed to
`[256, 256, 256, 256]`. Use the reused prototype training seeds beginning at
`8100`, the existing validation partitions, and a configured parent budget of
100,000 timesteps. Keep all PPO hyperparameters in a versioned config and
require CUDA for the active run.

Evaluate checkpoints with the deterministic directive executor on validation
episodes and select by mean completed tasks, using mean return as the declared
tie-breaker. Record directive frequencies, masked-action counts, entropy, and
whether the learned policy collapses to one directive. If fewer than two
non-wait directives are selected on validation episodes, revise the catalog
or reward-neutral executor semantics before involving the LLM.

Save the PPO checkpoint outside Git and write a small manifest containing the
catalog and executor versions, training and validation seeds, architecture,
hyperparameters, validation metrics, Git state, dependency versions, and
checkpoint digest. This checkpoint is a prototype artifact and must not become
the PPO checkpoint used by the main scientific experiment.

Phase 3 establishes the verified SB3-Contrib reference implementation. Before
actioner work begins, Phase 3.5 replaces SB3's internally owned policy, rollout
buffer, GAE calculation, PPO loss, and optimizer loop with explicit
project-owned components. The refactor keeps the environment, directive
semantics, seed partitions, validation rule, and public deterministic-prediction
contract fixed. Preserve the Phase 3 checkpoint and manifest as historical
reference artifacts; do not convert or overwrite them.

### 5.2 Demo runner

Add one argparse runner with a strict versioned configuration. It should:

1. Load and verify the directive PPO checkpoint, catalog, executor, and
   manifests.
2. Check the local LLM server and run the Qwen four-class preflight.
3. Run the frozen PPO policy with its deterministic training executor headlessly
   for 500 steps using matched-demo seed `9300`.
4. Reset the identical environment configuration to seed `9300`.
5. Run the same frozen PPO directive policy with the LLM actioner for 500
   rendered steps.
6. Display each directive and concrete LLM action as decisions occur.
7. Print and save a side-by-side descriptive summary.

The summary includes completed tasks, total return, clashes or stuck events,
episode wall time, directive-class counts, LLM call count, median and p95 LLM
latency, invalid action count, action-effect violations, and deterministic
executor fallback count. Per-decision records are JSONL; each controller
receives a small JSON episode summary.

Seeds used for catalog discovery, PPO training, PPO validation, or the demo
become permanently excluded from later scientific evaluation. The two displayed
episodes are not a statistical comparison.

## 6. Repository and documentation changes

- Add the isolated demo package, versioned directive YAML, deterministic
  directive executors, strict PPO configuration, training script, and demo
  runner.
- Add a prototype section to `PROJECT_SPEC.md` documenting this roadmap
  exception and its non-research status.
- Add a time-boxed prototype item to `ROADMAP.md` without marking existing phase
  gates complete.
- Document the directive-catalog decision in a dated research-log entry after
  the spike, including rejected alternatives and applicability coverage.
- Add README commands for training the directive PPO policy, starting the pinned
  local server, and running the demo.
- Ignore model files, generated trajectories, checkpoints, and large logs. Commit
  only code, configs, tests, manifests without personal paths, small summaries,
  and approved research logs.
- Record the first real end-to-end run in a separate dated research-log entry;
  do not pre-write successful results.

## 7. Detailed implementation blueprint

### 7.1 Package and file layout

Create the following prototype-only modules:

```text
configs/demo/
|-- directive-catalog.yaml
|-- directive-semantics-probe.yaml
|-- llm-actioner-few-shot-prompt.yaml
|-- llm-actioner-few-shot-prompt.txt
|-- ppo-model.yaml
|-- llm-actioner.yaml
|-- ppo-training-run.yaml
|-- ppo-llm-fine-tuning-run.yaml
`-- ppo-llm-evaluation-run.yaml
src/warehouse_ai/demo/
|-- contracts/         # Phase-grouped immutable records and configuration
|-- directives/        # Directive table, engine, and live probe
|-- ppo/               # Model, adapters, math, storage, training, validation
|-- inference/         # Backend protocol, llama.cpp, prompt, preflight
|-- evaluation/        # Episode loop, verification, matched runner
|-- controllers.py     # Shared action validator and strategy pipeline
`-- artifacts.py       # Canonical JSON, digests, atomic/append-only writes
scripts/
|-- probe_directives.py
|-- train_directive_ppo.py
|-- probe_llm_actioner.py
`-- run_directive_demo.py
```

All `__init__.py` files remain minimal; callers use direct module imports. Tests
mirror these boundaries under `tests/demo/`. The existing domain models and
`TarwareEnvWrapper` remain authoritative; do not create duplicate observation,
mask, or joint-action types.

Add bounded `httpx` and `PyYAML` runtime dependencies, then let `uv.lock` pin
their exact resolved versions. `httpx` communicates with the local `llama.cpp`
server and `PyYAML.safe_load()` supplies dictionaries to Pydantic. Do not add an
OpenAI SDK because this prototype only needs one local HTTP endpoint.

Phase 3.5 adds a bounded `lightning` development dependency. Keep
SB3-Contrib pinned while the reference checkpoint and parity tests remain
auditable, but the Phase 3.5 training path must not instantiate
`MaskablePPO` or use its rollout buffer, GAE, loss, or optimizer loop.

Class ownership is intentionally grouped by implementation phase:

| Phase | Class group | Primary modules |
| --- | --- | --- |
| 1 | Shared records, annotated field types, YAML loading, and artifact primitives | `contracts/`, `artifacts.py` |
| 2 | Directive specifications, shared execution engine, and probe | `directives/` |
| 3/3.5 | PPO environment, model/adapters, math, storage, collection, training, and validation | `ppo/` |
| 4 | Actioner request/result contracts, shared validation pipeline, and fake backends | `contracts/inference.py`, `controllers.py` |
| 5 | Backend protocol, llama.cpp transport, prompt, preflight, and provenance | `inference/` |
| 6 | Episode records, artifact verification, matched runner, JSONL, and summaries | `evaluation/`, `artifacts.py` |

Do not create placeholder implementations for later phases. Contracts are
grouped by subsystem so adding a phase does not recreate a monolithic model
module or introduce eager framework imports.

### 7.2 Existing code reuse and non-duplication rules

The repository already contains the simulator and domain boundary. Reuse it
directly:

| Existing API | Demo use | Do not recreate |
| --- | --- | --- |
| `TarwareEnvWrapper` | Own the only TA-RWARE instance; reset, observe, mask, decode, step, render, and close | No second simulator wrapper or direct `gym.make()` call in demo code |
| `WarehouseObservation` and `.encode()` | PPO observation and prompt-visible warehouse state | No demo observation model or alternate feature encoder |
| `ActionMask` | Destination validity and directive applicability input | No raw simulator-mask record |
| `AgentAction` and `JointAction` | Only internal concrete-action representation | No duplicate robot/action classes |
| `TarwareEnvWrapper.decode()` | Strictly validate executor and LLM destination vectors | No second destination validator |
| `TarwareEnvWrapper.no_op()` | Automatic no-op path | No hand-built all-zero `JointAction` |
| `TarwareEnvWrapper.step()` | Apply every accepted action | No access to `wrapper.unwrapped` outside compatibility tests |
| `TarwareEnvWrapper.render()` | Render the live demo | No separate renderer |
| `TarwareSb3Env` | Retain for the existing direct-joint-action PPO compatibility path | Do not modify its action semantics for directive PPO |

The only new environment-like class is `Sb3Env`. It is not another
TA-RWARE wrapper: it owns one existing `TarwareEnvWrapper` by composition and
translates a scalar directive ID into a validated `JointAction` through
`Registry`. Its purpose is solely to present
`Discrete(vocabulary_size)` to SB3.

Move the generic scalar mask-aware sampling helper into the existing
`warehouse_ai.env.sb3` module as `MaskAwareDiscrete`, beside the existing
`MaskAwareMultiDiscrete`. `Sb3Env` imports it. Do not create a separate
`MaskAwareDirectiveSpace` with duplicated sampling logic.

`WarehouseContext` is an immutable request snapshot, not an alternative
observation. It contains references to the existing `WarehouseObservation` and
`ActionMask` plus a state digest. `LlmActionResponse` is only the external JSON
wire model; it must be decoded immediately into the existing `JointAction` and
must never become a second internal action type.

### 7.3 Records and configuration (`contracts/`)

All models below inherit a prototype `DemoModel` configured with
`extra="forbid"` and `frozen=True`. Paths are resolved by the scripts and are
not stored as personal absolute paths in committed configs.

```python
class DemoModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class Directive(DemoModel):
    directive_id: NonNegativeInt
    command: str
    description: str
    behavior_key: str
    # Runtime-only fields attached by DirectiveCatalog.load().
    agents: AgentSelector = Field(exclude=True)
    destinations: DestinationSelector = Field(exclude=True)
    effect: EffectCardinality = Field(exclude=True)


class DirectiveCatalog(DemoModel):
    schema_version: Literal[1]
    vocabulary_id: str
    behavior_version: str
    wait_directive_id: NonNegativeInt
    directives: tuple[Directive, ...]

    @classmethod
    def load(cls, path: Path) -> DirectiveCatalog: ...
    def by_id(self, directive_id: int) -> Directive: ...
    @property
    def size(self) -> int: ...
    def digest(self) -> str: ...


class WarehouseContext(DemoModel):
    state_id: str
    observation: WarehouseObservation
    action_mask: ActionMask

    @classmethod
    def capture(cls, wrapper: TarwareEnvWrapper) -> WarehouseContext: ...


class PpoPolicyOutput(DemoModel):
    state_id: str
    episode_step: NonNegativeInt
    directive: Directive
    probabilities: tuple[float, ...]
    value_estimate: float
    policy_latency_ms: NonNegativeFloat
    checkpoint_sha256: str


class ActionerRequest(DemoModel):
    context: WarehouseContext
    prediction: PpoPolicyOutput


class LlmActionResponse(DemoModel):
    destinations: tuple[NonNegativeInt, ...]
    reason: str


class ActionerResult(DemoModel):
    response: LlmActionResponse
    raw_text: str
    model_id: str
    latency_ms: NonNegativeFloat
    prompt_tokens: NonNegativeInt | None = None
    completion_tokens: NonNegativeInt | None = None


class DemoDecision(DemoModel):
    prediction: PpoPolicyOutput
    action: JointAction
    reason: str
    source: Literal["llm", "directive_executor", "automatic_no_op"]
    model_id: str | None
    raw_response: str | None
    directive_effect_valid: bool
    llm_latency_ms: NonNegativeFloat | None
    failure_code: FailureCode | None
    failure_detail: str | None
    prompt_tokens: NonNegativeInt | None
    completion_tokens: NonNegativeInt | None


class DemoStepRecord(DemoModel):
    schema_version: Literal[2]
    decision: DemoDecision
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    simulator_info: dict[str, JsonValue]


class ModelHealth(DemoModel):
    reachable: bool
    advertised_model_id: str | None
    server_version: str | None


class ProbeReport(DemoModel):
    schema_version: Literal[1]
    environment_id: str
    vocabulary_sha256: str
    behavior_sha256: str
    probe_seeds: tuple[int, ...]
    states_checked: PositiveInt
    uncovered_states: NonNegativeInt
    executor_failures: NonNegativeInt
    effect_failures: NonNegativeInt
    passed: bool


class LlmPreflightReport(DemoModel):
    schema_version: Literal[1]
    model_id: str
    model_file_sha256: str
    server_version: str
    context_size: PositiveInt
    gpu_layers: NonNegativeInt
    prompt_schema_sha256: str
    hardware_summary: str
    valid_actions: NonNegativeInt
    measured_actions: PositiveInt
    sensitivity_cases_passed: NonNegativeInt
    sensitivity_cases_total: PositiveInt
    latency_median_ms: NonNegativeFloat
    latency_p95_ms: NonNegativeFloat
    passed: bool
```

`DirectiveCatalog` validation must require four to eight definitions,
contiguous IDs beginning at zero, unique commands and behavior keys, exactly one
valid wait ID, nonblank text, and stable tuple ordering by ID.

`DirectiveCatalog.digest()` hashes its canonical UTF-8 JSON serialization
with sorted object keys. `PpoPolicyOutput.probabilities` always has exactly
`catalog.size` entries in ID order, assigns `0.0` to masked directives, and
sums to one within floating-point tolerance.

`WarehouseContext.capture()` computes `state_id` as SHA-256 over canonical JSON
for the observation plus the action-mask bytes. The same function is used before
and after LLM inference to reject stale responses.

Configuration models:

```python
class ProbeConfig(DemoModel):
    schema_version: Literal[1, 2]
    environment_id: str
    vocabulary_path: Path
    probe_seeds: tuple[int, ...] = (8000, 8001, 8002, 8003, 8004)
    episode_steps: PositiveInt = 500


class PpoConfig(DemoModel):
    schema_version: Literal[1, 2, 3]
    environment_id: str
    vocabulary_path: Path
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int = 8100
    episode_steps: PositiveInt = 500
    total_timesteps: PositiveInt = 25_000
    learning_rate: PositiveFloat = 3e-4
    n_steps: PositiveInt = 256
    batch_size: PositiveInt = 64
    n_epochs: PositiveInt = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01


class LlamaCppBackendConfig(DemoModel):
    backend: Literal["llama_cpp"] = "llama_cpp"
    base_url: AnyHttpUrl
    model_id: str
    model_file_sha256: str
    model_revision: str | None = None
    model_filename: str | None = None
    quantization: str | None = None
    runtime_version: str | None = None
    server_command: str | None = None
    context_size: PositiveInt
    gpu_layers: NonNegativeInt
    request_timeout_seconds: PositiveFloat = 12.0
    max_new_tokens: PositiveInt = 192
    temperature: Literal[0] = 0
    seed: int = 0


class DemoConfig(DemoModel):
    schema_version: Literal[2]
    environment_id: str
    episode_seed: int = 9300
    episode_steps: PositiveInt = 500
    render_hybrid: bool = True
    vocabulary_path: Path
    directive_probe_report_path: Path
    ppo_checkpoint_path: Path
    ppo_manifest_path: Path
    selected_model: Literal["qwen3.5-2b-q8"]
    preflight_report_path: Path
    output_directory: Path
    inference: LlamaCppBackendConfig
    preflight_p95_limit_seconds: PositiveFloat = 3.92


ConfigT = TypeVar("ConfigT", bound=DemoModel)


def load_yaml_config(path: Path, model_type: type[ConfigT]) -> ConfigT: ...
```

Reject overlapping PPO training and validation seeds in configuration. The
active PPO config reuses training seeds `8100` through `8115` and validation
seeds `8200` through `8203`; confirmation uses `8202`-`8203`, matched demo uses
`9300`, and final evaluation reserves `9200`. The runner rejects any overlap
with the probe, PPO training, PPO validation, or final partitions.

The demo config names one already-running Qwen server. The pinned Apache-labeled
artifact is a hard compatibility gate: if preflight fails, stop for review and
do not silently swap models or quantizations. The episode runner never swaps or
launches model processes silently.

### 7.4 Catalog-owned directive engine (`directives/`)

YAML keeps the closed `behavior_key`; `DirectiveCatalog.load()` binds each key
through a private factory and returns complete immutable `Directive` records
containing the command metadata, agent selector, destination selector, and
effect cardinality. One `Registry` receives only that catalog and implements
applicability, deterministic proposal, wrapper decode, and effect validation for
all four commands.

```python
class Registry:
    engine_version: str
    def action_mask(self, context: WarehouseContext) -> NDArray[np.bool_]: ...
    def behavior_digest(self) -> str: ...
    def execute(
        self,
        directive_id: int,
        context: WarehouseContext,
        wrapper: TarwareEnvWrapper,
    ) -> JointAction: ...
    def validate_effect(
        self,
        directive_id: int,
        context: WarehouseContext,
        action: JointAction,
    ) -> None: ...
```

Wait accepts exactly zero nonzero assignments. Every other directive requires
at least one eligible legal assignment. Proposals iterate canonical agent order
and ascending numeric destination IDs, and enforce same-role destination
uniqueness. Effect validation deliberately permits unrelated legal assignments,
matching the frozen observable semantics.

`behavior_digest()` hashes the explicit engine version and canonical ordered
semantic descriptors. Formatting, comments, file splits, and unrelated source edits
do not invalidate a checkpoint. `SemanticsError` carries one stable
`SemanticsFailure` code (`unknown`, `inapplicable`, `executor`, or `effect`)
instead of using one exception subclass per condition.

### 7.5 Directive SB3 facade (`sb3.py`)

```python
class Sb3Env(gym.Env[NDArray[np.float32], int]):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        environment_id: str,
        catalog: DirectiveCatalog,
        registry: Registry,
        episode_seeds: Sequence[int],
        episode_steps: int = 500,
    ) -> None: ...

    @property
    def context(self) -> WarehouseContext: ...
    def action_masks(self) -> NDArray[np.bool_]: ...
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]: ...
    def step(
        self, action: int | NDArray[np.int64]
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...
```

`Sb3Env` constructs exactly one `TarwareEnvWrapper` and delegates all
simulator lifecycle and action validation to it. The observation space copies
the shape and dtype of `WarehouseObservation.encode()`. The action space is the
shared `MaskAwareDiscrete(catalog.size, self.action_masks)` from
`warehouse_ai.env.sb3`. Its `sample()` honors an explicit Gymnasium mask when
provided and otherwise samples from the live mask provider, allowing
`check_env()` to exercise only applicable directives. `reset(seed=...)` uses
that explicit seed; otherwise it cycles deterministically through
`episode_seeds`. The episode counter resets only when a new facade instance is
constructed.

`step()` accepts only a scalar integer, captures context, verifies the directive
mask, calls `registry.execute(..., wrapper=self.wrapper)`, and steps the wrapper.
Reward is the sum of TA-RWARE rewards with no shaping. It sets `truncated=True`
at `episode_steps`, preserves simulator termination separately, and adds only
serializable diagnostic fields to `info`: directive ID, command, raw executed
destinations, and executor status.

Executor failure is an invariant violation during PPO training and raises; it
does not silently fall back. Silent training recovery would teach against a
different transition function.

### 7.6 PPO adapters and training (`ppo/`)

```python
class PpoPolicy:
    def __init__(
        self,
        adapter: PolicyAdapter,
        catalog: DirectiveCatalog,
        checkpoint_sha256: str,
    ) -> None: ...

    @classmethod
    def load(
        cls,
        checkpoint: Path,
        catalog: DirectiveCatalog,
        manifest: PpoManifest,
    ) -> PpoPolicy: ...

    def predict(
        self,
        context: WarehouseContext,
        directive_mask: NDArray[np.bool_],
    ) -> PpoPolicyOutput: ...


def train_directive_ppo(
    config: PpoConfig,
    catalog: DirectiveCatalog,
    registry: Registry,
    output_directory: Path,
) -> PpoManifest: ...


def evaluate_directive_ppo(
    policy: PpoPolicy,
    config: PpoConfig,
    catalog: DirectiveCatalog,
    registry: Registry,
) -> PpoValidationSummary: ...
```

`predict()` verifies mask shape and at least one valid action. The selected
adapter returns masked probabilities and a value estimate; the facade chooses
the highest-probability valid ID and returns probabilities in directive-ID
order. Equal probabilities select the lowest directive ID. It must not sample
during validation or the demo.

`train_directive_ppo()` constructs separate training and validation environment
instances and a plain `PpoModel(torch.nn.Module)`. The explicit
collector, rollout buffer, GAE, clipped loss, and update loop use Lightning
Fabric only for device/optimizer orchestration. Evaluate every 5,000 training
steps, retain only the best checkpoint by the declared validation rule, and
write the manifest after hashing the saved file. The schema-v1 SB3 path is
loaded lazily only to reproduce the historical reference.

`PpoManifest` schema v3 additionally records the behavior engine
version and semantic behavior digest. It also records package versions, seeds,
hyperparameters, architecture/implementation identities, checkpoint SHA-256,
validation summary, Git commit, dirty flag, and UTC timestamp. Loading fails if
any current identity differs.

Schema v4 additionally requires the complete positive `hidden_sizes` sequence,
uses it in the architecture digest, and declares the
`directive-ppo-mlp-v1` implementation. Schema-v2/v3 manifests keep their scalar
`hidden_size` identity and are never rewritten.

### 7.7 PPO model and buffer ownership refactor (Phase 3.5)

Phase 3.5 is a controlled breaking refactor of the Phase 3 PPO implementation.
It does not change the directive MDP. `Sb3Env`, observation encoding,
directive masks, deterministic executors, rewards, episode limits, training and
validation seeds, checkpoint-selection rule, and validation metrics remain
fixed. Only ownership of the PPO algorithm moves from SB3-Contrib into explicit
project classes.

The refactor must create these modules and responsibilities:

| Module | Required ownership |
| --- | --- |
| `ppo/model.py` | Plain PyTorch actor/value networks, masking, action evaluation, and loss integration |
| `ppo/buffer.py` | Fixed-size rollout storage, lifecycle validation, finalized minibatch views, and deterministic shuffling |
| `ppo/math.py` | Pure GAE and PPO-loss calculations with no simulator, model, logging, or filesystem dependencies |
| `ppo/training.py` | Collection/update alternation, validation, checkpoint selection, and training-only Fabric orchestration |
| `ppo/policy.py` | Lazy SB3/PyTorch adapters and the deterministic inference facade |

Tensor-bearing training records are ordinary frozen dataclasses or named
tuples, not Pydantic models. Pydantic remains reserved for serialized configs,
manifests, predictions, and validation summaries.

```python
@dataclass(frozen=True)
class PpoForwardOutput:
    policy_logits: Tensor
    values: Tensor
    policy_latent: Tensor
    value_latent: Tensor


@dataclass(frozen=True)
class PpoActionSample:
    actions: Tensor
    log_probabilities: Tensor
    values: Tensor


@dataclass(frozen=True)
class PpoBatch:
    observations: Tensor
    action_masks: Tensor
    actions: Tensor
    old_log_probabilities: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor


@dataclass(frozen=True)
class PpoLossTerms:
    total: Tensor
    policy: Tensor
    value: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
```

The trainable model is a plain PyTorch module; Fabric remains training
orchestration only:

```python
class Mlp(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Sequence[int],
    ) -> None: ...


class PpoModel(nn.Module):
    def __init__(
        self,
        observation_size: int,
        directive_count: int,
        hidden_sizes: Sequence[int],
        learning_rate: float,
        clip_range: float,
        entropy_coefficient: float,
        value_coefficient: float,
        normalize_advantage: bool,
    ) -> None: ...

    def forward(
        self,
        observations: Tensor,
        action_masks: Tensor,
    ) -> PpoForwardOutput: ...

    def act(
        self,
        observations: Tensor,
        action_masks: Tensor,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> PpoActionSample: ...

    def evaluate_actions(self, batch: PpoBatch) -> tuple[Tensor, Tensor, Tensor]: ...
    def compute_loss(self, batch: PpoBatch) -> PpoLossTerms: ...
    def make_optimizer(self) -> Optimizer: ...
```

`Mlp.layers` is an `nn.ModuleList` containing every hidden and output linear
layer. Apply fixed `tanh` only after hidden layers and expose the last hidden
activation. Schema v4 requires a non-empty positive `hidden_sizes` sequence and
uses the same sequence for separate, non-weight-sharing policy and value MLPs.
`forward()` must expose the policy latent explicitly even
though this directive demo does not send it directly to the LLM. Invalid directive logits
are masked before constructing `torch.distributions.Categorical`; masked
directives must have exactly zero reported probability. `act(deterministic=True)`
uses lowest-ID argmax tie-breaking, while rollout collection samples from the
same masked probabilities through `torch.multinomial()` with the supplied
seeded PyTorch generator.

`PpoModel.compute_loss()` is the authoritative training calculation.
It delegates the pure arithmetic to `compute_ppo_loss()` and owns coefficients,
logging names, and optimizer configuration. It may not collect rollouts or
access an environment.

```python
def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    bootstrap_masks: Tensor,
    trace_masks: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]: ...


def compute_ppo_loss(
    batch: PpoBatch,
    new_log_probabilities: Tensor,
    new_values: Tensor,
    entropy: Tensor,
    *,
    clip_range: float,
    entropy_coefficient: float,
    value_coefficient: float,
    normalize_advantage: bool,
) -> PpoLossTerms: ...
```

Use the clipped surrogate policy loss, unclipped MSE value loss, and positive
mean categorical entropy. Define `total` as policy loss plus the weighted value
loss minus the weighted entropy. Normalize advantages within each minibatch when
configured, use the Phase 3 `1e-8` stability epsilon, and compute reverse
approximate KL and clip fraction for logging and optional early stopping. Do not
introduce value clipping or reward normalization during this refactor.

GAE must distinguish simulator termination, time-limit truncation, and an
ordinary rollout-buffer boundary. A terminated transition has no value
bootstrap and stops the recursive trace. A truncated transition bootstraps from
the returned terminal observation but stops the trace before the reset state.
An unfinished transition at the end of a rollout bootstraps and continues its
trace normally. Encode those rules explicitly through `bootstrap_masks` and
`trace_masks`; do not collapse them into one ambiguous `done` flag.

The rollout buffer owns storage only:

```python
class RolloutBuffer:
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        directive_count: int,
    ) -> None: ...

    def add(
        self,
        *,
        observation: NDArray[np.float32],
        action_mask: NDArray[np.bool_],
        action: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        old_value: float,
        next_value: float,
        old_log_probability: float,
    ) -> None: ...

    def finalize(self, *, gamma: float, gae_lambda: float) -> None: ...
    def iter_minibatches(
        self,
        batch_size: int,
        *,
        generator: torch.Generator,
    ) -> Iterator[PpoBatch]: ...
    def reset(self) -> None: ...
```

Preallocate observations as `float32`, masks as `bool`, actions as `int64`, and
scalar training fields as `float32`. Reject overflow, partial finalization,
minibatch access before finalization, and mutation after finalization. Store the
exact action mask used to produce each old log probability and reuse it when
evaluating the action during PPO updates. The buffer must not own an environment,
model, optimizer, logger, or checkpoint path.

Collection and optimization remain separate classes:

```python
class RolloutCollector:
    def collect(
        self,
        module: PpoModel,
        environment: Sb3Env,
        buffer: RolloutBuffer,
        *,
        rollout_steps: int,
        generator: torch.Generator,
    ) -> None: ...


class PpoTrainingLoop:
    def fit(
        self,
        module: PpoModel,
        config: PpoConfig,
        training_environment: Sb3Env,
        validation_environment: Sb3Env,
        output_directory: Path,
    ) -> PpoManifest: ...
```

`RolloutCollector` runs inference mode, samples only applicable
directives, records the returned next-state value before resetting a terminated
or truncated episode, and fills exactly one buffer. It never performs an
optimizer step. `PpoTrainingLoop` alternates one finalized rollout
with `n_epochs` of seeded minibatches, clips gradients, optionally stops an
update on a declared target KL, validates every 5,000 environment steps, and
selects checkpoints by the unchanged Phase 3 rule.

Use one long-lived model and optimizer for the entire run. Standard Lightning
`Trainer.fit()` assumes a comparatively stable dataset and must not be forced
to own simulator interaction through a DataLoader or private loop API. Use
Lightning Fabric for device placement, backward, gradient clipping, logging,
and checkpoint support around the explicit on-policy loop. The
`PpoModel` still owns forward, action evaluation, loss, metrics, and
optimizer configuration. Record the exact Lightning and Fabric versions in the
manifest.

Seed the environment, NumPy, PyTorch, minibatch generator, and rollout-action
generator from explicit config-derived values. Checkpoints are written only at
completed PPO-update boundaries, so no partially filled rollout buffer needs to
be serialized or resumed.

Refactor `PpoPolicy` to wrap a frozen `PpoModel` while
preserving its existing `predict(context, directive_mask)` behavior and
`PpoPolicyOutput` output. The new loader verifies the checkpoint,
catalog, behavior, architecture, and implementation digests before loading
the state dictionary on CPU.

Advance `PpoConfig` and `PpoManifest` to schema version 2 and
write a dedicated Phase 3.5 run config (since retired by the composed config
system). Add explicit fields for
`value_coefficient`, `max_grad_norm`, `normalize_advantage`, optional
`target_kl`, model implementation version, buffer schema version, GAE version,
loss version, and backend identity. Preserve the Phase 3 v1 config, SB3 zip,
manifest, and research-log entry unchanged. There is no checkpoint conversion:
Phase 3.5 retrains from the same declared seeds and budget and produces a new
checkpoint and manifest.

Exact weight or probability equality with SB3 is not expected because
initialization, minibatch order, and optimizer details can differ. Compare the
two implementations descriptively on the same prototype validation seeds and
record directive counts, completed tasks, returns, entropy, and invalid-action
behavior. The Phase 3.5 gate is correctness and reproducibility, not making the
new implementation outperform the SB3 reference.

Time-box the implementation to five working days, with up to two additional
days for deterministic reload and truncation fixes. If the explicit loop cannot
preserve optimizer state, masking, GAE semantics, and reproducible checkpoint
reload without private Lightning APIs, record the failed investigation and
stop before Phase 4 rather than hiding the problem behind the old SB3 model.

### 7.8 Prompt contracts and construction (`contracts/prompting.py`, `inference/prompting.py`)

```python
class ChatMessage(DemoModel):
    role: Literal["system", "user", "assistant"]
    content: str


class PromptEnvelope(DemoModel):
    messages: tuple[ChatMessage, ...]
    response_schema: dict[str, Any]


class LlmPromptBuilder:
    def __init__(self, prompt_spec=None, catalog=None) -> None: ...
    @property
    def contract_digest(self) -> str | None: ...
    def build(self, request: ActionerRequest) -> PromptEnvelope: ...
    def serialize_state(self, context: WarehouseContext) -> str: ...
    def response_schema(self, context: WarehouseContext) -> dict[str, Any]: ...
```

For schema v2, use the fixed system message and one user message containing
sections in this exact order: objective, directive, agents, requested shelves,
goals, legal destinations, and output instructions. For schema v3, use the
prompt-spec system message, four synthetic user/assistant pairs, and a final
live user message with the same full section order. Sort all records by their
existing canonical order. Do not include PPO probabilities, value estimates,
or experiment seed identifiers; the LLM should interpret the chosen token, not
second-guess PPO confidence.

The generated JSON Schema requires `destinations` and `reason`, forbids extra
keys, and gives `destinations` `minItems == maxItems == len(agents)`. Integer
bounds are `0` through `action_mask.destination_count - 1`. The prompt explains
which vector position belongs to each agent and lists each row's legal IDs.

### 7.9 Backend adapter and LLM actioner (`inference/`)

```python
class ModelBackend(Protocol):
    def health(self) -> ModelHealth: ...
    def generate(self, prompt: PromptEnvelope) -> GenerationResult: ...
    def close(self) -> None: ...


class Actioner(Protocol):
    def act(self, request: ActionerRequest) -> ActionerResult: ...


class LlmActioner:
    def __init__(
        self, backend: ModelBackend, prompts: LlmPromptBuilder
    ) -> None: ...
    def act(self, request: ActionerRequest) -> ActionerResult: ...
```

`LlamaCppBackend.health()` calls `/health`, then `/v1/models`, and rejects a
server whose advertised model does not match the configured model ID.
`generate()` posts to `/v1/chat/completions` with deterministic sampling and
this response wrapper:

```python
{
    "type": "json_schema",
    "json_schema": {
        "name": "warehouse_single_action",
        "strict": True,
        "schema": envelope.response_schema,
    },
}
```

The backend parses only the HTTP envelope and returns text plus provenance.
`LlmActioner` validates only `LlmActionResponse`; it never receives a
wrapper or directive registry. `ActionValidator` performs the one stale-state,
vector-length, wrapper-decode, canonical-order, and directive-effect pass.

`InferenceError`, `ActionerError`, and `DecisionFailure` each carry a stable
`FailureCode`: `health`, `timeout`, `transport`, `malformed_response`,
`invalid_action`, `effect_violation`, `stale_state`, or `executor_failure`.
There is no retry inside the adapter. An injected `httpx.Client` keeps envelope
tests independent of a real server.

### 7.10 Controllers (`controllers.py`)

```python
class DemoController(Protocol):
    @property
    def controller_name(self) -> Literal["ppo_executor", "ppo_llm"]: ...
    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...


class ExecutorController:
    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
    ) -> None: ...
    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...


class LlmController:
    def __init__(
        self,
        policy: PpoPolicy,
        registry: Registry,
        actioner: Actioner,
    ) -> None: ...
    def choose_action(self, wrapper: TarwareEnvWrapper) -> DemoDecision: ...
```

Both controllers capture context, compute the same directive mask, and obtain a
deterministic prediction from the same frozen PPO policy.

One `DecisionPipeline` owns prediction and fallback; executor and LLM are
replaceable `ActionStrategy` implementations. The LLM strategy bypasses the
LLM for wait and otherwise sends untrusted destinations through the shared
validator. Any structured failure invokes `registry.execute()` for the same
selected directive. If that fallback unexpectedly fails, apply
`wrapper.no_op()`, set `source="automatic_no_op"`, and record an invariant-level
failure; do not terminate the live demo.

Neither controller calls `wrapper.project()`. The returned `DemoDecision.action`
is the exact action subsequently sent to `wrapper.step()`.

### 7.11 Episode runner and artifacts (`evaluation/`, `artifacts.py`)

```python
class PpoValidationSummary(DemoModel):
    seeds: tuple[int, ...]
    mean_completed_tasks: NonNegativeFloat
    mean_return: float
    directive_counts: dict[str, NonNegativeInt]
    non_wait_directives_used: NonNegativeInt
    inapplicable_directives: NonNegativeInt


class PpoManifest(DemoModel):
    schema_version: Literal[1]
    environment_id: str
    vocabulary_sha256: str
    behavior_sha256: str
    checkpoint_sha256: str
    train_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    algorithm_seed: int
    hyperparameters: dict[str, JsonValue]
    validation: PpoValidationSummary
    package_versions: dict[str, str]
    git_commit: str
    dirty_worktree: bool
    created_at_utc: datetime


class EpisodeResult(DemoModel):
    controller: Literal["ppo_executor", "ppo_llm"]
    seed: int
    steps: PositiveInt
    completed_tasks: NonNegativeInt
    total_return: float
    clashes: NonNegativeInt
    stuck_events: NonNegativeInt
    wall_seconds: NonNegativeFloat
    directive_counts: dict[str, NonNegativeInt]
    llm_calls: NonNegativeInt
    invalid_actions: NonNegativeInt
    effect_violations: NonNegativeInt
    fallbacks: NonNegativeInt
    llm_latency_median_ms: NonNegativeFloat | None
    llm_latency_p95_ms: NonNegativeFloat | None


class DemoRunSummary(DemoModel):
    schema_version: Literal[1]
    demo_only: Literal[True]
    environment_id: str
    seed: int
    ppo_checkpoint_sha256: str
    vocabulary_sha256: str
    directive_probe_report_sha256: str
    preflight_report_sha256: str
    executor_episode: EpisodeResult
    llm_episode: EpisodeResult
    output_directory: Path


def run_episode(
    wrapper: TarwareEnvWrapper,
    controller: DemoController,
    *,
    seed: int,
    max_steps: int,
    render: bool,
    decision_log_path: Path,
) -> EpisodeResult: ...


def run_matched_demo(config: DemoConfig) -> DemoRunSummary: ...


def sha256_file(path: Path) -> str: ...
def capture_git_state(repository: Path) -> tuple[str, bool]: ...
def write_json_atomic(path: Path, model: DemoModel) -> None: ...
```

`run_episode()` resets explicitly, chooses exactly one action per simulator
step, optionally renders after each step, writes one flushed JSONL record per
decision, and stops on all-agent termination, all-agent truncation, or
`max_steps`. Derive completed tasks and collision/stuck metrics from the actual
pinned TA-RWARE `info` keys verified by the compatibility spike; do not invent
new simulator meanings.

`run_matched_demo()` verifies manifests and seed exclusions, runs the PPO plus
deterministic executor first, creates a fresh wrapper for the PPO plus LLM run,
and saves both episode results plus one comparison summary. It labels all output
`demo_only: true` and never computes significance or an improvement claim.

Manifest verification checks checkpoint, catalog, behavior-registry, and
preflight-report digests before either episode starts. `write_json_atomic()`
writes to a sibling temporary file and replaces only the final small JSON
artifact after serialization succeeds; JSONL decision records remain append-only
within a newly created run directory.

Write artifacts under
`results/demo/<run-id>/`, where the UTC run ID includes the controller name,
timestamp, and short Git commit. Small summaries may be committed; raw JSONL and
checkpoints remain ignored.

### 7.12 Script contracts

Each script returns exit code zero only when its stated artifact is complete:

```text
python scripts/probe_directives.py --config configs/demo/directive-semantics-probe.yaml --output results/summaries/directive-semantics-probe-report.json
python scripts/train_directive_ppo.py --config configs/demo/ppo-training-run.yaml
python scripts/probe_llm_actioner.py --config configs/demo/llm-actioner.yaml
pwsh -File scripts/start_local_llm.ps1
python scripts/run_directive_demo.py --config configs/demo/ppo-llm-evaluation-run.yaml
python scripts/finetune_directive_ppo_with_llm.py --config configs/demo/ppo-llm-fine-tuning-run.yaml
python scripts/evaluate_llm_finetuned_ppo.py --config configs/demo/ppo-llm-evaluation-run.yaml --partition selection
```

- `probe_directives.py` exercises applicability, executors, and action effects
  across representative seeded trajectories and writes coverage only; it does
  not train.
- `train_directive_ppo.py` resolves `ppo-training-run.yaml` with
  `ppo-model.yaml`, trains, evaluates, and writes a checkpoint manifest. The
  configured MLP successor remains untrained and must create new artifacts;
  historical checkpoint manifests keep their original schema identities.
- `probe_llm_actioner.py` performs the Qwen health, JSON, four-class sensitivity,
  model-identity, prompt-contract, and latency preflight.
- `start_local_llm.ps1` starts the pinned Qwen server with the configured context,
  offload, deterministic decoding, and reasoning-off settings. Model artifacts
  remain outside Git.
- `run_directive_demo.py` requires a passing actioner preflight artifact for the
  selected model, runs both matched episodes, and prints artifact locations.

### 7.13 Implementation order and gates

Implement the following class groups in order. Each phase should be one focused
change that leaves the repository runnable and tested.

The phase bullets below preserve the original implementation history. For the
current paths, contracts, schema versions, and runtime boundaries, Sections 0
and 7.1-7.12 are authoritative; the refactor did not rewrite old phase evidence.

#### Phase 1: Shared contracts and configuration

Create:

- In `models.py`: `DemoModel`, `Directive`,
  `DirectiveCatalog`, `WarehouseContext`, `ProbeConfig`,
  `PpoConfig`, `LlamaCppConfig`, and `DemoConfig`.
- In `artifacts.py`: `sha256_file()`, `capture_git_state()`,
  `write_json_atomic()`, and canonical JSON helpers used by model digests and
  `WarehouseContext.capture()`.
- In `models.py`: `ProbeReport` because it is the output contract for
  the next phase.

Do not create PPO, prompt, controller, or LLM classes yet. Phase 1 may import
only the existing project-owned domain records and `TarwareEnvWrapper`.

Deliverables:

- The four validated YAML config files parse through `load_yaml_config()`.
- Canonical catalog and state digests repeat byte-for-byte.
- Invalid IDs, duplicate behavior keys, overlapping seeds, absolute committed
  paths, and malformed URLs fail before simulator construction.

Gate: all Phase 1 model and artifact tests pass, and loading each committed
config performs no network, model, or simulator work.

#### Phase 2: Directive semantics and deterministic executors

Create in `directives.py`:

- `DirectiveBehavior` protocol.
- `Registry`.
- `SemanticsError`, `UnknownSemanticsError`,
  `InapplicableSemanticsError`, `DirectiveExecutorError`, and
  `DirectiveEffectError`.
- One concrete `DirectiveBehavior` class per command retained by the catalog
  spike. These concrete class names are decided by that spike and recorded in
  its research-log entry; they are not guessed in advance.

Create `probe_directives.py` only after the registry API is tested. The script
loads Phase 1 models, constructs real `WarehouseContext` values, and writes a
`ProbeReport`.

Phase 2 depends only on Phase 1 plus existing domain/environment classes. It
must not import SB3, HTTP clients, prompt builders, or controllers.

Deliverables:

- Every sampled state has at least one applicable directive, including wait.
- Every applicable executor returns a destination tuple that
  `TarwareEnvWrapper.decode()` converts into a valid `JointAction` satisfying
  its directive-effect rule.
- Each behavior is deterministic under repeated identical contexts.
- `behavior_digest()` changes when any behavior binding or implementation
  changes.

Gate: `probe_directives.py` produces a passing report for all committed probe
seeds. Freeze `directive-catalog.yaml` and the behavior version before Phase 3.

#### Phase 3: Directive PPO

Create:

- In existing `warehouse_ai.env.sb3`: add generic `MaskAwareDiscrete` beside
  `MaskAwareMultiDiscrete`; do not change `TarwareSb3Env` behavior.
- In `demo/sb3.py`: `Sb3Env`, which composes exactly one
  `TarwareEnvWrapper`.
- In the Phase 3 group of `models.py`: `PpoPolicyOutput`,
  `PpoValidationSummary`, and `PpoManifest`.
- In `ppo.py`: `PpoPolicy`, `train_directive_ppo()`, and
  `evaluate_directive_ppo()`.
- `train_directive_ppo.py` as the thin command-line adapter around those APIs.

Phase 3 may depend on Phases 1-2, Gymnasium, SB3-Contrib, and PyTorch. It must
not know about the LLM, prompts, HTTP, actioner plans, or demo rendering.

Deliverables:

- `Sb3Env` passes Gymnasium checks and MaskablePPO consumes its
  one-dimensional directive mask.
- Executor failures stop training instead of silently changing transitions.
- PPO save/reload preserves the deterministic directive and probabilities for
  fixed observation-mask pairs.
- Validation writes a manifest tied to catalog, behavior, checkpoint, seed,
  dependency, and Git digests.
- The selected policy uses at least two non-wait directives on validation
  episodes.

Gate: `train_directive_ppo.py` creates a verified checkpoint and passing
validation manifest from the committed config.

#### Phase 3.5: Repository-owned PPO model, buffer, and Lightning loop

Treat this as the one authorized breaking change to the completed Phase 3 PPO
implementation. Keep Phase 3's environment and prediction contracts stable,
but replace the `MaskablePPO` training core with the classes in Section 7.7.
Preserve the Phase 3 checkpoint, manifest, config, and research log as the SB3
reference; do not rewrite their history.

Create:

- In `ppo_model.py`: `PpoForwardOutput`, `PpoActionSample`, `PpoLossTerms`, and
  `PpoModel`.
- In `ppo_buffer.py`: `PpoBatch` and `RolloutBuffer`.
- In `ppo_math.py`: `compute_gae()` and `compute_ppo_loss()`.
- In `ppo_training.py`: `RolloutCollector` and
  `PpoTrainingLoop`.
- In `models.py`: schema-v2 `PpoConfig` and
  `PpoManifest` fields for implementation provenance.
- In `ppo.py`: the refactored `PpoPolicy` adapter and stable public
  training/evaluation functions.
- A historical schema-v2 run config using the same seeds, episode length,
  training budget, architecture, and checkpoint-selection rule as Phase 3.

Phase 3.5 may depend on Phases 1-3, PyTorch, and the bounded Lightning package.
It may use SB3 only in parity tests and reference-result loading. Production
Phase 3.5 model, collection, GAE, loss, and optimization code must not import or
instantiate `MaskablePPO` or `MaskableRolloutBuffer`.

Deliverables:

- Hand-calculated GAE tests pass for termination, truncation, and unfinished
  rollout boundaries.
- Hand-calculated PPO-loss tests cover clipping, value loss, entropy,
  approximate KL, and advantage normalization.
- Buffer lifecycle, dtypes, shapes, mask preservation, seeded shuffling, and
  minibatch coverage are deterministic and fully tested.
- The model assigns zero probability to masked directives, exposes its policy
  latent, and produces valid stochastic training actions and deterministic
  validation actions.
- One short training smoke run proves that rollout collection and optimizer
  state alternate without reset under the explicit Lightning/Fabric loop.
- Full-budget training writes a schema-v2 checkpoint manifest and the unchanged
  validation routine can load and evaluate it.
- Reloading the new checkpoint reproduces deterministic directives,
  probabilities, values, and validation summaries for fixed seeds.
- A dated research-log entry compares the new run descriptively with the Phase
  3 SB3 reference and records implementation cost or failed attempts.

Gate: the historical Phase 3.5 training run creates a
digest-verified schema-v2 checkpoint, the focused and repository quality gates
pass, and validation still selects at least two non-wait directives with zero
inapplicable selections. Phase 4 must consume only this Phase 3.5 checkpoint.

#### Phase 4: Actioner-neutral orchestration with a fake client

Create:

- In `contracts/inference.py` and `contracts/evaluation.py`: `ActionerRequest`,
  `LlmActionResponse`, non-generic `ActionerResult`, `DemoDecision`, and
  `DemoStepRecord`.
- In `inference/prompting.py`: `ChatMessage`, `PromptEnvelope`, and
  `LlmPromptBuilder`.
- In `inference/actioner.py`: the demo-specific `Actioner` protocol and
  structured `ActionerError`.
- In `controllers.py`: `DemoController`,
  `ExecutorController`, and `LlmController`.
- In tests only: `FakeActioner`, implemented against the public `Actioner`
  protocol rather than added to production code.

Phase 4 depends on Phases 1-3.5. Production Phase 4 code must not import
`httpx` or assume a particular model server.

Deliverables:

- Prompt text and JSON schema are deterministic for a fixed request.
- The executor controller and fake-actioner controller select the same frozen
  PPO directive from the same context.
- Valid fake plans reach the wrapper unchanged.
- Malformed, invalid, effect-violating, timed-out, and stale fake plans use the
  same directive's deterministic executor fallback.
- The demo-specific actioner seam works without defining any future
  sequence-plan class or trusting model output as a `JointAction`.

Gate: short fake-actioner episodes complete for success and every fallback
scenario with exact decision records.

#### Phase 5: Real local LLM actioner

Create:

- In `contracts/inference.py`: `ModelHealth` and
  `LlmPreflightReport`.
- In `inference/llama_cpp.py`: `LlamaCppBackend` using an injected or owned
  `httpx.Client`.
- `probe_llm_actioner.py` as the only script that decides whether a configured
  server passes preflight.

Phase 5 depends on the Phase 4 protocol and prompt contracts. Do not change PPO,
directive behavior, or controller APIs to accommodate model-specific quirks.

Deliverables:

- Health and advertised model identity are verified before generation.
- The pinned JSON-schema request shape returns strictly parsed single actions.
- Qwen passes validity, four-class directive sensitivity, model identity,
  prompt-contract, and p95 latency gates.
- Model file, prompt schema, server, offload, context, and hardware provenance
  are recorded.

The later schema-v3 preflight must additionally record the exact prompt-contract
digest and fail closed if it differs from the configured prompt spec. Its live
gate remains pending and must cover all four directive classes before a matched
episode is interpreted.

Gate: `probe_llm_actioner.py` produces a passing preflight report for the
exact server that will be used by the demo.

#### Phase 6: Episode evaluation and matched demonstration

Create:

- In the Phase 6 group of `models.py`: `EpisodeResult` and `DemoRunSummary`.
- In `evaluation.py`: `run_episode()` and `run_matched_demo()`.
- In `artifacts.py`: any remaining run-directory and append-only JSONL helpers.
- `run_directive_demo.py` as composition only; it must not contain controller
  or metrics logic.

Phase 6 composes the frozen public APIs from Phases 1-5. It may not revise the
directive catalog, behavior rules, PPO checkpoint, prompt schema, or
preflight result during an episode.

Deliverables:

- One headless PPO-plus-executor episode and one rendered
  PPO-plus-LLM-actioner episode use the same seed, settings, catalog, and
  frozen PPO checkpoint.
- Each decision is flushed to JSONL, and small episode and comparison summaries
  are atomically written.
- Digest mismatches stop before reset; runtime LLM failures remain visible as
  logged deterministic-executor fallbacks.
- README, project-spec exception, roadmap item, and dated research logs describe
  observed results without making a statistical improvement claim.

Gate: both 500-step episodes finish without manual intervention and satisfy all
acceptance criteria in Section 9.

Do not start a later phase when the current gate fails. Fix or revise the
current phase's contract instead of placing cross-phase workarounds in a later
module.

After a phase gate passes, treat that phase's public APIs as frozen for this
prototype. A necessary breaking change must rerun that phase's gate and every
later completed gate. Internal refactors that preserve signatures, serialized
schemas, digests, and behavior need only rerun affected tests.

## 8. Tests

### Unit tests

- Directive YAML rejects duplicate IDs, duplicate commands, invalid versions,
  and vocabularies outside the four-to-eight class limit.
- Applicability and action-effect rules are deterministic and covered for every
  frozen directive.
- Every reachable test state has at least one applicable directive.
- Each deterministic directive executor uses only controller-visible state and
  returns a strict valid action satisfying its action-effect rule.
- Directive masks match the frozen catalog and reject inapplicable actions.
- Shared `MaskAwareDiscrete.sample()` never returns a masked directive, and the
  directive facade passes Gymnasium's checker without changing
  `TarwareSb3Env`.
- PPO policy input, action distribution, and value shapes match the frozen
  observation and catalog.
- Phase 3.5 masked logits give inapplicable directives zero probability, and
  the model exposes stable policy- and value-latent shapes.
- `compute_gae()` matches hand-calculated termination, truncation, and rollout-
  boundary cases without leaking a reset state's value backward.
- `compute_ppo_loss()` matches hand-calculated unclipped and clipped policy
  terms, value loss, entropy, approximate KL, and clip fraction.
- `RolloutBuffer` rejects overflow and premature minibatch access,
  preserves the exact action mask and old log probability for each transition,
  and yields every finalized item once per seeded epoch.
- Fixed PPO checkpoints, observations, and masks produce identical directive
  predictions.
- Compact prompt serialization is deterministic and contains no hidden state.
- JSON schemas match the current canonical agent count and destination range.
- Valid LLM output decodes to the expected `JointAction`.
- Malformed JSON, wrong-length vectors, out-of-range or masked destinations,
  same-role conflicts, timeouts, server errors, and stale state all trigger the
  deterministic directive-executor fallback.
- Checkpoint, catalog, behavior, prompt-schema, model-file, and preflight
  digest mismatches stop before an episode begins.
- Decision logs, manifests, and episode summaries validate against their
  schemas.

### Integration tests

- Run the directive Gymnasium adapter through environment checks and a short
  multi-seed rollout.
- Train MaskablePPO briefly and save/reload an identical directive prediction.
- Run one Phase 3.5 collect/update cycle with the explicit buffer and
  Lightning/Fabric loop, verify that optimizer state advances without reset,
  and save/reload an identical deterministic prediction.
- Evaluate the SB3 reference and schema-v2 checkpoints through the same
  validation function and record descriptive differences without requiring
  identical weights or scores.
- Run a short TA-RWARE episode with a fake actioner returning valid actions.
- Force malformed and delayed fake responses and verify the episode completes
  with logged deterministic-executor fallbacks.
- Feed identical applicable states with different directives to the fake
  actioner and verify directive-specific action-effect validation.
- Run the real LLM preflight separately from ordinary automated tests and
  record it as a local hardware smoke test.

## 9. Acceptance criteria

The first prototype is complete when:

- The directive-catalog spike freezes a versioned four-to-eight command
  catalog with deterministic applicability, executor, and action-effect
  rules.
- A clean command trains or loads the directive PPO checkpoint and reproduces
  its manifest and validation summary.
- Phase 3.5 owns the actor/value model, rollout buffer, GAE, PPO loss, rollout
  collector, and optimization loop in separately tested modules; later demo
  phases load only its schema-v2 checkpoint.
- The selected PPO policy uses at least two non-wait directives on validation
  episodes and never submits an inapplicable directive.
- Qwen preflight passes for all four directive classes with the pinned model and
  prompt contract.
- The directive-sensitivity probe demonstrates that the selected local model
  can produce actions satisfying at least two distinct directive-effect rules;
  otherwise the prototype is reported as blocked rather than treating the
  neural output as meaningful.
- Both matched 500-step episodes complete without manual intervention.
- The rendered episode visibly reports the directive, LLM action, reason,
  validation result, latency, and fallback status.
- Every LLM action is strictly validated before reaching TA-RWARE.
- Exact configuration and artifact identities are preserved without committing
  generated trajectories, base models, or checkpoints.
- Documentation labels the result as a demonstration and describes future
  action sequences only as an unresolved main-project extension.
