# Fast PPO + Frozen Gemma Demo Prototype

## 1. Purpose and boundaries

Build a separate, unscored demonstration in which a small MaskablePPO policy
proposes valid warehouse joint actions and a frozen local Gemma model selects
among them. The demo exists to give the research team a working visual system
quickly; it is not evidence for or against the core projector-and-LoRA research
hypothesis.

This prototype must remain isolated from the authoritative experiment:

- Do not train Gemma, add LoRA adapters, inject soft tokens, or create a
  refinement dataset.
- Do not claim that the prototype improves on PPO, even if its one demonstration
  episode has a higher score.
- Do not count prototype work as completing a research roadmap phase or exit
  gate.
- Do not use prototype training or demonstration seeds in later validation or
  final evaluation.
- Keep the declared research metric and held-out protocol unchanged.

The first completed demo runs two matched 500-step episodes: a headless PPO-only
episode followed by a rendered PPO-plus-Gemma episode. Both use the same PPO
checkpoint, TA-RWARE configuration, environment seed, and episode length.

## 2. Architecture

### 2.1 Data flow

At each simulator step:

1. Read the canonical `WarehouseObservation` and current `ActionMask` from the
   existing TA-RWARE wrapper.
2. Use the frozen MaskablePPO policy to obtain masked categorical probabilities
   for each agent.
3. Construct up to three highest-scoring complete joint actions with a
   best-first beam search.
4. Strictly validate and deduplicate every candidate. Busy agents must select
   no-op, masked destinations are forbidden, and agents with the same role
   cannot select the same nonzero destination.
5. If fewer than two distinct candidates exist, or no idle agent has a nonzero
   choice, apply PPO candidate zero without calling Gemma.
6. Otherwise serialize the compact visible state, candidates, ranks, and scores
   into a deterministic prompt.
7. Ask the local Gemma server to choose one candidate using schema-constrained
   JSON.
8. Validate the selected candidate immediately before applying it. On any model
   or validation failure, apply PPO candidate zero and log the fallback.
9. Record the decision and advance TA-RWARE.

The PPO-only controller uses exactly the same candidate generator and always
chooses candidate zero. This ensures that the demonstration comparison differs
only in Gemma's reranking step.

### 2.2 Prototype interfaces

Add the prototype under an isolated `warehouse_ai.demo` package rather than
building the unfinished production hybrid package or full Typer CLI.

Define these strict immutable records:

- `ActionCandidate`
  - `candidate_id`: zero-based identifier local to one decision.
  - `action`: already validated `JointAction`.
  - `ppo_rank`: one-based PPO ranking.
  - `joint_log_score`: sum of the selected per-agent log probabilities.
  - `normalized_score`: candidate score normalized across the offered set.
- `CandidateProposal`
  - Observation step and stable identifier.
  - Between one and three unique `ActionCandidate` values.
- `GemmaSelection`
  - `candidate_id`: restricted to the IDs in the current proposal.
  - `reason`: presentation-only explanation limited to 160 characters.
- `DemoDecisionRecord`
  - Step, candidate vectors and scores, selected candidate, model reason,
    latency, model identity, and fallback status.

The Gemma response schema is:

```json
{
  "candidate_id": 0,
  "reason": "Short explanation"
}
```

The schema must forbid additional properties and dynamically restrict
`candidate_id` to the candidate IDs included in the current prompt. The reason
is displayed and logged but never changes validation or simulator behavior.

### 2.3 Candidate generation

Use the MaskablePPO policy distribution rather than repeatedly sampling
actions. Apply the current per-agent masks before ranking. Build joint actions
incrementally in canonical agent order, keeping the highest accumulated
log-probability partial assignments and pruning same-role destination conflicts
as soon as they appear.

Continue the search until three unique, strictly valid joint actions are found
or the valid search space is exhausted. Do not use `TarwareEnvWrapper.project`
for demo candidates: silent repair would make the action shown to Gemma differ
from the action executed by the simulator.

Use summed log-probability for ordering. Convert the final candidates to
normalized top-k scores for display and prompting. Stable tie-breaking uses the
lexicographic destination vector, so fixed observations and checkpoints always
produce the same candidate order.

### 2.4 Gemma prompt

The prompt contains only controller-visible information:

- Episode step and the objective of completing requested shelf work.
- Each agent's ID, role, position, busy state, target, and AGV carrying state.
- Requested shelf locations, goal destinations, and positions referenced by
  the candidates.
- Each candidate's ID, destination vector decoded by agent, PPO rank, and
  normalized score.
- A clear instruction to choose exactly one offered candidate and never invent
  an agent or destination.

Use deterministic decoding, a maximum of 64 generated tokens, and no corrective
retry. A timeout, HTTP error, malformed response, unexpected property, invalid
candidate ID, or stale action produces an immediate logged fallback to candidate
zero. Avoiding retries keeps the live demonstration responsive.

## 3. Model runtime and fallback

Use Google's official instruction-tuned Gemma E4B QAT Q4 GGUF checkpoint through
a pinned `llama.cpp` server and its local OpenAI-compatible endpoint. The model
is approximately 5.15 GB, so partial GPU offload is permitted on the laptop's
6 GB RTX 4050. The demo client should depend only on a small HTTP client; the
model server remains an explicit external process.

Before running TA-RWARE, perform an E4B preflight:

1. Verify the server health endpoint and requested model identity.
2. Send one warm-up selection followed by five measured schema-constrained
   selections.
3. Confirm all five responses validate against their dynamic schemas.
4. Record median and p95 decision latency.

Use E4B when the preflight completes without an out-of-memory error and p95
latency is at most 10 seconds. Otherwise use the official Gemma E2B QAT Q4 GGUF
checkpoint with the same prompt, schema, and server interface. Clearly display
and record any fallback to E2B; do not describe an E2B run as E4B.

Pin and record the exact model repository, immutable revision or file digest,
GGUF filename, quantization, `llama.cpp` version, GPU-offload settings, context
size, and server command. Model files remain outside Git.

## 4. Training and demonstration workflow

### 4.1 PPO checkpoint

Add a small argparse training script that:

- Uses the existing tiny TA-RWARE environment and `TarwareSb3Env` adapter.
- Trains MaskablePPO for 25,000 timesteps.
- Uses environment seed `8100` and an independently recorded algorithm seed.
- Saves the checkpoint plus a small JSON manifest outside Git.
- Records the Git commit, dirty state, environment ID, dependency versions,
  seeds, timesteps, and checkpoint digest.

The checkpoint is for the prototype only and must not become the frozen PPO
checkpoint used by the scientific experiment.

### 4.2 Demo runner

Add one argparse demo runner with a strict versioned configuration. It should:

1. Load the prototype PPO checkpoint and verify its manifest.
2. Check the local Gemma server and run the E4B/E2B preflight.
3. Run PPO-only headlessly for 500 steps with environment seed `9100`.
4. Reset the same environment configuration to seed `9100`.
5. Run the Gemma-reranked controller for 500 steps with rendering enabled.
6. Print and save a side-by-side summary.

The summary includes completed tasks, total return, clashes or stuck events,
episode wall time, LLM call count, model-selection count by candidate rank,
median and p95 LLM latency, invalid response count, and PPO fallback count.
Per-decision records are written as JSONL, and each controller receives a small
JSON episode summary.

Prototype seeds `8100` and `9100` become permanently excluded from future
validation and held-out final evaluation. Neither episode result is a
statistical comparison.

## 5. Repository and documentation changes

- Add the isolated demo package, strict configuration, training script, and
  runner without implementing the planned production CLI or hybrid trainer.
- Add a prototype section to `PROJECT_SPEC.md` documenting the explicit roadmap
  exception and its non-research status.
- Add a time-boxed prototype item to `ROADMAP.md` without marking existing phase
  gates complete.
- Add README commands for installing or starting the pinned local server,
  training the prototype PPO, and running the matched demonstration.
- Add model files, prototype checkpoints, manifests containing personal paths,
  and large logs to the appropriate ignore rules. Commit only configuration,
  code, tests, small summaries, and approved logs.
- Create a dated research-log entry after the first real run, recording model
  selection, latency, failures, hardware, observations, and the decision that
  follows. Do not pre-write successful results.

## 6. Tests

### Unit tests

- Masked top-three extraction returns deterministic candidates in descending
  score order.
- Busy agents receive only destination zero.
- Masked destinations and same-role duplicate destinations never appear.
- Candidate actions are unique and stable ties use lexicographic ordering.
- PPO-only candidate zero and hybrid candidate zero decode identically.
- Compact state serialization is deterministic and contains no hidden simulator
  state.
- Dynamic JSON schemas include only the current candidate IDs.
- Valid Gemma responses select the correct existing action.
- Unknown IDs, malformed JSON, additional properties, timeouts, server errors,
  and stale actions all fall back to PPO candidate zero.
- Decision logs and episode summaries validate against their schemas.

### Integration tests

- Run a short TA-RWARE episode using a stub selector that always chooses
  candidate zero.
- Run a short episode using a fake local HTTP server that selects different
  valid candidate IDs.
- Force malformed and delayed responses and verify the episode still completes
  using logged fallbacks.
- Save and reload the PPO checkpoint and confirm identical top-three candidates
  for a fixed observation.
- Run the real Gemma preflight separately from ordinary automated tests and
  record it as a local hardware smoke test.

## 7. Acceptance criteria

The first prototype is complete when:

- A documented clean command trains or loads the prototype PPO checkpoint.
- The E4B preflight either passes or deterministically selects and records the
  E2B fallback.
- Both matched 500-step episodes complete without manual intervention.
- The hybrid episode renders and visibly reports PPO candidates, Gemma's chosen
  candidate, its reason, latency, and any fallback.
- Every action applied by the hybrid is proven to equal an offered candidate or
  the logged PPO fallback.
- PPO-only and hybrid summaries use the same checkpoint, seed, environment, and
  episode length.
- Exact configuration and artifact identities are preserved without committing
  base models or checkpoints.
- Documentation clearly labels the result as a demonstration rather than
  evidence that the hybrid outperforms PPO.
