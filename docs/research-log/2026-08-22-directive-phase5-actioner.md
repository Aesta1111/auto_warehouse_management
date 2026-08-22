# 2026-08-22: Directive-demo Phase 5 local actioner

## Objective

Run the directive-demo actioner against the exact local Gemma E4B Q4_0 GGUF
checkpoint on the Windows RTX 4050, then decide the E4B/E2B gate using the
declared validity, directive-sensitivity, and latency criteria.

## Provenance

- Git commit: `fd41e53` plus the dirty Phase 5 implementation and local-runtime changes
- Config: `configs/demo/directive-demo-v1.yaml`
- PPO checkpoint: `results/demo/directive-ppo-v2/directive-ppo-v2.pt` and its manifest
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- Gemma repository: `google/gemma-4-E4B-it-qat-q4_0-gguf`
- Gemma revision: `4b4a2c1d584be7264f87aac328a1bc739ce81b6c`
- GGUF filename: `gemma-4-E4B_q4_0-it.gguf`
- GGUF SHA-256: `676c35070db6dbe52f93e9c864ee0fba4eddea94b9c875d9cb10daff453fbaee`
- Quantization: Q4_0 GGUF; local file size 5,154,941,280 bytes
- llama.cpp: release `b10569`, build `0.2.0-dev`, commit `5a32f7b66`
- llama.cpp runtime archive SHA-256: `f03eb16658f04ae259b759016746f1747d1e3c50c9689e407091aefc34f43635`
- CUDA 12.4 DLL archive SHA-256: `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6`
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB, driver 580.97
- Server settings: localhost:8080, context 4,096, 24 GPU layers, one slot, seed 0
- Estimated and actual cost: USD 0

## Method

The ignored `.local_runtime/` directory contains the pinned llama.cpp Windows
CUDA 12.4 runtime and model. `scripts/start_local_gemma.ps1` verifies both
model and runtime hashes, aliases the server as
`google/gemma-4-E4B-it-qat-q4_0-gguf:Q4_0`, binds it to `127.0.0.1:8080`, and
uses partial GPU offload. The server is started with `--reasoning off` and
`--reasoning-budget 0`; without this, Gemma 4 filled the 192-token response
budget in `reasoning_content` and left `message.content` empty.

`LlamaCppActioner` checks `/health` and `/v1/models`, sends deterministic
strict-schema chat-completion requests, and never retries or repairs an
invalid action. The prompt explicitly constrains same-role destination
duplicates because the TA-RWARE wrapper rejects them. The preflight uses the
frozen v2 PPO checkpoint, five measured requests across `wait` and
`service_requested_shelf`, and three fixed states with the applicable pair
`wait`/`service-requested-shelf`.

Commands:

```powershell
.\scripts\start_local_gemma.ps1 -Prepare
.\.venv\Scripts\python.exe scripts\probe_gemma_actioner.py --config configs\demo\directive-demo-v1.yaml
```

The preflight artifact is `results/demo/gemma-preflight-e4b.json`.

## Results

- Server health and exact model identity: passed.
- Warm-up: passed.
- Measured actions: 5/5 valid; classes covered `wait` and `service_requested_shelf`.
- Sensitivity: 3 states, 6/6 directive-specific responses valid.
- Latency: median 3.49 seconds; p95 3.56 seconds; limit 10 seconds.
- GPU evidence: `nvidia-smi` reported 1,964 MiB used of 6,141 MiB at report time.
- E4B preflight report: `passed: true`; no E2B substitution is needed.
- Offline Phase 5 actioner tests: 10 tests pass before the live run; the full
  suite remains subject to the managed Windows temporary-directory ACL issue
  documented in the earlier Phase 4/Phase 5 notes.

## Interpretation

The local RTX 4050 can serve the pinned E4B Q4_0 model with 24 layers offloaded
and enough headroom for the 4K action prompt. The measured p95 is comfortably
inside the declared gate, and the model produced strict, directive-sensitive
actions for this preflight. This is an operational Phase 5 result, not a claim
that the neural controller improves the research metric.

## Decision

Keep E4B selected. Phase 5 passes. The local server must be started before the
later rendered demo; the demo runner should continue to require and verify the
passing preflight artifact rather than launching a model process implicitly.

## Follow-up

- Preserve `.local_runtime/` outside Git; it is ignored and contains the large
  model/runtime artifacts.
- If the NVIDIA driver, llama.cpp build, context, or GPU-layer count changes,
  rerun the preflight and record a new report.
- The remaining work is the separate Phase 6 matched rendered demo and the
  main research project's prepared Linux/A40 compatibility run.
