# 2026-06-20: TA-RWARE source and installation audit

## Objective

Pin TA-RWARE, inspect its public environment contract, and determine whether its
smallest registered environment can run in the current native Windows setup.

## Hypothesis

Not applicable. This was a compatibility investigation.

## Provenance

- Project Git commit: `5097fb5`
- Dirty worktree at start: no
- TA-RWARE repository: <https://github.com/uoe-agents/task-assignment-robotic-warehouse>
- TA-RWARE commit: `6109c33f7228cace1e463c81f0568b754ff5f198`
- TA-RWARE package version: `1.1.0`
- `pyastar2d_TARWARE` commit resolved by `uv`:
  `ff9257df2166e659f6c0a51bfc09b119db58d3d8`
- Host: Windows, Python 3.12.4, `uv` 0.9.5, MSVC 14.50
- WSL: not installed
- Cost: USD 0
- Date: 2026-06-20 (Asia/Bangkok)

## Method

1. Queried the GitHub API for the upstream default branch and HEAD revision.
2. Read the pinned `README.md`, `pyproject.toml`, `LICENSE`, environment
   registration, warehouse implementation, heuristic, and runner.
3. Downloaded the pinned source archive into a temporary directory.
4. Created a disposable Python 3.12 virtual environment with `uv`.
5. Attempted a normal TA-RWARE installation.
6. After its legacy pathfinder build failed, preinstalled its build requirements
   and retried `pyastar2d_TARWARE` without build isolation.

No source or dependency was copied into the project repository.

## Results

### Verified upstream facts

- TA-RWARE commit `6109c33...` is MIT licensed and declares package version
  `1.1.0`.
- Its metadata advertises Python 3.9 but does not declare `requires-python`.
  Python 3.12 support therefore remains unproven.
- The smallest documented environment is
  `tarware-tiny-3agvs-2pickers-partialobs-v1`. Registrations use a 500-step
  limit and support `partial` and `global` observations.
- The action space is one discrete macro-action per agent. Actions select
  warehouse shelf or goal locations; navigation is handled internally with A*.
- `reset(seed=...)` returns only a tuple of agent observations, rather than the
  modern Gymnasium `(observation, info)` pair.
- `step(actions)` returns observations, per-agent rewards, per-agent termination
  flags, per-agent truncation flags, and a shared info dictionary. The source
  currently returns the same flags for termination and truncation.
- The info dictionary exposes shelf deliveries, clashes, stuck events, AGV and
  picker travel distance, idle counts, and agent busy flags.
- An upstream FIFO/nearest-agent heuristic is included in
  `tarware/heuristic.py`.
- No public simulator snapshot or restore API was found during source review.

### Installation outcome

The ordinary `uv pip install` failed because the unpinned Git dependency
`pyastar2d_TARWARE` uses legacy `setuptools` behavior that invokes `pip` inside
an isolated build environment where `pip` is absent.

Preinstalling `pip`, `setuptools`, `wheel`, NumPy, and Cython and disabling build
isolation progressed to C++ compilation. Compilation then failed under MSVC:

- the extension hard-codes GCC flags (`-O3`, `-Wall`, `-shared`, `-fpic`);
- MSVC ignored those flags; and
- `src/cpp/astar.cpp` failed to parse its use of `and` in the current toolchain.

TA-RWARE could not be imported because `pyastar2d` was not built, so the runtime
smoke episode and rendering test were not completed.

## Observations

- Upstream dependency pinning is incomplete: TA-RWARE points to the pathfinder
  repository without an immutable revision.
- The README's example names the two `step` flags in the opposite order from
  the implementation signature. Our adapter must follow tested behavior, not
  copy those variable names.
- TA-RWARE is multi-agent and does not directly satisfy Stable-Baselines3's
  single-agent Gymnasium contract. A project-owned wrapper is required.

## Interpretation

Native Windows is not a reliable authoritative execution environment for the
pinned upstream code without maintaining a patch or fork of its compiled
pathfinder dependency. That would add avoidable scope. Linux is the conservative
target because the extension is written and configured for a GCC-like compiler.

The observation/action interface is suitable for task-assignment research, but
Gymnasium and Stable-Baselines3 compatibility must be provided by an adapter.
State cloning should be treated as unavailable until a deterministic replay
test or a carefully validated snapshot mechanism proves otherwise.

## Decision

- Pin TA-RWARE to `6109c33f7228cace1e463c81f0568b754ff5f198` for the next test.
- Pin `pyastar2d_TARWARE` transitively to
  `ff9257df2166e659f6c0a51bfc09b119db58d3d8` rather than following its HEAD.
- Run the authoritative simulator smoke test on Linux, initially RunPod or a
  disposable Linux environment. Do not patch the dependency for native Windows
  unless Linux execution also fails or local execution becomes essential.
- Do not create the project lockfile until a working Python version and Linux
  installation command are verified.

## Problems and follow-up

1. Run the tiny partial-observation environment and upstream heuristic on Linux.
2. Record exact observation shapes, action cardinalities, reset/step values,
   episode metrics, determinism, and render behavior.
3. Test a thin Gymnasium wrapper suitable for Stable-Baselines3.
4. Test deterministic replay as the provisional substitute for state cloning.
5. Audit the exact Gemma checkpoint separately; it was not part of this slice.
