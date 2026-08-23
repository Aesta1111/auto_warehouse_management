#!/usr/bin/env bash
set -euo pipefail

# Run inside the pinned RunPod image:
# runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# Clone and check out the implementation commit before invoking this script.

uv sync --frozen
uv pip install --no-deps --force-reinstall \
  "pyastar2d @ git+https://github.com/raulsteleac/pyastar2d_TARWARE@ff9257df2166e659f6c0a51bfc09b119db58d3d8"
uv run --no-sync python scripts/probe_tarware_compatibility.py \
  --seeds 0 1 2 3 4 \
  --output results/summaries/phase1-tarware-linux-upstream.json
uv run --no-sync python scripts/compare_tarware_summaries.py \
  results/summaries/phase1-tarware-local.json \
  results/summaries/phase1-tarware-linux-upstream.json \
  --output results/summaries/phase1-tarware-platform-comparison.json
uv run --no-sync python scripts/probe_llm_cloud.py \
  --output results/summaries/phase1-llm-a40.json
