"""Small non-secret runtime provenance helpers."""

from __future__ import annotations

import platform
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def hardware_provenance() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            facts["cuda_device"] = torch.cuda.get_device_name(0)
    except (ImportError, RuntimeError) as exc:
        facts["torch"] = f"unavailable: {type(exc).__name__}"
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        try:
            result = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.used,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                facts["nvidia_smi"] = result.stdout.strip()
            else:
                facts["nvidia_smi_error"] = result.stderr.strip() or "non-zero exit"
        except (OSError, subprocess.TimeoutExpired) as exc:
            facts["nvidia_smi_error"] = type(exc).__name__
    return facts
