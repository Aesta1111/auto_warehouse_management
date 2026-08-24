"""Focused tests for the metric helpers in the TA-RWARE smoke runner."""

from pathlib import Path

import pytest

from scripts.smoke_tarware import (
    _guard_output_paths,
    parse_gpu_output,
    stable_sha256,
    summarize_gpu,
)


def test_parse_gpu_output_supports_nvidia_na_values() -> None:
    rows = parse_gpu_output(
        "0, NVIDIA GeForce RTX 4050 Laptop GPU, 7, 1234, 6141, [N/A]\n"
    )

    assert rows == [
        {
            "gpu_index": 0,
            "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
            "gpu_utilization_percent": 7.0,
            "gpu_memory_used_mib": 1234.0,
            "gpu_memory_total_mib": 6141.0,
            "gpu_power_watts": None,
        }
    ]


def test_stable_sha256_canonicalizes_mapping_order() -> None:
    assert stable_sha256({"b": [2, 3], "a": 1}) == stable_sha256(
        {"a": 1, "b": [2, 3]}
    )


def test_summarize_gpu_aggregates_first_gpu() -> None:
    summary = summarize_gpu(
        [
            {
                "gpu_index": 0,
                "gpu_name": "GPU",
                "sample_type": "start",
                "gpu_utilization_percent": 0.0,
                "gpu_memory_used_mib": 8.0,
                "gpu_memory_total_mib": 6141.0,
                "gpu_power_watts": 1.5,
            },
            {
                "gpu_index": 0,
                "gpu_name": "GPU",
                "sample_type": "interval",
                "gpu_utilization_percent": 10.0,
                "gpu_memory_used_mib": 10.0,
                "gpu_memory_total_mib": 6141.0,
                "gpu_power_watts": 2.5,
            },
        ],
        [],
    )

    assert summary["sample_count"] == 2
    assert summary["interval_sample_count"] == 1
    assert summary["gpu_utilization_percent"] == {
        "count": 2,
        "mean": 5.0,
        "min": 0.0,
        "max": 10.0,
    }
    assert summary["gpu_memory_used_mib"]["mean"] == 9.0


def test_guard_output_paths_refuses_late_overwrite(tmp_path: Path) -> None:
    existing = tmp_path / "summary.json"
    existing.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        _guard_output_paths((existing, None), overwrite=False)

    _guard_output_paths((existing,), overwrite=True)
