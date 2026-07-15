"""Bounded compatibility-spike helpers, not production hybrid training code."""

from warehouse_ai.compatibility.gemma import (
    GEMMA_FALLBACK_ID,
    GEMMA_FALLBACK_REVISION,
    GEMMA_PRIMARY_ID,
    GEMMA_PRIMARY_REVISION,
    SoftTokenProjector,
    run_tiny_gemma_probe,
)

__all__ = [
    "GEMMA_FALLBACK_ID",
    "GEMMA_FALLBACK_REVISION",
    "GEMMA_PRIMARY_ID",
    "GEMMA_PRIMARY_REVISION",
    "SoftTokenProjector",
    "run_tiny_gemma_probe",
]
