"""Declarative directive semantics and deterministic execution."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from warehouse_ai.demo.directives.registry import (
        BEHAVIOR_ENGINE_VERSION,
        Registry,
        SemanticsError,
        SemanticsFailure,
        build_registry,
    )

__all__ = [
    "BEHAVIOR_ENGINE_VERSION",
    "Registry",
    "SemanticsError",
    "SemanticsFailure",
    "build_registry",
]


def __getattr__(name: str) -> Any:
    """Load the execution engine lazily to keep contracts framework-free."""
    if name in __all__:
        from warehouse_ai.demo.directives import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
