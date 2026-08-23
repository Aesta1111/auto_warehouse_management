"""Compact deterministic rendering for validated few-shot examples."""

from __future__ import annotations

from warehouse_ai.demo.artifacts import canonical_json_text
from warehouse_ai.demo.contracts.prompting import DirectivePrompt


def render_example_user(directive: DirectivePrompt) -> str:
    example = directive.example
    sections = [
        "SYNTHETIC FEW-SHOT EXAMPLE",
        "This is a teaching example, not the live warehouse state.",
        "DIRECTIVE",
        f"command: {directive.command}",
        f"instruction: {directive.instruction}",
        "COMPACT STATE",
        f"agent roles by position: {canonical_json_text(example.agent_roles)}",
        (f"idle agent positions: {canonical_json_text(example.idle_agent_positions)}"),
        (
            "carrying-shelf positions: "
            f"{canonical_json_text(example.carrying_shelf_positions)}"
        ),
        (
            "requested destination IDs: "
            f"{canonical_json_text(example.requested_destination_ids)}"
        ),
        (f"goal destination IDs: {canonical_json_text(example.goal_destination_ids)}"),
        "LEGAL DESTINATIONS",
    ]
    sections.extend(
        f"position {row}: {canonical_json_text(destinations)}"
        for row, destinations in enumerate(example.legal_destinations)
    )
    sections.extend(
        (
            "EXPECTED OUTPUT",
            "Return only the demonstrated JSON shape for this synthetic state.",
        )
    )
    return "\n".join(sections)
