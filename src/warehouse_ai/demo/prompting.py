"""Deterministic, actioner-neutral prompt construction for the directive demo."""

from __future__ import annotations

from typing import Any, Literal

from warehouse_ai.demo.artifacts import canonical_json_text
from warehouse_ai.demo.models import (
    ActionerRequest,
    DemoModel,
    DirectiveContext,
)


class ChatMessage(DemoModel):
    """One message in the local chat-completions request."""

    role: Literal["system", "user"]
    content: str


class PromptEnvelope(DemoModel):
    """Prompt messages and the strict response schema sent to an actioner."""

    messages: tuple[ChatMessage, ...]
    response_schema: dict[str, Any]


_SYSTEM_MESSAGE = (
    "You are the actioner for a warehouse controller. Interpret the selected "
    "directive using only the visible warehouse state and legal destinations. "
    "Return one complete joint destination action. Every choice is validated "
    "before it reaches the simulator."
)


def _position_record(position: Any) -> list[int]:
    return [int(position.row), int(position.column)]


def _agent_record(agent: Any) -> dict[str, Any]:
    """Serialize only the controller-visible fields for one agent."""
    target = None if agent.target is None else _position_record(agent.target)
    return {
        "agent_id": agent.agent_id,
        "role": agent.role.value,
        "position": _position_record(agent.position),
        "target": target,
        "target_destination_id": agent.target_destination_id,
        "busy": agent.busy,
        "carrying_shelf": agent.carrying_shelf,
        "carrying_requested_shelf": agent.carrying_requested_shelf,
        "toggling_load": agent.toggling_load,
    }


def _shelf_record(shelf: Any) -> dict[str, Any]:
    return {
        "destination_id": shelf.destination_id,
        "position": _position_record(shelf.position),
    }


class GemmaPromptBuilder:
    """Build one reproducible prompt and schema from a request snapshot."""

    def build(self, request: ActionerRequest) -> PromptEnvelope:
        """Create the fixed two-message prompt used by Phase 4 and later."""
        context = request.context
        observation = context.observation
        directive = request.prediction.directive

        sections = [
            "OBJECTIVE",
            (
                f"At episode step {observation.step}, complete requested shelf work "
                "with one joint action. Use destination 0 for an agent that should "
                "receive no new assignment."
            ),
            "DIRECTIVE",
            f"command: {directive.command}",
            f"description: {directive.description}",
            "AGENTS",
        ]
        for index, agent in enumerate(observation.agents):
            sections.append(
                f"position {index} ({agent.agent_id}): "
                f"{canonical_json_text(_agent_record(agent))}"
            )

        sections.append("REQUESTED SHELVES")
        requested_shelves = tuple(
            shelf for shelf in observation.shelves if shelf.requested
        )
        if requested_shelves:
            sections.extend(
                f"{canonical_json_text(_shelf_record(shelf))}"
                for shelf in requested_shelves
            )
        else:
            sections.append("none")

        sections.append("GOALS")
        sections.append(
            f"goal destination IDs: "
            f"{canonical_json_text(tuple(int(value) for value in observation.goal_destination_ids))}"
        )

        sections.append("LEGAL DESTINATIONS")
        for index, agent in enumerate(observation.agents):
            legal_destinations = tuple(
                destination
                for destination, is_legal in enumerate(
                    context.action_mask.values[index]
                )
                if bool(is_legal)
            )
            sections.append(
                f"position {index} ({agent.agent_id}): "
                f"{canonical_json_text(legal_destinations)}"
            )

        sections.extend(
            (
                "OUTPUT INSTRUCTIONS",
                (
                    "Return exactly one JSON object with exactly the keys "
                    "destinations and reason. destinations must contain one "
                    f"integer for each agent in positions 0 through {len(observation.agents) - 1}, "
                    "in that order. Use only the legal destination IDs listed "
                    "above. Keep reason at 160 characters or fewer. Do not "
                    "return Markdown or additional keys."
                ),
            )
        )

        return PromptEnvelope(
            messages=(
                ChatMessage(role="system", content=_SYSTEM_MESSAGE),
                ChatMessage(role="user", content="\n".join(sections)),
            ),
            response_schema=self.response_schema(context),
        )

    def serialize_state(self, context: DirectiveContext) -> str:
        """Return compact deterministic JSON for the visible state and mask."""
        observation = context.observation
        state = {
            "agents": tuple(_agent_record(agent) for agent in observation.agents),
            "requested_shelves": tuple(
                _shelf_record(shelf) for shelf in observation.shelves if shelf.requested
            ),
            "goals": tuple(int(value) for value in observation.goal_destination_ids),
            "legal_destinations": tuple(
                {
                    "agent_id": agent.agent_id,
                    "destination_ids": tuple(
                        destination
                        for destination, is_legal in enumerate(
                            context.action_mask.values[index]
                        )
                        if bool(is_legal)
                    ),
                }
                for index, agent in enumerate(observation.agents)
            ),
        }
        return canonical_json_text(state)

    def response_schema(self, context: DirectiveContext) -> dict[str, Any]:
        """Build the strict schema for one current canonical agent vector."""
        agent_count = len(context.observation.agents)
        return {
            "type": "object",
            "properties": {
                "destinations": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": context.action_mask.destination_count - 1,
                    },
                    "minItems": agent_count,
                    "maxItems": agent_count,
                },
                "reason": {
                    "type": "string",
                    "maxLength": 160,
                },
            },
            "required": ["destinations", "reason"],
            "additionalProperties": False,
        }
