"""Deterministic prompt construction from one canonical state record."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from warehouse_ai.demo.artifacts import canonical_json_text
from warehouse_ai.demo.contracts.directives import WarehouseContext
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ChatMessage,
    PromptEnvelope,
)


_SYSTEM_MESSAGE = (
    "You are the actioner for a warehouse controller. Interpret the selected "
    "directive using only the visible warehouse state and legal destinations. "
    "Return one complete joint destination action. Every choice is validated "
    "before it reaches the simulator."
)


def _position(value: Any) -> list[int]:
    return [int(value.row), int(value.column)]


def _agent(agent: Any) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "role": agent.role.value,
        "position": _position(agent.position),
        "target": None if agent.target is None else _position(agent.target),
        "target_destination_id": agent.target_destination_id,
        "busy": agent.busy,
        "carrying_shelf": agent.carrying_shelf,
        "carrying_requested_shelf": agent.carrying_requested_shelf,
        "toggling_load": agent.toggling_load,
    }


class GemmaPromptBuilder:
    """Render proven wording from a compact canonical state representation."""

    def state_payload(
        self, context: WarehouseContext, *, directive: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        observation = context.observation
        role_groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, agent in enumerate(observation.agents):
            role_groups[agent.role.value].append(index)
        return {
            "step": observation.step,
            "directive": directive,
            "agents": [_agent(agent) for agent in observation.agents],
            "requested_shelves": [
                {
                    "destination_id": shelf.destination_id,
                    "position": _position(shelf.position),
                }
                for shelf in observation.shelves
                if shelf.requested
            ],
            "goal_destination_ids": [
                int(value) for value in observation.goal_destination_ids
            ],
            "legal_destinations": [
                [
                    destination
                    for destination, allowed in enumerate(
                        context.action_mask.values[row]
                    )
                    if bool(allowed)
                ]
                for row in range(len(observation.agents))
            ],
            "same_role_unique_nonzero_groups": dict(sorted(role_groups.items())),
        }

    def build(self, request: ActionerRequest) -> PromptEnvelope:
        directive = request.prediction.directive
        payload = self.state_payload(
            request.context,
            directive={
                "command": directive.command,
                "description": directive.description,
            },
        )
        sections = [
            "OBJECTIVE",
            (
                f"At episode step {payload['step']}, complete requested shelf work "
                "with one joint action. Use destination 0 for an agent that should "
                "receive no new assignment."
            ),
            "DIRECTIVE",
            f"command: {directive.command}",
            f"description: {directive.description}",
            "AGENTS",
        ]
        for index, agent in enumerate(payload["agents"]):
            sections.append(
                f"position {index} ({agent['agent_id']}): {canonical_json_text(agent)}"
            )
        sections.append("REQUESTED SHELVES")
        requested = payload["requested_shelves"]
        sections.extend(canonical_json_text(shelf) for shelf in requested)
        if not requested:
            sections.append("none")
        sections.extend(
            (
                "GOALS",
                f"goal destination IDs: {canonical_json_text(payload['goal_destination_ids'])}",
                "LEGAL DESTINATIONS",
            )
        )
        for index, destinations in enumerate(payload["legal_destinations"]):
            agent_id = payload["agents"][index]["agent_id"]
            sections.append(
                f"position {index} ({agent_id}): {canonical_json_text(destinations)}"
            )
        group_rules = "; ".join(
            f"{role} positions {positions} must have pairwise-distinct nonzero destinations"
            for role, positions in payload["same_role_unique_nonzero_groups"].items()
            if len(positions) > 1
        )
        sections.extend(
            (
                "JOINT-ACTION SAFETY RULES",
                (
                    "A nonzero destination may be selected by at most one agent with "
                    "the same role. If multiple same-role agents would choose the same "
                    "nonzero destination, keep one assignment and use destination 0 or "
                    "another legal destination for the others. Different roles may "
                    f"share a destination. In this layout, {group_rules}."
                ),
                "OUTPUT INSTRUCTIONS",
                (
                    "Return exactly one JSON object with exactly the keys destinations "
                    "and reason. destinations must contain one integer for each agent "
                    f"in positions 0 through {len(payload['agents']) - 1}, in that order. "
                    "Use only the legal destination IDs listed above. Keep reason at "
                    "160 characters or fewer. Do not return Markdown or additional keys."
                ),
            )
        )
        return PromptEnvelope(
            messages=(
                ChatMessage(role="system", content=_SYSTEM_MESSAGE),
                ChatMessage(role="user", content="\n".join(sections)),
            ),
            response_schema=self.response_schema(request.context),
        )

    def serialize_state(self, context: WarehouseContext) -> str:
        return canonical_json_text(self.state_payload(context))

    def response_schema(self, context: WarehouseContext) -> dict[str, Any]:
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
                "reason": {"type": "string", "maxLength": 160},
            },
            "required": ["destinations", "reason"],
            "additionalProperties": False,
        }
