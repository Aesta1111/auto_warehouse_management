"""Compact deterministic prompt construction."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from warehouse_ai.demo.artifacts import canonical_json_text
from warehouse_ai.demo.contracts.directives import DirectiveContext
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ChatMessage,
    PromptEnvelope,
)


_SYSTEM_MESSAGE = (
    "You control one warehouse step. Follow the selected directive using only "
    "the supplied state and legal destinations. Return the required JSON object."
)


def _position(value: Any) -> list[int]:
    return [int(value.row), int(value.column)]


def _agent(agent: Any) -> dict[str, Any]:
    return {
        "id": agent.agent_id,
        "role": agent.role.value,
        "position": _position(agent.position),
        "target": None if agent.target is None else _position(agent.target),
        "target_destination": agent.target_destination_id,
        "busy": agent.busy,
        "carrying_shelf": agent.carrying_shelf,
        "carrying_requested_shelf": agent.carrying_requested_shelf,
        "toggling_load": agent.toggling_load,
    }


class GemmaPromptBuilder:
    """Build one canonical JSON state prompt and strict response schema."""

    def state_payload(
        self, context: DirectiveContext, *, directive: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        observation = context.observation
        role_groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, agent in enumerate(observation.agents):
            role_groups[agent.role.value].append(index)
        return {
            "objective": "complete requested shelf work with one joint action",
            "step": observation.step,
            "directive": directive,
            "agents": [_agent(agent) for agent in observation.agents],
            "requested_shelves": [
                {
                    "destination": shelf.destination_id,
                    "position": _position(shelf.position),
                }
                for shelf in observation.shelves
                if shelf.requested
            ],
            "goal_destinations": [
                int(item) for item in observation.goal_destination_ids
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
            "rules": [
                "destination 0 means no new assignment",
                "nonzero destinations must be legal for that agent",
                "agents with the same role cannot share a nonzero destination",
            ],
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
        return PromptEnvelope(
            messages=(
                ChatMessage(role="system", content=_SYSTEM_MESSAGE),
                ChatMessage(role="user", content=canonical_json_text(payload)),
            ),
            response_schema=self.response_schema(request.context),
        )

    def serialize_state(self, context: DirectiveContext) -> str:
        return canonical_json_text(self.state_payload(context))

    def response_schema(self, context: DirectiveContext) -> dict[str, Any]:
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
