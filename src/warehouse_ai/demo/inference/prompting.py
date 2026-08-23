"""Deterministic prompt construction from one canonical state record."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from warehouse_ai.demo.artifacts import (
    canonical_json_bytes,
    canonical_json_text,
    sha256_bytes,
)
from warehouse_ai.demo.contracts.directives import WarehouseContext, DirectiveCatalog
from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ChatMessage,
    PromptEnvelope,
)
from warehouse_ai.demo.contracts.prompting import LlmPromptSpec
from warehouse_ai.demo.inference.prompt_examples import render_example_user


_SYSTEM_MESSAGE = (
    "You are the actioner for a warehouse controller. Interpret the selected "
    "directive using only the visible warehouse state and legal destinations. "
    "Return one complete joint destination action. Every choice is validated "
    "before it reaches the simulator."
)

_FEW_SHOT_TEMPLATE_VERSION = "directive-llm-few-shot-chat-v2"
_TEMPLATE_MARKERS = frozenset(
    {
        "{{SELECTED_DIRECTIVE_BEHAVIOR_KEY}}",
        "{{SELECTED_DIRECTIVE_COMMAND}}",
        "{{SELECTED_DIRECTIVE_INSTRUCTION}}",
        "{{LIVE_STATE_JSON}}",
        "{{AGENT_COUNT}}",
    }
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


class LlmPromptBuilder:
    """Render proven wording from a compact canonical state representation."""

    def __init__(
        self,
        prompt_spec: LlmPromptSpec | None = None,
        catalog: DirectiveCatalog | None = None,
        *,
        template_bytes: bytes | None = None,
    ) -> None:
        if (prompt_spec is None) != (catalog is None):
            raise ValueError("few-shot prompting requires both prompt spec and catalog")
        if prompt_spec is not None and catalog is not None:
            prompt_spec.validate_catalog(catalog)
        if prompt_spec is not None and prompt_spec.schema_version == 2:
            if template_bytes is None:
                raise ValueError("schema-v2 prompt specs require external template bytes")
            template = template_bytes.decode("utf-8")
            missing = sorted(
                marker for marker in _TEMPLATE_MARKERS if marker not in template
            )
            if missing:
                raise ValueError(f"prompt template is missing markers: {missing}")
            for behavior_key in prompt_spec.required_behavior_keys:
                heading = f"## {behavior_key.replace('_', ' ').upper()} EXAMPLE"
                if template.count(heading) != prompt_spec.examples_per_directive:
                    raise ValueError(
                        f"prompt template must contain {prompt_spec.examples_per_directive} "
                        f"examples for {behavior_key}"
                    )
        if prompt_spec is not None and prompt_spec.schema_version == 1:
            if template_bytes is not None:
                raise ValueError("schema-v1 prompt specs cannot provide template bytes")
        self.prompt_spec = prompt_spec
        self._template_bytes = template_bytes

    @classmethod
    def load(cls, path: Path, catalog: DirectiveCatalog) -> LlmPromptBuilder:
        spec = load_yaml_config(path, LlmPromptSpec)
        template_bytes = None
        if spec.schema_version == 2:
            assert spec.prompt_template_path is not None
            template_path = path.parent / spec.prompt_template_path
            try:
                template_bytes = template_path.read_bytes()
            except OSError as exc:
                raise ValueError(f"could not load prompt template {template_path}") from exc
        return cls(spec, catalog, template_bytes=template_bytes)

    @property
    def contract_digest(self) -> str | None:
        if self.prompt_spec is None:
            return None
        metadata = canonical_json_bytes(
            {
                "template_version": (
                    self.prompt_spec.template_version
                    or (
                        "directive-few-shot-chat-v1"
                        if self.prompt_spec.schema_version == 1
                        else _FEW_SHOT_TEMPLATE_VERSION
                    )
                ),
                "prompt_spec": self.prompt_spec,
            }
        )
        if self.prompt_spec.schema_version == 2:
            assert self._template_bytes is not None
            return sha256_bytes(metadata + b"\n" + self._template_bytes)
        return sha256_bytes(metadata)

    @property
    def template_digest(self) -> str | None:
        """Digest the exact external prompt bytes, preserving line endings."""
        if self._template_bytes is None:
            return None
        return sha256_bytes(self._template_bytes)

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
        directive = request.directive
        if self.prompt_spec is None:
            return self._build_legacy(request)
        if self.prompt_spec.schema_version == 2:
            assert self._template_bytes is not None
            template = self._template_bytes.decode("utf-8")
            payload = self.state_payload(
                request.context,
                directive={
                    "behavior_key": directive.behavior_key,
                    "command": directive.command,
                    "instruction": directive.description,
                },
            )
            rendered = template
            replacements = {
                "{{SELECTED_DIRECTIVE_BEHAVIOR_KEY}}": directive.behavior_key,
                "{{SELECTED_DIRECTIVE_COMMAND}}": directive.command,
                "{{SELECTED_DIRECTIVE_INSTRUCTION}}": directive.description,
                "{{LIVE_STATE_JSON}}": canonical_json_text(payload),
                "{{AGENT_COUNT}}": str(len(payload["agents"])),
            }
            for marker, value in replacements.items():
                rendered = rendered.replace(marker, value)
            if any(marker in rendered for marker in _TEMPLATE_MARKERS):
                raise ValueError("prompt template rendering left an unresolved marker")
            return PromptEnvelope(
                messages=(
                    ChatMessage(role="system", content=_SYSTEM_MESSAGE),
                    ChatMessage(role="user", content=rendered),
                ),
                response_schema=self.response_schema(request.context),
            )

        prompt_directive = self.prompt_spec.by_behavior_key(directive.behavior_key)
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=self.prompt_spec.system_message)
        ]
        for example_directive in self.prompt_spec.directives:
            messages.extend(
                (
                    ChatMessage(
                        role="user",
                        content=render_example_user(example_directive),
                    ),
                    ChatMessage(
                        role="assistant",
                        content=canonical_json_text(example_directive.example.response),
                    ),
                )
            )
        messages.append(
            ChatMessage(
                role="user",
                content=self._render_live_user(
                    request,
                    directive_label="instruction",
                    directive_text=prompt_directive.instruction,
                ),
            )
        )
        return PromptEnvelope(
            messages=tuple(messages),
            response_schema=self.response_schema(request.context),
        )

    def _build_legacy(self, request: ActionerRequest) -> PromptEnvelope:
        directive = request.directive
        return PromptEnvelope(
            messages=(
                ChatMessage(role="system", content=_SYSTEM_MESSAGE),
                ChatMessage(
                    role="user",
                    content=self._render_live_user(
                        request,
                        directive_label="description",
                        directive_text=directive.description,
                    ),
                ),
            ),
            response_schema=self.response_schema(request.context),
        )

    def _render_live_user(
        self,
        request: ActionerRequest,
        *,
        directive_label: str,
        directive_text: str,
    ) -> str:
        directive = request.directive
        payload = self.state_payload(
            request.context,
            directive={
                "command": directive.command,
                directive_label: directive_text,
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
            f"{directive_label}: {directive_text}",
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
        return "\n".join(sections)

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
