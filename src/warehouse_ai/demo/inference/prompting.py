"""Deterministic prompt construction from one canonical state record."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from warehouse_ai.demo.artifacts import (
    canonical_json_bytes,
    canonical_json_text,
    sha256_bytes,
)
from warehouse_ai.demo.contracts.directives import (
    Directive,
    DirectiveCatalog,
    WarehouseContext,
)
from warehouse_ai.demo.contracts.base import load_yaml_config
from warehouse_ai.demo.contracts.inference import (
    ActionerRequest,
    ChatMessage,
    PromptEnvelope,
)
from warehouse_ai.demo.contracts.prompting import LlmPromptSpec
from warehouse_ai.demo.inference.prompt_examples import render_example_user
from warehouse_ai.demo.directives.rules import (
    agent_is_eligible,
    eligible_destinations,
)


_SYSTEM_MESSAGE = (
    "You are the actioner for a warehouse controller. Interpret the selected "
    "directive using only the visible warehouse state and legal destinations. "
    "Return one complete joint destination action. Every choice is validated "
    "before it reaches the simulator."
)

_FEW_SHOT_TEMPLATE_VERSION = "directive-llm-few-shot-chat-v2"
_PER_DIRECTIVE_TEMPLATE_VERSION = "directive-llm-per-directive-chat-v1"
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
        template_bytes_by_behavior: Mapping[str, bytes] | None = None,
    ) -> None:
        if (prompt_spec is None) != (catalog is None):
            raise ValueError("few-shot prompting requires both prompt spec and catalog")
        if prompt_spec is not None and catalog is not None:
            prompt_spec.validate_catalog(catalog)
        if prompt_spec is not None and prompt_spec.schema_version == 2:
            if template_bytes is None:
                raise ValueError(
                    "schema-v2 prompt specs require external template bytes"
                )
            if template_bytes_by_behavior:
                raise ValueError(
                    "schema-v2 prompt specs cannot provide per-directive templates"
                )
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
        if prompt_spec is not None and prompt_spec.schema_version == 3:
            if template_bytes is not None:
                raise ValueError(
                    "schema-v3 prompt specs cannot provide one shared template"
                )
            templates = dict(template_bytes_by_behavior or {})
            if tuple(templates) != prompt_spec.required_behavior_keys:
                raise ValueError(
                    "schema-v3 prompt template bytes do not match required behaviors"
                )
            for behavior_key, raw_template in templates.items():
                template = raw_template.decode("utf-8")
                missing = sorted(
                    marker for marker in _TEMPLATE_MARKERS if marker not in template
                )
                if missing:
                    raise ValueError(
                        f"{behavior_key} prompt template is missing markers: {missing}"
                    )
                heading = f"## {behavior_key.replace('_', ' ').upper()} EXAMPLE"
                if template.count(heading) != prompt_spec.examples_per_directive:
                    raise ValueError(
                        f"{behavior_key} prompt template must contain "
                        f"{prompt_spec.examples_per_directive} matching examples"
                    )
                for other_key in (
                    *prompt_spec.required_behavior_keys,
                    *prompt_spec.deterministic_behavior_keys,
                ):
                    if other_key == behavior_key:
                        continue
                    other_heading = f"## {other_key.replace('_', ' ').upper()} EXAMPLE"
                    if other_heading in template:
                        raise ValueError(
                            f"{behavior_key} prompt template contains {other_key} examples"
                        )
        if prompt_spec is not None and prompt_spec.schema_version == 1:
            if template_bytes is not None or template_bytes_by_behavior:
                raise ValueError("schema-v1 prompt specs cannot provide template bytes")
        self.prompt_spec = prompt_spec
        self._template_bytes = template_bytes
        self._template_bytes_by_behavior = dict(template_bytes_by_behavior or {})

    @classmethod
    def load(cls, path: Path, catalog: DirectiveCatalog) -> LlmPromptBuilder:
        spec = load_yaml_config(path, LlmPromptSpec)
        template_bytes = None
        template_bytes_by_behavior: dict[str, bytes] | None = None
        if spec.schema_version == 2:
            assert spec.prompt_template_path is not None
            template_path = path.parent / spec.prompt_template_path
            try:
                template_bytes = template_path.read_bytes()
            except OSError as exc:
                raise ValueError(
                    f"could not load prompt template {template_path}"
                ) from exc
        elif spec.schema_version == 3:
            template_bytes_by_behavior = {}
            for behavior_key, relative_path in spec.prompt_template_paths.items():
                template_path = path.parent / relative_path
                try:
                    template_bytes_by_behavior[behavior_key] = (
                        template_path.read_bytes()
                    )
                except OSError as exc:
                    raise ValueError(
                        f"could not load prompt template {template_path}"
                    ) from exc
        return cls(
            spec,
            catalog,
            template_bytes=template_bytes,
            template_bytes_by_behavior=template_bytes_by_behavior,
        )

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
                        else (
                            _FEW_SHOT_TEMPLATE_VERSION
                            if self.prompt_spec.schema_version == 2
                            else _PER_DIRECTIVE_TEMPLATE_VERSION
                        )
                    )
                ),
                "prompt_spec": self.prompt_spec,
            }
        )
        if self.prompt_spec.schema_version == 2:
            assert self._template_bytes is not None
            return sha256_bytes(metadata + b"\n" + self._template_bytes)
        if self.prompt_spec.schema_version == 3:
            return sha256_bytes(metadata + b"\n" + self._template_bundle_bytes())
        return sha256_bytes(metadata)

    @property
    def template_digest(self) -> str | None:
        """Digest the exact external prompt bytes, preserving line endings."""
        if self._template_bytes is None:
            if not self._template_bytes_by_behavior:
                return None
            return sha256_bytes(self._template_bundle_bytes())
        return sha256_bytes(self._template_bytes)

    @property
    def template_digests(self) -> dict[str, str]:
        """Digest each directive-specific template independently."""
        return {
            behavior_key: sha256_bytes(template_bytes)
            for behavior_key, template_bytes in self._template_bytes_by_behavior.items()
        }

    def _template_bundle_bytes(self) -> bytes:
        chunks: list[bytes] = []
        for behavior_key, template_bytes in self._template_bytes_by_behavior.items():
            chunks.extend(
                (
                    behavior_key.encode("utf-8"),
                    b"\x00",
                    template_bytes,
                    b"\x00",
                )
            )
        return b"".join(chunks)

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

    @staticmethod
    def allowed_destinations_by_agent(
        context: WarehouseContext, directive: Directive
    ) -> tuple[tuple[int, ...], ...]:
        """Return directive-filtered destination IDs for each output position."""
        eligible = eligible_destinations(directive, context)
        rows: list[tuple[int, ...]] = []
        for row in range(len(context.observation.agents)):
            allowed = [0]
            if agent_is_eligible(directive, context, row):
                allowed.extend(
                    destination
                    for destination, legal in enumerate(context.action_mask.values[row])
                    if destination != 0 and bool(legal) and destination in eligible
                )
            rows.append(tuple(allowed))
        return tuple(rows)

    def action_state_payload(self, request: ActionerRequest) -> dict[str, Any]:
        """Render only the state needed to choose a directive-valid action."""
        context = request.context
        directive = request.directive
        role_groups: defaultdict[str, list[int]] = defaultdict(list)
        for index, agent in enumerate(context.observation.agents):
            role_groups[agent.role.value].append(index)
        allowed = self.allowed_destinations_by_agent(context, directive)
        candidates = self.safe_candidate_actions(request)
        return {
            "step": context.observation.step,
            "directive": {
                "behavior_key": directive.behavior_key,
                "command": directive.command,
                "instruction": directive.description,
            },
            "action_slots": [
                {
                    "output_index": row,
                    "agent_id": agent.agent_id,
                    "role": agent.role.value,
                    "allowed_destination_ids": list(allowed[row]),
                }
                for row, agent in enumerate(context.observation.agents)
            ],
            "candidate_actions": [list(candidate) for candidate in candidates],
            "minimum_nonzero_assignments": directive.effect.minimum,
            "maximum_nonzero_assignments": directive.effect.maximum,
            "same_role_unique_nonzero_groups": dict(sorted(role_groups.items())),
        }

    def safe_candidate_actions(
        self, request: ActionerRequest
    ) -> tuple[tuple[int, ...], ...]:
        """Enumerate a bounded set of complete actions that satisfy the directive."""
        context = request.context
        directive = request.directive
        allowed = self.allowed_destinations_by_agent(context, directive)
        agent_count = len(context.observation.agents)
        zero = (0,) * agent_count
        if directive.effect.minimum == 0:
            return (zero,)

        candidates: list[tuple[int, ...]] = []
        for row, destinations in enumerate(allowed):
            for destination in destinations:
                if destination == 0:
                    continue
                action: list[int] = list(zero)
                action[row] = destination
                candidates.append(tuple(action))

        greedy: list[int] = list(zero)
        selected: set[tuple[str, int]] = set()
        for row, destinations in enumerate(allowed):
            role = context.observation.agents[row].role.value
            for destination in destinations:
                if destination == 0 or (role, destination) in selected:
                    continue
                greedy[row] = destination
                selected.add((role, destination))
                break
        greedy_action = tuple(greedy)
        if greedy_action != zero and greedy_action not in candidates:
            candidates.append(greedy_action)
        if not candidates:
            raise ValueError(
                f"directive {directive.behavior_key} has no safe candidate action"
            )
        return tuple(candidates)

    def build(self, request: ActionerRequest) -> PromptEnvelope:
        directive = request.directive
        if self.prompt_spec is None:
            return self._build_legacy(request)
        if self.prompt_spec.schema_version in (2, 3):
            if self.prompt_spec.schema_version == 2:
                assert self._template_bytes is not None
                raw_template = self._template_bytes
            else:
                if (
                    directive.behavior_key
                    in self.prompt_spec.deterministic_behavior_keys
                ):
                    raise ValueError(
                        f"{directive.behavior_key} bypasses the LLM deterministically"
                    )
                try:
                    raw_template = self._template_bytes_by_behavior[
                        directive.behavior_key
                    ]
                except KeyError as exc:
                    raise ValueError(
                        f"no prompt template for {directive.behavior_key}"
                    ) from exc
            template = raw_template.decode("utf-8")
            payload = self.action_state_payload(request)
            rendered = template
            replacements = {
                "{{SELECTED_DIRECTIVE_BEHAVIOR_KEY}}": directive.behavior_key,
                "{{SELECTED_DIRECTIVE_COMMAND}}": directive.command,
                "{{SELECTED_DIRECTIVE_INSTRUCTION}}": directive.description,
                "{{LIVE_STATE_JSON}}": canonical_json_text(payload),
                "{{AGENT_COUNT}}": str(len(payload["action_slots"])),
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
                response_schema=self.response_schema(request),
            )

        prompt_directive = self.prompt_spec.by_behavior_key(directive.behavior_key)
        assert self.prompt_spec.system_message is not None
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
            response_schema=self.response_schema(request),
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
            response_schema=self.response_schema(request),
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

    def response_schema(self, request: ActionerRequest) -> dict[str, Any]:
        context = request.context
        agent_count = len(context.observation.agents)
        candidates = self.safe_candidate_actions(request)
        return {
            "type": "object",
            "properties": {
                "destinations": {
                    "type": "array",
                    "enum": [list(candidate) for candidate in candidates],
                    "minItems": agent_count,
                    "maxItems": agent_count,
                },
                "reason": {"type": "string", "maxLength": 160},
            },
            "required": ["destinations", "reason"],
            "additionalProperties": False,
        }
