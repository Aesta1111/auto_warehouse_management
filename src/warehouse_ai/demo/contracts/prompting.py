"""Versioned few-shot prompt specification contracts."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from warehouse_ai.demo.contracts.base import DemoModel, NonBlankStr, RepositoryPath
from warehouse_ai.demo.contracts.inference import LlmActionResponse


DirectiveBehaviorKey = Literal[
    "wait",
    "service_requested_shelf",
    "deliver_carried_shelf",
    "reposition_idle_agent",
]
ExampleAgentRole = Literal["agv", "picker"]


class FewShotExample(DemoModel):
    """One compact synthetic state and its fully validated action response."""

    behavior_key: DirectiveBehaviorKey
    agent_roles: tuple[ExampleAgentRole, ...] = Field(min_length=5, max_length=5)
    idle_agent_positions: tuple[NonNegativeInt, ...]
    carrying_shelf_positions: tuple[NonNegativeInt, ...] = ()
    requested_destination_ids: tuple[PositiveInt, ...] = ()
    goal_destination_ids: tuple[PositiveInt, ...] = ()
    legal_destinations: tuple[tuple[NonNegativeInt, ...], ...] = Field(
        min_length=5, max_length=5
    )
    response: LlmActionResponse

    @model_validator(mode="after")
    def validate_example(self) -> Self:
        agent_count = len(self.agent_roles)
        position_fields = {
            "idle_agent_positions": self.idle_agent_positions,
            "carrying_shelf_positions": self.carrying_shelf_positions,
        }
        for name, positions in position_fields.items():
            if len(set(positions)) != len(positions):
                raise ValueError(f"{name} must not contain duplicates")
            if any(position >= agent_count for position in positions):
                raise ValueError(f"{name} contains an out-of-range position")
        for name, destinations in (
            ("requested_destination_ids", self.requested_destination_ids),
            ("goal_destination_ids", self.goal_destination_ids),
        ):
            if len(set(destinations)) != len(destinations):
                raise ValueError(f"{name} must not contain duplicates")
        if len(self.response.destinations) != agent_count:
            raise ValueError("example response must contain one destination per agent")
        for row, legal in enumerate(self.legal_destinations):
            if not legal or 0 not in legal:
                raise ValueError(
                    "every example legal-destination row must contain zero"
                )
            if len(set(legal)) != len(legal):
                raise ValueError(
                    "example legal destinations must not contain duplicates"
                )
            if self.response.destinations[row] not in legal:
                raise ValueError("example response contains an illegal destination")
        selected_by_role: set[tuple[str, int]] = set()
        for role, destination in zip(
            self.agent_roles, self.response.destinations, strict=True
        ):
            if destination == 0:
                continue
            pair = (role, destination)
            if pair in selected_by_role:
                raise ValueError(
                    "example response contains a same-role duplicate destination"
                )
            selected_by_role.add(pair)
        if not self.response.reason.strip():
            raise ValueError("example response reason must not be blank")

        idle = set(self.idle_agent_positions)
        carrying = set(self.carrying_shelf_positions)
        requested = set(self.requested_destination_ids)
        goals = set(self.goal_destination_ids)
        selected = tuple(
            (row, destination)
            for row, destination in enumerate(self.response.destinations)
            if destination != 0
        )
        if self.behavior_key == "wait":
            valid = not selected
        elif self.behavior_key == "service_requested_shelf":
            valid = bool(selected) and all(
                row in idle and destination in requested
                for row, destination in selected
            )
        elif self.behavior_key == "deliver_carried_shelf":
            valid = bool(selected) and all(
                row in idle
                and row in carrying
                and self.agent_roles[row] == "agv"
                and destination in goals
                for row, destination in selected
            )
        else:
            valid = bool(selected) and all(
                row in idle and row not in carrying and destination not in requested
                for row, destination in selected
            )
        if not valid:
            raise ValueError(
                f"example response does not satisfy {self.behavior_key} instruction"
            )
        return self


class DirectivePrompt(DemoModel):
    behavior_key: DirectiveBehaviorKey
    command: NonBlankStr
    instruction: NonBlankStr
    example: FewShotExample

    @model_validator(mode="after")
    def validate_example_behavior(self) -> Self:
        if self.example.behavior_key != self.behavior_key:
            raise ValueError("example behavior_key must match its directive prompt")
        return self


class LlmPromptSpec(DemoModel):
    """Metadata for either a legacy inline prompt or an external template."""

    schema_version: Literal[1, 2]
    prompt_id: NonBlankStr
    system_message: NonBlankStr | None = None
    directives: tuple[DirectivePrompt, ...] = ()
    template_version: NonBlankStr | None = None
    prompt_template_path: RepositoryPath | None = None
    required_behavior_keys: tuple[DirectiveBehaviorKey, ...] = ()
    command_metadata: dict[str, NonBlankStr] = Field(default_factory=dict)
    examples_per_directive: PositiveInt = 1
    example_count: PositiveInt = 4

    @model_validator(mode="after")
    def validate_directives(self) -> Self:
        expected = (
            "wait",
            "service_requested_shelf",
            "deliver_carried_shelf",
            "reposition_idle_agent",
        )
        if self.schema_version == 1:
            actual = tuple(item.behavior_key for item in self.directives)
            if actual != expected:
                raise ValueError(
                    "prompt directives must contain all four behavior keys in canonical order"
                )
            commands = tuple(item.command.casefold() for item in self.directives)
            if len(set(commands)) != len(commands):
                raise ValueError("prompt directive commands must be unique")
            if self.system_message is None:
                raise ValueError("schema-v1 prompt specs require system_message")
            if (
                self.template_version is not None
                or self.prompt_template_path is not None
            ):
                raise ValueError(
                    "schema-v1 prompt specs cannot select an external template"
                )
            return self

        if self.system_message is not None or self.directives:
            raise ValueError(
                "schema-v2 prompt specs must keep prompt text in the TXT file"
            )
        if self.template_version is None or self.prompt_template_path is None:
            raise ValueError(
                "schema-v2 prompt specs require template metadata and a TXT path"
            )
        if self.required_behavior_keys != expected:
            raise ValueError("schema-v2 prompt specs must list all four behavior keys")
        if tuple(self.command_metadata) != expected:
            raise ValueError(
                "schema-v2 command metadata must use canonical behavior keys"
            )
        if self.example_count != len(expected) * self.examples_per_directive:
            raise ValueError("example_count must match the directive/example metadata")
        return self

    def validate_catalog(self, catalog: Any) -> None:
        """Reject prompt text that is not aligned with the frozen catalog."""
        expected = tuple(
            (item.behavior_key, item.command) for item in catalog.directives
        )
        if self.schema_version == 1:
            actual = tuple(
                (item.behavior_key, item.command) for item in self.directives
            )
            if actual != expected:
                raise ValueError("prompt directives do not match the directive catalog")
            return
        actual = tuple(
            (key, self.command_metadata[key]) for key in self.required_behavior_keys
        )
        if actual != expected:
            raise ValueError(
                "prompt command metadata does not match the directive catalog"
            )

    def by_behavior_key(self, behavior_key: str) -> DirectivePrompt:
        if self.schema_version != 1:
            raise KeyError(
                "schema-v2 prompt specs store directive text in the template"
            )
        for directive in self.directives:
            if directive.behavior_key == behavior_key:
                return directive
        raise KeyError(behavior_key)
