"""Clarification tool models and response helpers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClarificationRequest:
    """A clarification request created for an ambiguous design decision."""

    id: str = field(default_factory=lambda: f"clarify_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    question: str = ""
    options: list[str] = field(default_factory=list)
    context: str = ""
    agent_type: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "options": self.options,
            "context": self.context,
            "agent_type": self.agent_type,
            "created_at": self.created_at,
        }

    @classmethod
    def from_tool_call(
        cls,
        arguments: dict[str, Any],
        agent_type: str = "",
        context: str = "",
    ) -> "ClarificationRequest":
        """Build a request from a tool-call argument object."""
        return cls(
            question=arguments.get("question", ""),
            options=arguments.get("options", []),
            context=context,
            agent_type=agent_type,
        )


@dataclass
class ClarificationResponse:
    """A user's response to a clarification request."""

    request_id: str = ""
    answer: str = ""
    selected_option: Optional[int] = None
    responded_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "selected_option": self.selected_option,
            "responded_at": self.responded_at,
        }


ASK_CLARIFICATION_TOOL_SCHEMA = {
    "name": "ask_user_clarification",
    "description": """Ask the user to resolve an ambiguous or multi-option design decision.

Use this tool when:
- The user requests a style without specifying a direction.
- The color direction is unclear.
- The layout choice is ambiguous.
- Template constraints may conflict with the user's request.

Use it only when clarification is genuinely needed. Prefer known user
preferences, keep the question concise, and make options mutually exclusive
and comprehensive.""",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The concise, concrete question to ask the user.",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of 2-4 choices, for example "
                    "['A. Dark business', 'B. Light minimal']. "
                    "Omit it for an open-ended question."
                ),
            },
        },
        "required": ["question"],
    },
}


def format_tool_response(
    response: ClarificationResponse,
    request: ClarificationRequest | None = None,
) -> str:
    """Format a response for the agent without echoing every option."""
    parts = []
    if request:
        parts.append(f"[Question] {request.question}")
    if response.selected_option is not None:
        parts.append(f"[User selection] {response.answer}")
    else:
        parts.append(f"[User answer] {response.answer}")
    parts.append("Act on the user's choice immediately; do not ask for confirmation again.")
    return "\n".join(parts)


def is_clarification_tool_call(tool_call: Any) -> bool:
    """Return whether a tool call invokes the clarification tool."""
    if hasattr(tool_call, "function"):
        return getattr(tool_call.function, "name", "") == "ask_user_clarification"
    if isinstance(tool_call, dict):
        func = tool_call.get("function", {})
        return func.get("name", "") == "ask_user_clarification"
    return False


def parse_clarification_arguments(tool_call: Any) -> dict[str, Any]:
    """Parse clarification arguments from an OpenAI-style tool call."""
    if hasattr(tool_call, "function"):
        args_str = getattr(tool_call.function, "arguments", "{}")
    elif isinstance(tool_call, dict):
        args_str = tool_call.get("function", {}).get("arguments", "{}")
    else:
        return {}

    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        logger.warning("Failed to parse clarification arguments: %s", args_str)
        return {}


class ClarificationManager:
    """Manage pending and completed clarification requests in one session."""

    def __init__(self) -> None:
        self._pending_requests: dict[str, ClarificationRequest] = {}
        self._completed_requests: dict[
            str, tuple[ClarificationRequest, ClarificationResponse]
        ] = {}

    def create_request(
        self,
        question: str,
        options: list[str] | None = None,
        context: str = "",
        agent_type: str = "",
    ) -> ClarificationRequest:
        """Create and register a pending clarification request."""
        request = ClarificationRequest(
            question=question,
            options=options or [],
            context=context,
            agent_type=agent_type,
        )
        self._pending_requests[request.id] = request
        logger.info("Created clarification request: %s", request.id)
        return request

    def submit_response(
        self,
        request_id: str,
        answer: str,
        selected_option: int | None = None,
    ) -> ClarificationResponse | None:
        """Complete a pending request and return its stored response."""
        request = self._pending_requests.pop(request_id, None)
        if not request:
            logger.warning("Clarification request not found: %s", request_id)
            return None

        response = ClarificationResponse(
            request_id=request_id,
            answer=answer,
            selected_option=selected_option,
        )
        self._completed_requests[request_id] = (request, response)
        logger.info("Submitted clarification response: %s", request_id)
        return response

    def get_pending_request(self, request_id: str) -> ClarificationRequest | None:
        """Return a pending request by ID, if present."""
        return self._pending_requests.get(request_id)

    def get_completed(
        self,
        request_id: str,
    ) -> tuple[ClarificationRequest, ClarificationResponse] | None:
        """Return a completed request and response by ID, if present."""
        return self._completed_requests.get(request_id)

    def get_recent_completed(
        self,
        limit: int = 5,
    ) -> list[tuple[ClarificationRequest, ClarificationResponse]]:
        """Return the most recently completed requests."""
        items = list(self._completed_requests.values())
        return items[-limit:] if len(items) > limit else items
