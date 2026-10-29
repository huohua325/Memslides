from __future__ import annotations

import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from enum import Enum
from hashlib import md5
from pathlib import Path
from typing import Any, Literal

try:
    from enum import StrEnum
except ImportError:

    class StrEnum(str, Enum):
        pass

try:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )
except ImportError:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall as ChatCompletionMessageFunctionToolCall,
        Function,
    )

from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field

from memslides.utils.constants import PACKAGE_DIR
from memslides.utils.log import debug, warning


_PROJECT_ROOT = PACKAGE_DIR.parent.parent
_ENV_REFERENCE_RE = re.compile(r"\$([A-Z][A-Z0-9_]*[A-Z0-9])")


def _string_blocks(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value).strip()
    return [{"type": "text", "text": text}] if text else []


def _copy_files_to_dir(paths: list[str], target_dir: Path) -> list[str]:
    copied: list[str] = []
    if not paths:
        return copied
    target_dir.mkdir(parents=True, exist_ok=True)
    for raw_path in paths:
        source = Path(raw_path)
        if not source.exists():
            raise AssertionError(f"Attachment {raw_path} does not exist")
        target = target_dir / source.name
        if target.exists():
            warning("Attachment %s already exists in workspace", raw_path)
        else:
            shutil.copy(str(source), str(target))
        copied.append(str(target))
    return copied


def _append_if_present(lines: list[str], label: str, value: Any) -> None:
    if value is not None and value != "":
        lines.append(f"{label}: {value}")


def _prompt_language_note(lines: list[str], language: str | None) -> None:
    if language:
        lines.append(
            "Use this as the final slide text language except for source quotes, "
            "names, and technical terms."
        )


class MCPServer(BaseModel):
    """Configuration for one tool server entry in the MCP manifest."""

    name: str
    description: str
    command: str
    args: list[str]
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    header: dict[str, str] | None = None
    keep_tools: list[str] | None = None
    exclude_tools: list[str] | None = Field(default_factory=list)

    def _replace_env_refs(self, text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in self.env:
                value = self.env[key]
            elif key in os.environ:
                value = os.environ[key]
            else:
                raise ValueError(f"Environment variable {key} declared but not found")
            if f"${key}" in value:
                warning(
                    "Environment variable $%s references itself in mcp.json; "
                    "set it in the parent process before connecting MCP servers.",
                    key,
                )
                return match.group(0)
            debug("Resolved environment reference $%s", key)
            return value

        return _ENV_REFERENCE_RE.sub(replace, text)

    def _resolve_script_arg(self, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or not value.endswith(".py"):
            return value
        project_candidate = _PROJECT_ROOT / value
        if not candidate.exists() and project_candidate.exists():
            resolved = str(project_candidate)
            debug("Resolved MCP script argument to %s", resolved)
            return resolved
        return value

    def _build_process_env(self) -> dict[str, str]:
        merged = dict(os.environ)
        python_dir = str(Path(sys.executable).resolve().parent)
        existing_path = merged.get("PATH", "")
        path_parts = existing_path.split(os.pathsep) if existing_path else []
        if not path_parts or path_parts[0] != python_dir:
            merged["PATH"] = os.pathsep.join([python_dir, existing_path]) if existing_path else python_dir
        merged.update(self.env)
        return merged

    def _process_escape(self) -> None:
        self.args = [self._resolve_script_arg(self._replace_env_refs(arg)) for arg in self.args]
        self.env = {key: self._replace_env_refs(value) if "$" in value else value for key, value in self.env.items()}
        if self.url:
            self.url = self._replace_env_refs(self.url)
        if self.command == "python":
            self.command = sys.executable
        self.env = self._build_process_env()


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ChatMessage(BaseModel):
    role: Role
    content: None | str | list[dict]
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reasoning_content: None | str = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_error: bool = False
    from_tool: Function | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatCompletionMessageFunctionToolCall] | None = None
    extra_info: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, _) -> None:
        self.content = _string_blocks(self.content)
        for block in self.content:
            if block.get("type") == "text":
                block["text"] = str(block.get("text", "")).strip()

    @property
    def text(self) -> str:
        parts: list[str] = []
        for block in self.content or []:
            block_type = block.get("type")
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "image_url":
                parts.append("<image>")
        for tool_call in self.tool_calls or []:
            parts.append(tool_call.function.model_dump_json())
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else str(parts)

    @property
    def has_image(self) -> bool:
        return any(block.get("type") == "image_url" for block in self.content or [])


class RoleConfig(BaseModel):
    system: dict[str, str]
    instruction: str
    use_model: str
    include_tool_servers: list[str] | Literal["all"] = "all"
    exclude_tool_servers: list[str] = Field(default_factory=list)
    include_tools: list[str] = Field(default_factory=list)
    exclude_tools: list[str] = Field(default_factory=list)


class Cost(BaseModel):
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def __add__(self, other: CompletionUsage):
        self.prompt += other.prompt_tokens
        self.completion += other.completion_tokens
        self.total += other.total_tokens
        return self

    def __repr__(self) -> str:
        return f"{self.prompt / 1000:.1f}K prompt tokens and {self.completion / 1000:.1f}K completion tokens"


class ConvertType(StrEnum):
    MEMSLIDES = "memslides"
    TEMPLATE_PLANNER = "template_planner"


class PowerPointType(StrEnum):
    WIDE_SCREEN = "16:9"
    STANDARD_SCREEN = "4:3"
    POSTER = "A1"
    POSTER_A3 = "A3"
    POSTER_A2 = "A2"
    POSTER_A4 = "A4"


class InputRequest(BaseModel):
    instruction: str
    attachments: list[str] = Field(default_factory=list)
    num_pages: str | None = None
    language: Literal["zh", "en"] | None = None
    memory_intent: str = ""
    template: str | None = None
    template_as_reference: bool = False
    template_id: str | None = None
    powerpoint_type: PowerPointType = PowerPointType.WIDE_SCREEN
    convert_type: ConvertType = ConvertType.MEMSLIDES
    extra_info: dict[str, Any] = Field(default_factory=dict)

    def copy_to_workspace(self, workspace: Path) -> None:
        self.attachments = _copy_files_to_dir(self.attachments, workspace / "attachments")
        if not (self.template and self.template_as_reference):
            return
        template_path = Path(self.template)
        if not template_path.exists():
            return
        copied = _copy_files_to_dir([str(template_path)], workspace / "template")
        if copied:
            self.template = copied[0]
            debug("Template copied to workspace: %s", self.template)

    @property
    def request_id(self) -> str:
        fingerprint = self.instruction + "".join(self.attachments)
        return md5(fingerprint.encode()).hexdigest()[:8]

    @property
    def task_id(self) -> str:
        return self.request_id

    def _base_prompt_lines(self) -> list[str]:
        lines = [self.instruction]
        _append_if_present(lines, "Number of pages", self.num_pages)
        _append_if_present(lines, "Language", self.language)
        _prompt_language_note(lines, self.language)
        return lines

    @property
    def deepresearch_prompt(self) -> str:
        lines = self._base_prompt_lines()
        if self.attachments:
            lines.append("Attachments (primary content source): " + ", ".join(self.attachments))
            lines.append("Use the attachments as the factual basis for the draft before inventing any broader structure.")
        return "\n".join(lines)

    @property
    def template_planner_prompt(self) -> str:
        lines = [self.instruction]
        if self.template and self.template not in self.instruction:
            lines.append("PPT Template: " + self.template)
        _append_if_present(lines, "Number of pages", self.num_pages)
        _append_if_present(lines, "Language", self.language)
        _prompt_language_note(lines, self.language)
        if self.attachments:
            lines.append("Attachments (content source): " + ", ".join(self.attachments))
            lines.append("Ground the deck content in the attachments and use the template only for layout/style.")
        return "\n".join(lines)

    @property
    def designagent_prompt(self) -> str:
        lines = self._base_prompt_lines()
        lines.append("Aspect Ratio: " + self.powerpoint_type.value)
        if self.attachments:
            lines.append("Attachments (content source): " + ", ".join(self.attachments))
            lines.append("Inspect the attachments first and ground the slide content in them before drafting.")
        return "\n".join(lines)
