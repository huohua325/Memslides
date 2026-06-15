import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import docker
from docker.errors import DockerException, NotFound
from mcp.types import CallToolResult, TextContent
try:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageFunctionToolCall as ToolCall,
    )
except ImportError:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall as ToolCall,
    )
from pydantic import BaseModel

from memslides.utils.config import GLOBAL_CONFIG, MemSlidesConfig
from memslides.utils.constants import (
    CUTOFF_WARNING,
    LOGGING_LEVEL,
    MCP_CALL_TIMEOUT,
    TOOL_CACHE,
    TOOL_CUTOFF_LEN,
    WORKSPACE_BASE,
)
from memslides.utils.log import debug, info, timer, warning
from memslides.utils.mcp_client import MCPClient
from memslides.utils.typings import ChatMessage, MCPServer, Role


class ToolTiming(BaseModel):
    total_time: float = 0.0
    success_count: int = 0
    error_count: int = 0


_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")


def _snake_name(name: str) -> str:
    step = _CAMEL_BOUNDARY_1.sub(r"\1_\2", name)
    return _CAMEL_BOUNDARY_2.sub(r"\1_\2", step).lower()


def _normalize_tool_arguments(value: Any) -> Any:
    if isinstance(value, list):
        return [_normalize_tool_arguments(item) for item in value]
    if not isinstance(value, dict):
        return value

    converted: dict[Any, Any] = {}
    for raw_key, raw_value in value.items():
        key = _snake_name(raw_key) if isinstance(raw_key, str) else raw_key
        converted[key if key not in converted else raw_key] = _normalize_tool_arguments(raw_value)
    return converted


def _first_text_block(result: CallToolResult) -> str:
    for item in getattr(result, "content", []) or []:
        if getattr(item, "type", None) == "text":
            text = str(getattr(item, "text", "") or "").strip()
            if text:
                return text
    return ""


def _looks_like_error_text(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered.startswith(("error:", "failed:", "exception:")):
        return True
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("success") is False
        or bool(payload.get("error_code"))
        or (bool(payload.get("error")) and payload.get("success") is not True)
    )


def _coerce_textual_error_result(result: CallToolResult) -> CallToolResult:
    if getattr(result, "isError", False):
        return result
    if not _looks_like_error_text(_first_text_block(result)):
        return result

    try:
        return result.model_copy(update={"isError": True})
    except Exception:
        try:
            result.isError = True
        except Exception:
            pass
        return result


def _tool_error(message: str) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        isError=True,
    )


def _dump_model(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


class AgentEnv:
    def __init__(
        self,
        workspace: Path,
        config: MemSlidesConfig = GLOBAL_CONFIG,
        cutoff_len: int = TOOL_CUTOFF_LEN,
    ):
        self.workspace = Path(workspace).absolute()
        self.cutoff_len = cutoff_len
        self.mcp_configs = self._load_mcp_configs(config.mcp_config_file)

        self.client = MCPClient(envs=self._client_env(config))
        self.timing_dict: defaultdict[str, ToolTiming] = defaultdict(ToolTiming)
        self._tools_dict: dict[str, dict[str, Any]] = {}
        self._server_tools: defaultdict[str, list[str]] = defaultdict(list)
        self._tool_to_server: dict[str, str] = {}
        self.tool_history: list[tuple[ToolCall, ChatMessage]] = []
        self.tool_history_file = self.workspace / ".history" / "tool_history.jsonl"

    @staticmethod
    def _load_mcp_configs(config_file: str | Path) -> list[MCPServer]:
        with Path(config_file).open(encoding="utf-8") as handle:
            entries = json.load(handle)
        return [MCPServer(**entry) for entry in entries]

    def _host_workspace_path(self) -> str:
        host_root = os.environ.get("MEMSLIDES_HOST_WORKSPACE_BASE", "").strip()
        if not host_root:
            return str(self.workspace)
        mapped = str(self.workspace).replace(str(WORKSPACE_BASE), host_root, 1)
        debug("Host workspace mapping: %s -> %s", mapped, self.workspace)
        return mapped

    def _client_env(self, config: MemSlidesConfig) -> dict[str, str]:
        env = {
            "MEMSLIDES_WORKSPACE": str(self.workspace),
            "MEMSLIDES_HOST_WORKSPACE": self._host_workspace_path(),
            "MEMSLIDES_WORKSPACE_ID": self.workspace.stem,
            "MEMSLIDES_CONFIG_FILE": str(config.file_path),
            "FASTMCP_LOG_LEVEL": logging.getLevelName(LOGGING_LEVEL),
        }
        if config.offline_mode:
            env["MEMSLIDES_OFFLINE_MODE"] = "1"
        return env

    async def tool_execute(self, tool_call: ToolCall) -> ChatMessage:
        name = tool_call.function.name
        started_at = time.time()
        perf_start = time.perf_counter()
        arguments: dict[str, Any] | None = None

        try:
            arguments = self._decode_tool_arguments(tool_call)
            result = await self.client.tool_execute(
                self._tool_to_server[name],
                name,
                arguments,
            )
        except KeyError:
            result = _tool_error(f"Tool `{name}` is not registered.")
        except TimeoutError:
            result = _tool_error(
                f"Tool `{name}` timed out after {MCP_CALL_TIMEOUT} seconds."
            )
        except Exception as exc:
            result = _tool_error(f"Tool `{name}` failed: {exc}")

        elapsed = time.perf_counter() - perf_start
        duration_ms = max(0, round(elapsed * 1000))
        self.timing_dict[name].total_time += elapsed
        debug("Tool `%s` completed in %.2f seconds", name, elapsed)

        result = _coerce_textual_error_result(result)
        self._record_tool_outcome(name, tool_call.function.arguments, result)
        message = ChatMessage(
            role=Role.TOOL,
            content=self._message_content_from_result(name, arguments or {}, result),
            from_tool=tool_call.function,
            tool_call_id=tool_call.id,
            is_error=bool(getattr(result, "isError", False)),
            extra_info={
                "duration_ms": duration_ms,
                "started_at": started_at,
                "finished_at": started_at + elapsed,
            },
        )
        self.tool_history.append((tool_call, message))
        return message

    def _decode_tool_arguments(self, tool_call: ToolCall) -> dict[str, Any] | None:
        raw = tool_call.function.arguments or ""
        if not raw:
            return None
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return _normalize_tool_arguments(parsed)
        return parsed

    def _record_tool_outcome(
        self,
        name: str,
        raw_arguments: str,
        result: CallToolResult,
    ) -> None:
        timing = self.timing_dict[name]
        if getattr(result, "isError", False):
            timing.error_count += 1
            warning("Tool `%s` failed for args `%s`: %s", name, raw_arguments, result.content)
        else:
            timing.success_count += 1

    def _message_content_from_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: CallToolResult,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        unsupported = [
            getattr(block, "type", None)
            for block in result.content
            if getattr(block, "type", None) not in {"text", "image"}
        ]
        if unsupported:
            raise ValueError(f"Unsupported content type in tool result: {unsupported}")

        for block in result.content:
            if block.type == "image":
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": block.data, "detail": "low"},
                    }
                )
                continue
            converted.append(
                {
                    "type": "text",
                    "text": self._truncate_tool_text(tool_name, arguments, block.text),
                }
            )
        return converted

    def _truncate_tool_text(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        text: str,
    ) -> str:
        if len(text) <= self.cutoff_len:
            return text

        preview = text[: self.cutoff_len]
        last_break = preview.rfind("\n")
        if last_break > 0:
            preview = preview[:last_break]

        if tool_name == "read_file":
            resource_id = arguments.get("path") or arguments.get("file_path") or "unknown_file"
        else:
            resource_id = self.workspace / f"{tool_name}_{uuid.uuid4().hex[:4]}.txt"
            Path(resource_id).write_text(text, encoding="utf-8")

        return preview + CUTOFF_WARNING.format(
            line=preview.count("\n"),
            resource_id=str(resource_id),
        )

    async def __aenter__(self):
        self._remove_stale_sandbox_container()
        with timer("Connecting MCP servers"):
            await asyncio.gather(*(self.connect_server(server) for server in self.mcp_configs))
        self._register_builtin_tools()
        self._write_tool_cache_if_enabled()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for server_name in list(self._server_tools):
            await self.disconnect_server(server_name)
        self._append_tool_history()
        self._write_timing_summary()
        debug("Agent environment closed; tool history saved at %s", self.tool_history_file)

    def _remove_stale_sandbox_container(self) -> None:
        try:
            container = docker.from_env().containers.get(self.workspace.stem)
        except NotFound:
            return
        except DockerException as exc:
            warning("Docker is not accessible; sandbox cleanup skipped: %s", exc)
            return
        except Exception as exc:
            warning("Sandbox cleanup skipped after unexpected Docker error: %s", exc)
            return
        warning("Removing stale sandbox container id=%s", self.workspace.stem)
        container.remove(force=True)

    def _write_tool_cache_if_enabled(self) -> None:
        if LOGGING_LEVEL > logging.INFO:
            return
        tool_names = ", ".join(sorted(self._tools_dict))
        debug("Caching %d tool specs at %s: %s", len(self._tools_dict), TOOL_CACHE, tool_names)
        payload = {
            "server_tools": dict(self._server_tools),
            "tool_specs": list(self._tools_dict.values()),
        }
        TOOL_CACHE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_tool_history(self) -> None:
        self.tool_history_file.parent.mkdir(parents=True, exist_ok=True)
        with self.tool_history_file.open("a", encoding="utf-8") as handle:
            for tool_call, message in self.tool_history:
                record = [_dump_model(tool_call), _dump_model(message)]
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_timing_summary(self) -> None:
        history_dir = self.workspace / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ordered = sorted(
            self.timing_dict.items(),
            key=lambda item: item[1].total_time,
            reverse=True,
        )
        data = {name: timing.model_dump() for name, timing in ordered}
        (history_dir / "tools_time_cost.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    async def connect_server(self, server: MCPServer):
        server_name = server.name
        await self.client.connect_server(server_name, server)
        info("Connected to server %s", server_name)
        for tool_name, tool_info in (await self.client.list_tools(server_name)).items():
            if not self._server_allows_tool(server, tool_name):
                continue
            self._add_tool(
                server_name,
                tool_name,
                tool_info.description,
                tool_info.inputSchema,
            )

    @staticmethod
    def _server_allows_tool(server: MCPServer, tool_name: str) -> bool:
        keep = set(server.keep_tools or [])
        excluded = set(server.exclude_tools or [])
        return tool_name not in excluded and (not keep or tool_name in keep)

    def _add_tool(
        self,
        server_name: str,
        tool_name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        self._tools_dict[tool_name] = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._server_tools[server_name].append(tool_name)
        self._tool_to_server[tool_name] = server_name

    async def disconnect_server(self, server_name: str):
        tool_names = self._server_tools.pop(server_name, [])
        for tool_name in tool_names:
            self._tools_dict.pop(tool_name, None)
            self._tool_to_server.pop(tool_name, None)
        await self.client._close_server(server_name)
        info("Disconnected from server %s", server_name)

    def get_server_tools(self, server_name: str):
        return [
            self._tools_dict[tool_name]
            for tool_name in self._server_tools.get(server_name, [])
            if tool_name in self._tools_dict
        ]

    def _register_builtin_tools(self) -> None:
        return None
