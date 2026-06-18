import asyncio
import json
import uuid
from abc import abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import jsonlines
import yaml
from jinja2 import Template
from jinja2.runtime import StrictUndefined
from openai.types.chat.chat_completion_message import ChatCompletionMessage
try:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageFunctionToolCall as ToolCall,
    )
except ImportError:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall as ToolCall,
    )
from pydantic import BaseModel

from memslides.agents.env import AgentEnv
from memslides.memory.extract.tool_reasoning import (
    extract_text_from_content,
    normalize_reason_text,
)
from memslides.templates.runtime_state import load_template_runtime_state
from memslides.utils.config import (
    LLM,
    MemSlidesConfig,
    get_json_from_response,
)
from memslides.utils.constants import (
    AGENT_PROMPT,
    CONTEXT_MODE_PROMPT,
    CONTINUE_MSG,
    HALF_BUDGET_NOTICE_MSG,
    HIST_LOST_MSG,
    LAST_ITER_MSG,
    MAX_LOGGING_LENGTH,
    MAX_TOOL_RESULT_CHARS,
    MAX_TOOLCALL_PER_TURN,
    MEMORY_COMPACT_MSG,
    OFFLINE_PROMPT,
    OVERFLOW_MARKER,
    PACKAGE_DIR,
    TOOL_REASON_PROMPT,
    URGENT_BUDGET_NOTICE_MSG,
)
from memslides.utils.log import (
    debug,
    info,
    timer,
    warning,
)
from memslides.utils.typings import (
    ChatMessage,
    Cost,
    InputRequest,
    Role,
    RoleConfig,
)


class RoleToolContractError(RuntimeError):
    """Raised when a role prompt and runtime tool schema drift apart."""


class Agent:
    _DESIGN_FORBIDDEN_PATCH_TOOLS = frozenset(
        {
            "scan_slide_index",
            "batch_update_css_rule",
            "batch_update_semantic_style",
            "patch_semantic_inline_style",
        }
    )
    _LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS = {
        "replace_text": {"target_id", "text"},
        "replace_html": {"target_id", "html_fragment"},
        "merge_style": {"target_id", "declarations"},
        "remove_node": {"target_id"},
        "insert_html": {"anchor_target_id", "position", "html_fragment"},
        "set_attr": {"target_id", "attr_name", "value"},
        "remove_attr": {"target_id", "attr_name"},
        "merge_css_rule": {"rule_id", "declarations"},
        "replace_css_rule": {"rule_id", "declarations"},
    }
    _LEGACY_INSERT_POSITIONS = {"before", "after", "prepend", "append"}

    def __init__(
        self,
        config: MemSlidesConfig,
        agent_env: AgentEnv,
        workspace: Path,
        language: Literal["zh", "en"],
        config_file: str | None = None,
        keep_reasoning: bool = True,
    ):
        self.name = self.__class__.__name__
        self.cost = Cost()
        self.context_length = 0
        self.context_warning = 0
        self.workspace = workspace
        self.agent_env = agent_env
        self.language = language
        self.keep_reasoning = keep_reasoning
        self.context_window = config.context_window
        self.max_context_turns = config.max_context_folds
        config_file = (
            Path(config_file)
            if config_file
            else PACKAGE_DIR / "roles" / f"{self.name}.yaml"
        )
        if not config_file.exists():
            raise FileNotFoundError(f"Cannot found role config file at: {config_file} ")

        # Setting basic context
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        self.role_config = RoleConfig(**config_data)
        self.model_ref, self.llm = config.resolve_llm(self.role_config.use_model)
        if self.model_ref != self.role_config.use_model:
            info(
                f"{self.name} Agent requested model '{self.role_config.use_model}' "
                f"but is using fallback '{self.model_ref}'"
            )
        self.model = self.llm.model_name
        self._setup_toolset()
        self._base_tool_names = [
            str(tool.get("function", {}).get("name", "") or "").strip()
            for tool in self.tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and str(tool.get("function", {}).get("name", "") or "").strip()
        ]
        self._validate_role_tool_contract()
        if language not in self.role_config.system:
            raise ValueError(f"Language '{language}' not found in system prompts")
        self.error_history: list[ToolCall | ChatMessage] = []
        self._tool_result_callback = None
        self._memory_orchestrator = None  # Set by main.py for operation-level memory injection
        self._action_count = 0
        self.research_iter = 0
        if config.context_folding:
            self.context_warning = -1

        # Setting tools and interative context
        self.system = self.role_config.system[language]
        self.prompt: Template = Template(
            self.role_config.instruction, undefined=StrictUndefined
        )
        # ? for those agents equipped with sandbox only
        if any(t["function"]["name"] == "execute_command" for t in self.tools):
            self.system += AGENT_PROMPT.format(
                workspace=self.workspace,
                cutoff_len=self.agent_env.cutoff_len,
                time=datetime.now().strftime("%Y-%m-%d"),
                max_toolcall_per_turn=MAX_TOOLCALL_PER_TURN,
            )

        if config.offline_mode:
            self.system += OFFLINE_PROMPT

        if config.context_folding:
            self.system += CONTEXT_MODE_PROMPT

        if self.tools:
            self.system += TOOL_REASON_PROMPT

        self.chat_history: list[ChatMessage] = [
            ChatMessage(role=Role.SYSTEM, content=self.system)
        ]
        available_tools = [tool["function"]["name"] for tool in self.tools]
        debug(
            f"{self.name} Agent got {len(self.tools)} tools: {', '.join(available_tools)}"
        )
        info(
            f"{self.name} Agent model route: requested={self.role_config.use_model}, "
            f"resolved={self.model_ref}, model={self.model}"
        )

    @classmethod
    def _extract_reasoning_payload_from_openai_message(
        cls, message: Any
    ) -> tuple[str | None, str]:
        """Extract observable tool-use reason and its source."""
        reasoning = getattr(message, "reasoning_content", None)
        normalized = normalize_reason_text(reasoning)
        if normalized:
            return normalized, "reasoning_content"

        extra = getattr(message, "__pydantic_extra__", None) or {}
        for key in ("reasoning_content", "reasoning", "thinking", "reasoning_text", "reasoning_summary"):
            value = extra.get(key)
            if isinstance(value, str):
                normalized = normalize_reason_text(value)
                if normalized:
                    return normalized, "provider_extra_reasoning"
            elif isinstance(value, dict):
                for subkey in ("summary", "text", "content"):
                    nested = value.get(subkey)
                    if isinstance(nested, str):
                        normalized = normalize_reason_text(nested)
                        if normalized:
                            return normalized, "provider_extra_reasoning"

        content = extract_text_from_content(getattr(message, "content", None))
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls and content:
            normalized = normalize_reason_text(content)
            if normalized:
                return normalized, "assistant_content"
        if tool_calls:
            return None, "missing"
        return None, "not_applicable"

    @staticmethod
    def _extract_reasoning_from_openai_message(message: Any) -> str | None:
        """提取可观测的 reasoning 文本。"""
        reasoning, _ = Agent._extract_reasoning_payload_from_openai_message(message)
        return reasoning

    @classmethod
    def _reasoning_source_from_openai_message(cls, message: Any) -> str:
        """标记 tool-call reasoning 实际来自哪里，便于观测。"""
        _, source = cls._extract_reasoning_payload_from_openai_message(message)
        return source

    @staticmethod
    def _model_key_meta(obj: Any) -> tuple[list[str], list[str]]:
        """返回 pydantic/openai model 的声明字段与 extra 字段。"""
        model_fields = getattr(type(obj), "model_fields", {}) or {}
        declared_keys = list(model_fields.keys())
        extra_keys = list((getattr(obj, "__pydantic_extra__", None) or {}).keys())
        return declared_keys, extra_keys

    @classmethod
    def _build_llm_response_meta(cls, response: Any) -> dict[str, Any]:
        """构建轻量观测元数据，便于排查 reasoning 是否被返回。"""
        response_keys, response_extra_keys = cls._model_key_meta(response)
        response_dump = response.model_dump() if hasattr(response, "model_dump") else None
        response_dump_keys = list(response_dump.keys()) if isinstance(response_dump, dict) else []
        response_populated = (
            response.model_dump(exclude_none=True)
            if hasattr(response, "model_dump")
            else None
        )
        response_populated_keys = (
            list(response_populated.keys()) if isinstance(response_populated, dict) else []
        )
        message = response.choices[0].message if getattr(response, "choices", None) else None

        message_keys: list[str] = []
        message_extra_keys: list[str] = []
        message_dump_keys: list[str] = []
        message_populated_keys: list[str] = []
        if message is not None:
            message_keys, message_extra_keys = cls._model_key_meta(message)
            message_dump = message.model_dump() if hasattr(message, "model_dump") else None
            if isinstance(message_dump, dict):
                message_dump_keys = list(message_dump.keys())
            message_populated = (
                message.model_dump(exclude_none=True)
                if hasattr(message, "model_dump")
                else None
            )
            if isinstance(message_populated, dict):
                message_populated_keys = list(message_populated.keys())

        usage = getattr(response, "usage", None)
        usage_dump = usage.model_dump() if usage is not None else None
        usage_keys = list(usage_dump.keys()) if isinstance(usage_dump, dict) else []

        reasoning_tokens = None
        completion_tokens_detail_keys: list[str] = []
        output_tokens_detail_keys: list[str] = []
        if isinstance(usage_dump, dict):
            completion_details = usage_dump.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                completion_tokens_detail_keys = list(completion_details.keys())
                reasoning_tokens = completion_details.get("reasoning_tokens")
            if reasoning_tokens is None:
                output_details = usage_dump.get("output_tokens_details")
                if isinstance(output_details, dict):
                    output_tokens_detail_keys = list(output_details.keys())
                    reasoning_tokens = output_details.get("reasoning_tokens")
            elif isinstance(usage_dump.get("output_tokens_details"), dict):
                output_tokens_detail_keys = list(
                    usage_dump["output_tokens_details"].keys()
                )

        return {
            "response_type": type(response).__name__,
            "response_keys": response_keys,
            "response_dump_keys": response_dump_keys,
            "response_populated_keys": response_populated_keys,
            "response_extra_keys": response_extra_keys,
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "message_keys": message_keys,
            "message_dump_keys": message_dump_keys,
            "message_populated_keys": message_populated_keys,
            "message_extra_keys": message_extra_keys,
            "tool_call_count": len(getattr(message, "tool_calls", None) or []) if message is not None else 0,
            "tool_call_reason_source": cls._reasoning_source_from_openai_message(message) if message is not None else "not_applicable",
            "has_reasoning_content_attr": bool(message is not None and hasattr(message, "reasoning_content")),
            "reasoning_content_present": bool(cls._extract_reasoning_from_openai_message(message)) if message is not None else False,
            "usage_keys": usage_keys,
            "completion_tokens_detail_keys": completion_tokens_detail_keys,
            "output_tokens_detail_keys": output_tokens_detail_keys,
            "usage": usage_dump,
            "reasoning_tokens": reasoning_tokens,
        }

    def _setup_toolset(self):
        if self.role_config.include_tool_servers == "all":
            self.role_config.include_tool_servers = list(self.agent_env._server_tools)
        for server in self.role_config.include_tool_servers:
            assert server in self.agent_env._server_tools, (
                f"Server {server} is not available"
            )
        for tool in self.role_config.include_tools:
            assert tool in self.agent_env._tools_dict, f"Tool {tool} is not available"
        self.tools = []
        _added_tool_names = set()
        template_state = load_template_runtime_state(self.workspace)
        for server in self.role_config.include_tool_servers:
            if server not in self.role_config.exclude_tool_servers:
                if server == "memslides_template_tools" and not (
                    template_state and template_state.active
                ):
                    continue
                for tool in self.agent_env._server_tools[server]:
                    if tool not in self.role_config.exclude_tools:
                        self.tools.append(self.agent_env._tools_dict[tool])
                        _added_tool_names.add(tool)

        for tool_name, tool in self.agent_env._tools_dict.items():
            if tool_name in self.role_config.include_tools and tool_name not in _added_tool_names:
                self.tools.append(tool)
                _added_tool_names.add(tool_name)

    def set_tools_by_names(self, tool_names: list[str]) -> list[str]:
        """Replace the active tool list with a deterministic allowlist."""
        allowed: list[str] = []
        seen: set[str] = set()
        new_tools: list[dict[str, Any]] = []
        for tool_name in tool_names:
            tool_name = str(tool_name or "").strip()
            if not tool_name or tool_name in seen:
                continue
            tool = self.agent_env._tools_dict.get(tool_name)
            if tool is None:
                continue
            seen.add(tool_name)
            allowed.append(tool_name)
            new_tools.append(tool)
        self.tools = new_tools
        return allowed

    def restore_base_tools(self) -> list[str]:
        """Restore the role-configured tool list captured at initialization."""
        base_names = list(getattr(self, "_base_tool_names", []) or [])
        if not base_names:
            self._setup_toolset()
            base_names = [
                str(tool.get("function", {}).get("name", "") or "").strip()
                for tool in self.tools
                if isinstance(tool, dict)
                and isinstance(tool.get("function"), dict)
                and str(tool.get("function", {}).get("name", "") or "").strip()
            ]
            self._base_tool_names = base_names
            return base_names
        return self.set_tools_by_names(base_names)

    def remove_tools_by_names(self, tool_names: set[str]) -> list[str]:
        """Drop a set of tool names from the active tool list."""
        removed: list[str] = []
        retained: list[dict[str, Any]] = []
        for tool in self.tools:
            name = str(tool.get("function", {}).get("name", "") or "").strip()
            if name and name in tool_names:
                removed.append(name)
                continue
            retained.append(tool)
        self.tools = retained
        return removed

    @staticmethod
    def _tool_parameter_names(tool: dict[str, Any] | None) -> set[str]:
        if not isinstance(tool, dict):
            return set()
        function = tool.get("function", {})
        if not isinstance(function, dict):
            return set()
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            return set()
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return set()
        return {str(name) for name in properties.keys()}

    @staticmethod
    def _required_tool_parameter_names(tool: dict[str, Any] | None) -> set[str]:
        if not isinstance(tool, dict):
            return set()
        function = tool.get("function", {})
        if not isinstance(function, dict):
            return set()
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            return set()
        required = parameters.get("required", [])
        if not isinstance(required, list):
            return set()
        return {str(name) for name in required}

    @staticmethod
    def _tool_parameter_schema(tool: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(tool, dict):
            return {}
        function = tool.get("function", {})
        if not isinstance(function, dict):
            return {}
        parameters = function.get("parameters", {})
        if not isinstance(parameters, dict):
            return {}
        return parameters

    @staticmethod
    def _resolve_local_json_schema_ref(
        ref: str,
        definitions: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not isinstance(ref, str):
            return None
        prefix_to_strip = None
        if ref.startswith("#/$defs/"):
            prefix_to_strip = "#/$defs/"
        elif ref.startswith("#/definitions/"):
            prefix_to_strip = "#/definitions/"
        if prefix_to_strip is None:
            return None
        schema = definitions.get(ref[len(prefix_to_strip):])
        return schema if isinstance(schema, dict) else None

    @classmethod
    def _apply_slide_patch_contract(
        cls,
        tool: dict[str, Any] | None,
    ) -> tuple[dict[str, set[str]], set[str]]:
        parameters = cls._tool_parameter_schema(tool)
        properties = parameters.get("properties", {})
        if not isinstance(properties, dict):
            return (
                dict(cls._LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS),
                set(cls._LEGACY_INSERT_POSITIONS),
            )

        patch_ops_schema = properties.get("patch_ops", {})
        if not isinstance(patch_ops_schema, dict):
            return (
                dict(cls._LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS),
                set(cls._LEGACY_INSERT_POSITIONS),
            )

        items_schema = patch_ops_schema.get("items", {})
        if not isinstance(items_schema, dict):
            return (
                dict(cls._LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS),
                set(cls._LEGACY_INSERT_POSITIONS),
            )

        variants = items_schema.get("oneOf", [])
        if not isinstance(variants, list) or len(variants) == 0:
            return (
                dict(cls._LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS),
                set(cls._LEGACY_INSERT_POSITIONS),
            )

        definitions = parameters.get("$defs", {})
        if not isinstance(definitions, dict):
            definitions = parameters.get("definitions", {})
        if not isinstance(definitions, dict):
            definitions = {}

        required_by_op: dict[str, set[str]] = {}
        insert_positions = set(cls._LEGACY_INSERT_POSITIONS)
        for variant in variants:
            variant_schema = variant if isinstance(variant, dict) else None
            if variant_schema is None:
                continue

            ref = variant_schema.get("$ref")
            if ref:
                variant_schema = cls._resolve_local_json_schema_ref(ref, definitions)
            if not isinstance(variant_schema, dict):
                continue

            variant_properties = variant_schema.get("properties", {})
            if not isinstance(variant_properties, dict):
                continue
            op_schema = variant_properties.get("op", {})
            if not isinstance(op_schema, dict):
                continue

            op_name = op_schema.get("const")
            if not isinstance(op_name, str) or not op_name.strip():
                enum_values = op_schema.get("enum", [])
                if (
                    isinstance(enum_values, list)
                    and len(enum_values) == 1
                    and isinstance(enum_values[0], str)
                    and enum_values[0].strip()
                ):
                    op_name = enum_values[0].strip()
            if not isinstance(op_name, str) or not op_name.strip():
                continue

            required_fields = variant_schema.get("required", [])
            if not isinstance(required_fields, list):
                required_fields = []
            required_by_op[op_name] = {
                str(field)
                for field in required_fields
                if str(field) and str(field) != "op"
            }

            if op_name == "insert_html":
                position_schema = variant_properties.get("position", {})
                if isinstance(position_schema, dict):
                    enum_values = position_schema.get("enum", [])
                    parsed_positions = {
                        str(value).strip().lower()
                        for value in enum_values
                        if isinstance(value, str) and str(value).strip()
                    }
                    if parsed_positions:
                        insert_positions = parsed_positions

        if not required_by_op:
            return (
                dict(cls._LEGACY_APPLY_SLIDE_PATCH_REQUIREMENTS),
                set(cls._LEGACY_INSERT_POSITIONS),
            )
        return required_by_op, insert_positions

    def _tool_call_argument_error(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
    ) -> str | None:
        if not isinstance(arguments, dict):
            return None

        available_tools = {
            tool["function"]["name"]: tool
            for tool in self.tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name")
        }
        tool = available_tools.get(tool_name)
        required = self._required_tool_parameter_names(tool)
        missing_required = sorted(name for name in required if name not in arguments)
        if missing_required:
            if tool_name == "apply_slide_patch" and "patch_ops" in missing_required:
                return (
                    "Tool call `apply_slide_patch` is incomplete: missing `patch_ops`. "
                    "First call `read_slide_snapshot`, then send a non-empty `patch_ops` list "
                    "plus the matching `expected_hash`."
                )
            return (
                f"Tool call `{tool_name}` is missing required arguments: "
                + ", ".join(missing_required)
            )

        if tool_name != "apply_slide_patch":
            return None

        patch_ops = arguments.get("patch_ops")
        if not isinstance(patch_ops, list) or len(patch_ops) == 0:
            return (
                "Tool call `apply_slide_patch` requires a non-empty `patch_ops` list. "
                "Build one or more patch ops from `read_slide_snapshot.targets` before retrying."
            )

        required_by_op, valid_insert_positions = self._apply_slide_patch_contract(tool)
        for index, op in enumerate(patch_ops, start=1):
            if not isinstance(op, dict):
                return f"Tool call `apply_slide_patch` patch op #{index} must be an object."
            op_name = str(op.get("op", "") or "").strip()
            if op_name not in required_by_op:
                return (
                    f"Tool call `apply_slide_patch` patch op #{index} has unsupported or missing "
                    f"`op`: {op_name or '<empty>'}."
                )
            missing_fields = sorted(
                field for field in required_by_op[op_name] if field not in op
            )
            if missing_fields:
                return (
                    f"Tool call `apply_slide_patch` patch op #{index} ({op_name}) is missing fields: "
                    + ", ".join(missing_fields)
                )
            if op_name == "insert_html":
                position = str(op.get("position", "") or "").strip().lower()
                if position not in valid_insert_positions:
                    return (
                        f"Tool call `apply_slide_patch` patch op #{index} has invalid insert position: "
                        f"{position or '<empty>'}."
                    )
        return None

    def _validate_role_tool_contract(self) -> None:
        """Fail loudly when a role prompt depends on tools the runtime lacks."""
        if self.name not in {"RevisionEditor", "DeckDesigner"}:
            return

        available_tools = {
            tool["function"]["name"]: tool
            for tool in self.tools
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name")
        }
        if self.name == "DeckDesigner":
            forbidden_tools = sorted(
                tool_name
                for tool_name in self._DESIGN_FORBIDDEN_PATCH_TOOLS
                if tool_name in available_tools
            )
            if forbidden_tools:
                available_preview = ", ".join(sorted(available_tools.keys())[:20])
                raise RoleToolContractError(
                    "DeckDesigner protocol consistency check failed: unexpected modify-only tools: "
                    + ", ".join(forbidden_tools)
                    + ". DeckDesigner must not receive existing-slide patch/repair tools. "
                    + f"Available tools preview: {available_preview}"
                )
            missing_tools = [
                tool_name
                for tool_name in (
                    "write_html_file",
                    "read_slide_snapshot",
                    "apply_slide_patch",
                    "inspect_slide",
                    "finalize",
                )
                if tool_name not in available_tools
            ]
            schema_issues: list[str] = []
            write_html_params = self._tool_parameter_names(available_tools.get("write_html_file"))
            missing_write_params = sorted(
                {"force_regenerate", "expected_hash"} - write_html_params
            )
            if missing_write_params:
                schema_issues.append(
                    "write_html_file missing params: " + ", ".join(missing_write_params)
                )

            snapshot_params = self._tool_parameter_names(
                available_tools.get("read_slide_snapshot")
            )
            missing_snapshot_params = sorted({"slide_path"} - snapshot_params)
            if missing_snapshot_params:
                schema_issues.append(
                    "read_slide_snapshot missing params: "
                    + ", ".join(missing_snapshot_params)
                )

            patch_params = self._tool_parameter_names(
                available_tools.get("apply_slide_patch")
            )
            missing_patch_params = sorted(
                {"snapshot_id", "patch_ops", "expected_hash"} - patch_params
            )
            if missing_patch_params:
                schema_issues.append(
                    "apply_slide_patch missing params: "
                    + ", ".join(missing_patch_params)
                )

            if not missing_tools and not schema_issues:
                return

            available_preview = ", ".join(sorted(available_tools.keys())[:20])
            details: list[str] = []
            if missing_tools:
                details.append("missing tools: " + ", ".join(missing_tools))
            details.extend(schema_issues)
            raise RoleToolContractError(
                "DeckDesigner protocol consistency check failed: "
                + "; ".join(details)
                + ". Refusing to start DeckDesigner with a stale prompt/tool contract. "
                + f"Available tools preview: {available_preview}"
            )

        missing_tools = [
            tool_name
            for tool_name in (
                "read_slide_snapshot",
                "plan_slide_patch",
                "apply_slide_patch",
                "insert_slide",
                "write_new_slide_file",
                "inspect_slide",
                "finalize",
            )
            if tool_name not in available_tools
        ]

        schema_issues: list[str] = []
        new_slide_params = self._tool_parameter_names(available_tools.get("write_new_slide_file"))
        missing_new_slide_params = sorted({"file_path", "content"} - new_slide_params)
        if missing_new_slide_params:
            schema_issues.append(
                "write_new_slide_file missing params: " + ", ".join(missing_new_slide_params)
            )

        snapshot_params = self._tool_parameter_names(
            available_tools.get("read_slide_snapshot")
        )
        missing_snapshot_params = sorted({"slide_path"} - snapshot_params)
        if missing_snapshot_params:
            schema_issues.append(
                "read_slide_snapshot missing params: "
                + ", ".join(missing_snapshot_params)
            )

        planner_params = self._tool_parameter_names(
            available_tools.get("plan_slide_patch")
        )
        missing_planner_params = sorted({"slide_path", "edit_intent"} - planner_params)
        if missing_planner_params:
            schema_issues.append(
                "plan_slide_patch missing params: "
                + ", ".join(missing_planner_params)
            )

        patch_params = self._tool_parameter_names(
            available_tools.get("apply_slide_patch")
        )
        missing_patch_params = sorted(
            {"snapshot_id", "patch_ops", "expected_hash"} - patch_params
        )
        if missing_patch_params:
            schema_issues.append(
                "apply_slide_patch missing params: "
                + ", ".join(missing_patch_params)
            )

        if not missing_tools and not schema_issues:
            return

        available_preview = ", ".join(sorted(available_tools.keys())[:20])
        details: list[str] = []
        if missing_tools:
            details.append("missing tools: " + ", ".join(missing_tools))
        details.extend(schema_issues)
        raise RoleToolContractError(
            "RevisionEditor protocol consistency check failed: "
            + "; ".join(details)
            + ". Refusing to start RevisionEditor with a stale prompt/tool contract. "
            + f"Available tools preview: {available_preview}"
        )

    def _ensure_initial_user_turn(self, **chat_kwargs) -> None:
        if len(self.chat_history) != 1:
            return
        message = ChatMessage(
            role=Role.USER,
            content=self.prompt.render(**chat_kwargs),
        )
        self.chat_history.append(message)
        self.log_message(message)

    def _append_chat_message(self, message: ChatMessage) -> ChatMessage:
        self.chat_history.append(message)
        self.log_message(message)
        return message

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.cost += usage
        self.context_length = usage.prompt_tokens

    def _message_from_completion(self, response: Any) -> ChatMessage:
        completion_message = response.choices[0].message
        return ChatMessage(
            role=completion_message.role,
            content=completion_message.content,
            reasoning_content=self._extract_reasoning_from_openai_message(
                completion_message
            ),
            extra_info={
                "llm_response_meta": self._build_llm_response_meta(response),
            },
        )

    async def chat(
        self,
        message: ChatMessage,
        response_format: type[BaseModel] | None = None,
        **chat_kwargs,
    ) -> ChatMessage:
        self._ensure_initial_user_turn(**chat_kwargs)
        self._append_chat_message(message)
        with timer(f"{self.name} Agent LLM chat"):
            response = await self.llm.run(
                messages=self.chat_history,
                response_format=response_format,
            )
            self._record_usage(response)
            return self._append_chat_message(self._message_from_completion(response))

    async def action(
        self,
        **chat_kwargs,
    ):
        """Tool calling interface"""

        self._ensure_initial_user_turn(**chat_kwargs)

        with timer(f"{self.name} Agent LLM call"):
            request_kwargs = {}
            if self.name == "Researcher":
                request_kwargs["tool_choice"] = "auto"
            response = await self.llm.run(
                messages=self.chat_history,
                tools=self.tools,
                request_kwargs=request_kwargs or None,
            )
            if response.usage is not None:
                self.cost += response.usage
                self.context_length = response.usage.prompt_tokens
            agent_message: ChatCompletionMessage = response.choices[0].message
        reasoning, reasoning_source = self._extract_reasoning_payload_from_openai_message(
            agent_message
        )
        if self.keep_reasoning:
            visible_reasoning = reasoning
        else:
            visible_reasoning = None
        self.chat_history.append(
            ChatMessage(
                role=agent_message.role,
                content=agent_message.content,
                tool_calls=agent_message.tool_calls,
                reasoning_content=visible_reasoning,
                extra_info={
                    "tool_call_reason_source": reasoning_source,
                    "llm_response_meta": self._build_llm_response_meta(response),
                },
            )
        )
        self.log_message(self.chat_history[-1])
        return self.chat_history[-1]

    @abstractmethod
    async def loop(
        self, req: InputRequest, *args, **kwargs
    ) -> AsyncGenerator[str | ChatMessage, None]:
        """
        Loop interface, return the message or the outcome filepath of the agent.
        """

    @abstractmethod
    async def finish(self, result: str):
        """Define when and how an agent should finish its work, combined with outcome checks."""

    async def execute(self, tool_calls: list[ToolCall]) -> str | list[ChatMessage]:
        self._action_count += 1
        coros = []
        observations: list[ChatMessage] = []
        used_tools = set()
        finish_id = None
        outcome = None

        def _normalize_outcome_path(value: str | None) -> str | None:
            text = str(value or "").strip()
            if not text:
                return None
            if text.startswith("Error:") or text.startswith("Outcome "):
                return None
            try:
                path = Path(text)
                if not path.is_absolute():
                    path = self.workspace / path
                return str(path.resolve())
            except Exception:
                return None

        current_reasoning = ""
        current_reasoning_source = "missing"
        if self.chat_history:
            last_message = self.chat_history[-1]
            if last_message.role == Role.ASSISTANT:
                assistant_text = extract_text_from_content(last_message.content)
                current_reasoning = (
                    last_message.reasoning_content
                    or (
                        normalize_reason_text(assistant_text)
                        if last_message.tool_calls and assistant_text.strip()
                        else ""
                    )
                )
                current_reasoning_source = str(
                    last_message.extra_info.get(
                        "tool_call_reason_source",
                        "assistant_content" if current_reasoning else "missing",
                    )
                )

        # Handle tool call limit: execute first N, notify about skipped ones
        if len(tool_calls) > MAX_TOOLCALL_PER_TURN:
            info(f"Tool calls ({len(tool_calls)}) exceed limit ({MAX_TOOLCALL_PER_TURN}), executing first {MAX_TOOLCALL_PER_TURN}")
            executed_calls = tool_calls[:MAX_TOOLCALL_PER_TURN]
            skipped_calls = tool_calls[MAX_TOOLCALL_PER_TURN:]

            # Create error messages for skipped calls with actionable guidance
            skipped_tool_names = [t.function.name for t in skipped_calls]
            skipped_summary = ", ".join(set(skipped_tool_names))
            for idx, t in enumerate(skipped_calls):
                if idx == 0:
                    # First skipped call gets detailed message
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"⚠️ Tool call limit reached: {len(executed_calls)} executed, {len(skipped_calls)} skipped.\n"
                                    f"✅ Executed: {len(executed_calls)}/{len(tool_calls)} tools\n"
                                    f"⏭️ Skipped: {skipped_summary}\n"
                                    f"💡 Continue calling `{t.function.name}` in the NEXT turn to complete your exploration or work.",
                            tool_call_id=t.id,
                            is_error=True,
                        )
                    )
                else:
                    # Subsequent skipped calls get brief message
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Tool call skipped (exceeded limit). Continue in next turn.",
                            tool_call_id=t.id,
                            is_error=True,
                        )
                    )
            tool_calls = executed_calls

        for t in tool_calls:
            arguments = t.function.arguments
            if len(arguments) == 0:
                arguments = None
            else:
                try:
                    arguments = get_json_from_response(t.function.arguments)
                    if t.function.name == "finalize":
                        finish_id = t.id
                        assert "outcome" in arguments, (
                            "Finalize tool call must have an outcome"
                        )
                        outcome = arguments["outcome"]
                    assert isinstance(arguments, dict), (
                        f"Tool call arguments must be a dict or empty, while {arguments} is given"
                    )
                    validation_error = self._tool_call_argument_error(
                        t.function.name, arguments,
                    )
                    if validation_error:
                        observations.append(
                            ChatMessage(
                                role=Role.TOOL,
                                content=validation_error,
                                tool_call_id=t.id,
                                is_error=True,
                            )
                        )
                        info(f"Tool call `{t.function}` encountered error: {validation_error}")
                        continue
                    t.function.arguments = json.dumps(arguments, ensure_ascii=False)
                except AssertionError as e:
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=str(e),
                            tool_call_id=t.id,
                            is_error=True,
                        )
                    )
                    info(f"Tool call `{t.function}` encountered error: {e}")
                    continue
            used_tools.add(t.function.name)
            coros.append((t, self.agent_env.tool_execute(t)))

        # ── Letta-style two-level concurrency control ──
        # Tools that produce files or depend on file-system state must run
        # sequentially (in the order the LLM submitted them) to avoid race
        # conditions.  All other tools are parallel-safe and can use gather.
        _SEQUENTIAL_TOOLS = frozenset({
            "inspect_slide",      # validator — reads file-system state
            "read_slide_snapshot",# validator — reads bounded slide snapshot state
            "apply_slide_patch",  # producer — modifies existing slide files
            "write_html_file",    # producer — creates/modifies files
            "render_chart_asset", # producer — creates structured visual files
            "render_table_asset", # producer — creates structured visual files
            "image_generation",   # producer — creates image files
            "download_file",      # producer — creates files
        })

        sequential_coros = [(t, c) for t, c in coros if t.function.name in _SEQUENTIAL_TOOLS]
        parallel_coros   = [(t, c) for t, c in coros if t.function.name not in _SEQUENTIAL_TOOLS]

        def _collect_gather(tool_calls_list, gather_results):
            """Collect results from asyncio.gather, creating error messages for exceptions.

            Also applies a truncation guard to prevent oversized tool outputs.
            """
            results = []
            for _idx, _result in enumerate(gather_results):
                if isinstance(_result, BaseException):
                    _tc = tool_calls_list[_idx] if _idx < len(tool_calls_list) else None
                    _tc_id = _tc.id if _tc else f"unknown_{_idx}"
                    _tc_name = _tc.function.name if _tc else "unknown"
                    results.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Tool '{_tc_name}' execution failed: {_result}",
                            tool_call_id=_tc_id,
                            is_error=True,
                        )
                    )
                    info(f"Tool '{_tc_name}' raised {type(_result).__name__}: {_result}")
                else:
                    # ── Truncation guard: 防止超大输出导致 token 溢出 ──
                    if isinstance(_result, ChatMessage) and _result.content:
                        content_len = len(_result.content)
                        if content_len > MAX_TOOL_RESULT_CHARS:
                            truncated_content = _result.content[:MAX_TOOL_RESULT_CHARS]
                            overflow_msg = OVERFLOW_MARKER.format(
                                max_chars=MAX_TOOL_RESULT_CHARS,
                                actual_len=content_len
                            )
                            _result.content = truncated_content + "\n\n" + overflow_msg
                            info(
                                f"Tool result truncated: {content_len} → {MAX_TOOL_RESULT_CHARS} chars "
                                f"(tool_call_id={_result.tool_call_id})"
                            )
                    results.append(_result)
            return results

        if parallel_coros and sequential_coros:
            # Mixed batch: parallel-safe tools first, then sequential in order
            info(
                f"Phased execution: {len(parallel_coros)} parallel + "
                f"{len(sequential_coros)} sequential tools"
            )
            # Phase 1: parallel-safe tools
            _par_tcs = [p[0] for p in parallel_coros]
            _par_tasks = [p[1] for p in parallel_coros]
            _par_results = await asyncio.gather(*_par_tasks, return_exceptions=True)
            observations.extend(_collect_gather(_par_tcs, _par_results))

            # Phase 2: sequential tools — one by one in LLM-submitted order
            for _seq_tc, _seq_coro in sequential_coros:
                try:
                    _seq_result = await _seq_coro
                    observations.append(_seq_result)
                except BaseException as _seq_exc:
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Tool '{_seq_tc.function.name}' execution failed: {_seq_exc}",
                            tool_call_id=_seq_tc.id,
                            is_error=True,
                        )
                    )
                    info(f"Tool '{_seq_tc.function.name}' raised {type(_seq_exc).__name__}: {_seq_exc}")
        elif sequential_coros and not parallel_coros:
            # All sequential: execute in LLM-submitted order
            for _seq_tc, _seq_coro in sequential_coros:
                try:
                    _seq_result = await _seq_coro
                    observations.append(_seq_result)
                except BaseException as _seq_exc:
                    observations.append(
                        ChatMessage(
                            role=Role.TOOL,
                            content=f"Tool '{_seq_tc.function.name}' execution failed: {_seq_exc}",
                            tool_call_id=_seq_tc.id,
                            is_error=True,
                        )
                    )
                    info(f"Tool '{_seq_tc.function.name}' raised {type(_seq_exc).__name__}: {_seq_exc}")
        else:
            # All parallel-safe: original behavior
            _coro_tool_calls = [pair[0] for pair in coros]
            _coro_tasks = [pair[1] for pair in coros]
            _gather_results = await asyncio.gather(*_coro_tasks, return_exceptions=True)
            observations.extend(_collect_gather(_coro_tool_calls, _gather_results))

        # ── Step 2: Enhanced error diagnostics for "Image not found" ──
        _file_producers_in_batch = {
            t.function.name for t, _ in coros
        } & {
            "image_generation",
            "download_file",
            "write_html_file",
            "apply_slide_patch",
            "render_chart_asset",
            "render_table_asset",
        }
        if _file_producers_in_batch:
            for obs in observations:
                if obs.is_error and "Image not found" in obs.text:
                    self._ensure_list_content(obs)
                    obs.content.append({
                        "text": (
                            "⚠️ DIAGNOSTIC: This error may have occurred because file-producing tools "
                            f"({', '.join(_file_producers_in_batch)}) and file-validating tools ran in "
                            "the same batch. If the image was just generated, verify the file path is "
                            "correct and retry inspect_slide in the next turn."
                        ),
                        "type": "text",
                    })
        self._normalize_image_observations(observations)

        self.chat_history.extend(observations)

        # Match tool calls with observations by tool_call_id (not random UUID)
        _obs_by_tcid = {}
        for obs in observations:
            if obs.tool_call_id:
                _obs_by_tcid[obs.tool_call_id] = obs

        for t in tool_calls:
            o = _obs_by_tcid.get(t.id)
            if o is None:
                continue
            if o.is_error:
                self.error_history.append(t)
                self.error_history.append(o)

            if self._tool_result_callback:
                try:
                    _cb_result = self._tool_result_callback(
                        tool_name=t.function.name,
                        arguments=t.function.arguments,
                        result=o.text,
                        is_error=o.is_error,
                        duration_ms=int(o.extra_info.get("duration_ms", 0) or 0),
                        reasoning=current_reasoning,
                        reason_source=current_reasoning_source,
                    )
                    if asyncio.iscoroutine(_cb_result):
                        await _cb_result
                except Exception:
                    pass

            # Stage 13: remember_lesson → WM 写入
            if t.function.name == "remember_lesson" and not o.is_error and self._memory_orchestrator:
                try:
                    # 解析 arguments 获取 content, tool_name, keywords, category
                    import json as _json
                    _args = t.function.arguments
                    if isinstance(_args, str):
                        _args = _json.loads(_args)
                    _content = _args.get("content", "")
                    _tool_name = _args.get("tool_name", "")
                    _keywords = _args.get("keywords") or []
                    _category = _args.get("category", "tool_error")
                    if _content:
                        self._memory_orchestrator.on_remember_lesson(
                            content=_content,
                            tool_name=_tool_name,
                            keywords=_keywords,
                            category=_category,
                        )
                except Exception as _e:
                    pass  # Non-fatal

        if finish_id is not None:
            expected_path = _normalize_outcome_path(outcome)
            for obs in observations:
                observed_path = _normalize_outcome_path(obs.text)
                if obs.tool_call_id != finish_id:
                    continue
                if obs.text == outcome or (
                    expected_path is not None
                    and observed_path is not None
                    and observed_path == expected_path
                ):
                    info(f"{self.name} Agent finished with result: {obs.text}")
                    return obs.text

        # ── Operation-level memory injection (V2) ──
        # After tools execute, query LTM for relevant chain experiences
        # so the LLM can leverage them in the next decision turn.
        if self._memory_orchestrator and used_tools:
            _SKIP_MEMORY_TOOLS = frozenset({
                "finalize", "thinking", "todo_create", "todo_update", "todo_list",
                "list_files", "list_memory_artifacts", "remember_lesson",
            })
            _query_tools = used_tools - _SKIP_MEMORY_TOOLS
            if _query_tools:
                try:
                    import json as _json

                    _tool_contexts: dict[str, dict] = {}
                    for _tc in tool_calls:
                        _tool_name = _tc.function.name
                        if _tool_name not in _query_tools:
                            continue
                        _obs = _obs_by_tcid.get(_tc.id)
                        _raw_args = getattr(_tc.function, "arguments", None)
                        _parsed_args = None
                        if isinstance(_raw_args, str):
                            try:
                                _parsed_args = _json.loads(_raw_args)
                            except Exception:
                                _parsed_args = {"_raw": _raw_args[:400]}
                        elif isinstance(_raw_args, dict):
                            _parsed_args = _raw_args
                        elif _raw_args is not None:
                            _parsed_args = {"_raw": str(_raw_args)[:400]}

                        _current_target = ""
                        if isinstance(_parsed_args, dict):
                            for _key in (
                                "file_path",
                                "path",
                                "converted_dir",
                                "html_path",
                                "image_path",
                                "output_folder",
                                "directory",
                            ):
                                if _parsed_args.get(_key):
                                    _current_target = str(_parsed_args[_key])[:240]
                                    break

                        _tool_contexts[_tool_name] = {
                            "_parsed_args": _parsed_args,
                            "agent_phase": self.name.lower(),
                            "recent_error": bool(_obs.is_error) if _obs else False,
                            "recent_error_message": (_obs.text[:300] if _obs and _obs.is_error else ""),
                            "recent_observation": (_obs.text[:400] if _obs else ""),
                            "current_target": _current_target,
                            "used_tools": sorted(_query_tools),
                            "retry_count": 1 if _obs and _obs.is_error else 0,
                        }

                    _memory_parts: list[str] = []
                    _seen: set[str] = set()
                    for _tool in _query_tools:
                        _tool_ctx = _tool_contexts.get(
                            _tool,
                            {"agent_phase": self.name.lower(), "used_tools": sorted(_query_tools)},
                        )
                        _mem_text = await self._memory_orchestrator.get_memory_for_operation(
                            tool_name=_tool,
                            tool_args=_tool_ctx.get("_parsed_args"),
                            context=_tool_ctx,
                        )
                        if _mem_text and _mem_text not in _seen:
                            _seen.add(_mem_text)
                            _memory_parts.append(_mem_text)
                    if _memory_parts:
                        _combined = "\n".join(_memory_parts)
                        # Strip profile prompt to avoid duplication (already in SYSTEM)
                        # Only keep tool experience sections
                        _exp_lines = []
                        _in_exp = False
                        for _line in _combined.split("\n"):
                            if _line.startswith("## 工具使用经验") or _line.startswith("## Tool"):
                                _in_exp = True
                            if _in_exp:
                                _exp_lines.append(_line)
                        if _exp_lines:
                            _exp_text = "\n".join(_exp_lines)
                            self.chat_history.append(ChatMessage(
                                role=Role.USER,
                                content=f"<memory_hint>\n{_exp_text}\n</memory_hint>",
                            ))
                            info(f"[MemoryV2] Injected {len(_exp_lines)} lines of tool experience for: {', '.join(_query_tools)}")
                        else:
                            pass
                except Exception as _e:
                    debug(f"Operation-level memory injection failed (non-fatal): {_e}")

        if (
            self.context_warning == 0
            and self.context_length > self.context_window * 0.5
        ):
            self.context_warning += 1
            self._ensure_list_content(observations[0])
            observations[0].content.insert(0, HALF_BUDGET_NOTICE_MSG)
        elif (
            self.context_warning == 1
            and self.context_length > self.context_window * 0.8
        ):
            self._ensure_list_content(observations[0])
            observations[0].content.insert(0, URGENT_BUDGET_NOTICE_MSG)
            self.context_warning = 2

        for obs in observations:
            self.log_message(obs)

        if self.context_length > self.context_window:
            # compact 前先 expire 截图，避免 compact LLM 调用因截图过大而失败
            self._expire_screenshots(current_observations=observations)
            if self.context_warning == -1:
                await self.compact_history()
            else:
                # Fallback: try compact even without context_folding enabled
                try:
                    old_len = self.context_length
                    old_hist_len = len(self.chat_history)
                    await self.compact_history()
                    new_hist_len = len(self.chat_history)
                    if new_hist_len < old_hist_len:
                        # Estimate new context length from compacted history
                        # chars // 4 for message text + fixed overhead for tool definitions
                        _tools_overhead = sum(
                            len(str(t)) for t in self.tools
                        ) // 4 if self.tools else 0
                        est = sum(len(m.text) for m in self.chat_history) // 4 + _tools_overhead
                        self.context_length = est
                        warning(
                            f"{self.name}: emergency compact reduced context "
                            f"{old_len} → ~{est}/{self.context_window} "
                            f"(history {old_hist_len} → {new_hist_len} msgs)"
                        )
                    else:
                        raise RuntimeError(
                            f"{self.name} agent exceeded context window after compact: "
                            f"{self.context_length}/{self.context_window}"
                        )
                except RuntimeError:
                    raise
                except Exception:
                    raise RuntimeError(
                        f"{self.name} agent exceeded context window: "
                        f"{self.context_length}/{self.context_window}"
                    )

        # 滑动窗口截断 - 保留最近 N 轮完整内容，截断更早的 HTML/截图
        self._sliding_window_truncate(keep_recent=3)

        # 截图过期：agent 看过一次后立即替换为文本 caption，释放 context
        self._expire_screenshots(current_observations=observations)

        return observations

    @staticmethod
    def _ensure_list_content(msg: ChatMessage) -> None:
        """确保 msg.content 是 list 格式，以便安全 append/insert。"""
        if isinstance(msg.content, str):
            msg.content = [{"type": "text", "text": msg.content}]

    def log_message(self, msg: ChatMessage):
        if len(msg.text) < MAX_LOGGING_LENGTH:
            debug(f"{self.name}: {msg.text}")
        else:
            debug(f"{self.name}: {msg.text[:MAX_LOGGING_LENGTH]}...")

    def _normalize_image_observations(self, observations: list[ChatMessage]) -> None:
        model_name = self.model.lower()
        promote_to_user = "gemini" in model_name or "qwen" in model_name
        convert_for_claude = "claude" in model_name

        for observation in observations:
            if not observation.has_image:
                continue
            if promote_to_user:
                observation.role = Role.USER
            if convert_for_claude:
                self._convert_openai_image_block(observation)

    @staticmethod
    def _convert_openai_image_block(message: ChatMessage) -> None:
        if not isinstance(message.content, list) or not message.content:
            return
        block = message.content[0]
        if not isinstance(block, dict):
            return
        url = block.get("image_url", {}).get("url", "")
        if not isinstance(url, str) or "," not in url:
            return
        header, payload = url.split(",", 1)
        media_type = header.split(";", 1)[0].removeprefix("data:")
        message.content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": payload,
                },
            }
        ]

    @staticmethod
    def _extract_compact_summary_text(summary_message: ChatMessage) -> str:
        """Extract the actual compact summary text without tool-call JSON noise."""
        summary_text = extract_text_from_content(summary_message.content)
        if summary_text and summary_text.strip():
            return summary_text.strip()

        for tool_call in summary_message.tool_calls or []:
            if tool_call.function.name != "write_markdown_file":
                continue
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
            content = str(args.get("content", "") or "").strip()
            if content:
                return content
        return ""

    def _build_compact_resume_message(
        self,
        summary_text: str,
        *,
        summary_save_failed: bool = False,
    ) -> ChatMessage:
        """Build a single continuation message after history compaction.

        Keeping the full compact tool-call exchange in history can derail the
        RevisionEditor agent into repeatedly writing summary files instead of resuming
        slide edits. We only preserve the distilled state summary and the
        explicit continue instruction.
        """
        parts: list[str] = []
        if summary_text.strip():
            parts.append(f"<state_summary>\n{summary_text.strip()}\n</state_summary>")
        if summary_save_failed:
            parts.append(
                "<NOTICE>Saving the state summary file failed during compaction. "
                "Use the embedded state summary and continue the main work.</NOTICE>"
            )
        parts.append(CONTINUE_MSG["text"])
        parts.append(
            "<INSTRUCTION>State summary is embedded above. "
            "Do NOT write any more summary or state files. "
            "IMMEDIATELY continue your primary work.</INSTRUCTION>"
        )
        if self.research_iter == self.max_context_turns:
            parts.append(LAST_ITER_MSG["text"])

        return ChatMessage(
            id=f"context_resume_{uuid.uuid4().hex[:8]}",
            role=Role.SYSTEM,
            content=[{"type": "text", "text": part} for part in parts],
            extra_info={
                "context_compaction": True,
                "research_iter": self.research_iter,
                "summary_save_failed": summary_save_failed,
            },
        )

    async def compact_history(self, keep_head: int = 10, keep_tail: int = 8):
        """Summarize the history."""
        # ? it's 10 = system + user + (thinking, read, design, write)*2
        if keep_head + keep_tail > len(self.chat_history):
            return

        if self.research_iter == self.max_context_turns:
            return

        self.save_history(message_only=True)
        self.research_iter += 1
        head, tail = self._split_history(keep_head, keep_tail)

        # Stage 8: 截断大内容以减少 context 占用
        head = self._truncate_content_for_compact(head)
        tail = self._truncate_content_for_compact(tail)
        summary_ask = ChatMessage(
            role=Role.USER, content=MEMORY_COMPACT_MSG.format(language=self.language)
        )
        response = await self.llm.run(
            self.chat_history + [summary_ask],
            tools=self.tools,
        )
        agent_message = response.choices[0].message
        if self.keep_reasoning:
            reasoning = self._extract_reasoning_from_openai_message(agent_message)
        else:
            reasoning = None
        summary_message = ChatMessage(
            id=f"context_fold_{uuid.uuid4().hex[:8]}",
            role=agent_message.role,
            content=agent_message.content,
            tool_calls=agent_message.tool_calls,
            reasoning_content=reasoning,
            extra_info={
                "llm_response_meta": self._build_llm_response_meta(response),
            },
        )
        debug(
            f"Summary of Resarch Iter {self.research_iter:02d}: \n"
            + summary_message.text
        )

        # ── 方案 B: 直接注入摘要到 context，不依赖 agent 自己读写文件 ──
        # 1. 仍然执行 LLM 的工具调用（保存文件到磁盘用于调试/审计）
        # 2. 但把 LLM 生成的摘要文本直接嵌入 context
        # 3. 追加明确指令：禁止再写 summary，立即继续主任务
        _compact_tcs = summary_message.tool_calls or []
        tasks = [self.agent_env.tool_execute(tc) for tc in _compact_tcs]
        _compact_results = await asyncio.gather(*tasks, return_exceptions=True)
        summary_save_failed = False
        for _ci, _cr in enumerate(_compact_results):
            if isinstance(_cr, BaseException):
                _ctc = _compact_tcs[_ci] if _ci < len(_compact_tcs) else None
                summary_save_failed = True
                warning(
                    f"{self.name}: compact summary tool "
                    f"'{_ctc.function.name if _ctc else 'unknown'}' failed: {_cr}"
                )
        resume_message = self._build_compact_resume_message(
            self._extract_compact_summary_text(summary_message),
            summary_save_failed=summary_save_failed,
        )
        new_tail = [resume_message]
        # Stage 8 Phase 2: 保留被压缩掉的中间区域中的 session_preference 消息
        middle_start = len(head)
        middle_end = len(self.chat_history) - keep_tail
        preserved_prefs = [
            msg for msg in self.chat_history[middle_start:middle_end]
            if getattr(msg, 'role', None) == Role.SYSTEM
            and "<session_preference" in (getattr(msg, 'content', None) or "")
        ]
        self.chat_history = head + preserved_prefs + tail + new_tail

    def _split_history(self, keep_head, keep_tail):
        # ensure the left context window contains the paired tool call and tool call result
        head = []
        for msg in self.chat_history:
            if len(head) < keep_head or msg.role == Role.TOOL:
                head.append(msg)
            else:
                break
        self._ensure_list_content(head[-1])
        head[-1].content.append(HIST_LOST_MSG)

        tail = self.chat_history[-keep_tail:]
        for i, m in enumerate(tail):
            if m.role == Role.ASSISTANT and m not in head:
                tail = tail[i:]
                break
        else:
            tail = []

        return head, tail

    def _truncate_content_for_compact(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Stage 8: 截断大内容以减少 context 占用

        1. HTML 内容 > 2000 字符 → 截断并添加标记
        2. base64 图片 → 用 caption 替代
        """
        import re
        import copy

        result = []
        for msg in messages:
            # 深拷贝以避免修改原始消息
            new_msg = copy.deepcopy(msg)

            # 处理 content (可能是 str 或 list)
            content = new_msg.content
            if isinstance(content, str):
                content = self._truncate_single_content(content)
                new_msg.content = content
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if isinstance(item, str):
                        new_content.append(self._truncate_single_content(item))
                    elif isinstance(item, dict):
                        # 处理图片类型
                        if item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            if url.startswith("data:image"):
                                # 替换为 caption
                                new_content.append({
                                    "type": "text",
                                    "text": "[IMAGE: slide screenshot - base64 removed for context saving]"
                                })
                            else:
                                new_content.append(item)
                        else:
                            new_content.append(item)
                    else:
                        new_content.append(item)
                new_msg.content = new_content

            result.append(new_msg)
        return result

    def _truncate_single_content(self, text: str, max_len: int = 2000) -> str:
        """Stage 8: 截断单个内容字符串"""
        import re

        # 1. 移除 base64 图片 (data:image/...;base64,...)
        text = re.sub(
            r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
            '[IMAGE: base64 removed]',
            text
        )

        # 2. 截断过长 HTML
        if len(text) > max_len:
            # 检测是否是 HTML
            if '<html' in text.lower() or '<!doctype' in text.lower() or '<div' in text.lower():
                text = text[:max_len] + "\n\n[... HTML truncated for context saving ...]"
            else:
                text = text[:max_len] + "\n\n[... content truncated ...]"

        return text

    def _sliding_window_truncate(self, keep_recent: int = 3):
        """Stage 8: 滑动窗口截断 - 保留最近 N 轮完整内容，截断更早的 HTML/截图

        为什么 keep_recent=3？
        - 覆盖"生成→检查→修复→再检查"的典型迭代周期
        - LLM 需要看到近期工作来做增量修改
        - 更早的轮次只需知道"已完成"，不需要完整内容

        效果：per-turn 稳态消耗从 ~2000-4000 降至 ~600-1000 tokens，
              compact 触发从第 6-8 轮延后到第 20-30 轮
        """
        # 每轮大约 2 条消息（assistant + tool result），保守估计
        min_history_len = keep_recent * 2 + 2  # +2 for system + initial user
        if len(self.chat_history) <= min_history_len:
            return  # 历史太短，不截断

        # 计算截断边界：保留最近 keep_recent*2 条消息
        cutoff_idx = len(self.chat_history) - keep_recent * 2

        # 标记已截断的消息，避免重复处理
        for i, msg in enumerate(self.chat_history[:cutoff_idx]):
            if getattr(msg, '_sliding_truncated', False):
                continue  # 已处理过

            if msg.role == Role.TOOL:
                msg.content = self._truncate_old_tool_content(msg.content)
                msg._sliding_truncated = True
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "write_html_file":
                        tc.function.arguments = self._truncate_html_args(tc.function.arguments)
                msg._sliding_truncated = True

    def _expire_screenshots(self, current_observations: list[ChatMessage]):
        """截图过期：保留最近 1 张截图，替换更早的截图为文本 caption。

        保留策略：当前轮次的 observations + history 中最近的 1 张截图保留，
        其余全部替换为文本占位符。这样 LLM 始终能看到最新截图用于对比参考。

        效果：稳态最多 2 张截图（当前轮 + 上一轮），~8K tokens。
        """
        current_ids = {id(obs) for obs in current_observations}

        # 从后往前扫描，找到最近一张已有截图（非当前轮）并保留
        keep_one_id = None
        for msg in reversed(self.chat_history):
            if id(msg) in current_ids:
                continue
            if msg.role != Role.TOOL or not isinstance(msg.content, list):
                continue
            if any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in msg.content
            ):
                keep_one_id = id(msg)
                break

        for msg in self.chat_history:
            if id(msg) in current_ids or id(msg) == keep_one_id:
                continue  # 当前轮次 + 最近 1 张保留
            if msg.role != Role.TOOL:
                continue
            if not isinstance(msg.content, list):
                continue

            new_content = []
            replaced = False
            for item in msg.content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    new_content.append({
                        "type": "text",
                        "text": "[📸 Slide screenshot already reviewed - refer to text summary for details]",
                    })
                    replaced = True
                else:
                    new_content.append(item)
            if replaced:
                msg.content = new_content

    def _truncate_old_tool_content(self, content: str | list) -> str | list:
        """Stage 8: 截断旧 tool result 中的 HTML 和截图（比 compact 更激进）"""
        import re

        OLD_CONTENT_MAX_LEN = 500  # 比 compact 的 2000 更激进

        if isinstance(content, str):
            # 移除 base64 图片
            content = re.sub(
                r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                '[📸 旧截图已压缩]',
                content
            )
            # 截断长文本
            if len(content) > OLD_CONTENT_MAX_LEN:
                if '<html' in content.lower() or '<!doctype' in content.lower() or '<div' in content.lower():
                    content = content[:OLD_CONTENT_MAX_LEN] + "\n[... 旧 HTML 已压缩 ...]"
                else:
                    content = content[:OLD_CONTENT_MAX_LEN] + "\n[... 旧内容已压缩 ...]"
            return content

        elif isinstance(content, list):
            result = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "image_url":
                        # 截图 → 占位符
                        result.append({"type": "text", "text": "[📸 旧截图已压缩]"})
                    elif item.get("type") == "text":
                        text = item.get("text", "")
                        if len(text) > OLD_CONTENT_MAX_LEN:
                            text = text[:OLD_CONTENT_MAX_LEN] + "\n[... 旧内容已压缩 ...]"
                        result.append({"type": "text", "text": text})
                    else:
                        result.append(item)
                elif isinstance(item, str):
                    if len(item) > OLD_CONTENT_MAX_LEN:
                        item = item[:OLD_CONTENT_MAX_LEN] + "\n[... 旧内容已压缩 ...]"
                    result.append(item)
                else:
                    result.append(item)
            return result

        return content

    def _truncate_html_args(self, args_json: str) -> str:
        """Stage 8: 截断 write_html_file 参数中的 HTML 内容"""
        HTML_ARG_MAX_LEN = 200

        try:
            args = json.loads(args_json)
            if "content" in args and isinstance(args["content"], str):
                if len(args["content"]) > HTML_ARG_MAX_LEN:
                    # 保留开头，截断中间
                    args["content"] = args["content"][:HTML_ARG_MAX_LEN] + "... [旧 HTML 已压缩]"
            return json.dumps(args, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return args_json

    def reset_for_new_round(self, round_summary_prompt: str = "") -> None:
        """重置 chat_history，保留 SYSTEM prompt，注入历史 Round 摘要。

        每个 Round 开始前由 MemoryRuntime/MemoryOrchestrator 调用。
        """
        # Rebuild the base SYSTEM prompt from self.system when available.
        # chat_history[0] may already contain turn-specific memory injection.
        base_system = getattr(self, "system", "") or self.chat_history[0].text
        system_msg = ChatMessage(role=Role.SYSTEM, content=base_system)
        self.chat_history = [system_msg]

        if round_summary_prompt:
            self.chat_history.append(ChatMessage(
                role=Role.SYSTEM,
                content=round_summary_prompt,
            ))

        self._action_count = 0
        info(f"{self.name} chat_history reset for new round "
             f"(summary={len(round_summary_prompt)} chars)")

    def reset_for_new_task(self, task_summary_prompt: str = "") -> None:
        """Deprecated compatibility wrapper for ``reset_for_new_round``."""
        self.reset_for_new_round(task_summary_prompt)

    def _history_file_for_iter(self, hist_dir: Path) -> Path:
        if self.research_iter < 0:
            return hist_dir / f"{self.name}-history.jsonl"
        return hist_dir / f"{self.name}-{self.research_iter:02d}-history.jsonl"

    @staticmethod
    def _write_jsonl(path: Path, records: list[Any]) -> None:
        with jsonlines.open(path, mode="w") as writer:
            for record in records:
                writer.write(record.model_dump() if hasattr(record, "model_dump") else record)

    def _history_config_payload(self) -> dict[str, Any]:
        return {
            "agent": self.name,
            "model": self.model,
            "prompt_tokens": self.context_length,
            "cost": self.cost.model_dump(),
            "tools": self.tools,
        }

    def save_history(self, hist_dir: Path | None = None, message_only: bool = False):
        target_dir = hist_dir or self.workspace / ".history"
        target_dir.mkdir(parents=True, exist_ok=True)

        history_file = self._history_file_for_iter(target_dir)
        self._write_jsonl(history_file, self.chat_history)
        if message_only:
            return

        config_file = target_dir / f"{self.name}-config.json"
        config_file.write_text(
            json.dumps(self._history_config_payload(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.error_history:
            self._write_jsonl(target_dir / f"{self.name}-errors.jsonl", self.error_history)

        info(
            "%s done | cost:%s ctx:%s | history:%s config:%s",
            self.name,
            self.cost,
            self.context_length,
            history_file.name,
            config_file.name,
        )
