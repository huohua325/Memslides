"""Utilities for capturing observable tool-use reasons.

This module intentionally works with *observable* signals instead of hidden
reasoning tokens. The goal is to provide a stable "why this tool was used"
field for chain memory, regardless of model/API/provider differences.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_REASON_PREFIX_RE = re.compile(
    r"^\s*(reason|tool reason|rationale|why)[:：]\s*",
    re.IGNORECASE,
)
_ZH_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")
_MULTISPACE_RE = re.compile(r"\s+")

_TOOL_TARGET_KEYS = (
    "target_file",
    "file_path",
    "path",
    "html_file",
    "md_file",
    "directory",
    "converted_dir",
    "output_folder",
    "outcome",
)


def _infer_language(*texts: str) -> str:
    for text in texts:
        if text and _ZH_CHAR_RE.search(text):
            return "zh"
    return "en"


def extract_text_from_content(content: Any) -> str:
    """Extract plain text from provider-specific content shapes."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_type = str(part.get("type", ""))
                if part_type in {"text", "input_text", "output_text"}:
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                else:
                    text = part.get("content")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            elif isinstance(part, str) and part.strip():
                texts.append(part.strip())
        return "\n".join(texts).strip()

    return str(content).strip()


def normalize_reason_text(text: str | None) -> str:
    """Normalize observable rationale text and reject tool-call payload noise."""
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    text = _REASON_PREFIX_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    # Defensive guard: some fallbacks may accidentally capture serialized
    # tool-call payloads instead of natural-language rationale.
    if (
        text.startswith(("{", "["))
        and '"arguments"' in text
        and any(token in text for token in ('"function"', '"tool_calls"', '"name"'))
    ):
        return ""

    return text


def parse_tool_arguments(arguments: str | dict | None) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_tool_target(arguments: str | dict | None) -> str:
    args = parse_tool_arguments(arguments)
    for key in _TOOL_TARGET_KEYS:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            # Prefer compact file/dir names in synthesized rationale.
            if "/" in text or "\\" in text:
                return Path(text).name or text
            return text
    return ""


def synthesize_tool_reason(
    tool_name: str,
    arguments: str | dict | None = None,
    *,
    user_message: str = "",
    observation: str = "",
    language: str = "auto",
) -> str:
    """Synthesize a concise observable tool-use reason.

    This is a deterministic fallback for chain memory when the model/API does
    not expose a natural-language rationale alongside tool calls.
    """
    lang = _infer_language(user_message, observation) if language == "auto" else language
    target = extract_tool_target(arguments)
    tool = (tool_name or "").strip()
    lower_name = tool.lower()

    if lang == "zh":
        target_desc = target or "当前对象"
        if lower_name in {"read_file", "read_html_file"}:
            return f"需要先读取{target_desc}，确认当前内容和结构，再决定下一步操作。"
        if lower_name in {"write_file", "write_html_file", "write_markdown_file", "write_html"}:
            return f"需要把整理好的修改写入{target_desc}，让当前改动真正生效。"
        if lower_name in {"inspect_slide", "inspect_manuscript"}:
            return f"需要检查{target_desc}的渲染和校验结果，确认前一步修改是否正确生效。"
        if lower_name in {"list_files", "list_document_figures", "explore_workspace_images", "query_slide_layout"}:
            return f"需要先枚举{target_desc or '相关素材和结构'}，为后续选择内容或布局提供依据。"
        if lower_name == "convert_to_markdown":
            return "需要先把附件转换成 Markdown 和图片资源，便于后续抽取内容与版面素材。"
        if lower_name in {"image_caption", "search_image", "image_generation"}:
            return "需要先补充图像信息，帮助后续完成内容判断或视觉选择。"
        if lower_name == "remember_lesson":
            return "需要把这次可复用的经验写入记忆，便于后续任务直接复用。"
        if lower_name == "todo_create":
            return "需要先记录当前子任务，明确接下来的执行步骤。"
        if lower_name == "finalize":
            return "核心操作已完成，需要提交最终结果并结束当前任务。"
        return f"需要调用 {tool or '该工具'} 获取推进当前子任务所需的信息或执行结果。"

    target_desc = target or "the current target"
    if lower_name in {"read_file", "read_html_file"}:
        return f"I need to read {target_desc} first to understand the current content and structure before making changes."
    if lower_name in {"write_file", "write_html_file", "write_markdown_file", "write_html"}:
        return f"I need to write the prepared changes to {target_desc} so the update actually takes effect."
    if lower_name in {"inspect_slide", "inspect_manuscript"}:
        return f"I need to inspect {target_desc} to verify that the previous change rendered correctly."
    if lower_name in {"list_files", "list_document_figures", "explore_workspace_images", "query_slide_layout"}:
        return f"I need to enumerate the relevant assets or structure first to choose the next content or layout step."
    if lower_name == "convert_to_markdown":
        return "I need to convert the attachment into Markdown and extracted assets first so the content becomes easier to analyze."
    if lower_name in {"image_caption", "search_image", "image_generation"}:
        return "I need more image information to support the next content or visual decision."
    if lower_name == "remember_lesson":
        return "I need to store this reusable lesson in memory so later tasks can reuse it directly."
    if lower_name == "todo_create":
        return "I need to record the current subtask so the next execution steps stay explicit."
    if lower_name == "finalize":
        return "The core work is complete, so I need to submit the final result and end the task."
    return f"I need to call {tool or 'this tool'} to obtain the information or side effect required for the current subtask."


def normalize_tool_reason(
    *,
    tool_name: str,
    arguments: str | dict | None = None,
    raw_reason: str | None = None,
    user_message: str = "",
    observation: str = "",
    reason_source: str = "",
    language: str = "auto",
) -> tuple[str, str, str]:
    """Return ``(reason_text, source, quality)`` with deterministic fallback."""
    normalized = normalize_reason_text(raw_reason)
    if normalized:
        source = reason_source or "assistant_content"
        if source in {"reasoning_content", "reasoning_summary", "provider_extra_reasoning"}:
            quality = "high"
        elif source in {"assistant_content", "assistant_content_blocks"}:
            quality = "medium"
        else:
            quality = "low"
        return normalized, source, quality

    synthesized = synthesize_tool_reason(
        tool_name,
        arguments,
        user_message=user_message,
        observation=observation,
        language=language,
    )
    if synthesized:
        return synthesized, "synthesized_from_tool_context", "low"
    return "", "missing", "none"
