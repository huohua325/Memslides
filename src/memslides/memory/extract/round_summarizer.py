"""RoundSummarizer — Round 结束时生成结构化摘要

Design Doc §8.3: 规则模式 + 可选 LLM 精炼。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..core.models import RoundSummary
from .llm_compat import call_llm_with_prompt

logger = logging.getLogger(__name__)

LLM_SUMMARY_THRESHOLD = 20
_SLIDE_TOKEN_RE = re.compile(r"(slide_\d+)", re.IGNORECASE)
_WRITE_TOOLS = {"write_html_file"}
_PATCH_TOOLS = {"apply_slide_patch", "patch_semantic_inline_style"}
_BATCH_STYLE_TOOLS = {"batch_update_css_rule", "batch_update_semantic_style"}
_STRUCTURE_TOOLS = {"insert_slide", "delete_slide"}
_MODIFY_TOOLS = _WRITE_TOOLS | _PATCH_TOOLS | _BATCH_STYLE_TOOLS | _STRUCTURE_TOOLS


def _call_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or call.get("tool_name") or "")


def _call_args(call: dict[str, Any]) -> Any:
    if "args" in call:
        return call.get("args")
    return call.get("arguments", "")


def _call_result_text(call: dict[str, Any]) -> str:
    if "result_preview" in call and call.get("result_preview") is not None:
        return str(call.get("result_preview") or "")
    return str(call.get("result") or "")


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _extract_slide_tokens(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        slides: set[str] = set()
        for item in value:
            slides.update(_extract_slide_tokens(item))
        return slides
    return {match.group(1) for match in _SLIDE_TOKEN_RE.finditer(str(value))}


def _extract_inserted_slide_tokens(result_text: str) -> set[str]:
    path_match = re.search(r"New slide path:\s*([^\s]+)", result_text, re.IGNORECASE)
    if path_match:
        return _extract_slide_tokens(path_match.group(1))
    inserted_match = re.search(r"Inserted:\s*(slide_\d+)", result_text, re.IGNORECASE)
    if inserted_match:
        return {inserted_match.group(1)}
    return _extract_slide_tokens(result_text)


def _looks_like_soft_error(call: dict[str, Any]) -> bool:
    payload = _parse_json_object(call.get("result"))
    if payload is None:
        payload = _parse_json_object(call.get("result_preview"))
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return True
        if payload.get("error_code"):
            return True
        if payload.get("error") and payload.get("success") is not True:
            return True

    result_text = _call_result_text(call).strip()
    if not result_text:
        return False
    return result_text.lower().startswith("error:") or result_text.startswith("❌")


def _call_is_error(call: dict[str, Any]) -> bool:
    return bool(call.get("is_error")) or _looks_like_soft_error(call)


class RoundSummarizer:
    """Round 摘要生成器。

    两种模式：
    - 规则提取（快速）：从工具调用日志中提取 slides/actions/outcome
    - LLM 精炼（工具调用 > 20 次时）：调用 LLM 生成精炼摘要
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def summarize(
        self,
        round_index: int,
        user_message: str,
        tool_calls_log: list[dict],
        agent_response: str,
        round_id: str = "",
    ) -> RoundSummary:
        """生成 RoundSummary。"""
        slides_modified = self._extract_modified_slides(tool_calls_log)
        outcome = self._determine_outcome(tool_calls_log)

        if len(tool_calls_log) > LLM_SUMMARY_THRESHOLD and self._llm:
            key_actions, unresolved = await self._llm_summarize(
                user_message, tool_calls_log, agent_response)
        else:
            key_actions = self._extract_key_actions(tool_calls_log)
            unresolved = self._extract_unresolved(tool_calls_log)

        return RoundSummary(
            round_id=round_id,
            round_index=round_index,
            user_request=user_message,
            outcome=outcome,
            slides_modified=slides_modified,
            key_actions=key_actions[:5],
            unresolved_issues=unresolved,
        )

    def _extract_modified_slides(self, tool_calls_log: list[dict]) -> list[str]:
        """从实际修改类工具中提取 slide_N。"""
        slides = set()
        for call in tool_calls_log:
            name = _call_name(call)
            args = _call_args(call)
            result_text = _call_result_text(call)
            result_payload = _parse_json_object(call.get("result"))
            if result_payload is None:
                result_payload = _parse_json_object(result_text)

            if name == "write_html_file":
                slides.update(_extract_slide_tokens(args))
                continue

            if _call_is_error(call):
                continue

            if name == "apply_slide_patch":
                if isinstance(result_payload, dict) and result_payload.get("changed") is False:
                    continue
                slides.update(_extract_slide_tokens(
                    (result_payload or {}).get("slide_path") if isinstance(result_payload, dict) else result_text
                ))
                continue

            if name == "patch_semantic_inline_style":
                if isinstance(result_payload, dict):
                    if result_payload.get("success") is False or result_payload.get("changed") is False:
                        continue
                    slides.update(_extract_slide_tokens(result_payload.get("file_path")))
                else:
                    slides.update(_extract_slide_tokens(args))
                continue

            if name in _BATCH_STYLE_TOOLS:
                if isinstance(result_payload, dict):
                    batch_slides = set()
                    for entry in result_payload.get("results", []) or []:
                        status = str(entry.get("status", "") or "").strip().lower()
                        if status in {"", "error", "already_compliant"}:
                            continue
                        batch_slides.update(_extract_slide_tokens(entry.get("file_path")))
                    if batch_slides:
                        slides.update(batch_slides)
                        continue
                slides.update(_extract_slide_tokens(args))
                continue

            if name == "insert_slide":
                slides.update(_extract_inserted_slide_tokens(result_text))
                continue

            if name == "delete_slide":
                slides.update(_extract_slide_tokens(args))
        return sorted(slides)

    def _extract_key_actions(self, tool_calls_log: list[dict]) -> list[str]:
        """统计工具调用次数 + 检测 retry 模式。"""
        actions = []
        write_count = sum(1 for c in tool_calls_log if _call_name(c) == "write_html_file")
        patch_count = sum(1 for c in tool_calls_log if _call_name(c) == "apply_slide_patch")
        semantic_patch_count = sum(1 for c in tool_calls_log if _call_name(c) == "patch_semantic_inline_style")
        batch_style_count = sum(1 for c in tool_calls_log if _call_name(c) in _BATCH_STYLE_TOOLS)
        structure_change_count = sum(1 for c in tool_calls_log if _call_name(c) in _STRUCTURE_TOOLS)
        inspect_count = sum(1 for c in tool_calls_log if _call_name(c) == "inspect_slide")
        img_gen_count = sum(1 for c in tool_calls_log if _call_name(c) == "image_generation")
        if write_count:
            actions.append(f"写入了 {write_count} 次幻灯片文件")
        if patch_count:
            actions.append(f"执行了 {patch_count} 次结构化幻灯片补丁")
        if semantic_patch_count:
            actions.append(f"执行了 {semantic_patch_count} 次单页语义样式修补")
        if batch_style_count:
            actions.append(f"执行了 {batch_style_count} 次批量样式更新")
        if structure_change_count:
            actions.append(f"执行了 {structure_change_count} 次页面插入/删除")
        if inspect_count:
            actions.append(f"检查了 {inspect_count} 次幻灯片效果")
        if img_gen_count:
            actions.append(f"生成了 {img_gen_count} 张图片")
        # write→inspect retry 模式
        retry_count = sum(
            1 for i in range(1, len(tool_calls_log))
            if _call_name(tool_calls_log[i]) in _MODIFY_TOOLS
            and _call_name(tool_calls_log[i - 1]) == "inspect_slide"
        )
        if retry_count:
            actions.append(f"经历了 {retry_count} 次检查-修复迭代")
        return actions

    def _determine_outcome(self, tool_calls_log: list[dict]) -> str:
        """基于 is_error 和 finalize 判断 outcome。"""
        has_error = any(_call_is_error(c) for c in tool_calls_log)
        has_finalize = any(_call_name(c) == "finalize" for c in tool_calls_log)
        if has_error and not has_finalize:
            return "partial"
        return "success"

    def _extract_unresolved(self, tool_calls_log: list[dict]) -> list[str]:
        """检查最后一次 inspect_slide 结果中的溢出关键词。"""
        issues = []
        for call in reversed(tool_calls_log):
            if _call_name(call) == "inspect_slide":
                result = _call_result_text(call)
                result_lower = result.lower()
                if "render failed" in result_lower or "pptx_export failed" in result_lower:
                    issues.append("最后一次检查渲染失败")
                elif _call_is_error(call):
                    issues.append("最后一次检查失败")
                if "溢出" in result or "overflow" in result_lower:
                    issues.append("最后一次检查仍有文字溢出")
                if (
                    "nested `.slide` root" in result_lower
                    or "嵌套 `.slide`" in result
                    or "嵌套 .slide" in result
                ):
                    issues.append("最后一次检查仍有嵌套 .slide 根节点")
                if (
                    "invalid list structure" in result_lower
                    or "非 `<li>` 元素" in result
                    or "non-`<li>`" in result_lower
                ):
                    issues.append("最后一次检查仍有非法列表结构")
                if (
                    "suspicious slide geometry" in result_lower
                    or "误缩放" in result
                    or "画布尺寸" in result
                ):
                    issues.append("最后一次检查仍有可疑画布尺寸/缩放异常")
                if "structural warnings" in result_lower and not issues:
                    issues.append("最后一次检查仍有结构告警")
                break
        return issues

    async def _llm_summarize(
        self, user_message: str, tool_calls_log: list[dict], agent_response: str,
    ) -> tuple[list[str], list[str]]:
        """LLM 精炼摘要（工具调用 > LLM_SUMMARY_THRESHOLD 时）。"""
        from ..prompts.chain_prompts import TASK_SUMMARY_PROMPT

        # 压缩 tool_calls_log：每条只保留名称+是否出错+结果前100字符
        compressed = []
        for c in tool_calls_log:
            err_tag = " [ERROR]" if _call_is_error(c) else ""
            preview = _call_result_text(c)[:100]
            compressed.append(f"{_call_name(c) or '?'}{err_tag}: {preview}")

        prompt = TASK_SUMMARY_PROMPT.format(
            user_message=user_message[:300],
            tool_count=len(tool_calls_log),
            tool_calls_summary="\n".join(compressed),
            agent_response=agent_response[:500],
        )
        try:
            response_text = await call_llm_with_prompt(self._llm, prompt)
            data = json.loads(response_text)
            return (
                data.get("key_actions", [])[:5],
                data.get("unresolved_issues", []),
            )
        except Exception as e:
            logger.warning(f"LLM summarize failed, falling back to rules: {e}")
            return self._extract_key_actions(tool_calls_log), self._extract_unresolved(tool_calls_log)
