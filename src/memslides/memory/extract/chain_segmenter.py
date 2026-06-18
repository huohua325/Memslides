"""ChainSegmenter — 工具链分割器（Design Doc §6.1.1）

Task 结束时将 cycle 序列通过 LLM 划分为语义工具链。
使用完整的 reasoning + tool_call + observation 轨迹作为输入。
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from ..core.models import ChainSegment, ToolChain, make_chain_signature
from ..prompts.chain_prompts import CHAIN_SEGMENTATION_PROMPT
from .llm_compat import call_llm_with_prompt

logger = logging.getLogger(__name__)

# ── 智能截断工具 ──

# 大文本参数的工具名集合（这些工具的 content/html_content 字段需要摘要化）
_LARGE_CONTENT_TOOLS = {"write_html_file", "write_markdown_file", "write_file"}
# 大文本参数的字段名
_LARGE_CONTENT_FIELDS = {"content", "html_content", "markdown_content"}

# observation 截断上限（按工具类型区分）
_OBS_LIMITS = {
    "read_file": 800,
    "read_html_file": 800,
    "convert_to_markdown": 500,
    "list_document_figures": 500,
    "explore_workspace_images": 500,
    "inspect_slide": 600,
    "image_caption": 300,
    "query_slide_layout": 400,
    "_default": 1000,
}


def _smart_truncate_args(tool_name: str, args: str | dict, max_len: int = 2000) -> str:
    """智能截断工具参数。

    - 大文本类工具（write_html_file 等）：content 字段只保留前 200 字符 + 长度信息
    - 其他工具：保留完整参数，超长时截断
    """
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    else:
        args_str = str(args)

    if tool_name in _LARGE_CONTENT_TOOLS:
        try:
            parsed = json.loads(args_str) if isinstance(args_str, str) else args
            if isinstance(parsed, dict):
                for field in _LARGE_CONTENT_FIELDS:
                    if field in parsed and isinstance(parsed[field], str) and len(parsed[field]) > 300:
                        original_len = len(parsed[field])
                        parsed[field] = parsed[field][:200] + f"... [截断，原始长度 {original_len} 字符]"
                return json.dumps(parsed, ensure_ascii=False)[:max_len]
        except (json.JSONDecodeError, TypeError):
            pass

    if len(args_str) > max_len:
        return args_str[:max_len] + f"... [截断，原始长度 {len(args_str)} 字符]"
    return args_str


def _smart_truncate_observation(tool_name: str, result: str) -> str:
    """智能截断 observation。

    - 按工具类型使用不同的截断上限
    - 保留开头（通常包含关键状态信息）和结尾（通常包含摘要）
    """
    limit = _OBS_LIMITS.get(tool_name, _OBS_LIMITS["_default"])
    if len(result) <= limit:
        return result

    # 保留开头 70% + 结尾 30%
    head_len = int(limit * 0.7)
    tail_len = limit - head_len
    omitted = len(result) - head_len - tail_len
    return (
        result[:head_len]
        + f"\n... [省略 {omitted} 字符] ...\n"
        + result[-tail_len:]
    )


def build_rich_trace(
    tool_name: str,
    args: str | dict,
    result: str,
    is_error: bool = False,
    reasoning: str = "",
    tool_reason: str | None = None,
    reason_source: str = "",
    reason_quality: str = "",
) -> dict:
    """构建单个工具调用的 rich trace。

    - reasoning/tool_reason: 记录可观测的工具使用理由（显式理由优先，必要时使用合成兜底）
    - arguments: 大文本类做摘要
    - observation: 智能截断
    """
    final_reason = tool_reason if tool_reason is not None else reasoning
    return {
        "reasoning": final_reason,
        "tool_reason": final_reason,
        "reason_source": reason_source,
        "reason_quality": reason_quality,
        "tool_name": tool_name,
        "arguments": _smart_truncate_args(tool_name, args),
        "observation": _smart_truncate_observation(tool_name, result),
        "is_error": is_error,
    }


def format_rich_traces_for_llm(traces: list[dict]) -> str:
    """将 rich traces 格式化为 LLM 可读的文本。"""
    parts = []
    for i, t in enumerate(traces):
        lines = [f"[Cycle {i}] {t['tool_name']}"]
        tool_reason = t.get("tool_reason") or t.get("reasoning")
        if tool_reason:
            label = "Reasoning"
            reason_source = str(t.get("reason_source", "") or "")
            if reason_source and reason_source not in {"reasoning_content", "assistant_content"}:
                label += f" ({reason_source})"
            lines.append(f"  {label}: {tool_reason}")
        lines.append(f"  Arguments: {t['arguments']}")
        obs_preview = t.get("observation", "")
        if obs_preview:
            lines.append(f"  Observation: {obs_preview}")
        if t.get("is_error"):
            lines.append("  ⚠ ERROR")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


class ChainSegmenter:
    """Task 结束时将 cycle 序列划分为工具链。

    使用完整的 rich traces（reasoning + tool_call + observation）作为 LLM 输入，
    替代旧版的压缩摘要，保留更多语义信息用于后续经验提取。
    """

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def segment(
        self,
        compact_round: Any,
        round_id: str = "",
        rich_traces: list[dict] | None = None,
    ) -> list[ChainSegment]:
        """将工具调用序列划分为 ChainSegment 列表。

        Args:
            compact_round: AgentRound 实例（提供 segments 和 user_message）
            round_id: 当前 Round ID
            rich_traces: 完整执行轨迹列表。如果提供，优先使用；否则降级到压缩摘要。
        """
        if not compact_round or not self._llm:
            return []

        segments = getattr(compact_round, "segments", [])
        if not segments:
            return []

        # 构建 LLM 输入：优先使用 rich_traces
        if rich_traces:
            cycle_text = format_rich_traces_for_llm(rich_traces)
        else:
            # 降级：使用旧版压缩摘要
            cycle_lines = []
            for i, seg in enumerate(segments):
                name = getattr(seg, "tool_name", "unknown")
                summary = getattr(seg, "action_summary", "")[:100]
                is_error = getattr(seg, "is_error", False)
                err_tag = " [ERROR]" if is_error else ""
                err_detail = f" {getattr(seg, 'error_msg', '')[:80]}" if is_error else ""
                cycle_lines.append(f"[{i}] {name}: {summary}{err_tag}{err_detail}")
            cycle_text = "\n".join(cycle_lines)

        user_task = getattr(compact_round, "user_message", "")[:300]
        prompt = CHAIN_SEGMENTATION_PROMPT.format(
            user_task=user_task,
            cycle_count=len(rich_traces) if rich_traces else len(segments),
            cycle_summaries=cycle_text,
        )

        try:
            response_text = await call_llm_with_prompt(self._llm, prompt)
            chains_data = self._parse_response(response_text)
        except Exception as e:
            logger.warning(f"Chain segmentation LLM call failed: {e}")
            return [self._fallback_chain(segments, round_id, rich_traces)]

        chains: list[ChainSegment] = []
        for cd in chains_data:
            indices = cd.get("cycle_indices", [])
            semantic_label = cd.get("semantic_label", cd.get("chain_name", ""))
            outcome = cd.get("outcome", "success")
            tool_seq = [
                getattr(segments[i], "tool_name", "")
                for i in indices
                if i < len(segments)
            ]
            chain_name = make_chain_signature(tool_seq)

            # 提取该链对应的 rich_traces 子集
            chain_rich_traces = []
            if rich_traces:
                for idx in indices:
                    if idx < len(rich_traces):
                        chain_rich_traces.append(rich_traces[idx])

            chains.append(ChainSegment(
                chain_id=str(uuid4()),
                chain_name=chain_name,
                tool_sequence=tool_seq,
                cycle_indices=indices,
                rich_traces=chain_rich_traces,
                semantic_label=semantic_label,
                outcome=outcome,
                source_round_id=round_id,
            ))
        return chains

    def _parse_response(self, text: str) -> list[dict]:
        """解析 LLM JSON 响应。"""
        text = text.strip()
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return []

    def _fallback_chain(
        self,
        segments: list,
        round_id: str,
        rich_traces: list[dict] | None = None,
    ) -> ChainSegment:
        """降级：整个 Round 作为一条链。"""
        tool_seq = [getattr(s, "tool_name", "") for s in segments]
        return ChainSegment(
            chain_id=str(uuid4()),
            chain_name=make_chain_signature(tool_seq),
            tool_sequence=tool_seq,
            cycle_indices=list(range(len(segments))),
            rich_traces=rich_traces or [],
            semantic_label="完整任务",
            outcome="success",
            source_round_id=round_id,
        )
