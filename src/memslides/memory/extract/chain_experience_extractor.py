"""ChainExperienceExtractor — 链级经验提取（重构版）

核心变更：
- distill_single: 独立蒸馏单个 ChainSegment → ChainExperience（含 keywords）
- check_equivalence: LLM 等价判定两个 ChainExperience
- distill_and_merge: 独立蒸馏 + 等价判定 + 合并/追加

旧 extract 方法保留作为向后兼容。
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from ..core.models import ChainExperience, ChainSegment, ToolChain
from .llm_compat import call_llm_with_prompt, extract_response_text
from ..prompts.chain_prompts import (
    CHAIN_EXPERIENCE_PROMPT,
    CHAIN_INDEPENDENT_DISTILL_PROMPT,
    CHAIN_EQUIVALENCE_CHECK_PROMPT,
)

logger = logging.getLogger(__name__)


class ChainExperienceExtractor:
    """从工具链片段中提炼可复用经验。"""

    def __init__(self, llm: Any = None):
        self._llm = llm

    async def _call_llm(self, prompt: str) -> str:
        """兼容 callable LLM 与 `.run()` 风格的客户端。"""
        return await call_llm_with_prompt(self._llm, prompt)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """从不同 LLM 返回格式中提取文本。"""
        return extract_response_text(response)

    # ── 新接口 ──

    async def distill_single(
        self, chain_name: str, segment: ChainSegment,
    ) -> ChainExperience | None:
        """独立蒸馏单个 ChainSegment → ChainExperience（含 keywords）。

        使用 segment.rich_traces（完整的 reasoning + tool_call + observation）
        构建执行记录。

        Args:
            chain_name: 链签名
            segment: 单个 ChainSegment

        Returns:
            成功时返回一个 ChainExperience，LLM 失败时返回 None。
        """
        if not self._llm:
            return None

        # 使用 rich_traces 构建执行记录
        if segment.rich_traces:
            from .chain_segmenter import format_rich_traces_for_llm
            execution_text = format_rich_traces_for_llm(segment.rich_traces)
        else:
            execution_text = "(无)"

        prompt = CHAIN_INDEPENDENT_DISTILL_PROMPT.format(
            chain_name=chain_name,
            tool_sequence=" → ".join(segment.tool_sequence),
            execution_record=execution_text,
            outcome=segment.outcome,
        )

        try:
            response_text = await self._call_llm(prompt)
            return self._parse_distill_response(response_text, chain_name, segment.chain_id)
        except Exception as e:
            logger.warning(f"distill_single failed for '{chain_name}': {e}")
            return None

    async def check_equivalence(
        self, new_exp: ChainExperience, existing_exp: ChainExperience,
    ) -> tuple[bool, str]:
        """LLM 等价判定，返回 (is_equivalent, merged_lesson)。

        LLM 调用失败时降级为 (False, "")（视为不等价）。
        """
        if not self._llm:
            return False, ""

        prompt = CHAIN_EQUIVALENCE_CHECK_PROMPT.format(
            new_lesson=new_exp.lesson,
            new_applicable_when=new_exp.applicable_when,
            new_tool_pipeline=" → ".join(new_exp.tool_pipeline),
            existing_lesson=existing_exp.lesson,
            existing_applicable_when=existing_exp.applicable_when,
            existing_tool_pipeline=" → ".join(existing_exp.tool_pipeline),
        )

        try:
            response_text = await self._call_llm(prompt)
            return self._parse_equivalence_response(response_text)
        except Exception as e:
            logger.warning(f"check_equivalence failed: {e}")
            return False, ""

    async def distill_and_merge(
        self, chain_name: str, segment: ChainSegment,
        existing_experiences: list[ChainExperience],
    ) -> list[ChainExperience]:
        """独立蒸馏 + 等价判定 + 合并/追加，返回需要写入 LTM 的经验列表。

        流程：
        1. 调用 distill_single 蒸馏新经验
        2. 遍历 existing_experiences，对每个调用 check_equivalence
        3. 找到等价条目 → 合并（保留已有 subkey，更新 lesson/keywords/metadata）
        4. 无等价条目 → 生成新 subkey，追加
        """
        new_exp = await self.distill_single(chain_name, segment)
        if new_exp is None:
            return []

        # 无已有条目 → 直接追加
        if not existing_experiences:
            new_exp.subkey = uuid4().hex[:8]
            return [new_exp]

        # 遍历已有条目做等价判定
        for existing in existing_experiences:
            is_equiv, merged_lesson = await self.check_equivalence(new_exp, existing)
            if is_equiv:
                # 合并：保留已有 subkey，更新内容
                existing.lesson = merged_lesson or new_exp.lesson
                existing.keywords = new_exp.keywords or existing.keywords
                existing.source_chain_ids = list(
                    set(existing.source_chain_ids + new_exp.source_chain_ids)
                )
                existing.confidence = min(1.0, existing.confidence + 0.1)
                return [existing]

        # 不等价 → 生成新 subkey 追加
        new_exp.subkey = uuid4().hex[:8]
        return [new_exp]

    # ── 解析辅助 ──

    def _parse_distill_response(
        self, text: str, chain_name: str, source_chain_id: str,
    ) -> ChainExperience | None:
        """解析独立蒸馏的 LLM JSON 响应。"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        lesson = data.get("lesson", "")
        if not lesson:
            return None

        keywords = data.get("keywords", [])
        # 确保 keywords 在 [3, 8] 范围内
        if len(keywords) < 3:
            # 补充工具名作为关键词
            tool_pipeline = data.get("tool_pipeline", [])
            keywords = list(set(keywords + tool_pipeline + [chain_name]))[:8]
        keywords = keywords[:8]

        return ChainExperience(
            chain_name=chain_name,
            tool_pipeline=data.get("tool_pipeline", []),
            lesson=lesson,
            applicable_when=data.get("applicable_when", ""),
            anti_pattern=data.get("anti_pattern", ""),
            source_chain_ids=[source_chain_id],
            keywords=keywords,
        )

    def _parse_equivalence_response(self, text: str) -> tuple[bool, str]:
        """解析等价判定的 LLM JSON 响应。"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return False, ""
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return False, ""

        equivalent = data.get("equivalent", False)
        merged_lesson = data.get("merged_lesson", "")
        return bool(equivalent), str(merged_lesson)

    # ── 旧接口（向后兼容）──

    async def extract(
        self,
        chain_name: str,
        chains: list[ToolChain],
        existing_experiences: list[ChainExperience] | None = None,
    ) -> list[ChainExperience]:
        """从同名链批次中提炼经验（旧接口，保留向后兼容）。"""
        if not chains or not self._llm:
            return []

        # 构建已有经验段落
        existing_section = ""
        if existing_experiences:
            lines = ["已有经验（请在此基础上更新）："]
            for exp in existing_experiences:
                lines.append(f"- {exp.lesson}")
            existing_section = "\n".join(lines)

        # 构建链文本
        from .chain_segmenter import format_rich_traces_for_llm

        chains_text_parts = []
        for i, chain in enumerate(chains):
            parts = [f"### 执行 {i + 1} (outcome={chain.outcome})"]
            parts.append(f"工具序列: {' → '.join(chain.tool_sequence)}")
            if chain.rich_traces:
                parts.append(format_rich_traces_for_llm(chain.rich_traces))
            chains_text_parts.append("\n".join(parts))

        prompt = CHAIN_EXPERIENCE_PROMPT.format(
            chain_name=chain_name,
            chain_count=len(chains),
            existing_experience_section=existing_section,
            chains_text="\n\n".join(chains_text_parts),
        )

        try:
            response_text = await self._call_llm(prompt)
            return self._parse_response(
                response_text, chain_name,
                [c.chain_id for c in chains],
            )
        except Exception as e:
            logger.warning(f"Chain experience extraction failed for '{chain_name}': {e}")
            return []

    def _parse_response(
        self,
        text: str,
        chain_name: str,
        source_chain_ids: list[str],
    ) -> list[ChainExperience]:
        """解析旧接口的 LLM JSON 响应。"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return []
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

        experiences = []
        for item in data.get("experiences", []):
            if not item.get("lesson"):
                continue
            experiences.append(ChainExperience(
                chain_name=chain_name,
                tool_pipeline=item.get("tool_pipeline", []),
                lesson=item["lesson"],
                applicable_when=item.get("applicable_when", ""),
                anti_pattern=item.get("anti_pattern", ""),
                source_chain_ids=source_chain_ids,
            ))
        return experiences
