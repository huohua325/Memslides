"""
AgenticMemoryRetriever — 两阶段智能检索 (Stage 2, Task 15)

基于 memory_upgrade_design_v2.md §3.4 定义，借鉴 EverMemOS agentic_utils.py。

Round 1: HybridRetriever (RRF) → 快速充分性检查 (确定性, 0 LLM)
  → 如果充分 → 返回
Round 2: (仅在不充分时)
  → LLM 生成补充查询
  → 并行多查询搜索
  → 去重合并
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# Prompt 模板 — 迁移到 memory/prompts/retrieval.py
from ..prompts import MULTI_QUERY_PROMPT

# 充分性阈值
_SUFFICIENCY_SCORE_THRESHOLD = 0.85
_SUFFICIENCY_MIN_RESULTS = 3


class AgenticMemoryRetriever:
    """两阶段智能检索

    Round 1: HybridRetriever (RRF)
      → 快速充分性检查 (确定性, 0 LLM)
      → 如果充分 → 返回
    Round 2: (仅在不充分时)
      → LLM 生成补充查询
      → 并行多查询搜索
      → 去重合并
    """

    def __init__(
        self,
        hybrid: Any,
        llm: Any = None,
        enable_agentic: bool = True,
    ) -> None:
        self._hybrid = hybrid
        self._llm = llm
        self._enable_agentic = enable_agentic and (llm is not None)

    async def retrieve(
        self, query: str, user_id: str, top_k: int = 5,
    ) -> list:
        """两阶段检索"""
        # Round 1
        merged = await self._hybrid.retrieve(query, user_id, top_k)

        # 快速充分性检查
        if self._quick_sufficiency_check(query, merged) or not self._enable_agentic:
            return merged[:top_k]

        # Round 2: Multi-Query
        try:
            refined_queries = await self._generate_refined_queries(query, merged)
            if not refined_queries:
                return merged[:top_k]

            round2_results = await asyncio.gather(
                *[self._hybrid.retrieve(q, user_id, top_k) for q in refined_queries],
                return_exceptions=True,
            )
            valid_round2 = [r for r in round2_results if isinstance(r, list)]

            return self._deduplicate_merge(merged, valid_round2, top_k)
        except Exception as e:
            logger.warning("Agentic Round 2 failed (non-fatal): %s", e)
            return merged[:top_k]

    @staticmethod
    def _quick_sufficiency_check(query: str, results: list) -> bool:
        """确定性充分性检查 (0 LLM cost)"""
        if not results:
            return False
        # Top-1 分数足够高 — 兼容 dict 和 RetrievalResult
        top_score = getattr(results[0], 'score', 0)
        if isinstance(top_score, (int, float)) and top_score > _SUFFICIENCY_SCORE_THRESHOLD:
            return True
        # 结果数量足够
        if len(results) >= _SUFFICIENCY_MIN_RESULTS:
            return True
        return False

    async def _generate_refined_queries(
        self, original: str, results: list,
    ) -> list[str]:
        """LLM 生成 2-3 个补充查询"""
        if not self._llm:
            return []

        def _extract_text(r):
            """从 RetrievalResult 或 dict 中提取文本"""
            if hasattr(r, 'metadata') and isinstance(r.metadata, dict):
                return r.metadata.get('design_insight', '') or r.metadata.get('trigger', '')
            if hasattr(r, 'content') and isinstance(r.content, str):
                return r.content[:100]
            if isinstance(r, dict):
                return r.get('design_insight', r.get('trigger', ''))
            return str(r)[:100]

        context = "\n".join(
            f"- {_extract_text(r)}" for r in results[:3]
        )
        prompt = MULTI_QUERY_PROMPT.format(
            original_query=original,
            current_results=context or "(no results)",
        )
        try:
            response = await self._llm(prompt)
            return self._parse_queries(response)
        except Exception as e:
            logger.warning("LLM query generation failed: %s", e)
            return []

    @staticmethod
    def _parse_queries(response: Any) -> list[str]:
        """从 LLM 响应解析查询列表"""
        text = str(response).strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                queries = json.loads(text[start:end])
                if isinstance(queries, list):
                    return [str(q) for q in queries if q][:3]
            except json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def _deduplicate_merge(
        round1: list, round2_lists: list[list], top_k: int,
    ) -> list:
        """去重合并 Round 1 + Round 2 结果"""
        seen_ids: set[str] = set()
        merged: list = []

        def _get_id(item) -> str:
            if hasattr(item, 'id'):
                return item.id or ""
            if isinstance(item, dict):
                return item.get("id", "")
            return ""

        for item in round1:
            item_id = _get_id(item)
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(item)
            elif not item_id:
                merged.append(item)

        for result_list in round2_lists:
            for item in result_list:
                item_id = _get_id(item)
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    merged.append(item)
                elif not item_id:
                    merged.append(item)

        return merged[:top_k]

    async def close(self) -> None:
        close = getattr(self._hybrid, "close", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        self._hybrid = None
