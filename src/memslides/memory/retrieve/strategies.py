"""
检索策略层 — Vector / FTS5 / Hybrid RRF (Stage 2)

基于 memory_upgrade_design_v2.md §3.4 + §6.2 定义。
提供三个检索策略:
  - VectorSearchStrategy: 向量相似度检索
  - FTS5SearchStrategy: SQLite FTS5 全文检索
  - HybridRetriever: 组合多策略 + RRF 融合排序
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# 检索结果
# ═══════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """统一检索结果"""
    id: str
    content: Any = None
    score: float = 0.0
    source: str = ""
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════
# 检索策略基接口
# ═══════════════════════════════════════════════

@runtime_checkable
class SearchStrategy(Protocol):
    name: str

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        ...


# ═══════════════════════════════════════════════
# VectorSearchStrategy
# ═══════════════════════════════════════════════

class VectorSearchStrategy:
    """向量相似度检索 — 基于 NanoVectorDB"""

    name = "vector"

    def __init__(self, vector_db: Any, embedding_func: Callable) -> None:
        self._vector_db = vector_db
        self._embed = embedding_func

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        try:
            embedding = await self._embed(query)
            raw_results = self._vector_db.query(embedding, top_k=top_k)
            results = []
            for i, r in enumerate(raw_results or []):
                results.append(RetrievalResult(
                    id=r.get("__id__", ""),
                    score=r.get("__score__", 0.0),
                    source="vector",
                    metadata=r,
                ))
            return results
        except Exception as e:
            logger.warning("VectorSearch failed: %s", e)
            return []


# ═══════════════════════════════════════════════
# FTS5SearchStrategy
# ═══════════════════════════════════════════════

class FTS5SearchStrategy:
    """SQLite FTS5 全文检索"""

    name = "keyword"

    def __init__(self, db: Any, table_name: str, fts_table: str | None = None) -> None:
        self._db = db
        self._table = table_name
        self._fts_table = fts_table or f"{table_name}_fts"

    async def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if len(query.strip()) < 2:
            return []
        try:
            rows = await self._db.query(
                f"SELECT t.*, f.rank FROM {self._table} t "
                f"JOIN {self._fts_table} f ON t.rowid = f.rowid "
                f"WHERE {self._fts_table} MATCH ? "
                f"ORDER BY f.rank LIMIT ?",
                (query, top_k),
            )
            results = []
            for i, r in enumerate(rows):
                results.append(RetrievalResult(
                    id=r.get("id", ""),
                    content=r,
                    score=abs(r.get("rank", 0)),
                    source="fts5",
                    metadata=r,
                ))
            return results
        except Exception as e:
            logger.warning("FTS5Search failed: %s", e)
            return []


# ═══════════════════════════════════════════════
# RRF 融合 — 共享实现
# ═══════════════════════════════════════════════

def rrf_merge(
    result_lists: list[list[RetrievalResult]],
    top_k: int,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion (RRF) 合并多路检索结果

    RRF 公式: score(d) = Σ 1 / (k + rank_i + 1)
    其中 rank_i 是文档 d 在第 i 路结果中的排名 (0-indexed)。

    Args:
        result_lists: 多路检索结果
        top_k: 返回结果数
        rrf_k: RRF 融合参数 (默认 60)
    """
    scores: dict[str, float] = defaultdict(float)
    items: dict[str, RetrievalResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (rrf_k + rank + 1)
            scores[result.id] += rrf_score
            # 保留第一次出现的完整结果
            if result.id not in items:
                items[result.id] = result

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    merged: list[RetrievalResult] = []
    for doc_id in sorted_ids[:top_k]:
        item = items[doc_id]
        item.score = scores[doc_id]
        merged.append(item)
    return merged


# ═══════════════════════════════════════════════
# HybridRetriever — RRF 融合
# ═══════════════════════════════════════════════

class HybridRetriever:
    """组合多个 SearchStrategy + RRF (Reciprocal Rank Fusion) 融合

    RRF 公式: score(d) = Σ 1 / (k + rank_i + 1)
    其中 rank_i 是文档 d 在第 i 路结果中的排名 (0-indexed)。
    """

    def __init__(
        self,
        strategies: list[Any] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self._strategies = strategies or []
        self._rrf_k = rrf_k

    async def retrieve(
        self,
        query: str,
        user_id: str = "",
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """多路检索 → RRF 合并 → 返回 top_k"""
        if not self._strategies:
            return []

        result_lists: list[list[RetrievalResult]] = []
        for strategy in self._strategies:
            try:
                results = await strategy.search(query, top_k=top_k)
                if results:
                    result_lists.append(results)
            except Exception as e:
                logger.warning("Strategy '%s' failed: %s",
                               getattr(strategy, 'name', 'unknown'), e)

        if not result_lists:
            return []

        if len(result_lists) == 1:
            return result_lists[0][:top_k]

        return self._rrf_merge(result_lists, top_k)

    def _rrf_merge(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> list[RetrievalResult]:
        """RRF 合并多路结果 — 委托给共享 rrf_merge()"""
        return rrf_merge(result_lists, top_k, self._rrf_k)
