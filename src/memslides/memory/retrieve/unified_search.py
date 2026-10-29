"""统一搜索引擎 — 组合 ChromaDB + BM25 + RRF

学习 nemori UnifiedSearchEngine 的设计:
- 写入时双路索引 (ChromaDB + BM25)
- 检索时并行查询 + RRF 融合
- 支持 search_method: "hybrid" | "vector" | "bm25"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List

from .strategies import RetrievalResult, rrf_merge

logger = logging.getLogger(__name__)


class UnifiedSearchEngine:
    """统一搜索引擎

    学习 nemori UnifiedSearchEngine 的设计:
    - 写入时双路索引 (ChromaDB + BM25)
    - 检索时并行查询 + RRF 融合
    - 支持 search_method: "hybrid" | "vector" | "bm25"
    """

    def __init__(
        self,
        chroma_search: Any = None,
        bm25_search: Any = None,
        rrf_k: int = 60,
    ):
        self._chroma = chroma_search
        self._bm25 = bm25_search
        self._rrf_k = rrf_k

    async def add_episode(
        self,
        user_id: str,
        episode_id: str,
        document: str,
        metadata: dict,
    ):
        """双路写入 (学习 nemori: 写入时同步更新两个索引)"""
        # ChromaDB (async)
        if self._chroma:
            try:
                await self._chroma.add_episode(user_id, episode_id, document, metadata)
            except Exception as e:
                logger.warning("ChromaDB write failed: %s", e)

        # BM25 (sync, but fast)
        if self._bm25:
            try:
                self._bm25.add_episode(user_id, episode_id, document, metadata)
            except Exception as e:
                logger.warning("BM25 write failed: %s", e)

    async def search(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 5,
        method: str = "hybrid",
    ) -> List[RetrievalResult]:
        """统一检索入口"""
        if method == "vector":
            if self._chroma:
                return await self._chroma.search(query, top_k, user_id)
            return []
        elif method == "bm25":
            if self._bm25:
                return await self._bm25.search(query, top_k, user_id)
            return []
        else:  # hybrid
            return await self._hybrid_search(query, user_id, top_k)

    async def retrieve(
        self,
        query: str,
        user_id: str = "default",
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """兼容 HybridRetriever.retrieve() 接口 — AgenticRetriever 调用此方法"""
        return await self.search(query, user_id, top_k)

    async def _hybrid_search(
        self,
        query: str,
        user_id: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """并行 ChromaDB + BM25 → RRF 融合"""
        tasks = []
        if self._chroma:
            tasks.append(self._chroma.search(query, top_k * 2, user_id))
        if self._bm25:
            tasks.append(self._bm25.search(query, top_k * 2, user_id))

        if not tasks:
            return []

        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        result_lists: list[list[RetrievalResult]] = []
        for r in raw_results:
            if isinstance(r, list) and r:
                result_lists.append(r)

        if not result_lists:
            return []
        if len(result_lists) == 1:
            return result_lists[0][:top_k]

        return self._rrf_merge(result_lists, top_k)

    def _rrf_merge(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int,
    ) -> List[RetrievalResult]:
        """RRF 融合 — 委托给共享 rrf_merge()"""
        return rrf_merge(result_lists, top_k, self._rrf_k)

    async def close(self) -> None:
        for component in (self._chroma, self._bm25):
            close = getattr(component, "close", None)
            if not close:
                continue
            result = close()
            if asyncio.iscoroutine(result):
                await result
        self._chroma = None
        self._bm25 = None
