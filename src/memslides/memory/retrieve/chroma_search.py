"""ChromaDB 向量搜索引擎 — 替代 NanoVectorDB

学习 nemori + TraceMem 的设计:
- PersistentClient 持久化
- Per-user collection 隔离
- 空库优雅处理 (collection.count())
- 去重检查 (collection.get(ids=[...]))
- 异步 embedding 适配
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .strategies import RetrievalResult

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed, ChromaVectorSearch unavailable. pip install chromadb")


class ChromaVectorSearch:
    """基于 ChromaDB 的向量搜索策略

    特点 (学习自 nemori/TraceMem):
    - PersistentClient 持久化
    - Per-user collection 隔离
    - 空库优雅处理 (collection.count())
    - 去重检查 (collection.get(ids=[...]))
    - 异步 embedding 适配
    """

    name = "chroma_vector"

    def __init__(
        self,
        persist_dir: str,
        embedding_func: Callable,  # async (list[str]) -> np.ndarray
        collection_prefix: str = "dp",
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required for ChromaVectorSearch. pip install chromadb")

        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._embed = embedding_func
        self._prefix = collection_prefix
        self._collections: Dict[str, Any] = {}  # cache

    async def close(self) -> None:
        self._collections.clear()
        clear_cache = getattr(self._client, "clear_system_cache", None)
        if clear_cache:
            clear_cache()
        self._client = None

    def _get_collection(self, user_id: str, mem_type: str = "episodes"):
        """获取或创建 collection (学习 TraceMem 的线程安全模式)"""
        key = f"{self._prefix}_{user_id}_{mem_type}"
        if key not in self._collections:
            self._collections[key] = self._client.get_or_create_collection(
                name=key,
                metadata={"user_id": user_id, "type": mem_type},
            )
        return self._collections[key]

    async def add_episode(
        self,
        user_id: str,
        episode_id: str,
        document: str,
        metadata: dict,
        mem_type: str = "episodes",
    ) -> bool:
        """写入单个 episode 向量 (增量)

        Args:
            mem_type: Collection 类型，用于隔离不同记忆类型
                      "episodes" → DesignEpisode
                      "experiences" → ExperienceTrace

        Returns:
            True=新写入, False=已存在
        """
        collection = self._get_collection(user_id, mem_type)

        # 去重 (学习 nemori)
        existing = collection.get(ids=[episode_id])
        if existing["ids"]:
            return False

        # 生成 embedding
        embeddings = await self._embed([document])
        embedding = embeddings[0].tolist()

        collection.add(
            ids=[episode_id],
            documents=[document],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return True

    async def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: str = "default",
        mem_type: str = "episodes",
    ) -> List[RetrievalResult]:
        """向量搜索 (学习 TraceMem 的空库检查)

        Args:
            mem_type: Collection 类型，与 add_episode 保持一致
        """
        collection = self._get_collection(user_id, mem_type)

        if collection.count() == 0:
            logger.debug("ChromaDB collection empty for user %s", user_id)
            return []

        embeddings = await self._embed([query])
        query_embedding = embeddings[0].tolist()

        n_results = min(top_k, collection.count())
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        search_results: List[RetrievalResult] = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distances; convert to similarity
            distance = results["distances"][0][i] if results.get("distances") else 0.0
            score = 1.0 - distance
            search_results.append(
                RetrievalResult(
                    id=doc_id,
                    content=results["documents"][0][i] if results.get("documents") else "",
                    score=score,
                    source="chroma_vector",
                    metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                )
            )
        return search_results
