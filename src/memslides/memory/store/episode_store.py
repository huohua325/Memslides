"""
EpisodeStore — 情景记忆 SQLite 存储 (Stage 2)

基于 memory_upgrade_design_v2.md §5.1 + Task 06 定义。
实现 EpisodeStoreProtocol，使用 DatabaseBackend 进行 CRUD。
向量搜索通过 UnifiedSearchEngine (ChromaDB + BM25) 支持。
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import DesignEpisode

logger = logging.getLogger(__name__)


class EpisodeStore:
    """SQLite 情景记忆存储

    支持:
    - SQLite CRUD (add/get_by_session/get_active/update_status/archive)
    - FTS5 全文搜索 (search_by_text)
    - 可选向量相似度搜索 (search_similar, 需要 UnifiedSearchEngine)
    """

    TABLE = "design_episodes"

    def __init__(
        self,
        db: Any,
        unified_search: Any | None = None,
    ) -> None:
        self.db = db
        self._unified_search = unified_search

    async def add_episode(self, episode: DesignEpisode) -> str:
        """插入 Episode 到 SQLite（+ UnifiedSearchEngine 双路索引写入）"""
        d = episode.to_dict()
        await self.db.insert(self.TABLE, d)

        # ── 组合多字段作为检索文档 ──
        doc_parts = [
            episode.user_intent or "",
            episode.design_insight or "",
            episode.action_outcome or "",
        ]
        document = ". ".join(p for p in doc_parts if p.strip())

        # ── UnifiedSearchEngine 双路写入 (ChromaDB + BM25) ──
        if self._unified_search and document:
            try:
                metadata = {
                    "user_id": episode.user_id,
                    "session_id": episode.session_id or "",
                    "category": episode.category or "",
                    "user_intent": (episode.user_intent or "")[:200],
                    "design_insight": (episode.design_insight or "")[:200],
                }
                await self._unified_search.add_episode(
                    user_id=episode.user_id,
                    episode_id=episode.id,
                    document=document,
                    metadata=metadata,
                )
            except Exception as e:
                logger.warning("UnifiedSearch write failed for episode %s: %s", episode.id, e)

        return episode.id

    async def search_similar(
        self, query: str, limit: int = 5, user_id: str = "default",
    ) -> list[DesignEpisode]:
        """向量相似度搜索"""

        # ── UnifiedSearchEngine ──
        if self._unified_search:
            try:
                results = await self._unified_search.search(
                    query=query, user_id=user_id, top_k=limit,
                )
                if results:
                    ids = [r.id for r in results]
                    placeholders = ", ".join("?" for _ in ids)
                    rows = await self.db.query(
                        f"SELECT * FROM {self.TABLE} WHERE id IN ({placeholders})",
                        tuple(ids),
                    )
                    return [DesignEpisode.from_dict(r) for r in rows]
            except Exception as e:
                logger.warning("UnifiedSearch query failed: %s, falling back to FTS", e)

        # ── Fallback: FTS5 ──
        return await self.search_by_text(query, limit)

    async def search_by_text(self, query: str, limit: int = 5) -> list[DesignEpisode]:
        """FTS5 全文搜索 (fallback)"""
        if len(query) < 3:
            return []
        try:
            rows = await self.db.query(
                f"SELECT e.* FROM {self.TABLE} e "
                f"JOIN design_episodes_fts f ON e.rowid = f.rowid "
                f"WHERE design_episodes_fts MATCH ? LIMIT ?",
                (query, limit),
            )
            return [DesignEpisode.from_dict(r) for r in rows]
        except Exception as e:
            logger.warning("FTS search failed: %s", e)
            return []

    async def get_by_session(self, session_id: str) -> list[DesignEpisode]:
        """按 session_id 查询"""
        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        )
        return [DesignEpisode.from_dict(r) for r in rows]

    async def get_active(self, user_id: str) -> list[DesignEpisode]:
        """查询所有 status='active' 的 Episode"""
        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE user_id = ? AND status = 'active' "
            f"ORDER BY created_at DESC",
            (user_id,),
        )
        return [DesignEpisode.from_dict(r) for r in rows]

    async def update_status(self, episode_id: str, status: str) -> None:
        """更新状态"""
        await self.db.execute(
            f"UPDATE {self.TABLE} SET status = ? WHERE id = ?",
            (status, episode_id),
        )

    async def archive(self, episode_id: str) -> None:
        """归档"""
        await self.update_status(episode_id, "archived")

    async def list_episodes(
        self, user_id: str, limit: int = 100,
    ) -> list[DesignEpisode]:
        """按用户列出所有 episode（按创建时间倒序）"""
        rows = await self.db.query(
            f"SELECT * FROM {self.TABLE} WHERE user_id = ? "
            f"ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [DesignEpisode.from_dict(r) for r in rows]
