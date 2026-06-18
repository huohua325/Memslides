"""
AtomicPreferenceStore — 原子级用户偏好的 SQLite 存储和检索 (Stage 4)

接口:
- add(preference, auto_supersede=False): 添加偏好，可选自动覆盖同冲突组旧偏好
- search(query, user_id, limit, scope_filter, scope_value_filter): LIKE 搜索 + scope 过滤
- get_for_context(user_id, context, limit): 根据 context 中的 slide_type/element_type 匹配
- get_for_template(user_id, template_id, limit): 按 template scope 过滤
- get_active(user_id, limit): 获取所有 active 偏好
- verify(preference_id): 正向验证，增加 confidence
- contradict(preference_id): 反向矛盾，降低 confidence
- deprecate(preference_id): 废弃偏好
- get_by_conflict_group(user_id, conflict_group): 按冲突组获取最新偏好
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AtomicPreference
from ..core.db import DatabaseBackend

logger = logging.getLogger(__name__)

TABLE = "atomic_preferences"


class AtomicPreferenceStore:
    """原子级用户偏好的 SQLite 存储"""

    def __init__(self, db: DatabaseBackend) -> None:
        self.db = db

    # ── 写入 ──

    async def add(
        self,
        preference: AtomicPreference,
        auto_supersede: bool = False,
    ) -> AtomicPreference:
        """添加偏好

        Args:
            preference: AtomicPreference 实例
            auto_supersede: 若为 True 且 conflict_group 非空，自动废弃同组旧偏好

        Returns:
            存储后的 AtomicPreference
        """
        if auto_supersede and preference.conflict_group:
            await self.db.execute(
                f"UPDATE {TABLE} SET status = 'superseded' "
                "WHERE user_id = ? AND conflict_group = ? AND status = 'active'",
                (preference.user_id, preference.conflict_group),
            )

        await self.db.insert(TABLE, preference.to_dict())
        return preference

    # ── 检索 ──

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        scope_filter: str | None = None,
        scope_value_filter: str | None = None,
    ) -> list[AtomicPreference]:
        """LIKE 搜索 + scope 过滤

        Args:
            query: 搜索文本
            user_id: 用户 ID
            limit: 返回数量限制
            scope_filter: 作用域过滤 ("global" | "slide_type" | "element_type")
            scope_value_filter: 作用域值过滤
        """
        sql = (
            f"SELECT * FROM {TABLE} "
            "WHERE user_id = ? AND status = 'active'"
        )
        params: list[Any] = [user_id]

        if scope_filter:
            sql += " AND scope = ?"
            params.append(scope_filter)

        if scope_value_filter:
            sql += " AND scope_value = ?"
            params.append(scope_value_filter)

        if query:
            sql += " AND (preference LIKE ? OR trigger LIKE ? OR rationale LIKE ?)"
            like = f"%{query}%"
            params.extend([like, like, like])

        sql += " ORDER BY confidence DESC LIMIT ?"
        params.append(limit)

        rows = await self.db.query(sql, tuple(params))
        return [AtomicPreference.from_dict(r) for r in rows]

    async def get_for_context(
        self,
        user_id: str,
        context: dict,
        limit: int = 10,
    ) -> list[AtomicPreference]:
        """根据 context 中的 slide_type / element_type 匹配偏好

        匹配逻辑:
        - global 偏好始终匹配
        - slide_type 偏好在 context 包含对应 slide_type 时匹配
        - element_type 偏好在 context 包含对应 element_type 时匹配
        """
        slide_type = context.get("slide_type", "")
        element_type = context.get("element_type", "")
        params: list[Any] = [user_id]

        scope_conditions = ["scope = 'global'"]
        if slide_type:
            scope_conditions.append("(scope = 'slide_type' AND scope_value = ?)")
            params.append(slide_type)
        if element_type:
            scope_conditions.append("(scope = 'element_type' AND scope_value = ?)")
            params.append(element_type)

        scope_clause = " OR ".join(scope_conditions)
        sql = (
            f"SELECT * FROM {TABLE} "
            f"WHERE user_id = ? AND status = 'active' AND ({scope_clause}) "
            "ORDER BY confidence DESC LIMIT ?"
        )
        params.append(limit)

        rows = await self.db.query(sql, tuple(params))
        return [AtomicPreference.from_dict(r) for r in rows]

    async def get_for_template(
        self,
        user_id: str,
        template_id: str,
        limit: int = 5,
    ) -> list[AtomicPreference]:
        """按 template scope 过滤偏好"""
        sql = (
            f"SELECT * FROM {TABLE} "
            "WHERE user_id = ? AND status = 'active' "
            "AND scope = 'template' AND scope_value = ? "
            "ORDER BY confidence DESC LIMIT ?"
        )
        rows = await self.db.query(sql, (user_id, template_id, limit))
        return [AtomicPreference.from_dict(r) for r in rows]

    async def get_active(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[AtomicPreference]:
        """获取所有 active 偏好"""
        sql = (
            f"SELECT * FROM {TABLE} "
            "WHERE user_id = ? AND status = 'active' "
            "ORDER BY confidence DESC LIMIT ?"
        )
        rows = await self.db.query(sql, (user_id, limit))
        return [AtomicPreference.from_dict(r) for r in rows]

    async def get_by_conflict_group(
        self,
        user_id: str,
        conflict_group: str,
    ) -> AtomicPreference | None:
        """按冲突组获取最新 active 偏好"""
        row = await self.db.query_one(
            f"SELECT * FROM {TABLE} "
            "WHERE user_id = ? AND conflict_group = ? AND status = 'active' "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, conflict_group),
        )
        return AtomicPreference.from_dict(row) if row else None

    # ── 状态管理 ──

    async def verify(self, preference_id: str) -> None:
        """正向验证: verified_count += 1, confidence 微增"""
        await self.db.execute(
            f"UPDATE {TABLE} SET "
            "verified_count = verified_count + 1, "
            "confidence = MIN(1.0, confidence + 0.05) "
            "WHERE id = ?",
            (preference_id,),
        )

    async def contradict(self, preference_id: str) -> None:
        """反向矛盾: contradiction_count += 1, confidence 微降"""
        await self.db.execute(
            f"UPDATE {TABLE} SET "
            "contradiction_count = contradiction_count + 1, "
            "confidence = MAX(0.0, confidence - 0.1) "
            "WHERE id = ?",
            (preference_id,),
        )

    async def deprecate(self, preference_id: str) -> None:
        """废弃偏好"""
        await self.db.execute(
            f"UPDATE {TABLE} SET status = 'deprecated' WHERE id = ?",
            (preference_id,),
        )
