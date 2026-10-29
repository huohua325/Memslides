"""Implementation details."""

from __future__ import annotations

import logging
from typing import Any

from ..core.models import AtomicPreference
from ..core.db import DatabaseBackend

logger = logging.getLogger(__name__)

TABLE = "atomic_preferences"


class AtomicPreferenceStore:
    """Implementation details."""

    def __init__(self, db: DatabaseBackend) -> None:
        self.db = db



    async def add(
        self,
        preference: AtomicPreference,
        auto_supersede: bool = False,
    ) -> AtomicPreference:
        """Implementation details."""
        if auto_supersede and preference.conflict_group:
            await self.db.execute(
                f"UPDATE {TABLE} SET status = 'superseded' "
                "WHERE user_id = ? AND conflict_group = ? AND status = 'active'",
                (preference.user_id, preference.conflict_group),
            )

        await self.db.insert(TABLE, preference.to_dict())
        return preference



    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        scope_filter: str | None = None,
        scope_value_filter: str | None = None,
    ) -> list[AtomicPreference]:
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
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
        """Implementation details."""
        row = await self.db.query_one(
            f"SELECT * FROM {TABLE} "
            "WHERE user_id = ? AND conflict_group = ? AND status = 'active' "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, conflict_group),
        )
        return AtomicPreference.from_dict(row) if row else None



    async def verify(self, preference_id: str) -> None:
        """Implementation details."""
        await self.db.execute(
            f"UPDATE {TABLE} SET "
            "verified_count = verified_count + 1, "
            "confidence = MIN(1.0, confidence + 0.05) "
            "WHERE id = ?",
            (preference_id,),
        )

    async def contradict(self, preference_id: str) -> None:
        """Implementation details."""
        await self.db.execute(
            f"UPDATE {TABLE} SET "
            "contradiction_count = contradiction_count + 1, "
            "confidence = MAX(0.0, confidence - 0.1) "
            "WHERE id = ?",
            (preference_id,),
        )

    async def deprecate(self, preference_id: str) -> None:
        """Implementation details."""
        await self.db.execute(
            f"UPDATE {TABLE} SET status = 'deprecated' WHERE id = ?",
            (preference_id,),
        )
