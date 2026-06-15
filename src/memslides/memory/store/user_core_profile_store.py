"""UserCoreProfileStore — 稳定 persona / 跨 intent 核心画像存储。"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.models import UserCoreProfile

if TYPE_CHECKING:
    from ..core.db import DatabaseBackend

logger = logging.getLogger(__name__)


class UserCoreProfileStore:
    """按 user_id 读写稳定 persona 与跨 intent 画像。"""

    def __init__(self, db: DatabaseBackend):
        self._db = db

    @property
    def db(self) -> DatabaseBackend:
        return self._db

    @staticmethod
    def _normalize_profile(profile: UserCoreProfile, user_id: str) -> bool:
        changed = False
        normalized_user_id = str(user_id or "").strip()
        normalized_persona = str(profile.core_persona or "").strip()
        if profile.user_id != normalized_user_id:
            profile.user_id = normalized_user_id
            changed = True
        if profile.core_persona != normalized_persona:
            profile.core_persona = normalized_persona
            changed = True
        return changed

    async def ensure_schema(self) -> None:
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS user_core_profiles (
                user_id TEXT PRIMARY KEY,
                core_persona TEXT NOT NULL DEFAULT '',
                profile_json TEXT NOT NULL,
                version INTEGER DEFAULT 1,
                last_updated TEXT NOT NULL
            )
            """,
            (),
        )
        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_core_profiles_persona
            ON user_core_profiles(core_persona)
            """,
            (),
        )

    async def exists(self, user_id: str) -> bool:
        await self.ensure_schema()
        row = await self._db.query_one(
            "SELECT 1 FROM user_core_profiles WHERE user_id = ?",
            (user_id,),
        )
        return row is not None

    async def get(self, user_id: str) -> UserCoreProfile:
        await self.ensure_schema()
        try:
            row = await self._db.query_one(
                """
                SELECT core_persona, profile_json, version, last_updated
                FROM user_core_profiles
                WHERE user_id = ?
                """,
                (user_id,),
            )
            if row:
                try:
                    data = json.loads(row["profile_json"])
                except (TypeError, json.JSONDecodeError):
                    data = {}

                if not isinstance(data, dict):
                    data = {}

                data.setdefault("user_id", user_id)
                data.setdefault("core_persona", row.get("core_persona", ""))
                data.setdefault("version", row.get("version", 1))
                data.setdefault("last_updated", row.get("last_updated", ""))

                profile = UserCoreProfile.from_dict(data)
                self._normalize_profile(profile, user_id)
                if not profile.core_persona and row.get("core_persona"):
                    profile.core_persona = str(row["core_persona"]).strip()
                return profile
        except Exception as e:
            logger.warning("Failed to load user core profile for %s: %s", user_id, e)
        return UserCoreProfile(user_id=user_id)

    async def save(self, profile: UserCoreProfile) -> None:
        await self.ensure_schema()
        self._normalize_profile(profile, profile.user_id)
        profile.version += 1
        profile.last_updated = datetime.now().isoformat()
        try:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO user_core_profiles
                (user_id, core_persona, profile_json, version, last_updated)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    profile.user_id,
                    profile.core_persona,
                    json.dumps(asdict(profile), ensure_ascii=False),
                    profile.version,
                    profile.last_updated,
                ),
            )
        except Exception as e:
            logger.warning(
                "Failed to save user core profile for %s: %s",
                profile.user_id,
                e,
            )
