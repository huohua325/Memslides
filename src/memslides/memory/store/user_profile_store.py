"""UserProfileStore — 用户偏好画像存储（Design Doc §5.8, Intent-Based 扩展）

SQLite 表：user_profiles (user_id, intent, profile_json, version, last_updated)

Intent-Based 扩展:
- 同一用户按 intent（academic/business/education/creative/report/default）分别存储画像
- get/save 接口增加 intent 参数，默认 "default" 保持向后兼容
- 新增 get_all_intents() 查询用户所有 intent 画像
- 自动迁移：旧表（无 intent 列）的数据视为 intent="default"
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.models import DEFAULT_INTENT, IntentProfile, UserProfile

if TYPE_CHECKING:
    from ..core.db import DatabaseBackend

logger = logging.getLogger(__name__)


class UserProfileStore:
    """用户偏好画像的 SQLite 存储（支持 intent 分区）。"""

    def __init__(self, db: DatabaseBackend):
        self._db = db
        self._migrated = False
        self._payload_backfilled = False

    @staticmethod
    def _normalize_profile(profile: UserProfile, user_id: str) -> bool:
        """修正旧画像载荷中的 user_id / template 字段兼容问题。"""
        changed = False
        normalized_user_id = str(user_id or "").strip()
        if profile.user_id != normalized_user_id:
            profile.user_id = normalized_user_id
            changed = True
        if hasattr(profile, "template") and profile.template.normalize():
            changed = True
        return changed

    async def _ensure_profile_payload_compatibility(self) -> None:
        """为旧 user_profiles.profile_json 回填新增字段并修复脏 user_id。"""
        if self._payload_backfilled:
            return

        await self._ensure_intent_column()

        try:
            rows = await self._db.query(
                "SELECT user_id, intent, profile_json, version, last_updated FROM user_profiles",
                (),
            )
            for row in rows or []:
                try:
                    data = json.loads(row["profile_json"])
                except (TypeError, json.JSONDecodeError):
                    data = {}

                if not isinstance(data, dict):
                    data = {}

                profile = UserProfile.from_dict(data)
                changed = self._normalize_profile(profile, row["user_id"])
                normalized_data = asdict(profile)

                if not changed and normalized_data == data:
                    continue

                await self._db.execute(
                    """
                    UPDATE user_profiles
                    SET profile_json = ?, version = ?, last_updated = ?
                    WHERE user_id = ? AND intent = ?
                    """,
                    (
                        json.dumps(normalized_data, ensure_ascii=False),
                        row.get("version", profile.version),
                        row.get("last_updated", profile.last_updated),
                        row["user_id"],
                        row.get("intent", DEFAULT_INTENT),
                    ),
                )
            logger.info("user_profiles payload compatibility backfill complete")
        except Exception as e:
            logger.warning("user_profiles payload compatibility backfill failed: %s", e)

        self._payload_backfilled = True

    async def ensure_schema(self) -> None:
        """公开的 schema / payload 兼容入口。"""
        await self._ensure_intent_column()
        await self._ensure_profile_payload_compatibility()

    async def _ensure_intent_column(self) -> None:
        """确保 user_profiles 表有 intent 列且主键为 (user_id, intent)。

        旧 schema: PRIMARY KEY (user_id)
        新 schema: PRIMARY KEY (user_id, intent)

        SQLite 不支持 ALTER PRIMARY KEY，需要重建表。
        """
        if self._migrated:
            return
        try:
            rows = await self._db.query(
                "PRAGMA table_info(user_profiles)", ()
            )
            col_names = {r["name"] for r in rows} if rows else set()
            if not col_names:
                # 表不存在或为空，新 schema 会由 schema.sql 创建
                self._migrated = True
                return
            if "intent" not in col_names:
                # 旧表：需要重建以支持复合主键
                logger.info("Migrating user_profiles table: rebuilding with (user_id, intent) primary key")
                await self._db.execute(
                    """CREATE TABLE IF NOT EXISTS user_profiles_new (
                        user_id TEXT NOT NULL,
                        intent TEXT NOT NULL DEFAULT 'default',
                        profile_json TEXT NOT NULL,
                        version INTEGER DEFAULT 1,
                        last_updated TEXT NOT NULL,
                        PRIMARY KEY (user_id, intent)
                    )""", ()
                )
                await self._db.execute(
                    """INSERT OR IGNORE INTO user_profiles_new
                       (user_id, intent, profile_json, version, last_updated)
                       SELECT user_id, 'default', profile_json, version, last_updated
                       FROM user_profiles""", ()
                )
                await self._db.execute("DROP TABLE user_profiles", ())
                await self._db.execute(
                    "ALTER TABLE user_profiles_new RENAME TO user_profiles", ()
                )
                logger.info("user_profiles table migration complete")
        except Exception as e:
            logger.debug("Intent column migration check: %s", e)
        self._migrated = True

    async def get(self, user_id: str, intent: str = DEFAULT_INTENT) -> UserProfile:
        """加载指定 intent 的用户画像，不存在时返回空画像。"""
        await self.ensure_schema()
        try:
            row = await self._db.query_one(
                "SELECT profile_json FROM user_profiles WHERE user_id = ? AND intent = ?",
                (user_id, intent),
            )
            if row:
                data = json.loads(row["profile_json"])
                profile = UserProfile.from_dict(data)
                self._normalize_profile(profile, user_id)
                return profile
            # intent 不存在时，返回空画像（不从 default 继承，避免污染新 intent）
            # ProfileInjectionRouter 会在空维度上自动 skip
        except Exception as e:
            logger.warning(f"Failed to load user profile for {user_id}/{intent}: {e}")
        return UserProfile(user_id=user_id)

    async def save(self, profile: UserProfile, intent: str = DEFAULT_INTENT) -> None:
        """保存用户画像到指定 intent（INSERT OR REPLACE）。"""
        await self.ensure_schema()
        self._normalize_profile(profile, profile.user_id)
        profile.version += 1
        profile.last_updated = datetime.now().isoformat()
        try:
            await self._db.execute(
                """INSERT OR REPLACE INTO user_profiles
                   (user_id, intent, profile_json, version, last_updated)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    profile.user_id,
                    intent,
                    json.dumps(asdict(profile), ensure_ascii=False),
                    profile.version,
                    profile.last_updated,
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to save user profile for {profile.user_id}/{intent}: {e}")

    async def get_all_intents(self, user_id: str) -> IntentProfile:
        """加载用户所有 intent 的画像。"""
        await self.ensure_schema()
        ip = IntentProfile(user_id=user_id)
        try:
            rows = await self._db.query(
                "SELECT intent, profile_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            if rows:
                for row in rows:
                    intent = row.get("intent", DEFAULT_INTENT)
                    data = json.loads(row["profile_json"])
                    profile = UserProfile.from_dict(data)
                    self._normalize_profile(profile, user_id)
                    ip.profiles[intent] = profile
        except Exception as e:
            logger.warning(f"Failed to load all intents for {user_id}: {e}")
        return ip

    async def list_intents(self, user_id: str) -> list[str]:
        """列出用户已有的所有 intent。"""
        await self.ensure_schema()
        try:
            rows = await self._db.query(
                "SELECT intent FROM user_profiles WHERE user_id = ?",
                (user_id,),
            )
            return [r["intent"] for r in rows] if rows else []
        except Exception as e:
            logger.warning(f"Failed to list intents for {user_id}: {e}")
            return []
