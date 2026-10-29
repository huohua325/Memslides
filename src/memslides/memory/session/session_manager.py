"""Session lifecycle management.

Provides:
- SessionManager: create, retrieve, resume, end, and cleanup sessions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..core.db import DatabaseBackend
from .session import InteractiveSession

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器"""

    def __init__(self, db: DatabaseBackend):
        self.db = db
        self.sessions: dict[str, InteractiveSession] = {}

    async def get_or_create_session(
        self,
        user_id: str,
        project_id: str,
        workspace: Path | str,
        session_id: str | None = None,
        **kwargs,
    ) -> InteractiveSession:
        """获取或创建Session

        Args:
            session_id: Optional explicit session ID. If None, generates a UUID.
        """
        session_key = f"{user_id}:{project_id}"

        if session_key in self.sessions:
            existing = self.sessions[session_key]
            if not existing.is_ended and not existing.is_expired():
                existing.touch()
                return existing

        # 创建新Session
        if session_id is None:
            session_id = str(uuid4())
        session = InteractiveSession(
            session_id=session_id,
            user_id=user_id,
            project_id=project_id,
            workspace=Path(workspace),
            db=self.db,
            **kwargs,
        )

        # 记录到DB (use INSERT OR IGNORE for idempotency with global DB)
        try:
            existing = await self.db.query_one(
                "SELECT id FROM sessions WHERE id = ?", (session_id,)
            )
            if not existing:
                await self.db.insert(
                    "sessions",
                    {
                        "id": session_id,
                        "user_id": user_id,
                        "project_id": project_id,
                        "status": "active",
                        "created_at": datetime.now().isoformat(),
                    },
                )
            else:
                # Update status to active for resumed sessions
                await self.db.execute(
                    "UPDATE sessions SET status = 'active' WHERE id = ?",
                    (session_id,),
                )
        except Exception as e:
            logger.warning("Session DB insert/update failed (non-fatal): %s", e)

        self.sessions[session_key] = session
        logger.info("Created session %s for %s", session_id, session_key)
        return session

    async def end_session(self, session: InteractiveSession):
        """结束Session"""
        await session.end_session()
        session_key = f"{session.user_id}:{session.project_id}"
        self.sessions.pop(session_key, None)
        logger.info("Ended session %s", session.session_id)

    async def resume_session(
        self,
        user_id: str,
        project_id: str,
        workspace: Path | str,
        **kwargs,
    ) -> InteractiveSession:
        """检测并续接历史Session（Gap 1: Session续接机制）

        当用户隔天想继续修改同一PPT时，加载上次SessionSnapshot，
        恢复焦点状态并生成续接摘要注入新Session的system prompt。
        """
        snapshot = await self._load_latest_snapshot(user_id, project_id)
        session = await self.get_or_create_session(
            user_id, project_id, workspace, **kwargs
        )
        if snapshot:
            focus = (
                json.loads(snapshot["focus_context"])
                if snapshot.get("focus_context")
                else {}
            )
            session.focus_context = focus.get("slide", "")
            session.resume_context = self._build_resume_prompt(snapshot)
            logger.info(
                "Resumed session %s with snapshot from %s",
                session.session_id,
                snapshot.get("session_id", "?"),
            )
        return session

    async def _load_latest_snapshot(
        self, user_id: str, project_id: str
    ) -> dict | None:
        """加载最近的SessionSnapshot"""
        return await self.db.query_one(
            "SELECT * FROM session_snapshots WHERE user_id = ? AND project_id = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id, project_id),
        )

    def _build_resume_prompt(self, snapshot: dict) -> str:
        """基于SessionSnapshot生成续接摘要"""
        parts = ["## 上次修改记录（续接上下文）"]
        if snapshot.get("modification_summary"):
            parts.append(f"修改概要：{snapshot['modification_summary']}")
        if snapshot.get("last_episode_summary"):
            parts.append(f"最后修改：{snapshot['last_episode_summary']}")
        if snapshot.get("session_rules_summary"):
            parts.append(f"已建立规则：{snapshot['session_rules_summary']}")
        return "\n".join(parts)

    async def cleanup_expired(self):
        """清理过期Session"""
        expired = [k for k, s in self.sessions.items() if s.is_expired()]
        for key in expired:
            session = self.sessions.pop(key)
            await session.end_session()
            logger.info("Cleaned up expired session %s", session.session_id)
        return len(expired)
