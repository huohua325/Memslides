"""InteractiveSession: multi-turn session environment inheriting AgentEnv.

Provides:
- InteractiveSession(AgentEnv): session environment with MCP + memory capabilities

Two operating modes:
1. Full mode (config provided): inherits AgentEnv's MCP/Docker infrastructure
2. Standalone mode (config=None): session management only, for testing
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from memslides.agents.env import AgentEnv
from memslides.utils.config import MemSlidesConfig

from ..core.db import DatabaseBackend
from ..core.models import Message

logger = logging.getLogger(__name__)
CONSOLIDATION_TIMEOUT_S = float(os.getenv("MEMSLIDES_CONSOLIDATION_TIMEOUT", "0"))


def _env_flag_enabled(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


class InteractiveSession(AgentEnv):
    """Implementation details."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        project_id: str,
        workspace: Path | str,
        db: DatabaseBackend,
        config: MemSlidesConfig | None = None,
        memory_system: Any | None = None,
        context_window_size: int = 20,
    ):
        self._standalone_mode = config is None

        if config is not None:

            super().__init__(workspace, config)
        else:

            if isinstance(workspace, str):
                workspace = Path(workspace)
            self.workspace = workspace


        self.session_id = session_id
        self.user_id = user_id
        self.project_id = project_id
        self.db = db
        self.memory_system = memory_system


        self.is_ended: bool = False
        self.context_buffer: list[Message] = []
        self.context_window_size = context_window_size
        self.focus_context: str = ""
        self.resume_context: str = ""


        self._memory_dir = self.workspace / ".memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)


        self._created_at = datetime.now()
        self._last_activity = datetime.now()



    async def __aenter__(self):
        """Implementation details."""
        if not self._standalone_mode:
            await super().__aenter__()  # AgentEnv: Docker check + MCP connect



        await self._run_startup_consolidation()

        return self

    async def _run_startup_consolidation(self):
        """Implementation details."""
        if _env_flag_enabled("MEMSLIDES_SKIP_STARTUP_CONSOLIDATION"):
            logger.info("Session startup: memory consolidation skipped by MEMSLIDES_SKIP_STARTUP_CONSOLIDATION")
            return

        if not self.memory_system:
            return

        consolidator = getattr(self.memory_system, 'memory_consolidator', None)
        if not consolidator:
            return

        db = getattr(self.memory_system, 'db', None)
        if not db:
            return

        db_path = getattr(db, 'db_path', None)
        if not db_path:
            return

        isolated_db = None
        try:
            logger.info("Session startup: running memory consolidation...")


            await db.close()


            from ..core.db import SQLiteBackend
            from ..evolution.memory_consolidator import OfflineMemoryConsolidator

            isolated_db = SQLiteBackend(db_path)
            await isolated_db.connect()

            runtime_consolidator = OfflineMemoryConsolidator(
                db=isolated_db,
                llm=getattr(consolidator, "_llm", None),
                embedding_func=getattr(consolidator, "_embed", None),
                experience_vector_store=getattr(consolidator, "_vector_store", None),
                bm25_store=getattr(consolidator, "_bm25_store", None),
            )
            if CONSOLIDATION_TIMEOUT_S > 0:
                stats_obj = await asyncio.wait_for(
                    runtime_consolidator.consolidate(self.user_id),
                    timeout=CONSOLIDATION_TIMEOUT_S,
                )
            else:
                stats_obj = await runtime_consolidator.consolidate(self.user_id)
            stats = stats_obj.to_dict() if hasattr(stats_obj, "to_dict") else stats_obj
            logger.info("Session startup: consolidation completed - %s", stats)

        except asyncio.TimeoutError:
            logger.warning(
                "Session startup: consolidation timed out after %.0fs, fallback to exact dedup",
                CONSOLIDATION_TIMEOUT_S,
            )
            try:
                import sqlite3

                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                try:
                    stats = self._sync_consolidate(conn, self.user_id)
                    logger.info("Session startup: fallback exact dedup completed - %s", stats)
                finally:
                    conn.close()
            except Exception as fallback_error:
                logger.warning("Session startup: fallback exact dedup failed (non-fatal): %s", fallback_error)
        except Exception as e:

            logger.warning("Session startup: full consolidation failed, fallback to exact dedup: %s", e)
            try:
                import sqlite3

                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                try:
                    stats = self._sync_consolidate(conn, self.user_id)
                    logger.info("Session startup: fallback exact dedup completed - %s", stats)
                finally:
                    conn.close()
            except Exception as fallback_error:
                logger.warning("Session startup: fallback exact dedup failed (non-fatal): %s", fallback_error)
        finally:
            if isolated_db is not None:
                try:
                    await isolated_db.close()
                except Exception:
                    pass
            try:
                await db.connect()
            except Exception as reconnect_error:
                logger.warning("Session startup: DB reconnect failed (non-fatal): %s", reconnect_error)

    def _sync_consolidate(self, conn, user_id: str) -> dict:
        """Implementation details."""
        stats = {"experiences": 0, "episodes": 0, "preferences": 0}
        cursor = conn.cursor()


        cursor.execute("""
            SELECT et.lessons_learned, COUNT(*) as cnt, GROUP_CONCAT(et.id) as ids
            FROM experience_traces et
            JOIN sessions s ON et.session_id = s.id
            WHERE s.user_id = ?
              AND COALESCE(et.status, 'active') = 'active'
              AND et.lessons_learned IS NOT NULL AND et.lessons_learned != ''
            GROUP BY et.lessons_learned HAVING cnt > 1
        """, (user_id,))
        for row in cursor.fetchall():
            ids = row["ids"].split(",")
            if len(ids) > 1:
                placeholders = ",".join("?" * (len(ids) - 1))
                cursor.execute(f"DELETE FROM experience_traces WHERE id IN ({placeholders})", ids[1:])
                stats["experiences"] += len(ids) - 1


        cursor.execute("""
            SELECT design_insight, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
            FROM design_episodes
            WHERE user_id = ?
              AND design_insight IS NOT NULL AND design_insight != ''
              AND status = 'active'
            GROUP BY design_insight HAVING cnt > 1
        """, (user_id,))
        for row in cursor.fetchall():
            ids = row["ids"].split(",")
            if len(ids) > 1:
                placeholders = ",".join("?" * (len(ids) - 1))
                cursor.execute(f"UPDATE design_episodes SET status = 'archived' WHERE id IN ({placeholders})", ids[1:])
                stats["episodes"] += len(ids) - 1


        cursor.execute("""
            SELECT preference, COUNT(*) as cnt, GROUP_CONCAT(id) as ids
            FROM atomic_preferences
            WHERE preference IS NOT NULL AND preference != '' AND user_id = ?
            GROUP BY preference HAVING cnt > 1
        """, (user_id,))
        for row in cursor.fetchall():
            ids = row["ids"].split(",")
            if len(ids) > 1:
                placeholders = ",".join("?" * (len(ids) - 1))
                cursor.execute(f"DELETE FROM atomic_preferences WHERE id IN ({placeholders})", ids[1:])
                stats["preferences"] += len(ids) - 1

        conn.commit()
        return stats

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Implementation details."""
        pass

    async def end_session(self):
        """Implementation details."""
        self.is_ended = True


        await self._save_session_snapshot()


        if self.memory_system and hasattr(self.memory_system, "compactor"):
            try:
                compactor = self.memory_system.compactor
                if compactor is not None:
                    await compactor.run_gc(self.user_id)
            except Exception as e:
                logger.warning("Memory GC failed (non-fatal): %s", e)


        if not self._standalone_mode:
            await super().__aexit__(None, None, None)


        await self.db.update(
            "sessions",
            {"status": "ended", "ended_at": datetime.now().isoformat()},
            {"id": self.session_id},
        )

        logger.info("Session ended: %s", self.session_id)

    async def _save_session_snapshot(self):
        """Implementation details."""
        snapshot_data = {
            "id": str(uuid4()),
            "session_id": self.session_id,
            "user_id": self.user_id,
            "project_id": self.project_id,
            "focus_context": json.dumps(
                {"slide": self.focus_context}, ensure_ascii=False
            ),
            "session_rules_summary": "",
            "last_episode_summary": "",
            "modification_summary": "",
            "unfinished_items": json.dumps([], ensure_ascii=False),
            "total_episodes": 0,
            "total_edit_segments": 0,
        }
        await self.db.insert("session_snapshots", snapshot_data)

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """Implementation details."""
        if self.is_ended:
            return True
        elapsed = (datetime.now() - self._last_activity).total_seconds()
        return elapsed > timeout_minutes * 60

    def touch(self):
        """Implementation details."""
        self._last_activity = datetime.now()



    def add_message(self, message: Message):
        """Implementation details."""
        self.context_buffer.append(message)

        if len(self.context_buffer) > self.context_window_size:
            self.context_buffer = self.context_buffer[-self.context_window_size :]
        self.touch()

    def get_context_messages(self, last_n: int = 10) -> list[dict]:
        """Implementation details."""
        return [m.to_dict() for m in self.context_buffer[-last_n:]]



    def update_focus(self, intent: dict) -> None:
        """Implementation details."""
        new_focus = intent.get("target_slide", "")
        if new_focus:
            self.focus_context = new_focus



    def take_snapshot(self, turn: int) -> dict:
        """Implementation details."""
        snapshot = {
            "turn": turn,
            "session_id": self.session_id,
            "context_buffer_size": len(self.context_buffer),
            "session_rules_count": 0,
            "current_focus": self.focus_context,
            "is_ended": self.is_ended,
        }


        snapshot_path = self._memory_dir / "snapshots" / f"turn_{turn}.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return snapshot
