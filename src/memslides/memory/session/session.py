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
    """多轮交互会话环境 — 继承 AgentEnv，扩展记忆能力

    继承自 AgentEnv 的能力（完整模式）:
    - MCP client + tool_execute()
    - Docker 容器管理
    - tool_history 追踪

    新增能力:
    1. 上下文缓冲区（滑动窗口）
    2. focus_context 焦点追踪
    3. Episode管理
    4. 后台任务追踪
    5. 快照保存
    6. 记忆系统组件引用

    两种模式:
    - 完整模式 (config provided): 继承 AgentEnv 的 MCP/Docker 能力
    - 独立模式 (config=None): 仅会话管理，用于测试
    """

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
            # 完整模式: AgentEnv 初始化 (MCP, Docker, tools)
            super().__init__(workspace, config)
        else:
            # 独立模式: 手动设置 workspace，跳过 MCP/Docker
            if isinstance(workspace, str):
                workspace = Path(workspace)
            self.workspace = workspace

        # Session 标识
        self.session_id = session_id
        self.user_id = user_id
        self.project_id = project_id
        self.db = db
        self.memory_system = memory_system

        # 会话状态
        self.is_ended: bool = False
        self.context_buffer: list[Message] = []
        self.context_window_size = context_window_size
        self.focus_context: str = ""  # 当前焦点幻灯片
        self.resume_context: str = ""  # 续接上下文（Gap 1）

        # Prompt/日志目录
        self._memory_dir = self.workspace / ".memory"
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # 创建时间（用于过期检测）
        self._created_at = datetime.now()
        self._last_activity = datetime.now()

    # === 生命周期 ===

    async def __aenter__(self):
        """进入会话上下文

        完整模式: 连接 MCP servers (AgentEnv.__aenter__)
        独立模式: 直接返回

        Stage 4: Session 启动时执行记忆整合（从 session_end 移至此处）
        """
        if not self._standalone_mode:
            await super().__aenter__()  # AgentEnv: Docker check + MCP connect

        # Stage 4: 在 Session 启动时执行记忆整合
        # 此时 MCP 服务器刚启动，数据库连接稳定，无并发问题
        await self._run_startup_consolidation()

        return self

    async def _run_startup_consolidation(self):
        """Session 启动时执行记忆整合（去重 + 聚类合并）

        Stage 4: 关闭 aiosqlite 连接后执行整合，避免锁冲突
        """
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

            # 关闭 aiosqlite 连接，释放锁
            await db.close()

            # 使用独立 SQLiteBackend 执行完整整合（含聚类/合并）
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
            # 主路径失败时，降级到快速精确去重
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
        """同步执行精确去重"""
        stats = {"experiences": 0, "episodes": 0, "preferences": 0}
        cursor = conn.cursor()

        # ExperienceTrace 去重
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

        # DesignEpisode 去重
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

        # AtomicPreference 去重
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
        """退出上下文：不结束Session，保持MCP连接存活

        这是关键改造点：覆写AgentEnv的__aexit__，
        不断开MCP连接，仅在end_session()中显式断开。
        """
        pass

    async def end_session(self):
        """显式结束Session

        执行顺序：
        1. 保存SessionSnapshot（Gap 1: 跨Session续接）
        2. 运行记忆 GC（淘汰低置信度规则/经验）
        3. 断开MCP连接（完整模式）
        4. 保存Session元数据
        """
        self.is_ended = True

        # 1. 保存SessionSnapshot
        await self._save_session_snapshot()

        # 2. 运行记忆 GC
        if self.memory_system and hasattr(self.memory_system, "compactor"):
            try:
                compactor = self.memory_system.compactor
                if compactor is not None:
                    await compactor.run_gc(self.user_id)
            except Exception as e:
                logger.warning("Memory GC failed (non-fatal): %s", e)

        # 3. 断开MCP连接（完整模式下调用 AgentEnv 的原始 __aexit__）
        if not self._standalone_mode:
            await super().__aexit__(None, None, None)

        # 4. 更新Session状态到DB
        await self.db.update(
            "sessions",
            {"status": "ended", "ended_at": datetime.now().isoformat()},
            {"id": self.session_id},
        )

        logger.info("Session ended: %s", self.session_id)

    async def _save_session_snapshot(self):
        """保存SessionSnapshot用于跨Session续接（Gap 1）"""
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
        """检查Session是否过期"""
        if self.is_ended:
            return True
        elapsed = (datetime.now() - self._last_activity).total_seconds()
        return elapsed > timeout_minutes * 60

    def touch(self):
        """更新最后活动时间"""
        self._last_activity = datetime.now()

    # === 消息处理 ===

    def add_message(self, message: Message):
        """添加消息到上下文缓冲区"""
        self.context_buffer.append(message)
        # 滑动窗口
        if len(self.context_buffer) > self.context_window_size:
            self.context_buffer = self.context_buffer[-self.context_window_size :]
        self.touch()

    def get_context_messages(self, last_n: int = 10) -> list[dict]:
        """获取最近N条上下文消息（dict格式）"""
        return [m.to_dict() for m in self.context_buffer[-last_n:]]

    # === EditSegment管理 ===

    def update_focus(self, intent: dict) -> None:
        """根据意图更新当前焦点幻灯片"""
        new_focus = intent.get("target_slide", "")
        if new_focus:
            self.focus_context = new_focus

    # === 快照 ===

    def take_snapshot(self, turn: int) -> dict:
        """创建当前轮次的上下文快照"""
        snapshot = {
            "turn": turn,
            "session_id": self.session_id,
            "context_buffer_size": len(self.context_buffer),
            "session_rules_count": 0,
            "current_focus": self.focus_context,
            "is_ended": self.is_ended,
        }

        # 保存到文件
        snapshot_path = self._memory_dir / "snapshots" / f"turn_{turn}.json"
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return snapshot
