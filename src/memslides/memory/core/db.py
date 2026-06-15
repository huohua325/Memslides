"""Database abstraction layer for the memory system.

Provides:
- DatabaseBackend Protocol: abstract interface for any DB backend
- SQLiteBackend: async SQLite implementation using aiosqlite
"""

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import aiosqlite

logger = logging.getLogger(__name__)


@runtime_checkable
class DatabaseBackend(Protocol):
    """数据库后端协议"""

    async def connect(self) -> None:
        """建立连接"""
        ...

    async def close(self) -> None:
        """关闭连接"""
        ...

    async def execute(
        self,
        sql: str,
        params: tuple | dict = (),
        auto_commit: bool = True,
    ) -> int:
        """执行SQL，返回affected rows"""
        ...

    async def execute_many(self, sql: str, params_list: list[tuple | dict]) -> None:
        """批量执行SQL"""
        ...

    async def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        """查询，返回字典列表"""
        ...

    async def query_one(self, sql: str, params: tuple | dict = ()) -> dict | None:
        """查询单条记录"""
        ...

    async def insert(self, table: str, data: dict) -> str:
        """插入记录，返回ID"""
        ...

    async def update(self, table: str, data: dict, where: dict) -> int:
        """更新记录，返回affected rows"""
        ...

    async def delete(self, table: str, where: dict) -> int:
        """删除记录，返回affected rows"""
        ...

    async def init_schema(self) -> None:
        """初始化数据库Schema"""
        ...


class SQLiteBackend:
    """SQLite异步数据库后端

    Stage 4 改进：添加连接健康检查和自动重连机制
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._is_healthy = False

    async def connect(self) -> None:
        """建立连接 + 启用DELETE日志模式 + 启用外键

        Stage 4: 改用 DELETE 模式替代 WAL 模式
        原因：WAL 模式在并发访问时可能导致 "malformed" 错误
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        self._conn.row_factory = aiosqlite.Row
        # 使用 DELETE 模式替代 WAL，避免并发问题
        await self._conn.execute("PRAGMA journal_mode=DELETE")
        await self._conn.execute("PRAGMA busy_timeout=10000")  # 增加超时
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")  # 平衡性能和安全
        self._is_healthy = True
        logger.info("SQLiteBackend connected (DELETE mode): %s", self.db_path)

    async def close(self) -> None:
        """关闭连接"""
        if self._conn:
            try:
                await self._conn.close()
            except Exception:
                pass
            self._conn = None
            self._is_healthy = False
            logger.info("SQLiteBackend closed: %s", self.db_path)

    async def _check_and_reconnect(self) -> aiosqlite.Connection:
        """检查连接健康状态，必要时重连

        Stage 4: 解决长时间 LLM 调用后连接失效问题
        """
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")

        # 快速健康检查
        try:
            await self._conn.execute("SELECT 1")
            return self._conn
        except Exception as e:
            logger.warning("Database connection unhealthy, reconnecting: %s", e)
            self._is_healthy = False

        # 尝试重连
        try:
            if self._conn:
                try:
                    await self._conn.close()
                except Exception:
                    pass

            self._conn = await aiosqlite.connect(str(self.db_path))
            self._conn.row_factory = aiosqlite.Row
            await self._conn.execute("PRAGMA journal_mode=DELETE")
            await self._conn.execute("PRAGMA busy_timeout=10000")
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            self._is_healthy = True
            logger.info("SQLiteBackend reconnected (DELETE mode): %s", self.db_path)
            return self._conn
        except Exception as e:
            logger.error("Failed to reconnect database: %s", e)
            raise

    def _ensure_connected(self) -> aiosqlite.Connection:
        """同步检查连接（用于简单操作）"""
        if self._conn is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._conn

    async def execute(
        self,
        sql: str,
        params: tuple | dict = (),
        auto_commit: bool = True,
    ) -> int:
        """执行SQL

        Args:
            sql: SQL语句
            params: 参数
            auto_commit: 是否自动提交，批量操作时可设为False

        Returns:
            affected rows（非DML语句返回0）
        """
        conn = await self._check_and_reconnect()
        cursor = await conn.execute(sql, params)
        if auto_commit:
            await conn.commit()
        rowcount = getattr(cursor, "rowcount", -1)
        return rowcount if isinstance(rowcount, int) and rowcount >= 0 else 0

    async def commit(self) -> None:
        """手动提交事务"""
        conn = self._ensure_connected()
        await conn.commit()

    async def execute_many(self, sql: str, params_list: list[tuple | dict]) -> None:
        """批量执行SQL"""
        conn = await self._check_and_reconnect()
        await conn.executemany(sql, params_list)
        await conn.commit()

    async def query(self, sql: str, params: tuple | dict = ()) -> list[dict]:
        """查询，返回字典列表"""
        conn = await self._check_and_reconnect()
        cursor = await conn.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def query_one(self, sql: str, params: tuple | dict = ()) -> dict | None:
        """查询单条记录"""
        conn = await self._check_and_reconnect()
        cursor = await conn.execute(sql, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def insert(self, table: str, data: dict) -> str:
        """通用插入：自动从dict生成INSERT语句，返回ID"""
        conn = await self._check_and_reconnect()
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?"] * len(data))
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        await conn.execute(sql, tuple(data.values()))
        await conn.commit()
        return data.get("id", "")

    async def update(self, table: str, data: dict, where: dict) -> int:
        """通用更新：自动生成UPDATE ... WHERE语句，返回affected rows"""
        conn = await self._check_and_reconnect()
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        where_clause = " AND ".join(f"{k} = ?" for k in where.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where_clause}"
        params = tuple(data.values()) + tuple(where.values())
        cursor = await conn.execute(sql, params)
        await conn.commit()
        return cursor.rowcount

    async def delete(self, table: str, where: dict) -> int:
        """通用删除，返回affected rows"""
        conn = await self._check_and_reconnect()
        where_clause = " AND ".join(f"{k} = ?" for k in where.keys())
        sql = f"DELETE FROM {table} WHERE {where_clause}"
        cursor = await conn.execute(sql, tuple(where.values()))
        await conn.commit()
        return cursor.rowcount

    async def init_schema(self) -> None:
        """从 schema.sql 初始化全部表结构（sessions, experience_traces, design_episodes, atomic_preferences 等）"""
        conn = await self._check_and_reconnect()
        # Old global DBs may have an existing experience_traces table without the
        # Stage 15 columns. schema.sql now creates indexes on those new columns,
        # so we must patch legacy tables before executescript() touches the indexes.
        await self._ensure_stage15_experience_trace_schema(conn)
        schema_path = Path(__file__).parent / "schema.sql"
        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding="utf-8")
            await conn.executescript(schema_sql)
            await self._ensure_stage15_experience_trace_schema(conn)
            await self._ensure_chain_store_schema(conn)
            logger.info("Schema initialized from %s", schema_path)
        await conn.commit()

    async def _ensure_stage15_experience_trace_schema(self, conn: aiosqlite.Connection) -> None:
        """为已有 DB 追加 Stage 15 所需列与索引。"""
        cursor = await conn.execute("PRAGMA table_info(experience_traces)")
        rows = await cursor.fetchall()
        existing_columns = {row[1] for row in rows}
        if not existing_columns:
            return

        staged_columns = {
            "status": "TEXT DEFAULT 'active'",
            "superseded_by": "TEXT DEFAULT ''",
            "superseded_at": "TEXT DEFAULT ''",
            "merged_from_ids": "TEXT DEFAULT '[]'",
            "source_types": "TEXT DEFAULT '[]'",
            "consolidation_version": "TEXT DEFAULT ''",
        }
        for column, definition in staged_columns.items():
            if column not in existing_columns:
                await conn.execute(
                    f"ALTER TABLE experience_traces ADD COLUMN {column} {definition}"
                )

        await conn.execute(
            "UPDATE experience_traces SET status = 'active' "
            "WHERE status IS NULL OR status = ''"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experience_status "
            "ON experience_traces(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experience_type_status "
            "ON experience_traces(experience_type, status)"
        )

    async def _ensure_chain_store_schema(self, conn: aiosqlite.Connection) -> None:
        """为 chain_raw_data / chain_experiences 修补旧版 schema。"""
        await self._ensure_chain_raw_data_schema(conn)
        await self._ensure_chain_experience_schema(conn)

    async def _ensure_chain_raw_data_schema(self, conn: aiosqlite.Connection) -> None:
        """为旧版 chain_raw_data 增补缺失列与索引。"""
        cursor = await conn.execute("PRAGMA table_info(chain_raw_data)")
        rows = await cursor.fetchall()
        existing_columns = {row[1] for row in rows}
        if not existing_columns:
            return

        if "job_id" not in existing_columns:
            await conn.execute(
                "ALTER TABLE chain_raw_data ADD COLUMN job_id TEXT DEFAULT ''"
            )

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chain_raw_signature "
            "ON chain_raw_data(chain_signature)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chain_raw_user "
            "ON chain_raw_data(user_id)"
        )

    async def _ensure_chain_experience_schema(self, conn: aiosqlite.Connection) -> None:
        """将旧版单槽位 chain_experiences 迁移为多变体结构。"""
        cursor = await conn.execute("PRAGMA table_info(chain_experiences)")
        rows = await cursor.fetchall()
        existing_columns = {row[1]: row for row in rows}
        if not existing_columns:
            return

        # 旧版全局库曾把 chain_signature 设为 PRIMARY KEY，这会堵死 append。
        needs_recreate = (
            "id" not in existing_columns
            or "chain_signature" not in existing_columns
            or "experience_json" not in existing_columns
            or "user_id" not in existing_columns
            or "subkey" not in existing_columns
            or "keyword_embedding_json" not in existing_columns
            or bool(existing_columns["chain_signature"][5])
        )

        index_cursor = await conn.execute("PRAGMA index_list(chain_experiences)")
        index_rows = await index_cursor.fetchall()
        desired_unique = False
        for idx in index_rows:
            idx_name = idx[1]
            is_unique = bool(idx[2])
            if not is_unique:
                continue
            info_cursor = await conn.execute(f"PRAGMA index_info({idx_name!r})")
            info_rows = await info_cursor.fetchall()
            idx_columns = [row[2] for row in info_rows]
            if idx_columns == ["chain_signature", "subkey"]:
                desired_unique = True
                break

        if not needs_recreate:
            if "raw_chain_count" not in existing_columns:
                await conn.execute(
                    "ALTER TABLE chain_experiences "
                    "ADD COLUMN raw_chain_count INTEGER DEFAULT 0"
                )
            if "last_updated" not in existing_columns:
                await conn.execute(
                    "ALTER TABLE chain_experiences "
                    "ADD COLUMN last_updated TEXT DEFAULT ''"
                )
            if "subkey" not in existing_columns:
                await conn.execute(
                    "ALTER TABLE chain_experiences "
                    "ADD COLUMN subkey TEXT DEFAULT ''"
                )
            if "keyword_embedding_json" not in existing_columns:
                await conn.execute(
                    "ALTER TABLE chain_experiences "
                    "ADD COLUMN keyword_embedding_json TEXT DEFAULT ''"
                )

            if not desired_unique:
                await conn.execute("DROP INDEX IF EXISTS idx_chain_exp_unique")
                await conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_chain_exp_unique "
                    "ON chain_experiences(chain_signature, subkey)"
                )

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chain_exp_signature "
                "ON chain_experiences(chain_signature)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chain_exp_user "
                "ON chain_experiences(user_id)"
            )
            return

        await conn.execute("DROP TABLE IF EXISTS chain_experiences_legacy")
        await conn.execute("ALTER TABLE chain_experiences RENAME TO chain_experiences_legacy")
        await conn.execute(
            """
            CREATE TABLE chain_experiences (
                id TEXT PRIMARY KEY,
                chain_signature TEXT NOT NULL,
                user_id TEXT NOT NULL,
                experience_json TEXT NOT NULL,
                raw_chain_count INTEGER DEFAULT 0,
                last_updated TEXT NOT NULL,
                subkey TEXT DEFAULT '',
                keyword_embedding_json TEXT DEFAULT ''
            )
            """
        )

        legacy_cursor = await conn.execute("SELECT * FROM chain_experiences_legacy")
        legacy_rows = await legacy_cursor.fetchall()
        seen_pairs: set[tuple[str, str]] = set()

        for row in legacy_rows:
            row_dict = dict(row)
            exp_json = row_dict.get("experience_json", "") or "{}"
            try:
                exp_data = json.loads(exp_json)
            except (TypeError, json.JSONDecodeError):
                exp_data = {}

            chain_signature = (
                row_dict.get("chain_signature")
                or exp_data.get("chain_name")
                or ""
            )
            if not chain_signature:
                continue

            subkey = row_dict.get("subkey") or exp_data.get("subkey") or uuid.uuid4().hex[:8]
            pair = (chain_signature, subkey)
            while pair in seen_pairs:
                subkey = uuid.uuid4().hex[:8]
                pair = (chain_signature, subkey)
            seen_pairs.add(pair)

            exp_data.setdefault("chain_name", chain_signature)
            exp_data["subkey"] = subkey
            source_chain_ids = exp_data.get("source_chain_ids") or []
            support_count = exp_data.get("support_count") or len(set(source_chain_ids)) or 1
            exp_data["support_count"] = max(1, int(support_count))
            exp_data["merge_count"] = max(
                int(exp_data.get("merge_count") or 0),
                exp_data["support_count"] - 1,
            )

            user_id = row_dict.get("user_id") or ""
            source_users = exp_data.get("source_users") or []
            if user_id and user_id not in source_users:
                source_users = [*source_users, user_id]
            exp_data["source_users"] = source_users
            exp_data.setdefault("source_job_ids", [])
            exp_data.setdefault("inject_summary", "")

            row_id = row_dict.get("id") or uuid.uuid4().hex
            raw_chain_count = int(row_dict.get("raw_chain_count") or 0)
            last_updated = (
                row_dict.get("last_updated")
                or exp_data.get("timestamp")
                or datetime.now().isoformat()
            )
            keyword_embedding_json = row_dict.get("keyword_embedding_json") or ""

            await conn.execute(
                """
                INSERT INTO chain_experiences (
                    id, chain_signature, user_id, experience_json,
                    raw_chain_count, last_updated, subkey, keyword_embedding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row_id,
                    chain_signature,
                    user_id,
                    json.dumps(exp_data, ensure_ascii=False),
                    raw_chain_count,
                    last_updated,
                    subkey,
                    keyword_embedding_json,
                ),
            )

        await conn.execute("DROP TABLE chain_experiences_legacy")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chain_exp_signature "
            "ON chain_experiences(chain_signature)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chain_exp_user "
            "ON chain_experiences(user_id)"
        )
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chain_exp_unique "
            "ON chain_experiences(chain_signature, subkey)"
        )

    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        conn = self._ensure_connected()
        await conn.execute("BEGIN")
        try:
            yield
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
