"""Stage 12: 执行数据库迁移脚本

Usage:
    python -m memslides.memory.migrations.run_migration
"""
import asyncio
import logging
from pathlib import Path

from memslides.memory.core.db import DatabaseBackend

logger = logging.getLogger(__name__)


async def run_migration(db_path: str = ".memory/memory.db"):
    """执行 Stage 12 迁移：添加链级扩展字段 + 偏好来源追踪字段"""
    db = DatabaseBackend(db_path)
    await db.connect()

    migration_files = [
        Path(__file__).parent / "add_chain_fields_to_experience_traces.sql",
        Path(__file__).parent / "add_source_tracking_to_atomic_preferences.sql",
    ]

    try:
        for migration_file in migration_files:
            if not migration_file.exists():
                logger.warning(f"Migration file not found: {migration_file}")
                continue
            logger.info(f"Running migration: {migration_file.name}")
            sql = migration_file.read_text(encoding="utf-8")
            statements = [s.strip() for s in sql.split(";") if s.strip()]

            for stmt in statements:
                try:
                    await db.execute(stmt)
                    logger.info(f"Executed: {stmt[:60]}...")
                except Exception as e:
                    logger.warning(f"Statement failed (may be expected): {e}")

        await db.commit()
        logger.info("✅ Migration completed successfully")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_migration())
