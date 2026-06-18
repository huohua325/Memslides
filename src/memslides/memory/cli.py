"""Memory CLI — 手动执行记忆整合

Usage:
    python -m memslides.memory.cli consolidate [--db-path PATH] [--user-id USER]
    python -m memslides.memory.cli status [--db-path PATH]

Stage 4: 提供独立的整合命令，避免 Session 中的并发问题
"""

import asyncio
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path.home() / ".cache" / "memslides" / ".memory"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "global_memory.db"
DEFAULT_DB_PATH_V2 = DEFAULT_DB_DIR / "global_memory_v2.db"


async def consolidate(db_path: Path, user_id: str = "default"):
    """执行记忆整合（独立进程，无并发问题）"""
    from .core.db import SQLiteBackend
    from .evolution.memory_consolidator import OfflineMemoryConsolidator

    logger.info("Starting consolidation...")
    logger.info("Database: %s", db_path)
    logger.info("User ID: %s", user_id)

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return

    # 连接数据库
    db = SQLiteBackend(db_path)
    await db.connect()

    try:
        # 创建整合器（无 LLM，仅执行精确去重）
        consolidator = OfflineMemoryConsolidator(
            db=db,
            llm=None,  # 不使用 LLM，避免长时间操作
            embedding_func=None,
        )

        # 执行整合
        stats = await consolidator.consolidate(user_id)

        logger.info("=" * 50)
        logger.info("Consolidation completed!")
        logger.info("ExperienceTrace: %d → %d (removed %d duplicates)",
                   stats.total_before, stats.total_after, stats.duplicates_removed)
        logger.info("Episodes: %d → %d (removed %d duplicates)",
                   stats.episodes_before, stats.episodes.get("after", 0), stats.episodes_duplicates)
        logger.info("Preferences: %d → %d (removed %d duplicates)",
                   stats.preferences_before, stats.preferences.get("after", 0), stats.preferences_duplicates)
        if stats.errors:
            logger.warning("Errors: %s", stats.errors)
        logger.info("=" * 50)

    finally:
        await db.close()


async def status(db_path: Path):
    """显示记忆状态"""
    from .core.db import SQLiteBackend

    logger.info("Database: %s", db_path)

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return

    db = SQLiteBackend(db_path)
    await db.connect()

    try:
        # 查询统计
        exp_count = await db.query_one("SELECT COUNT(*) as cnt FROM experience_traces")
        ep_count = await db.query_one("SELECT COUNT(*) as cnt FROM design_episodes")
        pref_count = await db.query_one("SELECT COUNT(*) as cnt FROM atomic_preferences")

        logger.info("=" * 50)
        logger.info("Memory Status")
        logger.info("ExperienceTrace: %d", exp_count["cnt"] if exp_count else 0)
        logger.info("DesignEpisode: %d", ep_count["cnt"] if ep_count else 0)
        logger.info("AtomicPreference: %d", pref_count["cnt"] if pref_count else 0)
        logger.info("=" * 50)

    finally:
        await db.close()


def main():
    parser = argparse.ArgumentParser(description="Memory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # consolidate 命令
    consolidate_parser = subparsers.add_parser("consolidate", help="执行记忆整合")
    consolidate_parser.add_argument("--db-path", type=Path, default=None,
                                   help="数据库路径 (默认根据 --v2 自动选择)")
    consolidate_parser.add_argument("--user-id", default="default", help="用户 ID")
    consolidate_parser.add_argument("--v2", action="store_true",
                                   help="使用 memory_v2 数据库 (global_memory_v2.db)")

    # status 命令
    status_parser = subparsers.add_parser("status", help="显示记忆状态")
    status_parser.add_argument("--db-path", type=Path, default=None,
                              help="数据库路径 (默认根据 --v2 自动选择)")
    status_parser.add_argument("--v2", action="store_true",
                              help="使用 memory_v2 数据库 (global_memory_v2.db)")

    args = parser.parse_args()

    # 如果未显式指定 db-path，根据 --v2 选择默认路径
    if args.db_path is None:
        args.db_path = DEFAULT_DB_PATH_V2 if args.v2 else DEFAULT_DB_PATH

    if args.command == "consolidate":
        asyncio.run(consolidate(args.db_path, args.user_id))
    elif args.command == "status":
        asyncio.run(status(args.db_path))


if __name__ == "__main__":
    main()
