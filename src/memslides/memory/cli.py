"""Command-line utilities for consolidating and inspecting memory."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path.home() / ".cache" / "memslides" / ".memory"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "global_memory.db"
DEFAULT_DB_PATH_V2 = DEFAULT_DB_DIR / "global_memory_v2.db"


async def consolidate(db_path: Path, user_id: str = "default") -> None:
    """Consolidate memory records in a standalone process."""
    from .core.db import SQLiteBackend
    from .evolution.memory_consolidator import OfflineMemoryConsolidator

    logger.info("Starting consolidation...")
    logger.info("Database: %s", db_path)
    logger.info("User ID: %s", user_id)

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return

    db = SQLiteBackend(db_path)
    await db.connect()
    try:
        consolidator = OfflineMemoryConsolidator(
            db=db,
            llm=None,
            embedding_func=None,
        )
        stats = await consolidator.consolidate(user_id)

        logger.info("=" * 50)
        logger.info("Consolidation completed!")
        logger.info(
            "ExperienceTrace: %d -> %d (removed %d duplicates)",
            stats.total_before,
            stats.total_after,
            stats.duplicates_removed,
        )
        logger.info(
            "Episodes: %d -> %d (removed %d duplicates)",
            stats.episodes_before,
            stats.episodes.get("after", 0),
            stats.episodes_duplicates,
        )
        logger.info(
            "Preferences: %d -> %d (removed %d duplicates)",
            stats.preferences_before,
            stats.preferences.get("after", 0),
            stats.preferences_duplicates,
        )
        if stats.errors:
            logger.warning("Errors: %s", stats.errors)
        logger.info("=" * 50)
    finally:
        await db.close()


async def status(db_path: Path) -> None:
    """Report record counts for a memory database."""
    from .core.db import SQLiteBackend

    logger.info("Database: %s", db_path)
    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        return

    db = SQLiteBackend(db_path)
    await db.connect()
    try:
        exp_count = await db.query_one("SELECT COUNT(*) as cnt FROM experience_traces")
        ep_count = await db.query_one("SELECT COUNT(*) as cnt FROM design_episodes")
        pref_count = await db.query_one("SELECT COUNT(*) as cnt FROM atomic_preferences")

        logger.info("=" * 50)
        logger.info("Memory status")
        logger.info("ExperienceTrace: %d", exp_count["cnt"] if exp_count else 0)
        logger.info("DesignEpisode: %d", ep_count["cnt"] if ep_count else 0)
        logger.info("AtomicPreference: %d", pref_count["cnt"] if pref_count else 0)
        logger.info("=" * 50)
    finally:
        await db.close()


def main() -> None:
    """Parse arguments and run the selected memory operation."""
    parser = argparse.ArgumentParser(description="Inspect and consolidate MemSlides memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    consolidate_parser = subparsers.add_parser(
        "consolidate",
        help="Consolidate duplicate memory records",
    )
    consolidate_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Database path (selected automatically with --v2 when omitted)",
    )
    consolidate_parser.add_argument("--user-id", default="default", help="User ID")
    consolidate_parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the memory_v2 database (global_memory_v2.db)",
    )

    status_parser = subparsers.add_parser(
        "status",
        help="Display memory record counts",
    )
    status_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Database path (selected automatically with --v2 when omitted)",
    )
    status_parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the memory_v2 database (global_memory_v2.db)",
    )

    args = parser.parse_args()
    if args.db_path is None:
        args.db_path = DEFAULT_DB_PATH_V2 if args.v2 else DEFAULT_DB_PATH

    if args.command == "consolidate":
        asyncio.run(consolidate(args.db_path, args.user_id))
    elif args.command == "status":
        asyncio.run(status(args.db_path))


if __name__ == "__main__":
    main()
