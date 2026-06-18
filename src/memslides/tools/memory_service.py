"""Memory MCP Server launcher — exposes memory tools to agents.

Tools:
    1. search_rules       — multi-strategy rule retrieval
    2. save_rule           — save rule with deduplication
    3. query_memory        — unified search across rules/facts/foresights
    4. get_modification_history — parameter change history
    5. search_experiences  — experience trace search
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure direct script execution imports the local repo package first.
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from memslides.memory.server.server import create_mcp_app
from memslides.utils.config import MemSlidesConfig
from memslides.utils.log import set_logger


def _resolve_global_db_dir(
    env_global_db_dir: str,
    cfg: Any | None = None,
) -> Path | None:
    """Resolve the global memory directory from env first, then config."""
    raw_global_db_dir = (env_global_db_dir or "").strip()
    if not raw_global_db_dir and cfg is not None:
        memory_cfg = getattr(cfg, "memory", None)
        raw_global_db_dir = getattr(memory_cfg, "global_db_dir", "") or ""
    if not raw_global_db_dir:
        return None
    return Path(raw_global_db_dir).expanduser()


def _resolve_memory_db_path(work_dir: Path, cfg: Any | None = None) -> Path:
    """Resolve the DB path without creating a per-session fallback unnecessarily."""
    raw_db_path = os.environ.get("MEMSLIDES_MEMORY_DB_PATH", "").strip()
    if raw_db_path:
        return Path(raw_db_path).expanduser()

    global_db_dir = _resolve_global_db_dir(
        os.environ.get("MEMSLIDES_MEMORY_GLOBAL_DB_DIR", ""),
        cfg=cfg,
    )
    memory_cfg = getattr(cfg, "memory", None)
    memory_v2 = bool(memory_cfg and getattr(memory_cfg, "memory_v2", False))

    if global_db_dir is not None:
        db_name = "global_memory_v2.db" if memory_v2 else "global_memory.db"
        return global_db_dir / db_name

    return work_dir / ".memory" / "memory.db"


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python -m memslides.tools.memory_tools <workspace>"
    work_dir = Path(sys.argv[1])
    assert work_dir.exists(), f"Workspace {work_dir} does not exist."
    os.chdir(work_dir)
    set_logger(
        f"memslides-memory-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_memory_tools.log",
    )

    # Set environment variables for the memory server's lazy init
    memory_v2 = False
    cfg = None
    try:
        cfg = MemSlidesConfig.load_from_file(os.getenv("MEMSLIDES_CONFIG_FILE"))
        memory_v2 = cfg.memory and cfg.memory.memory_v2
    except Exception:
        pass

    resolved_global_db_dir = _resolve_global_db_dir(
        os.environ.get("MEMSLIDES_MEMORY_GLOBAL_DB_DIR", ""),
        cfg=cfg,
    )
    if resolved_global_db_dir is not None:
        os.environ["MEMSLIDES_MEMORY_GLOBAL_DB_DIR"] = str(resolved_global_db_dir)
    os.environ["MEMSLIDES_MEMORY_DB_PATH"] = str(_resolve_memory_db_path(work_dir, cfg=cfg))
    os.environ["MEMSLIDES_MEMORY_V2"] = "1" if memory_v2 else "0"

    # Pass embedding config from MemSlides config to env vars for server.py's lazy init
    try:
        if cfg is None:
            cfg = MemSlidesConfig.load_from_file(os.getenv("MEMSLIDES_CONFIG_FILE"))
        if cfg.memory and cfg.memory.enabled and cfg.memory.embedding:
            emb = cfg.memory.embedding
            os.environ.setdefault("MEMSLIDES_EMBEDDING_PROVIDER", getattr(emb, "provider", "") or "openai-compatible")
            os.environ.setdefault("MEMSLIDES_EMBEDDING_MODEL", emb.model or "BAAI/bge-m3")
            os.environ.setdefault("MEMSLIDES_EMBEDDING_DIM", str(getattr(emb, "dim", 1024)))
            if getattr(emb, "api_model", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_API_MODEL", emb.api_model)
            if getattr(emb, "api_fallback_model", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_FALLBACK_API_MODEL", emb.api_fallback_model)
            # OpenAI provider settings
            if getattr(emb, "api_key", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_API_KEY", emb.api_key)
                os.environ.setdefault("MEMSLIDES_OPENAI_API_KEY", emb.api_key)
            if getattr(emb, "api_fallback_api_key", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_FALLBACK_API_KEY", emb.api_fallback_api_key)
            if getattr(emb, "api_base_url", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_API_BASE_URL", emb.api_base_url)
            if getattr(emb, "api_fallback_base_url", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_FALLBACK_API_BASE_URL", emb.api_fallback_base_url)
            if getattr(emb, "base_url", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_BASE_URL", emb.base_url)
                os.environ.setdefault("MEMSLIDES_OPENAI_BASE_URL", emb.base_url)
    except Exception:
        pass  # Non-fatal: server will use defaults or env vars

    mcp = create_mcp_app()
    mcp.run(show_banner=False)
