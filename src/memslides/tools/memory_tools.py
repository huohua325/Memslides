from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from memslides.memory.server.server import create_mcp_app
from memslides.tools.memory_service import (
    _resolve_global_db_dir,
    _resolve_memory_db_path,
)
from memslides.utils.config import MemSlidesConfig
from memslides.utils.log import set_logger


def _load_config() -> Any | None:
    try:
        return MemSlidesConfig.load_from_file(os.getenv("MEMSLIDES_CONFIG_FILE"))
    except Exception:
        return None


def _configure_memory_environment(work_dir: Path, cfg: Any | None) -> None:
    memory_v2 = bool(getattr(getattr(cfg, "memory", None), "memory_v2", False))

    resolved_global_db_dir = _resolve_global_db_dir(
        os.environ.get("MEMSLIDES_MEMORY_GLOBAL_DB_DIR", ""),
        cfg=cfg,
    )
    if resolved_global_db_dir is not None:
        os.environ["MEMSLIDES_MEMORY_GLOBAL_DB_DIR"] = str(resolved_global_db_dir)
    os.environ["MEMSLIDES_MEMORY_DB_PATH"] = str(_resolve_memory_db_path(work_dir, cfg=cfg))
    os.environ["MEMSLIDES_MEMORY_V2"] = "1" if memory_v2 else "0"

    try:
        if cfg and cfg.memory and cfg.memory.enabled and cfg.memory.embedding:
            emb = cfg.memory.embedding
            os.environ.setdefault("MEMSLIDES_EMBEDDING_PROVIDER", getattr(emb, "provider", "") or "openai-compatible")
            os.environ.setdefault("MEMSLIDES_EMBEDDING_MODEL", emb.model or "BAAI/bge-m3")
            os.environ.setdefault("MEMSLIDES_EMBEDDING_DIM", str(getattr(emb, "dim", 1024)))
            if getattr(emb, "api_model", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_API_MODEL", emb.api_model)
            if getattr(emb, "api_fallback_model", ""):
                os.environ.setdefault("MEMSLIDES_EMBEDDING_FALLBACK_API_MODEL", emb.api_fallback_model)
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
        pass


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        raise SystemExit("Usage: python -m memslides.tools.memory_tools <workspace>")

    work_dir = Path(args[0])
    if not work_dir.exists():
        raise FileNotFoundError(f"Workspace {work_dir} does not exist.")
    os.chdir(work_dir)
    set_logger(
        f"memslides-memory-tools-{work_dir.stem}",
        work_dir / ".history" / "memslides_memory_tools.log",
    )

    cfg = _load_config()
    _configure_memory_environment(work_dir, cfg)
    create_mcp_app().run(show_banner=False)


if __name__ == "__main__":
    main()
