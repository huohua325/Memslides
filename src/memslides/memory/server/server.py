"""Memory MCP Server — 独立FastMCP进程

启动方式：
    fastmcp dev memslides/memory/server.py
    或通过mcp.json配置自动启动

暴露工具：
    1. search_experiences — 检索经验轨迹
    2. search_episodes    — 检索设计情景 (Stage 2)
    3. remember_lesson    — 记录重要经验到经验库
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from ..core.db import SQLiteBackend
from ..core.embedding import get_embedding_func
from ..core.models import ExperienceTrace, infer_task_experience_category

try:
    from ..store.episode_store import EpisodeStore
    from ..experience_writer import ExperienceTraceWriter
    STAGE2_AVAILABLE = True
except ImportError:
    STAGE2_AVAILABLE = False

logger = logging.getLogger(__name__)

# 全局实例（惰性初始化）
_db: SQLiteBackend | None = None

# Stage 2 stores
_episode_store = None
_exp_writer = None


def _resolve_db_path() -> Path:
    """Resolve and normalize the memory DB path from environment variables."""
    raw_db_path = os.environ.get("MEMSLIDES_MEMORY_DB_PATH", "").strip()
    if raw_db_path:
        return Path(raw_db_path).expanduser()

    raw_global_db_dir = os.environ.get("MEMSLIDES_MEMORY_GLOBAL_DB_DIR", "").strip()
    if raw_global_db_dir:
        memory_v2 = os.environ.get("MEMSLIDES_MEMORY_V2", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        db_name = "global_memory_v2.db" if memory_v2 else "global_memory.db"
        return Path(raw_global_db_dir).expanduser() / db_name

    return Path(".memory/memory.db")


async def _ensure_initialized():
    """惰性初始化数据库和存储"""
    global _db, _episode_store, _exp_writer
    if _db is None:
        db_path = _resolve_db_path()
        _db = SQLiteBackend(db_path)
        await _db.connect()
        await _db.init_schema()

        if STAGE2_AVAILABLE:
            embedding_func = get_embedding_func(
                provider=os.environ.get("MEMSLIDES_EMBEDDING_PROVIDER", "openai-compatible"),
                model=os.environ.get("MEMSLIDES_EMBEDDING_MODEL", "BAAI/bge-m3"),
                model_name=os.environ.get("MEMSLIDES_EMBEDDING_MODEL", "BAAI/bge-m3"),
                api_model=(
                    os.environ.get("MEMSLIDES_EMBEDDING_API_MODEL")
                    or os.environ.get("MEMSLIDES_EMBEDDING_MODEL", "BAAI/bge-m3")
                ),
                api_key=(
                    os.environ.get("MEMSLIDES_EMBEDDING_API_KEY")
                    or os.environ.get("SILICONFLOW_API_KEY")
                    or os.environ.get("MEMSLIDES_OPENAI_API_KEY")
                ),
                api_base_url=(
                    os.environ.get("MEMSLIDES_EMBEDDING_API_BASE_URL")
                    or os.environ.get("MEMSLIDES_EMBEDDING_BASE_URL")
                    or os.environ.get("SILICONFLOW_BASE_URL")
                    or os.environ.get("MEMSLIDES_OPENAI_BASE_URL")
                ),
                base_url=(
                    os.environ.get("MEMSLIDES_EMBEDDING_BASE_URL")
                    or os.environ.get("SILICONFLOW_BASE_URL")
                    or os.environ.get("MEMSLIDES_OPENAI_BASE_URL")
                ),
                api_fallback_model=(
                    os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_MODEL")
                    or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_MODEL")
                ),
                api_fallback_base_url=(
                    os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_BASE_URL")
                    or os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_BASE_URL")
                ),
                api_fallback_api_key=(
                    os.environ.get("MEMSLIDES_EMBEDDING_FALLBACK_API_KEY")
                    or os.environ.get("OPENROUTER_API_KEY")
                ),
            )
            _episode_store = EpisodeStore(_db)
            _exp_writer = ExperienceTraceWriter(db=_db)
            logger.info("MemSlides memory MCP server initialized (Stage 2): %s", db_path)
        else:
            logger.info("MemSlides memory MCP server initialized: %s", db_path)


async def _get_db() -> SQLiteBackend:
    await _ensure_initialized()
    assert _db is not None
    return _db


async def _get_episode_store():
    await _ensure_initialized()
    assert _episode_store is not None, "Stage 2 not available"
    return _episode_store


async def _get_exp_writer():
    """获取 ExperienceTraceWriter"""
    await _ensure_initialized()
    assert _exp_writer is not None, "Stage 2 not available"
    return _exp_writer


# ── Tool implementations (decoupled from FastMCP for testability) ──


async def tool_search_experiences(
    session_id: str = "",
    task_description: str = "",
    scenarios: str = "",
    outcome: str = "",
    limit: int = 10,
) -> str:
    """检索经验轨迹"""
    db = await _get_db()

    where_clauses = []
    params: list = []

    if session_id:
        where_clauses.append("session_id = ?")
        params.append(session_id)
    if outcome:
        where_clauses.append("final_outcome = ?")
        params.append(outcome)
    if scenarios:
        for tag in scenarios.split(","):
            tag = tag.strip()
            if tag:
                where_clauses.append("applicable_scenarios LIKE ?")
                params.append(f"%{tag}%")
    if task_description:
        where_clauses.append("task_description LIKE ?")
        params.append(f"%{task_description}%")

    where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
    query_sql = (
        "SELECT * FROM experience_traces "
        f"WHERE COALESCE(status, 'active') = 'active' AND {where_sql} "
        "ORDER BY reuse_count DESC, created_at DESC LIMIT ?"
    )
    params.append(limit)

    rows = await db.query(query_sql, tuple(params))
    traces = [ExperienceTrace.from_dict(row) for row in rows]

    return json.dumps(
        [
            {
                "id": t.id,
                "task_description": t.task_description,
                "reasoning_steps": json.loads(t.reasoning_steps)
                if isinstance(t.reasoning_steps, str)
                else t.reasoning_steps,
                "tools_used": json.loads(t.tools_used)
                if isinstance(t.tools_used, str)
                else t.tools_used,
                "final_outcome": t.final_outcome,
                "lessons_learned": t.lessons_learned,
                "applicable_scenarios": json.loads(t.applicable_scenarios)
                if isinstance(t.applicable_scenarios, str)
                else t.applicable_scenarios,
                "confidence": t.confidence,
                "reuse_count": t.reuse_count,
            }
            for t in traces
        ],
        ensure_ascii=False,
        indent=2,
    )


# ── Stage 2 Tool Implementations ──


async def tool_search_episodes(user_id: str, query: str, limit: int = 10) -> str:
    """检索设计情景（Stage 2 EpisodeStore）"""
    if not STAGE2_AVAILABLE:
        return json.dumps({"error": "Stage 2 not available"}, ensure_ascii=False)

    episode_store = await _get_episode_store()
    episodes = await episode_store.search_similar(query, limit, user_id)
    return json.dumps(
        [
            {
                "id": e.id,
                "user_intent": e.user_intent,
                "design_insight": e.design_insight,
                "action_outcome": e.action_outcome,
                "category": e.category,
                "created_at": e.created_at,
            }
            for e in episodes
        ],
        ensure_ascii=False,
        indent=2,
    )


async def tool_remember_lesson(
    content: str,
    tool_name: str = "",
    keywords: list[str] | None = None,
    category: str = "tool_error",
    session_id: str = "",
) -> str:
    """记录一条重要经验到经验库 (Stage 14 重构)

    Agent 在推理过程中主动调用，记录发现的可复用教训。
    MCP 工具仅返回确认 JSON，不直接操作 DB。
    实际写入由 agent.py 的 _tool_result_callback → orchestrator.on_remember_lesson() → WM 完成。
    LTM 持久化统一由 Job 结束时 _consolidate_experiences_to_ltm() 处理。

    Stage 14: 移除 from_tool_error() hack，remember_lesson 不再双写 LTM。
    """
    normalized_category = infer_task_experience_category(
        category,
        keywords,
        "",
        content,
    )

    # 自动生成 keywords（如果未提供）
    if not keywords:
        keywords = _auto_generate_keywords(content, tool_name)

    return json.dumps(
        {
            "status": "recorded",
            "content": content,
            "tool_name": tool_name,
            "keywords": keywords,
            "category": normalized_category,
        },
        ensure_ascii=False,
    )


def _auto_generate_keywords(content: str, tool_name: str) -> list[str]:
    """从 content 和 tool_name 自动提取关键词"""
    keywords = []
    if tool_name:
        keywords.append(tool_name)

    # 常见模式词
    patterns = [
        "overflow", "layout", "spacing", "font", "color", "image",
        "text", "title", "card", "grid", "flex", "margin", "padding",
        "error", "failed", "limit", "constraint", "avoid", "must",
        "溢出", "布局", "间距", "字体", "颜色", "图片", "文本", "标题",
    ]
    content_lower = content.lower()
    for p in patterns:
        if p in content_lower:
            keywords.append(p)

    return keywords[:10]  # 限制数量


def reset_global_state():
    """Reset global state (for testing)"""
    global _db, _episode_store, _exp_writer
    _db = None
    _episode_store = None
    _exp_writer = None


def create_mcp_app():
    """Create FastMCP application with all tools registered.

    Usage:
        mcp = create_mcp_app()
        mcp.run()
    """
    try:
        from fastmcp import FastMCP
    except ImportError:
        raise ImportError("fastmcp is required: pip install fastmcp")

    mcp = FastMCP("MemSlidesMemoryTools")

    @mcp.tool()
    async def search_experiences(
        session_id: str = "",
        task_description: str = "",
        scenarios: str = "",
        outcome: str = "",
        limit: int = 10,
    ) -> str:
        """检索经验轨迹"""
        return await tool_search_experiences(
            session_id, task_description, scenarios, outcome, limit
        )

    # ── Stage 2 Tools ──

    @mcp.tool()
    async def search_episodes(user_id: str, query: str, limit: int = 10) -> str:
        """检索设计情景（Stage 2）"""
        return await tool_search_episodes(user_id, query, limit)

    @mcp.tool()
    async def remember_lesson(
        content: str,
        tool_name: str = "",
        keywords: list[str] | None = None,
        category: str = "tool_error",
        session_id: str = "",
    ) -> str:
        """记录一条重要经验到经验库。当你发现某个可复用的工具教训或成功模式时调用此工具。

        Args:
            content: 教训内容
            tool_name: 相关工具名（可选）
            keywords: 检索关键词列表（可选，用于后续向量检索）
            category: 分类标签，只能是以下之一：
                - "tool_error": 直接工具失败、输入错误、前置条件不满足
                - "tool_misuse": 工具选错/顺序错/角色误用导致软失败
                - "tool_limitation": 工具盲区、能力边界、观测缺口
                - "pattern": 成功且可复用的工作流或最佳实践
            session_id: 会话ID（可选）
        """
        return await tool_remember_lesson(content, tool_name, keywords, category, session_id)

    return mcp


if __name__ == "__main__":
    app = create_mcp_app()
    app.run()
