"""Memory System Prompt Registry — 集中管理所有 LLM Prompt

按功能分组:
- extraction: 意图分类、Episode提取、工具约束学习、用户消息分析、原子偏好提取
- experience: 经验轨迹提取、工具错误合并
- retrieval: 多查询生成

Usage:
    from memslides.memory.prompts import (
        EPISODE_EXTRACTION_PROMPT,
        TURN_EXPERIENCE_PROMPT,
        ...
    )
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# Extraction Prompts
# ══════════════════════════════════════════════════════════════════════════════

from .extraction import (
    build_intent_classification_prompt,
    EPISODE_EXTRACTION_PROMPT,
    ATOMIC_PREFERENCE_EXTRACTION_PROMPT,
)

# ══════════════════════════════════════════════════════════════════════════════
# Experience Prompts
# ══════════════════════════════════════════════════════════════════════════════

from .experience import (
    TURN_EXPERIENCE_PROMPT,
    MERGE_TOOL_ERROR_PROMPT,
)

# ══════════════════════════════════════════════════════════════════════════════
# Retrieval Prompts
# ══════════════════════════════════════════════════════════════════════════════

from .retrieval import (
    MULTI_QUERY_PROMPT,
)

__all__ = [
    # Extraction
    "build_intent_classification_prompt",
    "EPISODE_EXTRACTION_PROMPT",
    "ATOMIC_PREFERENCE_EXTRACTION_PROMPT",
    # Experience
    "TURN_EXPERIENCE_PROMPT",
    "MERGE_TOOL_ERROR_PROMPT",
    # Retrieval
    "MULTI_QUERY_PROMPT",
]
