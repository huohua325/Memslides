"""
Event Bus — 轻量异步事件总线 (Stage 2)

基于 memory_upgrade_design_v2.md §3.3 定义。
解耦组件通信：RollbackManager、Episode 提取等通过事件交互。
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class MemoryEvent:
    """预定义事件类型常量"""

    ROLLBACK_EXECUTED = "rollback_executed"
    EPISODE_EXTRACTED = "episode_extracted"
    PROFILE_UPDATED = "profile_updated"
    RULE_INDUCED = "rule_induced"
    TOOL_ERROR = "tool_error"
    TOOL_RESULT = "tool_result"
    FORESIGHT_ACCEPTED = "foresight_accepted"
    FORESIGHT_IGNORED = "foresight_ignored"


class MemoryEventBus:
    """轻量异步事件总线

    - 并行执行所有 handler（asyncio.gather）
    - 单个 handler 失败不影响其他（try/except 隔离）
    - 日志记录发布和错误
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """注册事件处理器"""
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", getattr(handler, "__qualname__", handler), event_type)

    def unsubscribe(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """移除事件处理器"""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """发布事件，并行执行所有 handler"""
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            return
        logger.debug("Publishing %s to %d handlers", event_type, len(handlers))
        results = await asyncio.gather(
            *[self._safe_call(h, event_type, data) for h in handlers],
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Event handler error for %s: %s", event_type, r)

    async def _safe_call(
        self,
        handler: Callable[..., Awaitable[None]],
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """隔离执行单个 handler"""
        try:
            await handler(data)
        except Exception as e:
            logger.warning("Event %s handler %s failed: %s", event_type,
                           getattr(handler, "__qualname__", handler), e)
            raise

    @property
    def handler_count(self) -> int:
        """已注册的 handler 总数"""
        return sum(len(hs) for hs in self._handlers.values())

    def get_handler_count(self, event_type: str) -> int:
        """某事件类型的 handler 数量"""
        return len(self._handlers.get(event_type, []))
