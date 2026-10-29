"""Implementation details."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class MemoryEvent:
    """Implementation details."""

    ROLLBACK_EXECUTED = "rollback_executed"
    EPISODE_EXTRACTED = "episode_extracted"
    PROFILE_UPDATED = "profile_updated"
    RULE_INDUCED = "rule_induced"
    TOOL_ERROR = "tool_error"
    TOOL_RESULT = "tool_result"
    FORESIGHT_ACCEPTED = "foresight_accepted"
    FORESIGHT_IGNORED = "foresight_ignored"


class MemoryEventBus:
    """Implementation details."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[..., Awaitable[None]]]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Implementation details."""
        self._handlers[event_type].append(handler)
        logger.debug("Subscribed %s to %s", getattr(handler, "__qualname__", handler), event_type)

    def unsubscribe(self, event_type: str, handler: Callable[..., Awaitable[None]]) -> None:
        """Implementation details."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Implementation details."""
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
        """Implementation details."""
        try:
            await handler(data)
        except Exception as e:
            logger.warning("Event %s handler %s failed: %s", event_type,
                           getattr(handler, "__qualname__", handler), e)
            raise

    @property
    def handler_count(self) -> int:
        """Implementation details."""
        return sum(len(hs) for hs in self._handlers.values())

    def get_handler_count(self, event_type: str) -> int:
        """Implementation details."""
        return len(self._handlers.get(event_type, []))
