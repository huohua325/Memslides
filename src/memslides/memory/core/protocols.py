"""
Protocol Layer — 所有记忆组件的接口契约 (Stage 2)

纯接口定义，零实现代码。不 import 任何具体类（SQLite, ChromaDB 等）。

Protocol 组:
1. 组件生命周期 (Startable)
2. 记忆存储 (Episode/Preference Store)
3. 提取器 (Extractor/EpisodeExtractor)
4. 检索器 (MemoryRetriever/RetrievalStrategy)
5. 事件总线 (EventBus)
6. 中间件 (Middleware)
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

try:
    from .models import (
        DesignEpisode,
        AtomicPreference,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        DesignEpisode,
        AtomicPreference,
    )


# ═══════════════════════════════════════════════
# 1. 组件生命周期
# ═══════════════════════════════════════════════

@runtime_checkable
class Startable(Protocol):
    """可启动/停止的组件"""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def health_check(self) -> dict[str, Any]: ...


# ═══════════════════════════════════════════════
# 2. 五类记忆的存储协议
# ═══════════════════════════════════════════════

@runtime_checkable
class EpisodeStoreProtocol(Protocol):
    """情景记忆存储"""

    async def add_episode(self, episode: DesignEpisode) -> str: ...

    async def search_similar(self, query: str, limit: int = 5) -> list[DesignEpisode]: ...

    async def get_by_session(self, session_id: str) -> list[DesignEpisode]: ...

    async def get_active(self, user_id: str) -> list[DesignEpisode]: ...

    async def update_status(self, episode_id: str, status: str) -> None: ...

    async def archive(self, episode_id: str) -> None: ...

    async def list_episodes(self, user_id: str, limit: int = 100) -> list: ...


@runtime_checkable
class AtomicPreferenceStoreProtocol(Protocol):
    """原子级用户偏好存储 (Stage 4)"""

    async def add(self, preference: AtomicPreference) -> AtomicPreference: ...

    async def search(self, query: str, user_id: str, limit: int = 10, **kwargs: Any) -> list[AtomicPreference]: ...

    async def get_for_context(self, user_id: str, context: dict, limit: int = 10) -> list[AtomicPreference]: ...

    async def get_for_template(self, user_id: str, template_id: str, limit: int = 5) -> list[AtomicPreference]: ...

    async def get_active(self, user_id: str, limit: int = 100) -> list[AtomicPreference]: ...

    async def verify(self, preference_id: str) -> None: ...

    async def contradict(self, preference_id: str) -> None: ...

    async def deprecate(self, preference_id: str) -> None: ...


# ═══════════════════════════════════════════════
# 3. 提取器协议
# ═══════════════════════════════════════════════

@runtime_checkable
class ExtractorProtocol(Protocol):
    """通用提取器（Template Method 基类）"""

    async def extract(self, input_data: Any, **kwargs: Any) -> list: ...

    def should_extract(self, input_data: Any) -> bool: ...


@runtime_checkable
class EpisodeExtractorProtocol(Protocol):
    """情景记忆提取器"""

    async def extract(
        self,
        batch_text: str,
        satisfaction_signals: list | None = None,
    ) -> list[DesignEpisode]: ...

    def should_extract(self, input_data: Any) -> bool: ...


# ═══════════════════════════════════════════════
# 4. 检索器协议
# ═══════════════════════════════════════════════

@runtime_checkable
class MemoryRetrieverProtocol(Protocol):
    """记忆检索器"""

    async def retrieve(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list: ...


@runtime_checkable
class RetrievalStrategy(Protocol):
    """检索策略（Strategy Pattern）"""

    async def search(self, query: str, top_k: int) -> list: ...

    @property
    def name(self) -> str: ...


# ═══════════════════════════════════════════════
# 5. 事件总线协议
# ═══════════════════════════════════════════════

@runtime_checkable
class EventBusProtocol(Protocol):
    """异步事件总线"""

    def subscribe(
        self,
        event_type: str,
        handler: Callable[..., Awaitable[None]],
    ) -> None: ...

    async def publish(
        self,
        event_type: str,
        data: dict[str, Any],
    ) -> None: ...


# ═══════════════════════════════════════════════
# 6. 中间件协议
# ═══════════════════════════════════════════════

@runtime_checkable
class MiddlewareProtocol(Protocol):
    """中间件（横切关注点：日志/指标/重试）"""

    async def __call__(self, context: dict, next_fn: Callable) -> Any: ...


# ═══════════════════════════════════════════════
# 便捷导出
# ═══════════════════════════════════════════════

__all__ = [
    # 生命周期
    "Startable",
    # 存储
    "EpisodeStoreProtocol",
    "AtomicPreferenceStoreProtocol",
    # 提取器
    "ExtractorProtocol",
    "EpisodeExtractorProtocol",
    # 检索器
    "MemoryRetrieverProtocol",
    "RetrievalStrategy",
    # 事件总线
    "EventBusProtocol",
    # 中间件
    "MiddlewareProtocol",
]
