from __future__ import annotations

from pathlib import Path
from typing import Any

from memslides.memory.config_helper import MemorySystem


class MemoryRuntime:
    """Session-scoped memory runtime facade."""

    def __init__(self, system: MemorySystem | None = None):
        self.system = system

    @classmethod
    async def from_config(cls, config: Any, workspace: Path | None = None) -> "MemoryRuntime":
        if not getattr(config, "memory", None) or not config.memory.enabled:
            return cls(None)
        system = await MemorySystem.from_config(config, project_dir=workspace)
        return cls(system)

    @property
    def enabled(self) -> bool:
        return self.system is not None

    @property
    def orchestrator(self) -> Any | None:
        if self.system is None:
            return None
        return getattr(self.system, "orchestrator", None)

    def bind_orchestrator(self, orchestrator: Any) -> None:
        if self.system is not None:
            setattr(self.system, "orchestrator", orchestrator)

    async def on_job_start(self, *args: Any, **kwargs: Any) -> None:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "on_job_start"):
            await orchestrator.on_job_start(*args, **kwargs)

    async def on_job_end(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "on_job_end"):
            result = await orchestrator.on_job_end(*args, **kwargs)
            return result if isinstance(result, dict) else {}
        return {}

    async def on_round_start(self, *args: Any, **kwargs: Any) -> None:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "on_round_start"):
            await orchestrator.on_round_start(*args, **kwargs)

    async def on_round_end(self, *args: Any, **kwargs: Any) -> None:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "on_round_end"):
            await orchestrator.on_round_end(*args, **kwargs)

    async def get_memory_for_operation(self, *args: Any, **kwargs: Any) -> str:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "get_memory_for_operation"):
            return await orchestrator.get_memory_for_operation(*args, **kwargs)
        return ""

    def on_operation_complete(self, *args: Any, **kwargs: Any) -> None:
        orchestrator = self.orchestrator
        if orchestrator and hasattr(orchestrator, "on_operation_complete"):
            orchestrator.on_operation_complete(*args, **kwargs)

    async def close(self) -> None:
        if self.system is not None and hasattr(self.system, "close"):
            await self.system.close()
            self.system = None


__all__ = ["MemoryRuntime"]
