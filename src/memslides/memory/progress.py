"""Lightweight progress callbacks for memory save/consolidation flows."""

from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import Any, Callable


logger = logging.getLogger(__name__)


ProgressCallback = Callable[[dict[str, Any]], Any]


async def emit_memory_save_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    progress: float | int,
    message: str = "",
    status: str = "saving",
    **extra: Any,
) -> None:
    """Emit a best-effort progress update.

    The callback is intentionally optional and non-fatal so memory saving never
    depends on the Web UI being present.
    """
    if not progress_callback:
        return
    pct = max(0, min(100, int(round(float(progress)))))
    payload: dict[str, Any] = {
        "status": status,
        "stage": stage,
        "progress": pct,
        "message": message,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    try:
        maybe_awaitable = progress_callback(payload)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable
    except Exception:
        logger.debug("Memory save progress callback failed", exc_info=True)
