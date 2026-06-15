from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator


def read_recent_events(workspace: Path, *, limit: int = 80) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_file = workspace / ".memory" / "debug" / "events.jsonl"
    per_source_limit = max(10, limit // 2)
    if event_file.exists():
        for line in event_file.read_text(encoding="utf-8", errors="replace").splitlines()[-per_source_limit:]:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append({"kind": "memory", "payload": payload})

    for log_file in _candidate_log_files(workspace):
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-per_source_limit:]:
            stripped = line.strip()
            if stripped and _is_user_relevant_log_line(stripped):
                events.append({"kind": "log", "message": stripped})
    return events[-limit:]


def read_full_events(
    workspace: Path,
    *,
    limit: int = 400,
    include_filtered: bool = False,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_file = workspace / ".memory" / "debug" / "events.jsonl"
    if event_file.exists():
        for line_no, line in enumerate(
            event_file.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:],
            start=1,
        ):
            raw = line.rstrip("\n")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {"raw": raw}
            events.append(
                {
                    "kind": "memory",
                    "payload": payload,
                    "raw": raw,
                    "source": str(event_file.relative_to(workspace)),
                    "line": line_no,
                }
            )

    per_source_limit = max(50, limit)
    for log_file in _candidate_log_files(workspace):
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-per_source_limit:]
        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            relevant = _is_user_relevant_log_line(stripped)
            if not relevant and not include_filtered:
                continue
            events.append(
                {
                    "kind": "log",
                    "message": stripped,
                    "raw": line,
                    "source": str(log_file.relative_to(workspace)),
                    "line": line_no,
                    "filtered": not relevant,
                }
            )
    return events[-limit:]


async def stream_workspace_events(workspace: Path) -> AsyncIterator[str]:
    cursor = 0
    while True:
        events = read_recent_events(workspace, limit=200)
        if cursor < len(events):
            for event in events[cursor:]:
                yield _format_sse(event)
            cursor = len(events)
        else:
            yield _format_sse({"kind": "heartbeat", "ts": time.time()})
        await asyncio.sleep(1.5)


def _format_sse(payload: dict[str, Any]) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _candidate_log_files(workspace: Path) -> list[Path]:
    history = workspace / ".history"
    names = ("runtime.log", "web.log", "memslides-loop.log")
    return [history / name for name in names if (history / name).exists()]


def _is_user_relevant_log_line(line: str) -> bool:
    compact = line.strip()
    if compact in {"{", "}", "},", "],", "[", "]"}:
        return False
    lowered = compact.lower()
    noisy_tokens = (
        '"retry_times"',
        '"client_kwargs"',
        '"sampling_parameters"',
        '"endpoints"',
        '"secret_logging"',
        '"embedding"',
        '"cache_enabled"',
        '"batch_size"',
        '"max_length"',
        '"device"',
        '"modules"',
        '"memory"',
        '"tool_memory"',
    )
    if any(token in lowered for token in noisy_tokens):
        return False
    relevant_tokens = (
        "recovery",
        "reverted",
        "interrupted revision",
        "error",
        "warning",
        "failed",
        "unavailable",
        "skipped",
        "memory",
        "template",
        "slide_",
        "write_html_file",
        "inspect_slide",
        "read_slide_snapshot",
        "researcher",
        "deckdesigner",
        "revision",
        "export",
        "finalize",
        "manuscript",
        "pptx",
        "pdf",
    )
    return any(token in lowered for token in relevant_tokens)
