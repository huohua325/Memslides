from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import Future
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from memslides.contracts import DeckRequest, MemoryOptions, RevisionRequest, SessionOptions, TemplateOptions
from memslides.session import MemSlidesSession
from memslides.templates.induction import induct_template as run_template_induction
from memslides.utils.constants import WORKSPACE_BASE
from memslides.utils.log import reset_context_logger, set_logger
from memslides.web.api_profiles import ApiProfileError, ApiProfileStore


JobState = Literal["idle", "running", "succeeded", "failed"]
OperationPhase = Literal["generation", "revision"]
SESSION_META_FILENAME = "session_meta.json"
SESSIONS_INDEX_FILENAME = "sessions_index.jsonl"
OPERATION_STATE_FILENAME = "operation_state.json"
OPERATIONS_DIRNAME = ".operations"
INTERRUPTED_DIRNAME = ".interrupted"
DEFAULT_MAX_ACTIVE_OPERATIONS = 2
DEFAULT_OPERATION_HEARTBEAT_INTERVAL_SECONDS = 30
DEFAULT_OPERATION_STALE_TIMEOUT_SECONDS = 600
RECOVERY_REVERTED_MESSAGE = "Interrupted revision was reverted to the last stable deck."
RECOVERY_INTERRUPTED_MESSAGE = (
    "Previous revision was interrupted before a stable checkpoint could be restored."
)
LOGGER = logging.getLogger(__name__)
TERMINAL_OPERATION_STATES = {"succeeded", "failed", "interrupted", "reverted", "cancelled"}


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def normalize_export_warnings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return normalize_export_warnings(parsed)
        return [text]
    if isinstance(value, (list, tuple, set)):
        warnings: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    warnings.append(text)
            elif item is not None:
                warnings.append(str(item))
        if warnings == ["[", "]"]:
            return []
        return warnings
    return [str(value)]


class SessionWorker:
    """Dedicated asyncio loop for one UI session.

    Generation/revision can contain sync-heavy model/tool work. Running them on
    Uvicorn's loop would freeze status polling and SSE, so each Web session owns
    a small background loop where MemSlides runtime calls are serialized.
    """

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run_loop, name="memslides-web-session", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        if self._loop is None:
            raise RuntimeError("Failed to start Web session worker")

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def submit(self, coro: Any) -> Future:
        if self._closed or self._loop is None:
            raise RuntimeError("Web session worker is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _log_unhandled(done: Future) -> None:
            if done.cancelled():
                LOGGER.warning("Web session worker task was cancelled before it completed")
                return
            try:
                exc = done.exception()
            except BaseException:  # noqa: BLE001
                LOGGER.exception("Could not inspect Web session worker result")
                return
            if exc is not None:
                LOGGER.error(
                    "Unhandled Web session worker task failure",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        future.add_done_callback(_log_unhandled)
        return future

    async def run(self, coro: Any) -> Any:
        if self._closed or self._loop is None:
            raise RuntimeError("Web session worker is closed")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    async def close(self) -> None:
        if self._closed or self._loop is None:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        await asyncio.to_thread(self._thread.join, 5)


class OperationHeartbeat:
    """Persist progress for embedded Web operations from a side thread."""

    def __init__(self, record: "WebSessionRecord", *, interval_seconds: int) -> None:
        self._record = record
        self._interval_seconds = max(1, int(interval_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"memslides-operation-heartbeat-{record.session_id}",
            daemon=True,
        )

    def start(self) -> None:
        self.pulse()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(5, self._interval_seconds + 2))
        if self._thread.is_alive():
            LOGGER.warning("Operation heartbeat did not stop cleanly for %s", self._record.session_id)

    def pulse(self) -> None:
        with self._record.state_lock:
            operation = self._record.last_operation
            if not isinstance(operation, dict):
                return
            if str(operation.get("state") or "").lower() in TERMINAL_OPERATION_STATES:
                return
            heartbeat_at = now_iso()
            self._record.last_heartbeat_at = heartbeat_at
            operation = update_operation_state(
                self._record.workspace,
                operation,
                heartbeat_at=heartbeat_at,
            )
            self._record.last_operation = operation

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            try:
                self.pulse()
            except Exception:  # noqa: BLE001
                LOGGER.warning("Failed to persist operation heartbeat for %s", self._record.session_id, exc_info=True)


@dataclass
class WebSessionRecord:
    session_id: str
    workspace: Path
    session: MemSlidesSession
    memory_db_dir: Path
    state: JobState = "idle"
    phase: str = "ready"
    message: str = ""
    error: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    display_name: str = ""
    instruction_summary: str = ""
    num_pages: int | str | None = None
    language: str = "en"
    user_id: str = "web-demo"
    memory_intent: str = ""
    memory_profile_id: str = ""
    memory_enabled: bool = True
    api_profile_id: str = ""
    api_profile_display_name: str = ""
    slide_count: int = 0
    artifact_counts: dict[str, int] = field(default_factory=dict)
    has_exports: bool = False
    has_outputs: bool = False
    last_artifact_at: str = ""
    relative_workspace: str = ""
    recovery_status: str = ""
    latest_export_fresh: bool = False
    export_status: str = "none"
    export_warnings: list[str] = field(default_factory=list)
    last_heartbeat_at: str = ""
    last_operation: dict[str, Any] | None = None
    queued_operation: str = ""
    queued_at: str = ""
    memory_save_busy: bool = False
    memory_last_saved_at: str = ""
    memory_save_status: str = ""
    memory_save_progress: int = 0
    memory_save_stage: str = ""
    memory_save_message: str = ""
    memory_save_started_at: str = ""
    memory_save_updated_at: str = ""
    memory_save_error: str = ""
    memory_save_pending_backup: str = ""
    memory_autosave_last_checked_at: str = ""
    memory_autosave_last_saved_at: str = ""
    memory_autosave_skipped_at: str = ""
    memory_autosave_skip_reason: str = ""
    memory_autosave_manual_required: bool = False
    summary_cache_signature: str = ""
    summary_cache_payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    last_opened_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    busy: bool = False
    runtime_resumed: bool = False
    state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    worker: SessionWorker = field(default_factory=SessionWorker, repr=False)

    def to_dict(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "session_id": self.session_id,
                "workspace": str(self.workspace),
                "state": self.state,
                "phase": self.phase,
                "message": self.message,
                "error": self.error,
                "result": self.result,
                "display_name": self.display_name,
                "instruction_summary": self.instruction_summary,
                "num_pages": self.num_pages,
                "language": self.language,
                "user_id": self.user_id,
                "memory_intent": self.memory_intent,
                "memory_profile_id": self.memory_profile_id,
                "memory_enabled": self.memory_enabled,
                "api_profile_id": self.api_profile_id,
                "api_profile_display_name": self.api_profile_display_name,
                "service_profile_id": self.api_profile_id,
                "service_profile_display_name": self.api_profile_display_name,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "last_opened_at": self.last_opened_at,
                "memory_db_dir": str(self.memory_db_dir),
                "slide_count": self.slide_count,
                "artifact_counts": self.artifact_counts,
                "has_exports": self.has_exports,
                "has_outputs": self.has_outputs,
                "last_artifact_at": self.last_artifact_at,
                "relative_workspace": self.relative_workspace,
                "recovery_status": self.recovery_status,
                "latest_export_fresh": self.latest_export_fresh,
                "export_status": self.export_status,
                "export_warnings": self.export_warnings,
                "last_heartbeat_at": self.last_heartbeat_at,
                "last_operation": self.last_operation,
                "queued_operation": self.queued_operation,
                "queued_at": self.queued_at,
                "memory_save_busy": self.memory_save_busy,
                "memory_last_saved_at": self.memory_last_saved_at,
                "memory_save_status": self.memory_save_status,
                "memory_save_progress": self.memory_save_progress,
                "memory_save_stage": self.memory_save_stage,
                "memory_save_message": self.memory_save_message,
                "memory_save_started_at": self.memory_save_started_at,
                "memory_save_updated_at": self.memory_save_updated_at,
                "memory_save_error": self.memory_save_error,
                "memory_save_pending_backup": self.memory_save_pending_backup,
                "memory_autosave_last_checked_at": self.memory_autosave_last_checked_at,
                "memory_autosave_last_saved_at": self.memory_autosave_last_saved_at,
                "memory_autosave_skipped_at": self.memory_autosave_skipped_at,
                "memory_autosave_skip_reason": self.memory_autosave_skip_reason,
                "memory_autosave_manual_required": self.memory_autosave_manual_required,
            }


@dataclass
class QueuedOperation:
    record: WebSessionRecord
    operation_type: OperationPhase
    payload: dict[str, Any] | None = None
    attachments: list[Path] = field(default_factory=list)
    feedback: str = ""
    memory_intent: str = ""
    queued_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())


class WebSessionStore:
    def __init__(
        self,
        *,
        workspace_base: Path | None = None,
        config_file: Path | None = None,
        memory_db_dir: Path | None = None,
        service_profile_encryption_key: str = "",
        max_active_operations: int | None = None,
        operation_heartbeat_interval_seconds: int | None = None,
        operation_stale_timeout_seconds: int | None = None,
        user_isolation: bool = False,
    ):
        self.workspace_base = workspace_base or WORKSPACE_BASE / "web"
        self.cache_root = self.workspace_base.parent if self.workspace_base.name == "web" else self.workspace_base
        self.memory_db_dir = memory_db_dir or self.cache_root / ".memory"
        self.memory_users_root = self.memory_db_dir / "users"
        self.api_profiles = ApiProfileStore(
            self.memory_db_dir,
            encryption_secret=service_profile_encryption_key,
        )
        self.config_file = config_file
        self.user_isolation = user_isolation
        self.records: dict[str, WebSessionRecord] = {}
        self.index_path = self.workspace_base / SESSIONS_INDEX_FILENAME
        self.max_active_operations = max(1, int(max_active_operations or os.getenv("MEMSLIDES_WEB_MAX_ACTIVE_OPERATIONS", DEFAULT_MAX_ACTIVE_OPERATIONS)))
        self.operation_heartbeat_interval_seconds = max(
            1,
            int(
                operation_heartbeat_interval_seconds
                or os.getenv("MEMSLIDES_OPERATION_HEARTBEAT_INTERVAL_SEC", DEFAULT_OPERATION_HEARTBEAT_INTERVAL_SECONDS)
            ),
        )
        self.operation_stale_timeout_seconds = max(
            self.operation_heartbeat_interval_seconds * 2,
            int(
                operation_stale_timeout_seconds
                or os.getenv("MEMSLIDES_OPERATION_STALE_TIMEOUT_SEC", DEFAULT_OPERATION_STALE_TIMEOUT_SECONDS)
            ),
        )
        self._scheduler_lock = threading.RLock()
        self._active_sessions: set[str] = set()
        self._active_session_profiles: dict[str, str] = {}
        self._active_profile_counts: dict[str, int] = {}
        self._queue: list[QueuedOperation] = []
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.memory_users_root.mkdir(parents=True, exist_ok=True)
        self.discover_legacy_sessions()

    async def create_session(
        self,
        *,
        language: str = "en",
        user_id: str = "web-demo",
        api_profile_id: str = "",
        memory_profile_id: str = "",
        memory_intent: str = "",
        memory_enabled: bool = True,
    ) -> WebSessionRecord:
        session_id = uuid.uuid4().hex[:8]
        display_name = "Untitled deck"
        workspace = self._build_workspace_path(session_id=session_id, instruction="", user_id=user_id)
        workspace.mkdir(parents=True, exist_ok=True)
        runtime_profile = self._resolve_runtime_llm_profile(api_profile_id, user_id=user_id)
        user_memory_dir = self._memory_dir_for_user(user_id)
        options = SessionOptions(
            config_file=self.config_file,
            workspace=workspace,
            session_id=session_id,
            language=language,  # type: ignore[arg-type]
            memory=MemoryOptions(enabled=bool(memory_enabled), user_id=user_id, global_db_dir=user_memory_dir),
            api_profile_id=str(api_profile_id or ""),
            service_profile_id=str(api_profile_id or ""),
            runtime_llm_profile=runtime_profile,
            runtime_service_profile=runtime_profile,
        )
        reset_context_logger()
        session = MemSlidesSession(options=options)
        record = WebSessionRecord(
            session_id=session_id,
            workspace=workspace,
            session=session,
            memory_db_dir=user_memory_dir,
            display_name=display_name,
            instruction_summary="",
            language=language,
            user_id=user_id,
            memory_intent=str(memory_intent or ""),
            memory_profile_id=str(memory_profile_id or ""),
            memory_enabled=bool(memory_enabled),
            api_profile_id=str(runtime_profile.get("profile_id") or api_profile_id or ""),
            api_profile_display_name=str(runtime_profile.get("display_name") or ""),
        )
        self.records[session_id] = record
        self.refresh_session_metadata(record)
        return record

    def get(self, session_id: str) -> WebSessionRecord:
        try:
            return self.records[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session_id: {session_id}") from exc

    def _resolve_runtime_llm_profile(self, api_profile_id: str = "", *, user_id: str = "web-demo") -> dict[str, Any]:
        profile_id = str(api_profile_id or "").strip()
        if not profile_id:
            return {}
        return self.api_profiles.get_runtime_profile(profile_id, user_id=user_id)

    def _api_profile_display_name(self, api_profile_id: str = "", *, user_id: str = "web-demo") -> str:
        if not api_profile_id:
            return ""
        try:
            return str(self.api_profiles.get_public_profile(api_profile_id, user_id=user_id).get("display_name") or "")
        except ApiProfileError:
            return ""

    def _make_session(
        self,
        *,
        workspace: Path,
        session_id: str,
        language: str,
        user_id: str,
        api_profile_id: str = "",
        memory_enabled: bool = True,
    ) -> tuple[MemSlidesSession, dict[str, Any]]:
        runtime_profile = self._resolve_runtime_llm_profile(api_profile_id, user_id=user_id)
        user_memory_dir = self._memory_dir_for_user(user_id)
        options = SessionOptions(
            config_file=self.config_file,
            workspace=workspace,
            session_id=session_id,
            language=language,  # type: ignore[arg-type]
            memory=MemoryOptions(enabled=bool(memory_enabled), user_id=user_id, global_db_dir=user_memory_dir),
            api_profile_id=str(runtime_profile.get("profile_id") or api_profile_id or ""),
            service_profile_id=str(runtime_profile.get("profile_id") or api_profile_id or ""),
            runtime_llm_profile=runtime_profile,
            runtime_service_profile=runtime_profile,
        )
        reset_context_logger()
        return MemSlidesSession(options=options), runtime_profile

    def _memory_dir_for_user(self, user_id: str) -> Path:
        safe_user = slugify_user_id(user_id or "web-demo")
        target = self.memory_users_root / safe_user
        target.mkdir(parents=True, exist_ok=True)
        return target

    async def _apply_api_profile_to_record(
        self,
        record: WebSessionRecord,
        api_profile_id: str | None = None,
        memory_enabled: bool | None = None,
    ) -> None:
        requested = (
            str(record.api_profile_id or "").strip()
            if api_profile_id is None
            else str(api_profile_id or "").strip()
        )
        requested_memory_enabled = record.memory_enabled if memory_enabled is None else _coerce_bool(memory_enabled, record.memory_enabled)
        current_config = Path(str(getattr(record.session.config, "file_path", "") or ""))
        config_matches_workspace = (
            not requested
            or (
                current_config.exists()
                and path_within_or_same(current_config, record.workspace)
            )
        )
        if (
            requested == str(record.api_profile_id or "").strip()
            and requested_memory_enabled == bool(record.memory_enabled)
            and config_matches_workspace
        ):
            return
        if record.busy or record.queued_operation:
            raise RuntimeError("Session is already running")
        old_session = record.session
        try:
            await record.worker.run(old_session.close())
        except Exception:
            pass
        reset_context_logger()
        session, runtime_profile = self._make_session(
            workspace=record.workspace,
            session_id=record.session_id,
            language=record.language,
            user_id=record.user_id,
            api_profile_id=requested,
            memory_enabled=requested_memory_enabled,
        )
        with record.state_lock:
            record.session = session
            record.memory_enabled = requested_memory_enabled
            record.api_profile_id = str(runtime_profile.get("profile_id") or requested or "")
            record.api_profile_display_name = str(runtime_profile.get("display_name") or "")
            record.runtime_resumed = False

    async def generate(self, session_id: str, payload: dict[str, Any], attachments: list[Path]) -> None:
        record = self.get(session_id)
        with record.state_lock:
            if record.memory_save_busy:
                raise RuntimeError("Memory save is already running")
        self.assert_can_generate(record)
        self._ensure_named_workspace(record, str(payload.get("instruction") or ""))
        requested_profile = (
            str(payload.get("service_profile_id") or payload.get("api_profile_id") or "")
            if ("service_profile_id" in payload or "api_profile_id" in payload)
            else None
        )
        requested_memory_enabled = (
            _coerce_bool(payload.get("memory_enabled"), record.memory_enabled)
            if "memory_enabled" in payload
            else None
        )
        await self._apply_api_profile_to_record(record, requested_profile, requested_memory_enabled)
        language = normalize_language(payload.get("language"), record.language)
        with record.state_lock:
            record.instruction_summary = summarize_instruction(str(payload.get("instruction") or ""))
            record.display_name = humanize_display_name(record.instruction_summary)
            record.num_pages = payload.get("num_pages") or None
            record.language = language
            record.session.runtime.language = language
            record.session.options.language = language  # type: ignore[assignment]
            record.memory_intent = str(payload.get("memory_intent") or "")
            if "memory_profile_id" in payload:
                record.memory_profile_id = str(payload.get("memory_profile_id") or "")
            record.last_opened_at = now_iso()
        self.refresh_session_metadata(record)
        item = QueuedOperation(record=record, operation_type="generation", payload=payload, attachments=attachments)
        if self._enqueue_or_start(item):
            self._dispatch_operation(item)

    async def revise(
        self,
        session_id: str,
        feedback: str,
        memory_intent: str = "",
        memory_profile_id: str = "",
        api_profile_id: str | None = None,
    ) -> None:
        record = self.get(session_id)
        await self._apply_api_profile_to_record(record, api_profile_id)
        effective_memory_intent = str(memory_intent or record.memory_intent or "").strip()
        with record.state_lock:
            if record.memory_save_busy:
                raise RuntimeError("Memory save is already running")
            if effective_memory_intent:
                record.memory_intent = effective_memory_intent
            record.memory_profile_id = str(memory_profile_id or "")
            record.last_opened_at = now_iso()
        self.refresh_session_metadata(record)
        item = QueuedOperation(
            record=record,
            operation_type="revision",
            feedback=feedback,
            memory_intent=effective_memory_intent,
        )
        if self._enqueue_or_start(item):
            self._dispatch_operation(item)

    def _memory_save_disabled_payload(self, record: WebSessionRecord) -> dict[str, Any]:
        return {
            "status": "disabled",
            "saved": False,
            "saved_at": now_iso(),
            "session_id": record.session_id,
            "memory_db_dir": str(record.memory_db_dir),
            "message": "Memory is disabled for this session.",
        }

    def _prepare_memory_save(self, record: WebSessionRecord) -> dict[str, Any]:
        if not record.memory_enabled:
            raise RuntimeError("Memory is disabled for this session.")
        started_at = now_iso()
        previous_failure = {
            "status": record.memory_save_status,
            "error": record.memory_save_error,
            "pending_backup": record.memory_save_pending_backup,
        }
        with record.state_lock:
            if record.busy or record.queued_operation:
                raise RuntimeError("Session is already running")
            if record.memory_save_busy:
                raise RuntimeError("Memory save is already running")
            record.memory_save_busy = True
            record.memory_save_status = "saving"
            record.memory_save_progress = 1
            record.memory_save_stage = "starting"
            record.memory_save_message = "Starting memory save."
            record.memory_save_started_at = started_at
            record.memory_save_updated_at = started_at
            record.memory_save_error = ""
            # Keep the previous pending backup visible while the backend decides
            # whether this save is a fresh flush or a retry-from-backup.
            record.memory_save_pending_backup = str(previous_failure.get("pending_backup") or "")
            record.updated_at = now_iso()
        self.refresh_session_metadata(record)
        return previous_failure

    async def start_memory_save(self, session_id: str) -> dict[str, Any]:
        record = self.get(session_id)
        if not record.memory_enabled:
            return self._memory_save_disabled_payload(record)
        previous_failure = self._prepare_memory_save(record)
        record.worker.submit(self._run_memory_save(record, previous_failure))
        return {
            "status": "saving",
            "accepted": True,
            "background": True,
            "saved": False,
            "session_id": record.session_id,
            "memory_db_dir": str(record.memory_db_dir),
            "message": "Memory consolidation started. You can switch sessions or close this tab.",
            "stage": "starting",
            "memory_save_busy": True,
        }

    async def save_memory(self, session_id: str) -> dict[str, Any]:
        record = self.get(session_id)
        if not record.memory_enabled:
            return self._memory_save_disabled_payload(record)
        previous_failure = self._prepare_memory_save(record)
        return await asyncio.wrap_future(record.worker.submit(self._run_memory_save(record, previous_failure)))

    async def _run_memory_save(self, record: WebSessionRecord, previous_failure: dict[str, Any]) -> dict[str, Any]:

        async def progress_callback(payload: dict[str, Any]) -> None:
            update_memory_save_progress(record, payload)
            append_memory_save_progress_trace(record, payload)

        try:
            save_memory = record.session.save_memory
            signature = inspect.signature(save_memory)
            accepts_progress = (
                "progress_callback" in signature.parameters
                or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
            )
            save_coro = (
                save_memory(progress_callback=progress_callback)
                if accepts_progress
                else save_memory()
            )
            result = await save_coro
        except Exception as exc:
            result = {
                "status": "failed",
                "saved": False,
                "saved_at": now_iso(),
                "session_id": record.session_id,
                "memory_db_dir": str(record.memory_db_dir),
                "message": str(exc),
            }
            append_memory_save_trace(record, result)
        result.setdefault("session_id", record.session_id)
        result.setdefault("saved_at", now_iso())
        if not result.get("memory_db_dir"):
            result["memory_db_dir"] = str(record.memory_db_dir)
        if (
            str(result.get("status") or "") == "nothing_to_save"
            and str(previous_failure.get("status") or "") == "failed"
            and str(previous_failure.get("pending_backup") or "")
        ):
            result["status"] = "failed"
            result["saved"] = False
            result["consolidation_failed"] = True
            result["pending_consolidation_backup"] = str(previous_failure["pending_backup"])
            result["retry_available"] = True
            result["message"] = (
                "Previous memory save failed and no active working-memory job remains. "
                "Use Retry from backup. "
                f"Pending backup: {previous_failure['pending_backup']}"
            )
        with record.state_lock:
            previous_progress_stage = record.memory_save_stage
            previous_progress_message = record.memory_save_message
            record.memory_save_busy = False
            record.memory_save_progress = 100
            if bool(result.get("saved")):
                record.memory_last_saved_at = str(result.get("saved_at") or now_iso())
                record.memory_save_error = ""
                record.memory_save_pending_backup = ""
                record.memory_save_message = str(result.get("message") or "Working memory was saved to long-term memory.")
            else:
                record.memory_save_error = str(result.get("message") or "")
                record.memory_save_pending_backup = str(result.get("pending_consolidation_backup") or "")
                record.memory_save_message = str(result.get("message") or result.get("status") or "")
            record.memory_save_status = str(result.get("status") or "unknown")
            final_stage = str(result.get("stage") or result.get("status") or "complete")
            if not bool(result.get("saved")) and final_stage in {"", "failed"} and previous_progress_stage:
                final_stage = previous_progress_stage
                if previous_progress_message and previous_progress_message not in record.memory_save_message:
                    record.memory_save_message = (
                        f"{record.memory_save_message} Last stage: "
                        f"{previous_progress_stage} — {previous_progress_message}"
                    ).strip()
            record.memory_save_stage = final_stage
            record.memory_save_updated_at = now_iso()
            if bool(result.get("saved")) or record.memory_save_status in {"no_memory", "nothing_to_save", "disabled"}:
                record.memory_autosave_manual_required = False
                record.memory_autosave_skip_reason = ""
                if bool(result.get("saved")):
                    record.memory_autosave_last_saved_at = str(result.get("saved_at") or now_iso())
            record.updated_at = now_iso()
        self.refresh_session_metadata(record)
        append_memory_save_event(record, result)
        return result

    async def get_working_memory(self, session_id: str) -> dict[str, Any]:
        record = self.get(session_id)
        result = await record.worker.run(record.session.get_working_memory_snapshot())
        result.setdefault("session_id", record.session_id)
        result.setdefault("memory_db_dir", str(record.memory_db_dir))
        return result

    async def update_working_memory(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.get(session_id)
        result = await record.worker.run(record.session.update_working_memory_snapshot(payload))
        result.setdefault("session_id", record.session_id)
        result.setdefault("memory_db_dir", str(record.memory_db_dir))
        record.updated_at = now_iso()
        self.refresh_session_metadata(record)
        return result

    async def get_user_profile(self, session_id: str, intent: str = "") -> dict[str, Any]:
        record = self.get(session_id)
        resolved_intent = str(intent or record.memory_intent or "").strip()
        result = await record.worker.run(record.session.get_user_profile(resolved_intent))
        result.setdefault("session_id", record.session_id)
        result.setdefault("memory_db_dir", str(record.memory_db_dir))
        return result

    async def get_global_user_profile(
        self,
        *,
        user_id: str = "web-demo",
        intent: str = "",
        api_profile_id: str = "",
    ) -> dict[str, Any]:
        session = self._create_profile_session(user_id=user_id, api_profile_id=api_profile_id)
        try:
            result = await session.get_user_profile(intent)
        finally:
            await session.close()
        result.setdefault("session_id", "")
        result.setdefault("memory_db_dir", str(self.memory_db_dir))
        return result

    async def save_user_profile(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = self.get(session_id)
        with record.state_lock:
            if record.busy or record.queued_operation:
                raise RuntimeError("Session is already running")
            if record.memory_save_busy:
                raise RuntimeError("Memory save is already running")
        editor_payload = dict(payload or {})
        editor_payload["intent"] = str(editor_payload.get("intent") or record.memory_intent or "").strip()
        result = await record.worker.run(record.session.save_user_profile(editor_payload))
        result.setdefault("session_id", record.session_id)
        result.setdefault("memory_db_dir", str(record.memory_db_dir))
        append_profile_editor_trace(record, result)
        append_profile_editor_event(record, result)
        self.refresh_session_metadata(record)
        return result

    async def save_global_user_profile(
        self,
        payload: dict[str, Any],
        *,
        user_id: str = "web-demo",
        api_profile_id: str = "",
    ) -> dict[str, Any]:
        editor_payload = dict(payload or {})
        editor_payload["intent"] = str(editor_payload.get("intent") or "").strip()
        session = self._create_profile_session(user_id=user_id, api_profile_id=api_profile_id)
        try:
            result = await session.save_user_profile(editor_payload)
        finally:
            await session.close()
        result.setdefault("session_id", "")
        result.setdefault("memory_db_dir", str(self._memory_dir_for_user(user_id)))
        return result

    def _create_profile_session(self, *, user_id: str = "web-demo", api_profile_id: str = "") -> MemSlidesSession:
        normalized_user_id = str(user_id or "web-demo").strip() or "web-demo"
        safe_user_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", normalized_user_id).strip("._-") or "web-demo"
        workspace = self.workspace_base / ".profile_preview" / safe_user_id
        workspace.mkdir(parents=True, exist_ok=True)
        session, _ = self._make_session(
            workspace=workspace,
            session_id=f"profile-{hashlib.sha1(normalized_user_id.encode('utf-8')).hexdigest()[:8]}",
            language="en",
            user_id=normalized_user_id,
            api_profile_id=api_profile_id,
        )
        return session

    async def open_session(self, *, session_id: str | None = None, workspace: Path | None = None) -> WebSessionRecord:
        workspace = workspace.expanduser() if workspace is not None else None
        if session_id and session_id in self.records:
            record = self.records[session_id]
            with record.state_lock:
                record.last_opened_at = now_iso()
            self.refresh_session_metadata(record)
            return record

        meta = self._find_session_meta(session_id=session_id, workspace=workspace)
        if meta is None:
            raise KeyError(f"Unknown session: {session_id or workspace}")

        resolved_workspace = Path(meta["workspace"]).expanduser()
        resolved_session_id = str(meta.get("session_id") or resolved_workspace.name.rsplit("_", 1)[-1])
        resolved_user_id = str(meta.get("user_id") or "web-demo")
        resolved_language = str(meta.get("language") or "en")
        resolved_memory_enabled = _coerce_bool(meta.get("memory_enabled", True), True)
        api_profile_id = str(meta.get("service_profile_id") or meta.get("api_profile_id") or "")
        try:
            session, runtime_profile = self._make_session(
                workspace=resolved_workspace,
                session_id=resolved_session_id,
                language=resolved_language,
                user_id=resolved_user_id,
                api_profile_id=api_profile_id,
                memory_enabled=resolved_memory_enabled,
            )
        except ApiProfileError:
            session, runtime_profile = self._make_session(
                workspace=resolved_workspace,
                session_id=resolved_session_id,
                language=resolved_language,
                user_id=resolved_user_id,
                api_profile_id="",
                memory_enabled=resolved_memory_enabled,
            )
            api_profile_id = ""
        record = WebSessionRecord(
            session_id=resolved_session_id,
            workspace=resolved_workspace,
            session=session,
            memory_db_dir=self._memory_dir_for_user(resolved_user_id),
            state=str(meta.get("state") or "idle"),  # type: ignore[arg-type]
            phase=str(meta.get("phase") or "ready"),
            message=str(meta.get("message") or ""),
            display_name=str(meta.get("display_name") or humanize_display_name(str(meta.get("instruction_summary") or ""))),
            instruction_summary=str(meta.get("instruction_summary") or ""),
            num_pages=meta.get("num_pages"),
            language=resolved_language,
            user_id=resolved_user_id,
            memory_intent=str(meta.get("memory_intent") or ""),
            memory_profile_id=str(meta.get("memory_profile_id") or ""),
            memory_enabled=resolved_memory_enabled,
            api_profile_id=str(runtime_profile.get("profile_id") or api_profile_id or ""),
            api_profile_display_name=str(runtime_profile.get("display_name") or meta.get("api_profile_display_name") or ""),
            created_at=str(meta.get("created_at") or now_iso()),
            updated_at=str(meta.get("updated_at") or now_iso()),
            last_opened_at=now_iso(),
            recovery_status=str(meta.get("recovery_status") or ""),
            latest_export_fresh=bool(meta.get("latest_export_fresh", False)),
            export_status=str(meta.get("export_status") or "none"),
            export_warnings=normalize_export_warnings(meta.get("export_warnings")),
            last_heartbeat_at=str(meta.get("last_heartbeat_at") or ""),
            last_operation=meta.get("last_operation") if isinstance(meta.get("last_operation"), dict) else None,
            queued_operation=str(meta.get("queued_operation") or ""),
            queued_at=str(meta.get("queued_at") or ""),
            memory_save_busy=False,
            memory_last_saved_at=str(meta.get("memory_last_saved_at") or ""),
            memory_save_status=str(meta.get("memory_save_status") or ""),
            memory_save_progress=int(meta.get("memory_save_progress") or 0),
            memory_save_stage=str(meta.get("memory_save_stage") or ""),
            memory_save_message=str(meta.get("memory_save_message") or ""),
            memory_save_started_at=str(meta.get("memory_save_started_at") or ""),
            memory_save_updated_at=str(meta.get("memory_save_updated_at") or ""),
            memory_save_error=str(meta.get("memory_save_error") or ""),
            memory_save_pending_backup=str(meta.get("memory_save_pending_backup") or ""),
            memory_autosave_last_checked_at=str(meta.get("memory_autosave_last_checked_at") or ""),
            memory_autosave_last_saved_at=str(meta.get("memory_autosave_last_saved_at") or ""),
            memory_autosave_skipped_at=str(meta.get("memory_autosave_skipped_at") or ""),
            memory_autosave_skip_reason=str(meta.get("memory_autosave_skip_reason") or ""),
            memory_autosave_manual_required=_coerce_bool(meta.get("memory_autosave_manual_required"), False),
        )
        self.records[resolved_session_id] = record
        self.refresh_session_metadata(record)
        return record

    def list_sessions(self, *, limit: int = 100) -> list[dict[str, Any]]:
        self.discover_legacy_sessions()
        items = self._load_index_entries()
        items.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return items[:limit]

    def active_session_ids(self) -> set[str]:
        with self._scheduler_lock:
            return set(self._active_sessions)

    def active_api_profiles(self) -> dict[str, int]:
        with self._scheduler_lock:
            return dict(self._active_profile_counts)

    def queued_operation_count(self) -> int:
        with self._scheduler_lock:
            return len(self._queue)

    def mark_operation_cancelled(
        self,
        record: WebSessionRecord,
        *,
        operation_id: str = "",
        message: str = "This run was stopped.",
    ) -> dict[str, Any] | None:
        with record.state_lock:
            record.busy = False
            record.queued_operation = ""
            record.queued_at = ""
            record.state = "cancelled"
            record.phase = "cancelled"
            record.message = message
            record.error = {"status": "cancelled", "message": message, "recoverable": True}
            record.recovery_status = "cancelled"
            record.updated_at = now_iso()
            operation = record.last_operation
            if operation_id and (not isinstance(operation, dict) or str(operation.get("operation_id") or "") != operation_id):
                operation = read_operation_state(record.workspace)
            if isinstance(operation, dict):
                record.last_operation = update_operation_state(
                    record.workspace,
                    operation,
                    state="cancelled",
                    phase="cancelled",
                    message=message,
                    error=record.error,
                    recovery_status="cancelled",
                )
            else:
                record.last_operation = None
        with self._scheduler_lock:
            self._queue = [item for item in self._queue if item.record.session_id != record.session_id]
            self._release_operation_slot(record)
        self.refresh_session_metadata(record)
        self._start_queued_operations()
        return record.last_operation

    def reconcile_active_sessions(self) -> None:
        for record in list(self.records.values()):
            self.refresh_session_metadata(record)

    async def induct_template(self, template_file: Path, output_dir: Path | None = None, *, replace_existing: bool = False) -> dict[str, Any]:
        output_dir = output_dir or self.workspace_base / "templates" / template_file.stem
        final_dir = output_dir
        work_dir = final_dir
        staging_dir: Path | None = None
        backup_dir: Path | None = None
        token = uuid.uuid4().hex[:10]
        if replace_existing:
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staging_dir = final_dir.parent / f".{final_dir.name}.staging-{token}"
            backup_dir = final_dir.parent / f".{final_dir.name}.backup-{token}"
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            work_dir = staging_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        operation = create_operation_state(
            work_dir,
            operation_type="template_induct",
            phase="template_induct",
            message="Inducting template",
        )
        try:
            analysis = await asyncio.to_thread(
                lambda: asyncio.run(run_template_induction(template_file, output_dir=work_dir, workspace=work_dir))
            )
            update_operation_state(work_dir, operation, state="succeeded", phase="template_induct_complete", message="Template inducted")
            if staging_dir is not None and backup_dir is not None:
                if final_dir.exists():
                    final_dir.rename(backup_dir)
                try:
                    staging_dir.rename(final_dir)
                except Exception:
                    if backup_dir.exists() and not final_dir.exists():
                        backup_dir.rename(final_dir)
                    raise
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
            return analysis.to_dict() if hasattr(analysis, "to_dict") else dict(analysis)
        except Exception as exc:
            update_operation_state(
                work_dir,
                operation,
                state="failed",
                phase="template_induct_error",
                message=str(exc),
                error={"message": str(exc), "detail": traceback.format_exc(limit=20)},
            )
            if staging_dir is not None and staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    async def close_all(self) -> None:
        for record in list(self.records.values()):
            try:
                self.refresh_session_metadata(record)
                await record.worker.run(record.session.close())
            finally:
                await record.worker.close()

    async def _run_generate(self, record: WebSessionRecord, payload: dict[str, Any], attachments: list[Path]) -> None:
        set_session_runtime_logger(record)
        heartbeat = OperationHeartbeat(record, interval_seconds=self.operation_heartbeat_interval_seconds)
        heartbeat.start()
        try:
            request = DeckRequest(
                instruction=str(payload.get("instruction") or ""),
                attachments=attachments,
                num_pages=payload.get("num_pages") or None,
                language=record.language,
                memory_intent=str(payload.get("memory_intent") or ""),
                template=Path(payload["template"]) if payload.get("template") else None,
                template_id=payload.get("template_id") or None,
                template_as_reference=bool(payload.get("template_as_reference", False)),
                extra_info={
                    "source": "web",
                    "memory_profile_id": str(payload.get("memory_profile_id") or ""),
                    "service_profile_id": str(payload.get("service_profile_id") or payload.get("api_profile_id") or ""),
                },
            )
            result = await record.session.generate(request)
            with record.state_lock:
                record.result = result.model_dump(mode="json")
            self._snapshot_exports(record, "generation")
            self._set_success(record, "generation_complete", "Deck generated")
        except asyncio.CancelledError as exc:
            self._set_error(record, exc)
        except Exception as exc:
            self._set_error(record, exc)
        finally:
            heartbeat.stop()
            self._finish_operation(record)

    async def _run_revise(self, record: WebSessionRecord, feedback: str, memory_intent: str) -> None:
        set_session_runtime_logger(record)
        operation: dict[str, Any] = dict(record.last_operation or {})
        heartbeat = OperationHeartbeat(record, interval_seconds=self.operation_heartbeat_interval_seconds)
        heartbeat.start()
        try:
            await self._ensure_runtime_resumed(record)
            before = revision_state_snapshot(record.workspace)
            operation = dict(record.last_operation or {})
            result = await record.session.revise(RevisionRequest(feedback=feedback, memory_intent=memory_intent))
            with record.state_lock:
                record.result = result.model_dump(mode="json")
            self._validate_revision_result(record, result, before)
            self._snapshot_exports(record, "revision")
            self._set_success(record, "revision_complete", "Deck revised")
        except asyncio.CancelledError as exc:
            self._set_error(record, exc)
        except Exception as exc:
            try:
                if rollback_interrupted_revision(record.workspace, operation or record.last_operation or {}):
                    append_recovery_event(
                        record.workspace,
                        "Failed revision was rolled back to the pre-revision checkpoint.",
                        operation=operation or record.last_operation,
                        status="reverted",
                    )
            except Exception:
                logging.getLogger(__name__).debug("Revision rollback after failure failed", exc_info=True)
            self._set_error(record, exc)
        finally:
            heartbeat.stop()
            self._finish_operation(record)

    async def _ensure_runtime_resumed(self, record: WebSessionRecord) -> None:
        if record.runtime_resumed:
            return
        if not (record.workspace / "intermediate_output.json").exists():
            raise RuntimeError(
                "This Web session has no generated deck state to revise. Generate a deck first."
            )
        final = await record.session.resume()
        runtime = record.session.runtime
        if runtime.agent_env is None or runtime.designagent is None:
            raise RuntimeError(
                "Could not restore the generated deck runtime for revision. "
                "Please reopen the session or regenerate the deck before revising."
            )
        with record.state_lock:
            record.runtime_resumed = True
            if final:
                record.result = {"resumed_final": str(final)}

    def _validate_revision_result(
        self,
        record: WebSessionRecord,
        result: Any,
        before_summary: dict[str, Any],
    ) -> None:
        messages = [str(message) for message in getattr(result, "messages", []) if str(message).strip()]
        joined = "\n".join(messages).lower()
        blockers = (
            "please run initial generation first",
            "agent environment not available",
            "revisioneditor is unavailable",
            "no deckdesigner agent available",
        )
        if any(blocker in joined for blocker in blockers):
            raise RuntimeError(messages[-1] if messages else "Revision could not start.")

        after_snapshot = revision_state_snapshot(record.workspace)
        if after_snapshot == before_summary:
            raise RuntimeError(
                "Revision finished without changing slides, exports, or round artifacts. "
                "No slide change was detected."
            )
        slide_changed = after_snapshot.get("slides") != before_summary.get("slides")
        fresh_export = revision_has_fresh_export(record.workspace, before_summary, after_snapshot)
        failure_markers = (
            "modification did not finalize successfully",
            "export was skipped",
            "slides modified but export failed",
            "visual_quality_failed",
            "new elements created in this modify turn do not yet follow",
        )
        if any(marker in joined for marker in failure_markers):
            if fresh_export:
                export_status, export_warnings = infer_export_status(record.workspace, operation=record.last_operation)
                warning_text = "Revision reported finalize/QA warnings after producing a fresh export."
                if warning_text not in export_warnings:
                    export_warnings = [*export_warnings, warning_text]
                LOGGER.warning(warning_text)
                if record.last_operation:
                    record.last_operation = update_operation_state(
                        record.workspace,
                        record.last_operation,
                        export_status=export_status,
                        export_warnings=export_warnings,
                        recovery_status="success_with_warnings_after_fresh_export",
                    )
                return
            raise RuntimeError(messages[-1] if messages else "Revision did not finalize/export successfully.")

        memory_only_success = (
            not slide_changed
            and not fresh_export
            and any("preference_update" in message.lower() or "memory-only" in message.lower() for message in messages)
        )
        if not fresh_export and not memory_only_success:
            raise RuntimeError(
                "Revision finished without a fresh modification export. "
                "No new modification_N.pptx/pdf was created after this revision started."
            )

    def save_uploads(self, record: WebSessionRecord, files: list[tuple[str, bytes]]) -> list[Path]:
        upload_dir = record.workspace / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for filename, content in files:
            safe_name = Path(filename).name
            if not safe_name:
                continue
            target = upload_dir / safe_name
            target.write_bytes(content)
            saved.append(target)
        return saved

    def list_saved_uploads(self, record: WebSessionRecord) -> list[Path]:
        upload_dir = record.workspace / "uploads"
        if not upload_dir.exists() or not upload_dir.is_dir():
            return []
        return sorted(path for path in upload_dir.rglob("*") if path.is_file())

    def save_reference_template(self, record: WebSessionRecord, filename: str, content: bytes) -> Path:
        template_dir = record.workspace / "reference_template"
        template_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "reference_template.pptx"
        if not safe_name.lower().endswith(".pptx"):
            safe_name = f"{Path(safe_name).stem}.pptx"
        target = template_dir / safe_name
        target.write_bytes(content)
        return target

    def save_template_upload(self, filename: str, content: bytes) -> Path:
        upload_dir = self.cache_root / "template_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        target = upload_dir / Path(filename).name
        target.write_bytes(content)
        return target

    def prepare_workspace(self, record: WebSessionRecord, instruction: str) -> Path:
        self.assert_can_generate(record)
        self._ensure_named_workspace(record, instruction)
        self.refresh_session_metadata(record)
        return record.workspace

    def assert_can_generate(self, record: WebSessionRecord) -> None:
        summary = self.refresh_session_metadata(record)
        has_generated_artifacts = (
            bool(summary.get("has_outputs"))
            or int(summary.get("slide_count") or 0) > 0
            or bool(_candidate_export_files(record.workspace))
        )
        if not has_generated_artifacts:
            return
        if self.can_regenerate_after_cancelled_generation(record):
            self._clear_cancelled_generation_outputs(record)
            return
        raise RuntimeError(
            "This session already has a generated deck. Click New before generating another deck."
        )

    @staticmethod
    def can_regenerate_after_cancelled_generation(record: WebSessionRecord) -> bool:
        operation = record.last_operation if isinstance(record.last_operation, dict) else read_operation_state(record.workspace)
        operation_type = str((operation or {}).get("operation_type") or "").lower()
        states = {
            str(record.state or "").lower(),
            str(record.phase or "").lower(),
            str(record.recovery_status or "").lower(),
            str((operation or {}).get("state") or "").lower(),
            str((operation or {}).get("phase") or "").lower(),
            str((operation or {}).get("recovery_status") or "").lower(),
        }
        return (
            not record.busy
            and not record.queued_operation
            and operation_type in {"generation", "generate"}
            and bool(states.intersection({"cancelled", "canceled"}))
        )

    def _clear_cancelled_generation_outputs(self, record: WebSessionRecord) -> None:
        """Remove partial generation artifacts while keeping user inputs/config."""
        workspace = record.workspace
        for dirname in (
            "outputs",
            "slides",
            ".slide_images",
            "attachments",
            "downloads",
            "template_shell",
            ".interrupted",
            ".rollback",
        ):
            target = workspace / dirname
            if target.exists() and target.is_dir():
                shutil.rmtree(target, ignore_errors=True)

        for filename in (
            "intermediate_output.json",
            "deck_execution_state.json",
            "design_plan.md",
            "layout_mapping.yaml",
            "page_asset_plan.json",
            "page_execution_plan.md",
            "template_reference_profile.json",
            "resolved_intent.json",
            "template_match.json",
            ".template_profile.json",
            ".template_runtime_state.json",
        ):
            target = workspace / filename
            if target.exists() and target.is_file():
                try:
                    target.unlink()
                except OSError:
                    LOGGER.debug("Could not remove stale generation artifact %s", target, exc_info=True)

        for path in _candidate_export_files(workspace):
            try:
                path.unlink()
            except OSError:
                LOGGER.debug("Could not remove stale export %s", path, exc_info=True)

        runtime = getattr(record.session, "runtime", None)
        if runtime is not None and hasattr(runtime, "intermediate_output"):
            try:
                runtime.intermediate_output.clear()
            except Exception:
                LOGGER.debug("Could not clear runtime intermediate output for %s", record.session_id, exc_info=True)

        with record.state_lock:
            record.result = None
            record.runtime_resumed = False
            record.slide_count = 0
            record.artifact_counts = {}
            record.has_exports = False
            record.has_outputs = False
            record.latest_export_fresh = False
            record.export_status = "none"
            record.export_warnings = []
            record.last_artifact_at = ""
            record.updated_at = now_iso()
        self.refresh_session_metadata(record)

    def _snapshot_exports(self, record: WebSessionRecord, label: str) -> list[Path]:
        exports = _candidate_export_files(record.workspace)
        if not exports:
            return []
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = record.workspace / "downloads" / f"{label}_{stamp}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for path in exports:
            target = snapshot_dir / path.name
            if target.exists():
                target = snapshot_dir / f"{path.stem}_{uuid.uuid4().hex[:6]}{path.suffix}"
            try:
                shutil.copy2(path, target)
            except OSError:
                continue
            copied.append(target)
        return copied

    def _create_revision_checkpoint(self, record: WebSessionRecord, operation: dict[str, Any]) -> None:
        checkpoint_dir = operation_checkpoint_dir(record.workspace, operation)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        copied_dirs: list[str] = []
        for dirname in ("outputs", "slides", ".slide_images"):
            source = record.workspace / dirname
            if source.exists() and source.is_dir():
                shutil.copytree(source, checkpoint_dir / dirname, dirs_exist_ok=True)
                copied_dirs.append(dirname)

        copied_files: list[str] = []
        for rel in revision_checkpoint_files(record.workspace):
            source = record.workspace / rel
            if not source.exists() or not source.is_file():
                continue
            target = checkpoint_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(source, target)
            except OSError:
                continue
            copied_files.append(rel.as_posix())

        manifest = {
            "operation_id": operation.get("operation_id"),
            "created_at": now_iso(),
            "copied_dirs": copied_dirs,
            "copied_files": copied_files,
        }
        (checkpoint_dir / "checkpoint_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        operation["checkpoint_dir"] = str(checkpoint_dir)
        operation["checkpoint_created"] = bool(copied_dirs or copied_files)
        update_operation_state(record.workspace, operation, state="running", message="Revising deck")

    @staticmethod
    def copy_template_artifacts(source: Path, target: Path) -> None:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(source, target)

    def _reconcile_memory_save_state(self, record: WebSessionRecord) -> None:
        """Clear stale failed save state after a pending backup was retried.

        A running Web process can keep an old failed SessionRecord in memory even
        after a pending_consolidation_*.json file has been retried and removed
        by another request/process. In that case the stale record would keep
        rewriting `session_meta.json` as failed and the Activity panel would
        continue to show a retry card. Treat a matching saved retry trace as the
        source of truth and make the UI state converge back to saved.
        """
        if record.memory_save_status != "failed" or not record.memory_save_pending_backup:
            return

        pending_path = Path(record.memory_save_pending_backup).expanduser()
        if pending_path.exists():
            return

        recovery_event = self._find_memory_backup_recovery_event(record.workspace, pending_path)
        if not recovery_event:
            return

        saved_at = str(
            recovery_event.get("saved_at")
            or recovery_event.get("timestamp")
            or recovery_event.get("ts")
            or now_iso()
        )
        record.memory_save_busy = False
        record.memory_save_status = "saved"
        record.memory_save_progress = 100
        record.memory_save_stage = str(recovery_event.get("stage") or "complete")
        record.memory_save_message = str(
            recovery_event.get("message")
            or "Pending memory backup was saved to long-term memory."
        )
        record.memory_save_error = ""
        record.memory_save_pending_backup = ""
        record.memory_last_saved_at = saved_at
        record.memory_save_updated_at = saved_at
        record.updated_at = now_iso()

    @staticmethod
    def _find_memory_backup_recovery_event(workspace: Path, pending_path: Path) -> dict[str, Any] | None:
        pending_str = str(pending_path)
        pending_job_id = ""
        match = re.fullmatch(r"pending_consolidation_(.+)\.json", pending_path.name)
        if match:
            pending_job_id = match.group(1)

        trace_files = [
            workspace / ".history" / "memory_save_trace.jsonl",
            workspace / ".history" / "memory_save_progress.jsonl",
            workspace / ".memory" / "debug" / "events.jsonl",
        ]
        for trace_file in trace_files:
            if not trace_file.exists():
                continue
            try:
                lines = trace_file.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines[-1000:]):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                status = str(payload.get("status") or "")
                if status != "saved" and not bool(payload.get("saved")):
                    continue
                if str(payload.get("pending_consolidation_backup") or ""):
                    continue
                if str(payload.get("original_pending_backup") or "") == pending_str:
                    return payload
                if (
                    bool(payload.get("retried_pending_backup"))
                    and pending_job_id
                    and str(payload.get("job_id") or "") == pending_job_id
                ):
                    return payload
        return None

    def refresh_session_metadata(self, record: WebSessionRecord) -> dict[str, Any]:
        release_stale_slot = False
        with record.state_lock:
            signature = _workspace_summary_signature(record.workspace)
            if record.summary_cache_signature == signature and record.summary_cache_payload:
                cached = dict(record.summary_cache_payload)
                record.slide_count = int(cached.get("slide_count") or record.slide_count or 0)
                record.artifact_counts = dict(cached.get("artifact_counts") or record.artifact_counts or {})
                record.has_exports = bool(cached.get("has_exports"))
                record.has_outputs = bool(cached.get("has_outputs"))
                record.last_artifact_at = str(cached.get("last_artifact_at") or record.last_artifact_at or "")
                record.latest_export_fresh = bool(cached.get("latest_export_fresh", record.latest_export_fresh))
                record.export_status = str(cached.get("export_status") or record.export_status or "none")
                record.export_warnings = normalize_export_warnings(cached.get("export_warnings") or record.export_warnings)
                record.last_heartbeat_at = str(cached.get("last_heartbeat_at") or record.last_heartbeat_at or "")
                record.relative_workspace = str(cached.get("relative_workspace") or record.relative_workspace or workspace_relative_to(self.workspace_base, record.workspace))
                record.updated_at = str(cached.get("updated_at") or record.updated_at or now_iso())
                if isinstance(cached.get("last_operation"), dict):
                    record.last_operation = dict(cached["last_operation"])
                return dict(cached)
            summary = summarize_workspace(record.workspace)
            existing_meta = read_json_file(record.workspace / SESSION_META_FILENAME) or {}
            if isinstance(existing_meta, dict):
                record_has_autosave_state = any(
                    [
                        record.memory_autosave_last_checked_at,
                        record.memory_autosave_last_saved_at,
                        record.memory_autosave_skipped_at,
                        record.memory_autosave_skip_reason,
                        record.memory_autosave_manual_required,
                    ]
                )
                if existing_meta.get("memory_autosave_last_checked_at") and not record.memory_autosave_last_checked_at:
                    record.memory_autosave_last_checked_at = str(existing_meta.get("memory_autosave_last_checked_at") or "")
                if existing_meta.get("memory_autosave_last_saved_at") and not record.memory_autosave_last_saved_at:
                    record.memory_autosave_last_saved_at = str(existing_meta.get("memory_autosave_last_saved_at") or "")
                if existing_meta.get("memory_autosave_skipped_at") and not record.memory_autosave_skipped_at:
                    record.memory_autosave_skipped_at = str(existing_meta.get("memory_autosave_skipped_at") or "")
                if existing_meta.get("memory_autosave_skip_reason") and not record.memory_autosave_skip_reason:
                    record.memory_autosave_skip_reason = str(existing_meta.get("memory_autosave_skip_reason") or "")
                if _coerce_bool(existing_meta.get("memory_autosave_manual_required"), False) and not record_has_autosave_state:
                    record.memory_autosave_manual_required = True
            if record.last_operation is None:
                record.last_operation = read_operation_state(record.workspace)
            if record.state == "running" and not record.busy and not record.queued_operation:
                recovered = recover_stale_running_state(record.workspace, record.phase, summary)
                record.state = recovered["state"]  # type: ignore[assignment]
                record.phase = recovered["phase"]
                record.message = recovered["message"]
                record.error = recovered.get("error")
                record.recovery_status = str(recovered.get("recovery_status") or "")
                record.export_status = str(recovered.get("export_status") or record.export_status or "none")
                record.export_warnings = normalize_export_warnings(
                    recovered.get("export_warnings") or record.export_warnings
                )
                record.last_operation = recovered.get("last_operation") if isinstance(recovered.get("last_operation"), dict) else read_operation_state(record.workspace)
                record.updated_at = now_iso()
            elif record.state == "running" and record.busy and operation_is_stale(record.last_operation, self.operation_stale_timeout_seconds):
                recovered = recover_stale_running_state(record.workspace, record.phase, summary)
                if recovered.get("state") == "idle":
                    message = "Operation stopped without producing recoverable artifacts."
                    failed_operation = update_operation_state(
                        record.workspace,
                        record.last_operation,
                        state="failed",
                        phase="error",
                        message=message,
                        error={"status": "failed", "message": message, "recoverable": True},
                        recovery_status="stale_timeout",
                        export_status="none",
                        export_warnings=[],
                    )
                    recovered = {
                        "state": "failed",
                        "phase": "error",
                        "message": message,
                        "error": {"status": "failed", "message": message, "recoverable": True},
                        "recovery_status": "stale_timeout",
                        "export_status": "none",
                        "export_warnings": [],
                        "last_operation": failed_operation,
                    }
                record.busy = False
                record.queued_operation = ""
                record.queued_at = ""
                record.state = recovered["state"]  # type: ignore[assignment]
                record.phase = recovered["phase"]
                record.message = recovered["message"]
                record.error = recovered.get("error")
                record.recovery_status = str(recovered.get("recovery_status") or "")
                record.export_status = str(recovered.get("export_status") or record.export_status or "none")
                record.export_warnings = normalize_export_warnings(
                    recovered.get("export_warnings") or record.export_warnings
                )
                record.last_operation = recovered.get("last_operation") if isinstance(recovered.get("last_operation"), dict) else read_operation_state(record.workspace)
                record.updated_at = now_iso()
                release_stale_slot = True
            record.slide_count = int(summary["slide_count"])
            record.artifact_counts = dict(summary["artifact_counts"])
            record.has_exports = bool(summary["has_exports"])
            record.has_outputs = bool(summary["has_outputs"])
            record.last_artifact_at = str(summary["last_artifact_at"])
            record.latest_export_fresh = latest_export_fresh(record.workspace)
            record.export_status, record.export_warnings = infer_export_status(record.workspace, summary, record.last_operation)
            if record.state == "succeeded" and record.phase == "export_partial" and record.export_status == "strict":
                operation_type = str((record.last_operation or {}).get("operation_type") or "generation")
                record.phase = "revision_complete" if operation_type == "revision" else "generation_complete"
                record.message = "Deck revised" if operation_type == "revision" else "Deck generated"
                if record.last_operation:
                    record.last_operation = update_operation_state(
                        record.workspace,
                        record.last_operation,
                        phase=record.phase,
                        message=record.message,
                        export_status=record.export_status,
                        export_warnings=record.export_warnings,
                    )
            record.last_heartbeat_at = str((record.last_operation or {}).get("heartbeat_at") or "")
            if record.state == "running":
                operation_type = str((record.last_operation or {}).get("operation_type") or "").strip().lower()
                operation_phase = str((record.last_operation or {}).get("phase") or "").strip()
                operation_message = str((record.last_operation or {}).get("message") or "").strip()
                if operation_type in {"revision", "revise"}:
                    if record.phase == "generation_complete" or not record.phase or record.phase in {"ready", "generation"}:
                        record.phase = operation_phase if operation_phase and operation_phase != "generation_complete" else "revision"
                    if not record.message or record.message == "Deck generated":
                        record.message = operation_message or "Revising deck"
                elif operation_phase in {"revision", "revision_complete"}:
                    record.phase = operation_phase
                    if operation_message:
                        record.message = operation_message
            self._reconcile_memory_save_state(record)
            with self._scheduler_lock:
                queued_session_ids = {item.record.session_id for item in self._queue}
            if record.session_id not in queued_session_ids:
                record.queued_operation = ""
                record.queued_at = ""
            record.relative_workspace = workspace_relative_to(self.workspace_base, record.workspace)
            meta = {
                "session_id": record.session_id,
                "workspace": str(record.workspace),
                "display_name": record.display_name or humanize_display_name(record.instruction_summary),
                "instruction_summary": record.instruction_summary,
                "num_pages": record.num_pages,
                "language": record.language,
                "user_id": record.user_id,
                "memory_intent": record.memory_intent,
                "memory_profile_id": record.memory_profile_id,
                "memory_enabled": record.memory_enabled,
                "api_profile_id": record.api_profile_id,
                "api_profile_display_name": record.api_profile_display_name,
                "service_profile_id": record.api_profile_id,
                "service_profile_display_name": record.api_profile_display_name,
                "state": record.state,
                "phase": record.phase,
                "message": record.message,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
                "last_opened_at": record.last_opened_at,
                "slide_count": record.slide_count,
                "artifact_counts": record.artifact_counts,
                "has_exports": record.has_exports,
                "has_outputs": record.has_outputs,
                "last_artifact_at": record.last_artifact_at,
                "relative_workspace": record.relative_workspace,
                "recovery_status": record.recovery_status,
                "latest_export_fresh": record.latest_export_fresh,
                "export_status": record.export_status,
                "export_warnings": record.export_warnings,
                "last_heartbeat_at": record.last_heartbeat_at,
                "last_operation": record.last_operation,
                "queued_operation": record.queued_operation,
                "queued_at": record.queued_at,
                "memory_save_busy": record.memory_save_busy,
                "memory_last_saved_at": record.memory_last_saved_at,
                "memory_save_status": record.memory_save_status,
                "memory_save_progress": record.memory_save_progress,
                "memory_save_stage": record.memory_save_stage,
                "memory_save_message": record.memory_save_message,
                "memory_save_started_at": record.memory_save_started_at,
                "memory_save_updated_at": record.memory_save_updated_at,
                "memory_save_error": record.memory_save_error,
                "memory_save_pending_backup": record.memory_save_pending_backup,
                "memory_autosave_last_checked_at": record.memory_autosave_last_checked_at,
                "memory_autosave_last_saved_at": record.memory_autosave_last_saved_at,
                "memory_autosave_skipped_at": record.memory_autosave_skipped_at,
                "memory_autosave_skip_reason": record.memory_autosave_skip_reason,
                "memory_autosave_manual_required": record.memory_autosave_manual_required,
        }
        self._write_session_meta(record.workspace, meta)
        record.summary_cache_signature = signature
        record.summary_cache_payload = dict(meta)
        self._upsert_index_entry(meta)
        if release_stale_slot:
            with self._scheduler_lock:
                self._release_operation_slot(record)
            self._start_queued_operations()
        return meta

    def discover_legacy_sessions(self) -> None:
        if not self.workspace_base.exists():
            return
        for workspace in iter_known_workspaces(self.workspace_base):
            meta = self._load_or_infer_meta(workspace)
            if meta is None:
                continue
            self._upsert_index_entry(meta)

    def _begin_operation(self, record: WebSessionRecord, operation_type: str, phase: str, message: str) -> dict[str, Any]:
        with record.state_lock:
            if record.busy:
                raise RuntimeError("Session is already running")
            record.busy = True
            record.recovery_status = ""
            operation = create_operation_state(
                record.workspace,
                operation_type=operation_type,
                phase=phase,
                message=message,
                session_id=record.session_id,
            )
            record.last_operation = operation
            self._set_running(record, phase, message)
        self.refresh_session_metadata(record)
        return operation

    def mark_external_operation_queued(
        self,
        record: WebSessionRecord,
        *,
        operation_id: str,
        operation_type: str,
        message: str,
        queue_name: str = "",
    ) -> dict[str, Any]:
        with record.state_lock:
            if record.busy or record.queued_operation or record.memory_save_busy:
                raise RuntimeError("Session is already running")
            operation = create_operation_state(
                record.workspace,
                operation_id=operation_id,
                operation_type=operation_type,
                phase="queued",
                message=message,
                session_id=record.session_id,
                state="queued",
            )
            if queue_name:
                operation["queue_name"] = queue_name
                operation = write_operation_state(record.workspace, operation)
            record.last_operation = operation
            record.queued_operation = operation_type
            record.queued_at = str(operation.get("started_at") or now_iso())
            record.state = "running"
            record.phase = "queued"
            record.message = message
            record.error = None
            record.updated_at = now_iso()
        self.refresh_session_metadata(record)
        return operation

    def activate_external_operation(
        self,
        record: WebSessionRecord,
        *,
        operation_id: str,
        operation_type: str,
        phase: str,
        message: str,
        queue_name: str = "",
    ) -> dict[str, Any]:
        with record.state_lock:
            if record.busy:
                raise RuntimeError("Session is already running")
            record.busy = True
            record.recovery_status = ""
            record.queued_operation = ""
            record.queued_at = ""
            operation = create_operation_state(
                record.workspace,
                operation_id=operation_id,
                operation_type=operation_type,
                phase=phase,
                message=message,
                session_id=record.session_id,
                state="running",
            )
            if queue_name:
                operation["queue_name"] = queue_name
                operation = write_operation_state(record.workspace, operation)
            record.last_operation = operation
            self._set_running(record, phase, message)
        self.refresh_session_metadata(record)
        return operation

    def apply_external_operation_payload(
        self,
        record: WebSessionRecord,
        *,
        operation_type: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist the user-visible run snapshot before an external worker starts."""
        with record.state_lock:
            if operation_type == "generation":
                instruction = str(payload.get("instruction") or "")
                record.instruction_summary = summarize_instruction(instruction)
                record.display_name = humanize_display_name(record.instruction_summary)
                record.num_pages = payload.get("num_pages") or None
                record.language = normalize_language(payload.get("language"), record.language)
                record.session.runtime.language = record.language
                record.session.options.language = record.language  # type: ignore[assignment]
                if "memory_enabled" in payload:
                    record.memory_enabled = _coerce_bool(payload.get("memory_enabled"), record.memory_enabled)
            elif operation_type == "revision":
                feedback = str(payload.get("feedback") or "")
                if feedback:
                    record.message = feedback

            if "memory_intent" in payload:
                record.memory_intent = str(payload.get("memory_intent") or "")
            if "memory_profile_id" in payload:
                record.memory_profile_id = str(payload.get("memory_profile_id") or "")
            if "service_profile_id" in payload or "api_profile_id" in payload:
                record.api_profile_id = str(payload.get("service_profile_id") or payload.get("api_profile_id") or "")
            record.last_opened_at = now_iso()
            record.updated_at = now_iso()
        self.refresh_session_metadata(record)

    async def run_external_generate(
        self,
        record: WebSessionRecord,
        *,
        operation_id: str,
        payload: dict[str, Any],
        attachments: list[Path],
        queue_name: str = "",
    ) -> WebSessionRecord:
        requested_profile = (
            str(payload.get("service_profile_id") or payload.get("api_profile_id") or "")
            if ("service_profile_id" in payload or "api_profile_id" in payload)
            else None
        )
        requested_memory_enabled = (
            _coerce_bool(payload.get("memory_enabled"), record.memory_enabled)
            if "memory_enabled" in payload
            else None
        )
        self._ensure_named_workspace(record, str(payload.get("instruction") or ""))
        await self._apply_api_profile_to_record(record, requested_profile, requested_memory_enabled)
        language = normalize_language(payload.get("language"), record.language)
        with record.state_lock:
            record.instruction_summary = summarize_instruction(str(payload.get("instruction") or ""))
            record.display_name = humanize_display_name(record.instruction_summary)
            record.num_pages = payload.get("num_pages") or None
            record.language = language
            record.session.runtime.language = language
            record.session.options.language = language  # type: ignore[assignment]
            record.memory_intent = str(payload.get("memory_intent") or "")
            if "memory_profile_id" in payload:
                record.memory_profile_id = str(payload.get("memory_profile_id") or "")
            record.last_opened_at = now_iso()
        self.refresh_session_metadata(record)
        self.activate_external_operation(
            record,
            operation_id=operation_id,
            operation_type="generation",
            phase="generation",
            message="Generating deck",
            queue_name=queue_name,
        )
        await self._run_generate(record, payload, attachments)
        return record

    async def run_external_revision(
        self,
        record: WebSessionRecord,
        *,
        operation_id: str,
        feedback: str,
        memory_intent: str,
        memory_profile_id: str = "",
        api_profile_id: str | None = None,
        queue_name: str = "",
    ) -> WebSessionRecord:
        await self._apply_api_profile_to_record(record, api_profile_id)
        effective_memory_intent = str(memory_intent or record.memory_intent or "").strip()
        with record.state_lock:
            if effective_memory_intent:
                record.memory_intent = effective_memory_intent
            record.memory_profile_id = str(memory_profile_id or "")
            record.last_opened_at = now_iso()
        self.refresh_session_metadata(record)
        self._snapshot_exports(record, "before_revision")
        operation = self.activate_external_operation(
            record,
            operation_id=operation_id,
            operation_type="revision",
            phase="revision",
            message="Revising deck",
            queue_name=queue_name,
        )
        self._create_revision_checkpoint(record, operation)
        await self._run_revise(record, feedback, effective_memory_intent)
        return record

    def _enqueue_or_start(self, item: QueuedOperation) -> bool:
        with item.record.state_lock:
            if item.record.busy or item.record.queued_operation or item.record.memory_save_busy:
                raise RuntimeError("Session is already running")
        with self._scheduler_lock:
            if self._can_start_operation(item.record):
                self._claim_operation_slot(item.record)
                return True
            self._queue.append(item)
            queue_phase = self._queue_phase(item.record)
        with item.record.state_lock:
            item.record.queued_operation = item.operation_type
            item.record.queued_at = item.queued_at
            item.record.state = "running"
            item.record.phase = queue_phase
            item.record.message = (
                "Queued behind other operations using this API profile"
                if queue_phase == "queued_api_profile"
                else "Queued behind other active Web Studio operations"
            )
            item.record.updated_at = now_iso()
            item.record.last_operation = create_operation_state(
                item.record.workspace,
                operation_type=item.operation_type,
                phase=queue_phase,
                message=item.record.message,
                session_id=item.record.session_id,
                state="queued",
            )
        self.refresh_session_metadata(item.record)
        return False

    def _can_start_operation(self, record: WebSessionRecord) -> bool:
        return (
            record.session_id not in self._active_sessions
            and len(self._active_sessions) < self.max_active_operations
            and self._profile_slot_available(record)
        )

    def _profile_slot_key(self, record: WebSessionRecord) -> str:
        return str(record.api_profile_id or "").strip()

    def _profile_slot_limit(self, record: WebSessionRecord) -> int:
        key = self._profile_slot_key(record)
        if not key:
            return self.max_active_operations
        return self.api_profiles.max_concurrent(key, user_id=record.user_id)

    def _profile_slot_available(self, record: WebSessionRecord) -> bool:
        key = self._profile_slot_key(record)
        if not key:
            return True
        return self._active_profile_counts.get(key, 0) < self._profile_slot_limit(record)

    def _queue_phase(self, record: WebSessionRecord) -> str:
        if len(self._active_sessions) < self.max_active_operations and not self._profile_slot_available(record):
            return "queued_api_profile"
        return "queued"

    def _claim_operation_slot(self, record: WebSessionRecord) -> None:
        self._active_sessions.add(record.session_id)
        key = self._profile_slot_key(record)
        if key:
            self._active_session_profiles[record.session_id] = key
            self._active_profile_counts[key] = self._active_profile_counts.get(key, 0) + 1

    def _release_operation_slot(self, record: WebSessionRecord) -> None:
        self._active_sessions.discard(record.session_id)
        key = self._active_session_profiles.pop(record.session_id, "")
        if key:
            remaining = max(0, self._active_profile_counts.get(key, 0) - 1)
            if remaining:
                self._active_profile_counts[key] = remaining
            else:
                self._active_profile_counts.pop(key, None)

    def _start_queued_operations(self) -> None:
        while True:
            with self._scheduler_lock:
                if len(self._active_sessions) >= self.max_active_operations or not self._queue:
                    return
                next_index = next(
                    (
                        idx for idx, item in enumerate(self._queue)
                        if item.record.session_id not in self._active_sessions and not item.record.busy
                    ),
                    -1,
                )
                if next_index < 0:
                    return
                item = self._queue.pop(next_index)
                self._claim_operation_slot(item.record)
            with item.record.state_lock:
                item.record.queued_operation = ""
                item.record.queued_at = ""
            self._dispatch_operation(item)

    def _dispatch_operation(self, item: QueuedOperation) -> None:
        try:
            if item.operation_type == "generation":
                self._begin_operation(item.record, "generation", "generation", "Generating deck")
                future = item.record.worker.submit(self._run_generate(item.record, item.payload or {}, item.attachments))
                future.add_done_callback(lambda done, record=item.record: self._handle_background_future_done(record, done))
                return
            if item.operation_type == "revision":
                self._snapshot_exports(item.record, "before_revision")
                operation = self._begin_operation(item.record, "revision", "revision", "Revising deck")
                self._create_revision_checkpoint(item.record, operation)
                future = item.record.worker.submit(self._run_revise(item.record, item.feedback, item.memory_intent))
                future.add_done_callback(lambda done, record=item.record: self._handle_background_future_done(record, done))
                return
            raise RuntimeError(f"Unsupported queued operation: {item.operation_type}")
        except Exception:
            with self._scheduler_lock:
                self._release_operation_slot(item.record)
            with item.record.state_lock:
                item.record.busy = False
                item.record.queued_operation = ""
                item.record.queued_at = ""
            self.refresh_session_metadata(item.record)
            self._start_queued_operations()
            raise

    def _handle_background_future_done(self, record: WebSessionRecord, future: Future) -> None:
        if not record.busy:
            return
        try:
            exc = future.exception()
        except BaseException as err:  # noqa: BLE001
            exc = err
        if exc is None:
            return
        with record.state_lock:
            if not record.busy:
                return
        self._set_error(record, exc)
        self._finish_operation(record)

    def _finish_operation(self, record: WebSessionRecord) -> None:
        with record.state_lock:
            record.busy = False
            record.queued_operation = ""
            record.queued_at = ""
        with self._scheduler_lock:
            self._release_operation_slot(record)
        self.refresh_session_metadata(record)
        self._start_queued_operations()

    def _set_running(self, record: WebSessionRecord, phase: str, message: str) -> None:
        with record.state_lock:
            record.state = "running"
            record.phase = phase
            record.message = message
            record.error = None
            record.updated_at = now_iso()

    def _set_success(self, record: WebSessionRecord, phase: str, message: str) -> None:
        with record.state_lock:
            export_status, export_warnings = infer_export_status(record.workspace, operation=record.last_operation)
            if export_status == "partial":
                phase = "export_partial"
                message = f"{message} with export warnings"
            record.state = "succeeded"
            record.phase = phase
            record.message = message
            record.export_status = export_status
            record.export_warnings = export_warnings
            record.updated_at = now_iso()
            if record.last_operation:
                record.last_operation = update_operation_state(
                    record.workspace,
                    record.last_operation,
                    state="succeeded",
                    phase=phase,
                    message=message,
                    export_status=export_status,
                    export_warnings=export_warnings,
                )

    def _set_error(self, record: WebSessionRecord, exc: BaseException) -> None:
        with record.state_lock:
            export_status, export_warnings = infer_export_status(record.workspace, operation=record.last_operation)
            operation_type = str((record.last_operation or {}).get("operation_type") or "")
            partial_success_allowed = (
                operation_type == "generation" and export_status in {"strict", "partial"}
            ) or (
                operation_type == "revision"
                and export_status in {"strict", "partial"}
                and operation_has_fresh_export_since_start(record.workspace, record.last_operation)
            )
            if partial_success_allowed:
                warning = str(exc) or exc.__class__.__name__
                if warning and warning not in export_warnings:
                    export_warnings = [*export_warnings, warning]
                record.state = "succeeded"
                record.phase = "export_partial"
                record.message = (
                    "Deck revised with export warnings"
                    if operation_type == "revision"
                    else "Deck generated with export warnings"
                )
                record.error = None
                record.export_status = "partial"
                record.export_warnings = export_warnings
                record.updated_at = now_iso()
                if record.last_operation:
                    record.last_operation = update_operation_state(
                        record.workspace,
                        record.last_operation,
                        state="succeeded",
                        phase="export_partial",
                        message=record.message,
                        recovery_status="partial_export_after_error",
                        export_status="partial",
                        export_warnings=export_warnings,
                    )
                return
            record.state = "failed"
            record.phase = "error"
            record.message = str(exc)
            record.export_status = export_status
            record.export_warnings = export_warnings
            record.error = {
                "status": "failed",
                "message": str(exc),
                "detail": traceback.format_exc(limit=20),
                "recoverable": True,
            }
            record.updated_at = now_iso()
            if record.last_operation:
                record.last_operation = update_operation_state(
                    record.workspace,
                    record.last_operation,
                    state="failed",
                    phase="error",
                    message=str(exc),
                    error=record.error,
                    export_status=export_status,
                    export_warnings=export_warnings,
                )

    def _build_workspace_path(self, *, session_id: str, instruction: str, user_id: str = "") -> Path:
        now = datetime.now().astimezone()
        date_part = now.strftime("%Y%m%d")
        time_part = now.strftime("%H%M%S")
        slug = slugify_instruction(instruction)
        if self.user_isolation:
            safe_user = slugify_user_id(user_id or "web-demo")
            base_dir = self.workspace_base / "users" / safe_user / "sessions" / date_part
        else:
            base_dir = self.workspace_base / date_part
        candidate = base_dir / f"{time_part}_{slug}_{session_id}"
        suffix = 1
        while candidate.exists():
            candidate = base_dir / f"{time_part}_{slug}_{session_id}_{suffix:02d}"
            suffix += 1
        return candidate

    def _ensure_named_workspace(self, record: WebSessionRecord, instruction: str) -> None:
        current = record.workspace
        desired_slug = slugify_instruction(instruction)
        current_slug = extract_workspace_slug(current.name, record.session_id)
        if current_slug and current_slug == desired_slug and current.exists():
            return
        renamed = self._build_workspace_path(session_id=record.session_id, instruction=instruction, user_id=record.user_id)
        if current.resolve() == renamed.resolve():
            return
        renamed.parent.mkdir(parents=True, exist_ok=True)
        try:
            current.rename(renamed)
        except OSError:
            LOGGER.warning(
                "Could not rename workspace %s to %s; continuing with the existing workspace",
                current,
                renamed,
                exc_info=True,
            )
            return
        record.workspace = renamed
        record.session.workspace = renamed
        record.session.runtime.workspace = renamed
        record.last_operation = read_operation_state(record.workspace)
        if getattr(record.session.runtime, "session_id", None) is None:
            record.session.runtime.session_id = record.session_id

    def _write_session_meta(self, workspace: Path, meta: dict[str, Any]) -> None:
        try:
            (workspace / SESSION_META_FILENAME).write_text(
                json.dumps(meta, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _upsert_index_entry(self, meta: dict[str, Any]) -> None:
        entries = self._load_index_entries()
        session_id = str(meta.get("session_id") or "")
        workspace = str(meta.get("workspace") or "")
        filtered = [
            item for item in entries
            if str(item.get("session_id") or "") != session_id and str(item.get("workspace") or "") != workspace
        ]
        filtered.append(meta)
        filtered.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("w", encoding="utf-8") as fh:
            for item in filtered:
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")

    def _load_index_entries(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                entries.append(item)
        return entries

    def _find_session_meta(self, *, session_id: str | None = None, workspace: Path | None = None) -> dict[str, Any] | None:
        for item in self._load_index_entries():
            if session_id and str(item.get("session_id") or "") == session_id:
                return item
            if workspace is not None and Path(str(item.get("workspace") or "")).expanduser() == workspace:
                return item
        if workspace is not None:
            return self._load_or_infer_meta(workspace)
        return None

    def _load_or_infer_meta(self, workspace: Path) -> dict[str, Any] | None:
        meta_path = workspace / SESSION_META_FILENAME
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = None
            if isinstance(data, dict):
                data.setdefault("workspace", str(workspace))
                data.setdefault("relative_workspace", workspace_relative_to(self.workspace_base, workspace))
                session_id = str(data.get("session_id") or infer_session_id_from_workspace(workspace))
                active_record = self.records.get(session_id)
                if str(data.get("phase") or "") in {"queued", "queued_api_profile"} and not (active_record and active_record.queued_operation):
                    data["state"] = "idle"
                    data["phase"] = "ready"
                    data["message"] = ""
                    data["queued_operation"] = ""
                    data["queued_at"] = ""
                if str(data.get("state") or "") == "running" and not (active_record and active_record.busy):
                    data.update(recover_stale_running_state(workspace, str(data.get("phase") or "")))
                if not _session_has_active_operation(data, active_record=active_record):
                    data["last_operation"] = reconcile_inactive_last_operation(data.get("last_operation"))
                request = read_json_file(workspace / ".input_request.json")
                resolved_intent = read_json_file(workspace / "resolved_intent.json")
                recovered_memory_intent = recover_workspace_memory_intent(
                    data.get("memory_intent"),
                    request,
                    resolved_intent,
                )
                if recovered_memory_intent:
                    data["memory_intent"] = recovered_memory_intent
                data.setdefault("last_operation", reconcile_inactive_last_operation(read_operation_state(workspace)))
                summary = summarize_workspace(workspace)
                data["slide_count"] = int(summary.get("slide_count") or data.get("slide_count") or 0)
                data["artifact_counts"] = dict(summary.get("artifact_counts") or data.get("artifact_counts") or {})
                data["has_exports"] = bool(summary.get("has_exports"))
                data["has_outputs"] = bool(summary.get("has_outputs"))
                data["last_artifact_at"] = str(summary.get("last_artifact_at") or data.get("last_artifact_at") or "")
                data["latest_export_fresh"] = latest_export_fresh(workspace)
                export_status, export_warnings = infer_export_status(workspace, operation=data.get("last_operation"))
                data["export_status"] = export_status
                data["export_warnings"] = export_warnings
                data["last_heartbeat_at"] = str((data.get("last_operation") or {}).get("heartbeat_at") or "")
                data.setdefault("user_id", "web-demo")
                data.setdefault("memory_enabled", True)
                data.setdefault("memory_profile_id", "")
                data.setdefault("api_profile_id", "")
                data.setdefault("api_profile_display_name", "")
                data.setdefault("service_profile_id", data.get("api_profile_id") or "")
                data.setdefault("service_profile_display_name", data.get("api_profile_display_name") or "")
                data.setdefault("queued_operation", "")
                data.setdefault("queued_at", "")
                data.setdefault("memory_save_busy", False)
                data.setdefault("memory_last_saved_at", "")
                data.setdefault("memory_save_status", "")
                data.setdefault("memory_save_progress", 0)
                data.setdefault("memory_save_stage", "")
                data.setdefault("memory_save_message", "")
                data.setdefault("memory_save_started_at", "")
                data.setdefault("memory_save_updated_at", "")
                data.setdefault("memory_save_error", "")
                data.setdefault("memory_save_pending_backup", "")
                data.setdefault("memory_autosave_last_checked_at", "")
                data.setdefault("memory_autosave_last_saved_at", "")
                data.setdefault("memory_autosave_skipped_at", "")
                data.setdefault("memory_autosave_skip_reason", "")
                data.setdefault("memory_autosave_manual_required", False)
                return data

        session_id = infer_session_id_from_workspace(workspace)
        created_at = datetime.fromtimestamp(workspace.stat().st_ctime).astimezone().isoformat()
        updated_at = datetime.fromtimestamp(workspace.stat().st_mtime).astimezone().isoformat()
        request = read_json_file(workspace / ".input_request.json")
        instruction = ""
        num_pages = None
        memory_intent = ""
        resolved_intent = read_json_file(workspace / "resolved_intent.json")
        if isinstance(request, dict):
            instruction = str(request.get("instruction") or "")
            num_pages = request.get("num_pages")
            memory_intent = str(request.get("memory_intent") or "")
        memory_intent = recover_workspace_memory_intent(memory_intent, request, resolved_intent)
        summary = summarize_workspace(workspace)
        operation = read_operation_state(workspace)
        export_status, export_warnings = infer_export_status(workspace, summary, operation)
        return {
            "session_id": session_id,
            "workspace": str(workspace),
            "display_name": humanize_display_name(summarize_instruction(instruction) or workspace.name),
            "instruction_summary": summarize_instruction(instruction),
            "num_pages": num_pages,
            "language": "en",
            "user_id": "web-demo",
            "memory_intent": memory_intent,
            "memory_profile_id": "",
            "memory_enabled": True,
            "api_profile_id": "",
            "api_profile_display_name": "",
            "service_profile_id": "",
            "service_profile_display_name": "",
            "state": infer_workspace_state(workspace),
            "phase": infer_workspace_phase(workspace),
            "message": "",
            "created_at": created_at,
            "updated_at": updated_at,
            "last_opened_at": updated_at,
            "slide_count": summary["slide_count"],
            "artifact_counts": summary["artifact_counts"],
            "has_exports": summary["has_exports"],
            "has_outputs": summary["has_outputs"],
            "last_artifact_at": summary["last_artifact_at"],
            "relative_workspace": workspace_relative_to(self.workspace_base, workspace),
            "recovery_status": "",
            "latest_export_fresh": latest_export_fresh(workspace),
            "export_status": export_status,
            "export_warnings": export_warnings,
            "last_heartbeat_at": str((operation or {}).get("heartbeat_at") or ""),
            "last_operation": operation,
            "memory_save_busy": False,
            "memory_last_saved_at": "",
            "memory_save_status": "",
            "memory_save_progress": 0,
            "memory_save_stage": "",
            "memory_save_message": "",
            "memory_save_started_at": "",
            "memory_save_updated_at": "",
            "memory_save_error": "",
            "memory_save_pending_backup": "",
            "memory_autosave_last_checked_at": "",
            "memory_autosave_last_saved_at": "",
            "memory_autosave_skipped_at": "",
            "memory_autosave_skip_reason": "",
            "memory_autosave_manual_required": False,
        }


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def path_within_or_same(path: Path, root: Path) -> bool:
    try:
        resolved = path.resolve()
        resolved_root = root.resolve()
        if resolved == resolved_root:
            return True
        resolved.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def append_memory_save_trace(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    try:
        history_dir = record.workspace / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "timestamp": now_iso(),
            "session_id": record.session_id,
            "reason": "manual_web_save",
            **payload,
        }
        with (history_dir / "memory_save_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).debug("Failed to append Web memory save trace", exc_info=True)


def update_memory_save_progress(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    progress = payload.get("progress", record.memory_save_progress)
    try:
        progress_int = int(progress)
    except (TypeError, ValueError):
        progress_int = record.memory_save_progress
    with record.state_lock:
        record.memory_save_status = str(payload.get("status") or record.memory_save_status or "saving")
        record.memory_save_progress = max(0, min(100, progress_int))
        record.memory_save_stage = str(payload.get("stage") or record.memory_save_stage)
        record.memory_save_message = str(payload.get("message") or record.memory_save_message)
        record.memory_save_updated_at = str(payload.get("updated_at") or now_iso())
        if payload.get("pending_consolidation_backup"):
            record.memory_save_pending_backup = str(payload.get("pending_consolidation_backup") or "")
        if record.memory_save_status == "failed":
            record.memory_save_error = str(payload.get("message") or record.memory_save_error)
        record.updated_at = now_iso()
        meta_path = record.workspace / SESSION_META_FILENAME
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        except (json.JSONDecodeError, OSError):
            data = {}
        if isinstance(data, dict):
            data.update(
                {
                    "session_id": record.session_id,
                    "workspace": str(record.workspace),
                    "updated_at": record.updated_at,
                    "memory_save_busy": record.memory_save_busy,
                    "memory_save_status": record.memory_save_status,
                    "memory_save_progress": record.memory_save_progress,
                    "memory_save_stage": record.memory_save_stage,
                    "memory_save_message": record.memory_save_message,
                    "memory_save_started_at": record.memory_save_started_at,
                    "memory_save_updated_at": record.memory_save_updated_at,
                    "memory_save_error": record.memory_save_error,
                    "memory_save_pending_backup": record.memory_save_pending_backup,
                }
            )
            meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_memory_save_progress_trace(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    try:
        history_dir = record.workspace / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "timestamp": now_iso(),
            "session_id": record.session_id,
            "reason": "manual_web_save",
            **payload,
        }
        with (history_dir / "memory_save_progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).debug("Failed to append Web memory save progress trace", exc_info=True)


def append_memory_save_event(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    try:
        event_dir = record.workspace / ".memory" / "debug"
        event_dir.mkdir(parents=True, exist_ok=True)
        saved = bool(payload.get("saved"))
        event_payload = {
            "event": "memory_save",
            "status": "saved" if saved else str(payload.get("status") or "failed"),
            "message": payload.get("message") or (
                "Working memory was saved to long-term memory."
                if saved
                else "Memory save did not complete."
            ),
            "session_id": record.session_id,
            "job_id": payload.get("job_id") or "",
            "round_count": payload.get("round_count") or 0,
            "pending_consolidation_backup": payload.get("pending_consolidation_backup") or "",
            "ts": now_iso(),
        }
        with (event_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).debug("Failed to append Web memory save event", exc_info=True)


def append_profile_editor_trace(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    try:
        history_dir = record.workspace / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        trace_payload = {
            "timestamp": now_iso(),
            "session_id": record.session_id,
            "event": "profile_editor_save",
            "user_id": payload.get("user_id") or record.user_id,
            "intent": payload.get("intent") or record.memory_intent or "default",
            "version": payload.get("version") or {},
            "last_updated": payload.get("last_updated") or {},
        }
        with (history_dir / "profile_editor_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace_payload, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).debug("Failed to append Web profile editor trace", exc_info=True)


def append_profile_editor_event(record: WebSessionRecord, payload: dict[str, Any]) -> None:
    try:
        event_dir = record.workspace / ".memory" / "debug"
        event_dir.mkdir(parents=True, exist_ok=True)
        event_payload = {
            "event": "profile_editor_save",
            "status": "saved",
            "message": "User profile was saved from Web Studio.",
            "session_id": record.session_id,
            "user_id": payload.get("user_id") or record.user_id,
            "intent": payload.get("intent") or record.memory_intent or "default",
            "version": payload.get("version") or {},
            "ts": now_iso(),
        }
        with (event_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")
    except Exception:
        logging.getLogger(__name__).debug("Failed to append Web profile editor event", exc_info=True)


def set_session_runtime_logger(record: WebSessionRecord) -> None:
    history = record.workspace / ".history"
    history.mkdir(parents=True, exist_ok=True)
    reset_context_logger()
    logger = set_logger(
        f"memslides-web-runtime-{record.session_id}",
        history / "runtime.log",
    )
    mirror_log_file(record.workspace / ".history" / "memslides-loop.log", history / "runtime.log")
    web_logger = logging.getLogger("memslides.web")
    if not any(
        isinstance(handler, logging.FileHandler)
        and str(Path(getattr(handler, "baseFilename", ""))) == str(history / "web.log")
        for handler in web_logger.handlers
    ):
        web_logger.setLevel(logging.DEBUG)
        web_logger.propagate = False
        file_handler = logging.FileHandler(history / "web.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(levelname)-4s %(asctime)s [%(name)s] %(message)s"))
        web_logger.addHandler(file_handler)
    logger.info(
        "Web operation context session_id=%s workspace=%s operation_id=%s",
        record.session_id,
        record.workspace,
        (record.last_operation or {}).get("operation_id"),
    )


def mirror_log_file(source: Path, target: Path) -> None:
    if source == target or not source.exists() or target.exists():
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    except OSError:
        pass


def read_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def recover_workspace_memory_intent(
    current: Any,
    request: dict[str, Any] | None,
    resolved_intent: dict[str, Any] | None = None,
) -> str:
    current_text = str(current or "").strip()
    request_text = str((request or {}).get("memory_intent") or "").strip() if isinstance(request, dict) else ""
    resolved_candidates = [
        (resolved_intent or {}).get("resolved_memory_intent") if isinstance(resolved_intent, dict) else "",
        (resolved_intent or {}).get("resolved_scenario_intent") if isinstance(resolved_intent, dict) else "",
        (resolved_intent or {}).get("explicit_request_intent") if isinstance(resolved_intent, dict) else "",
    ]
    weak_defaults = {
        "demo_style_memory",
        "read_template_and_style_guidance",
        "smoke_full_memory",
        "presentation_revision_memory",
        "web_custom_memory",
    }
    if request_text and (not current_text or current_text in weak_defaults):
        return request_text
    if current_text:
        return current_text
    if request_text:
        return request_text
    for value in resolved_candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def summarize_instruction(instruction: str) -> str:
    text = " ".join(str(instruction or "").strip().split())
    if not text:
        return ""
    if len(text) <= 96:
        return text
    return text[:93].rstrip() + "..."


def humanize_display_name(summary: str) -> str:
    cleaned = (summary or "").strip()
    if not cleaned:
        return "Untitled deck"
    return cleaned[:120]


def normalize_language(value: Any, default: str = "en") -> str:
    text = str(value or "").strip().lower()
    if text in {"zh", "cn", "chinese", "中文", "汉语"}:
        return "zh"
    if text in {"en", "english"}:
        return "en"
    return default if default in {"zh", "en"} else "en"


def slugify_instruction(instruction: str, *, max_length: int = 48) -> str:
    summary = summarize_instruction(instruction)
    if not summary:
        return "untitled_deck"
    ascii_text = summary.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        return "untitled_deck"
    return slug[:max_length].rstrip("_") or "untitled_deck"


def slugify_user_id(user_id: str, *, max_length: int = 48) -> str:
    text = str(user_id or "web-demo").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        return "web_demo"
    return text[:max_length].rstrip("_") or "web_demo"


def iter_known_workspaces(workspace_base: Path):
    if not workspace_base.exists():
        return
    for date_dir in sorted(workspace_base.iterdir()):
        if not date_dir.is_dir() or not re.fullmatch(r"\d{8}", date_dir.name):
            continue
        for workspace in sorted(date_dir.iterdir()):
            if workspace.is_dir():
                yield workspace
    users_root = workspace_base / "users"
    if not users_root.exists():
        return
    for user_dir in sorted(users_root.iterdir()):
        sessions_root = user_dir / "sessions"
        if not sessions_root.exists():
            continue
        for date_dir in sorted(sessions_root.iterdir()):
            if not date_dir.is_dir() or not re.fullmatch(r"\d{8}", date_dir.name):
                continue
            for workspace in sorted(date_dir.iterdir()):
                if workspace.is_dir():
                    yield workspace


def summarize_workspace(workspace: Path) -> dict[str, Any]:
    counts = {"slide": 0, "pptx": 0, "pdf": 0, "memory": 0, "image": 0, "log": 0, "file": 0}
    last_artifact_at = 0.0
    has_outputs = False
    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".locks", "__pycache__"} for part in path.parts):
            continue
        suffix = path.suffix.lower()
        kind = "file"
        if ".memory" in path.parts or "memory" in path.parts:
            kind = "memory"
        elif ".history" in path.parts or suffix in {".log", ".jsonl"}:
            kind = "log"
        elif suffix == ".html":
            kind = "slide"
        elif suffix == ".pptx":
            kind = "pptx"
        elif suffix == ".pdf":
            kind = "pdf"
        elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
            kind = "image"
        counts[kind] = counts.get(kind, 0) + 1
        last_artifact_at = max(last_artifact_at, path.stat().st_mtime)
        if path.parts and "outputs" in path.parts:
            has_outputs = True
    return {
        "slide_count": count_current_slides(workspace),
        "artifact_counts": counts,
        "has_exports": bool(counts["pptx"] or counts["pdf"]),
        "has_outputs": has_outputs or bool(counts["slide"]),
        "last_artifact_at": datetime.fromtimestamp(last_artifact_at).astimezone().isoformat() if last_artifact_at else "",
    }


def count_current_slides(workspace: Path) -> int:
    for slide_dir in (workspace / "outputs", workspace / "slides", workspace):
        if not slide_dir.exists() or not slide_dir.is_dir():
            continue
        count = len(list(slide_dir.glob("*.html")))
        if count:
            return count
    return 0


def revision_state_snapshot(workspace: Path) -> dict[str, Any]:
    """Capture only artifacts that prove a revision did real deck work."""
    slide_hashes: dict[str, str] = {}
    for path in sorted((workspace / "outputs").glob("slide_*.html")):
        slide_hashes[path.name] = file_digest(path)

    exports: dict[str, int] = {}
    for pattern in ("modification_*.pptx", "modification_*.pdf", "*.pptx", "*.pdf"):
        for path in sorted(workspace.glob(pattern)):
            try:
                exports[path.name] = int(path.stat().st_mtime_ns)
            except OSError:
                continue

    round_artifacts: dict[str, int] = {}
    rounds_dir = workspace / ".memory" / "rounds"
    if rounds_dir.exists():
        for path in sorted(rounds_dir.rglob("*")):
            if not path.is_file():
                continue
            try:
                round_artifacts[str(path.relative_to(workspace))] = int(path.stat().st_mtime_ns)
            except OSError:
                continue

    return {
        "slides": slide_hashes,
        "exports": exports,
        "round_artifacts": round_artifacts,
    }


def revision_has_fresh_export(
    workspace: Path,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any] | None = None,
) -> bool:
    """Return whether this revision produced a new or newer modification export."""
    after_snapshot = after_snapshot or revision_state_snapshot(workspace)
    before_exports = before_snapshot.get("exports", {}) if isinstance(before_snapshot, dict) else {}
    after_exports = after_snapshot.get("exports", {}) if isinstance(after_snapshot, dict) else {}
    if not isinstance(before_exports, dict) or not isinstance(after_exports, dict):
        return False
    for name, after_mtime in after_exports.items():
        if not re.fullmatch(r"modification_\d+\.(pptx|pdf)", str(name), re.IGNORECASE):
            continue
        try:
            after_value = int(after_mtime)
            before_value = int(before_exports.get(name, 0) or 0)
        except (TypeError, ValueError):
            continue
        if after_value > before_value:
            return True
    return False


def _candidate_export_files(workspace: Path) -> list[Path]:
    seen: set[Path] = set()
    exports: list[Path] = []
    for pattern in ("*.pptx", "*.pdf", "outputs/*.pptx", "outputs/*.pdf"):
        for path in sorted(workspace.glob(pattern)):
            if not path.is_file() or "downloads" in path.parts:
                continue
            if INTERRUPTED_DIRNAME in path.parts:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            exports.append(path)
    return exports


def create_operation_state(
    workspace: Path,
    *,
    operation_id: str = "",
    operation_type: str,
    phase: str,
    message: str,
    session_id: str = "",
    state: str = "running",
) -> dict[str, Any]:
    operation = {
        "operation_id": operation_id or uuid.uuid4().hex[:12],
        "session_id": session_id,
        "operation_type": operation_type,
        "state": state,
        "phase": phase,
        "message": message,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "heartbeat_at": now_iso(),
        "export_status": "none",
        "export_warnings": [],
        "checkpoint_dir": "",
        "checkpoint_created": False,
        "recovery_status": "",
        "error": None,
    }
    return write_operation_state(workspace, operation)


def read_operation_state(workspace: Path) -> dict[str, Any] | None:
    data = read_json_file(workspace / OPERATION_STATE_FILENAME)
    return data if isinstance(data, dict) else None


def write_operation_state(workspace: Path, operation: dict[str, Any]) -> dict[str, Any]:
    operation = dict(operation)
    operation["updated_at"] = now_iso()
    path = workspace / OPERATION_STATE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(operation, ensure_ascii=True, indent=2), encoding="utf-8")
    operation_dir = workspace / OPERATIONS_DIRNAME / str(operation.get("operation_id") or "unknown")
    operation_dir.mkdir(parents=True, exist_ok=True)
    (operation_dir / OPERATION_STATE_FILENAME).write_text(
        json.dumps(operation, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return operation


def update_operation_state(
    workspace: Path,
    operation: dict[str, Any] | None,
    *,
    state: str | None = None,
    phase: str | None = None,
    message: str | None = None,
    error: dict[str, Any] | None = None,
    recovery_status: str | None = None,
    heartbeat_at: str | None = None,
    export_status: str | None = None,
    export_warnings: list[str] | None = None,
) -> dict[str, Any]:
    current = dict(operation or read_operation_state(workspace) or {})
    if not current:
        current = {
            "operation_id": uuid.uuid4().hex[:12],
            "operation_type": "unknown",
            "started_at": now_iso(),
        }
    if state is not None:
        current["state"] = state
    if phase is not None:
        current["phase"] = phase
    if message is not None:
        current["message"] = message
    if error is not None:
        current["error"] = error
    if recovery_status is not None:
        current["recovery_status"] = recovery_status
    if heartbeat_at is not None:
        current["heartbeat_at"] = heartbeat_at
    if export_status is not None:
        current["export_status"] = export_status
    if export_warnings is not None:
        current["export_warnings"] = normalize_export_warnings(export_warnings)
    if state in TERMINAL_OPERATION_STATES:
        current["finished_at"] = now_iso()
    if state == "succeeded":
        current["error"] = None
    return write_operation_state(workspace, current)


def append_recovery_event(workspace: Path, message: str, *, operation: dict[str, Any] | None = None, status: str = "reverted") -> None:
    event_dir = workspace / ".memory" / "debug"
    event_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": "recovery",
        "status": status,
        "message": message,
        "operation_id": (operation or {}).get("operation_id"),
        "operation_type": (operation or {}).get("operation_type"),
        "ts": now_iso(),
    }
    with (event_dir / "events.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


def operation_checkpoint_dir(workspace: Path, operation: dict[str, Any]) -> Path:
    explicit = str(operation.get("checkpoint_dir") or "").strip()
    if explicit:
        return Path(explicit)
    operation_id = str(operation.get("operation_id") or "unknown")
    return workspace / OPERATIONS_DIRNAME / operation_id / "checkpoint"


def revision_checkpoint_files(workspace: Path) -> list[Path]:
    rels = [
        Path("intermediate_output.json"),
        Path(".memory") / "pending_rounds.json",
        Path(".memory") / "collector" / "pending_rounds.json",
    ]
    rels.extend(path.relative_to(workspace) for path in _candidate_export_files(workspace))
    return rels


def latest_export_fresh(workspace: Path) -> bool:
    newest_slide = latest_current_slide_mtime(workspace)
    newest_export = latest_current_export_mtime(workspace)
    return bool(newest_export and (not newest_slide or newest_export + 1.0 >= newest_slide))


def infer_export_status(
    workspace: Path,
    summary: dict[str, Any] | None = None,
    operation: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    summary = summary or summarize_workspace(workspace)
    operation = operation if isinstance(operation, dict) else {}
    warnings = normalize_export_warnings(operation.get("export_warnings"))
    intermediate = read_json_file(workspace / "intermediate_output.json")
    if isinstance(intermediate, dict):
        warnings.extend(normalize_export_warnings(intermediate.get("export_warnings")))
    warnings = list(dict.fromkeys(warnings))

    explicit_status = str(operation.get("export_status") or "").strip().lower()
    if not explicit_status and isinstance(intermediate, dict):
        explicit_status = str(intermediate.get("export_status") or "").strip().lower()
    candidate_exports = _candidate_export_files(workspace)
    has_candidate_exports = bool(candidate_exports)

    if explicit_status in {"strict", "failed", "none"}:
        if explicit_status == "none" and (has_candidate_exports or summary.get("has_outputs")):
            pass
        else:
            return explicit_status, warnings
    elif explicit_status == "partial" and warnings:
        return explicit_status, warnings

    has_pptx = any(path.suffix.lower() == ".pptx" for path in candidate_exports)
    has_pdf = any(path.suffix.lower() == ".pdf" for path in candidate_exports)
    has_slides = int(summary.get("slide_count") or 0) > 0
    fresh_export = latest_export_fresh(workspace)

    if has_pptx and fresh_export and not warnings:
        return "strict", []
    if has_pptx or has_pdf:
        if not warnings:
            if not has_pptx:
                warnings.append("PPTX export is unavailable; PDF/HTML preview is available.")
            elif not fresh_export:
                warnings.append("Latest export may be older than the current slide HTML.")
        return "partial", warnings
    if has_slides:
        if not warnings:
            warnings.append("Slides are available for preview, but PPTX/PDF export did not complete.")
        return "partial", warnings
    return "none", warnings


def operation_age_seconds(operation: dict[str, Any] | None) -> float | None:
    if not isinstance(operation, dict):
        return None
    raw = (
        str(operation.get("heartbeat_at") or "").strip()
        or str(operation.get("updated_at") or "").strip()
        or str(operation.get("started_at") or "").strip()
    )
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    try:
        return max(0.0, time.time() - stamp.timestamp())
    except OSError:
        return None


def operation_has_fresh_export_since_start(workspace: Path, operation: dict[str, Any] | None) -> bool:
    if not isinstance(operation, dict):
        return False
    started = str(operation.get("started_at") or "").strip()
    if not started:
        return False
    try:
        started_ts = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return False
    newest_export = latest_current_export_mtime(workspace)
    return bool(newest_export and newest_export + 1.0 >= started_ts)


def operation_is_stale(operation: dict[str, Any] | None, timeout_seconds: int) -> bool:
    if not isinstance(operation, dict):
        return False
    state = str(operation.get("state") or "").strip().lower()
    phase = str(operation.get("phase") or "").strip().lower()
    if state not in {"running", "queued"} and phase not in {"queued", "queued_api_profile"}:
        return False
    age = operation_age_seconds(operation)
    return bool(age is not None and age >= timeout_seconds)


def rollback_interrupted_revision(workspace: Path, operation: dict[str, Any]) -> bool:
    checkpoint = operation_checkpoint_dir(workspace, operation)
    manifest = checkpoint / "checkpoint_manifest.json"
    if not checkpoint.exists() or not manifest.exists():
        return False

    operation_id = str(operation.get("operation_id") or uuid.uuid4().hex[:12])
    interrupted_dir = workspace / INTERRUPTED_DIRNAME / operation_id
    interrupted_dir.mkdir(parents=True, exist_ok=True)

    for dirname in ("outputs", "slides", ".slide_images"):
        target = workspace / dirname
        source = checkpoint / dirname
        if target.exists():
            archive = interrupted_dir / dirname
            if archive.exists():
                shutil.rmtree(archive, ignore_errors=True)
            shutil.move(str(target), str(archive))
        if source.exists() and source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)

    for rel in revision_checkpoint_files(workspace):
        if rel.parts and rel.parts[0] == INTERRUPTED_DIRNAME:
            continue
        target = workspace / rel
        source = checkpoint / rel
        if target.exists() and target.is_file():
            archive = interrupted_dir / rel
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                archive.unlink()
            shutil.move(str(target), str(archive))
        if source.exists() and source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    (interrupted_dir / "recovery_manifest.json").write_text(
        json.dumps(
            {
                "operation_id": operation_id,
                "recovered_at": now_iso(),
                "checkpoint_dir": str(checkpoint),
                "message": RECOVERY_REVERTED_MESSAGE,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return True


def file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def infer_workspace_state(workspace: Path) -> str:
    summary = summarize_workspace(workspace)
    if summary["slide_count"] or summary["has_exports"]:
        return "succeeded"
    return "idle"


def infer_workspace_phase(workspace: Path) -> str:
    if (workspace / "design_plan.md").exists():
        return "generation"
    if (workspace / "outputs" / "manuscript.md").exists():
        return "research_complete"
    return "ready"


def recover_stale_running_state(
    workspace: Path,
    phase: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a persisted running state from a dead Web worker into a stable UI state."""
    summary = summary or summarize_workspace(workspace)
    phase_text = str(phase or "").lower()
    newest_slide = latest_current_slide_mtime(workspace)
    newest_export = latest_current_export_mtime(workspace)
    operation = read_operation_state(workspace) or {}
    operation_state = str(operation.get("state") or "").lower()
    export_status, export_warnings = infer_export_status(workspace, summary, operation)

    if "revision" in phase_text:
        if newest_export and (not newest_slide or newest_export + 1.0 >= newest_slide):
            operation = update_operation_state(
                workspace,
                operation or None,
                state="succeeded" if operation_state == "running" else None,
                phase="revision_complete",
                message="Deck revised",
                recovery_status="fresh_export_detected",
                export_status=export_status,
                export_warnings=export_warnings,
            )
            return {
                "state": "succeeded",
                "phase": "revision_complete",
                "message": "Deck revised",
                "error": None,
                "recovery_status": "fresh_export_detected",
                "export_status": export_status,
                "export_warnings": export_warnings,
                "last_operation": operation,
            }

        reverted = rollback_interrupted_revision(workspace, operation)
        if reverted:
            operation = update_operation_state(
                workspace,
                operation,
                state="reverted",
                phase="reverted",
                message=RECOVERY_REVERTED_MESSAGE,
                recovery_status="reverted",
                export_status=export_status,
                export_warnings=export_warnings,
            )
            append_recovery_event(workspace, RECOVERY_REVERTED_MESSAGE, operation=operation, status="reverted")
            return {
                "state": "succeeded",
                "phase": "reverted",
                "message": RECOVERY_REVERTED_MESSAGE,
                "error": None,
                "recovery_status": "reverted",
                "export_status": export_status,
                "export_warnings": export_warnings,
                "last_operation": operation,
            }
        operation = update_operation_state(
            workspace,
            operation or None,
            state="interrupted",
            phase="interrupted",
            message=RECOVERY_INTERRUPTED_MESSAGE,
            error={"status": "interrupted", "message": RECOVERY_INTERRUPTED_MESSAGE, "recoverable": True},
            recovery_status="interrupted",
            export_status=export_status,
            export_warnings=export_warnings,
        )
        append_recovery_event(workspace, RECOVERY_INTERRUPTED_MESSAGE, operation=operation, status="interrupted")
        return {
            "state": "failed",
            "phase": "interrupted",
            "message": RECOVERY_INTERRUPTED_MESSAGE,
            "error": {"status": "interrupted", "message": RECOVERY_INTERRUPTED_MESSAGE, "recoverable": True},
            "recovery_status": "interrupted",
            "export_status": export_status,
            "export_warnings": export_warnings,
            "last_operation": operation,
        }

    has_current_outputs = int(summary.get("slide_count") or 0) > 0 or bool(_candidate_export_files(workspace))
    if has_current_outputs:
        message = (
            "Recovered completed workspace"
            if export_status == "strict"
            else "Recovered workspace with export warnings"
        )
        recovered_phase = "generation_complete" if export_status == "strict" else "export_partial"
        operation = update_operation_state(
            workspace,
            operation or None,
            state="succeeded" if operation_state == "running" else None,
            phase=recovered_phase,
            message=message,
            recovery_status="completed",
            export_status=export_status,
            export_warnings=export_warnings,
        )
        return {
            "state": "succeeded",
            "phase": recovered_phase,
            "message": message,
            "error": None,
            "recovery_status": "completed",
            "export_status": export_status,
            "export_warnings": export_warnings,
            "last_operation": operation,
        }
    return {
        "state": "idle",
        "phase": "ready",
        "message": "",
        "error": None,
        "recovery_status": "",
        "export_status": export_status,
        "export_warnings": export_warnings,
        "last_operation": operation or None,
    }


def _session_has_active_operation(data: dict[str, Any], *, active_record: WebSessionRecord | None) -> bool:
    if active_record and (active_record.busy or active_record.queued_operation):
        return True
    state = str(data.get("state") or "").strip().lower()
    phase = str(data.get("phase") or "").strip().lower()
    return bool(data.get("queued_operation")) or state == "running" or phase in {"queued", "queued_api_profile"}


def reconcile_inactive_last_operation(operation: Any) -> dict[str, Any] | None:
    if not isinstance(operation, dict):
        return None
    state = str(operation.get("state") or "").strip().lower()
    phase = str(operation.get("phase") or "").strip().lower()
    if state not in {"queued", "running"} and phase not in {"queued", "queued_api_profile"}:
        return operation
    cleaned = dict(operation)
    cleaned["state"] = "idle"
    cleaned["phase"] = "ready"
    cleaned["message"] = str(cleaned.get("message") or "Recovered inactive operation")
    cleaned["recovery_status"] = str(cleaned.get("recovery_status") or "inactive")
    return cleaned


def latest_current_slide_mtime(workspace: Path) -> float:
    latest = 0.0
    for slide_dir in (workspace / "outputs", workspace / "slides", workspace):
        if not slide_dir.exists() or not slide_dir.is_dir():
            continue
        for path in slide_dir.glob("slide_*.html"):
            if path.is_file():
                latest = max(latest, path.stat().st_mtime)
        if latest:
            return latest
    return latest


def latest_current_export_mtime(workspace: Path) -> float:
    latest = 0.0
    for path in _candidate_export_files(workspace):
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
    return latest


def _path_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


def _summary_slide_files(workspace: Path, *, limit: int = 80) -> list[Path]:
    for slide_dir in (workspace / "outputs", workspace / "slides", workspace):
        if not slide_dir.exists() or not slide_dir.is_dir():
            continue
        html_files = sorted(
            slide_dir.glob("slide_*.html"),
            key=lambda path: (path.stat().st_mtime_ns if path.exists() else 0, path.name),
            reverse=True,
        )
        if html_files:
            return html_files[:limit]
    return []


def _workspace_summary_signature(workspace: Path) -> str:
    slide_files = _summary_slide_files(workspace, limit=80)
    slide_signature = ";".join(
        f"{path.relative_to(workspace).as_posix()}:{_path_mtime_ns(path)}:{int(path.stat().st_size) if path.exists() else 0}"
        for path in slide_files
    )
    return "|".join(
        str(value)
        for value in (
            _path_mtime_ns(workspace),
            _path_mtime_ns(workspace / "session_meta.json"),
            latest_current_slide_mtime(workspace),
            latest_current_export_mtime(workspace),
            _path_mtime_ns(workspace / OPERATION_STATE_FILENAME),
            len(slide_files),
            slide_signature,
        )
    )


def infer_session_id_from_workspace(workspace: Path) -> str:
    name = workspace.name
    parts = name.split("_")
    if parts:
        tail = parts[-1]
        if re.fullmatch(r"[0-9a-f]{8}", tail):
            return tail
    if re.fullmatch(r"[0-9a-f]{8}", name):
        return name
    return name[:8]


def workspace_relative_to(root: Path, workspace: Path) -> str:
    try:
        return workspace.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(workspace)


def extract_workspace_slug(name: str, session_id: str) -> str:
    pattern = re.compile(rf"^\d{{6}}_(?P<slug>.+)_{re.escape(session_id)}(?:_\d{{2}})?$")
    match = pattern.fullmatch(name)
    if match:
        return match.group("slug")
    return ""
