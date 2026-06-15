from __future__ import annotations

import asyncio
import json
import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from memslides.utils.constants import PACKAGE_DIR, WORKSPACE_BASE
from memslides.web.artifacts import (
    list_artifacts,
    list_files_tree,
    list_slides,
    list_templates,
    render_preview_html,
    resolve_workspace_file,
    template_summary,
)
from memslides.web.api_profiles import ApiProfileError
from memslides.web.events import read_full_events, read_recent_events, stream_workspace_events
from memslides.web.session_store import WebSessionStore
from memslides.web.settings import WebAppSettings, load_web_settings


STATIC_DIR = PACKAGE_DIR / "web" / "static"
PUBLIC_USER_ID = "web-demo"
LOGGER = logging.getLogger(__name__)


def resolve_web_cache_root(workspace_base: Path | None = None) -> Path:
    if workspace_base is not None:
        return Path(workspace_base).expanduser()
    env_workspace = _nonempty_env_path("MEMSLIDES_WORKSPACE_BASE")
    if env_workspace is not None:
        return env_workspace
    env_cache = _nonempty_env_path("MEMSLIDES_DEFAULT_CACHE_ROOT")
    if env_cache is not None:
        return env_cache / "web"
    return WORKSPACE_BASE / "web"


def _nonempty_env_path(name: str) -> Path | None:
    import os

    value = os.getenv(name, "").strip()
    if not value or value in {"./workspace", "workspace"}:
        return None
    return Path(value).expanduser()


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def create_app(
    *,
    workspace_base: Path | None = None,
    data_root: Path | None = None,
    config_file: Path | None = None,
    max_active_operations: int | None = None,
) -> FastAPI:
    resolved_workspace_base = resolve_web_cache_root(workspace_base)
    resolved_data_root = (
        data_root.expanduser()
        if data_root is not None
        else resolved_workspace_base.parent if resolved_workspace_base.name == "web" else resolved_workspace_base
    )
    settings = load_web_settings(
        data_root=resolved_data_root,
        workspace_base=resolved_workspace_base,
        max_active_operations=max_active_operations,
    )
    store = WebSessionStore(
        workspace_base=settings.workspace_base,
        config_file=config_file,
        memory_db_dir=settings.memory_root,
        service_profile_encryption_key=settings.service_profile_encryption_key,
        max_active_operations=settings.max_active_operations,
        operation_heartbeat_interval_seconds=settings.operation_heartbeat_interval_seconds,
        operation_stale_timeout_seconds=settings.operation_stale_timeout_seconds,
        user_isolation=False,
    )
    memory_profiles = LocalMemoryProfileStore(settings.data_root / "memory_profiles.json")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await store.close_all()

    app = FastAPI(title="MemSlides Local Studio", version="1.0.0", lifespan=lifespan)
    app.state.store = store
    app.state.settings = settings
    app.state.memory_profiles = memory_profiles

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
        assets_dir = STATIC_DIR / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="web_assets")

    def _spa_index_response() -> FileResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=503, detail="Web static assets are not installed. Build the frontend first.")
        return FileResponse(index_path)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        return _spa_index_response()

    @app.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mode": "single-user-local",
            "workspace_base": str(settings.workspace_base),
        }

    @app.get("/api/service-profiles")
    async def list_service_profiles() -> dict[str, Any]:
        return {"profiles": store.api_profiles.list_profiles(user_id=PUBLIC_USER_ID)}

    @app.post("/api/service-profiles")
    async def save_service_profile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return store.api_profiles.upsert_profile(payload, user_id=PUBLIC_USER_ID)
        except ApiProfileError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/service-profiles/validate")
    async def validate_service_profile_payload(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await store.api_profiles.validate_payload(payload, user_id=PUBLIC_USER_ID)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/service-profiles/{profile_id}/validate")
    async def validate_service_profile(profile_id: str) -> dict[str, Any]:
        try:
            return await store.api_profiles.validate_profile(profile_id, user_id=PUBLIC_USER_ID)
        except ApiProfileError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/service-profiles/{profile_id}")
    async def delete_service_profile(profile_id: str) -> dict[str, Any]:
        deleted = store.api_profiles.delete_profile(profile_id, user_id=PUBLIC_USER_ID)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Unknown service profile: {profile_id}")
        return {"deleted": True, "profile_id": profile_id}

    @app.post("/api/sessions")
    async def create_session(payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        memory_profile = memory_profiles.get(str(payload.get("memory_profile_id") or ""))
        record = await store.create_session(
            language=str(payload.get("language") or "en"),
            user_id=PUBLIC_USER_ID,
            api_profile_id=str(payload.get("service_profile_id") or payload.get("api_profile_id") or ""),
            memory_profile_id=str(memory_profile.get("memory_profile_id") or ""),
            memory_intent=str(memory_profile.get("intent") or payload.get("memory_intent") or ""),
            memory_enabled=_coerce_bool(payload.get("memory_enabled"), True),
        )
        if memory_profile.get("profile"):
            await store.save_global_user_profile(dict(memory_profile["profile"]), user_id=PUBLIC_USER_ID)
        return record.to_dict()

    @app.get("/api/sessions")
    async def list_sessions(limit: int = 100) -> dict[str, Any]:
        sessions = [store.refresh_session_metadata(await _open_record(store, item)) for item in store.list_sessions(limit=limit)]
        return {"sessions": sessions}

    @app.post("/api/sessions/open")
    async def open_session(payload: dict[str, Any]) -> dict[str, Any]:
        record = await store.open_session(
            session_id=str(payload.get("session_id") or "").strip() or None,
            workspace=Path(str(payload.get("workspace"))).expanduser() if payload.get("workspace") else None,
        )
        return record.to_dict()

    @app.post("/api/sessions/{session_id}/cancel")
    async def cancel_session_operation(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        operation = store.mark_operation_cancelled(record, message="This run was stopped.")
        return {**store.refresh_session_metadata(record), "cancelled": True, "last_operation": operation}

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        if record.busy or record.queued_operation:
            store.mark_operation_cancelled(record, message="This run was stopped before deleting the session.")
        await record.worker.run(record.session.close())
        await record.worker.close()
        import shutil

        shutil.rmtree(record.workspace, ignore_errors=True)
        store.records.pop(session_id, None)
        return {"deleted": True, "session_id": session_id}

    @app.post("/api/sessions/{session_id}/generate")
    async def generate(
        session_id: str,
        instruction: Annotated[str, Form()],
        num_pages: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        memory_intent: Annotated[str, Form()] = "",
        template: Annotated[str, Form()] = "",
        template_id: Annotated[str, Form()] = "",
        template_as_reference: Annotated[bool, Form()] = False,
        api_profile_id: Annotated[str, Form()] = "",
        service_profile_id: Annotated[str, Form()] = "",
        memory_profile_id: Annotated[str, Form()] = "",
        memory_enabled: Annotated[str | None, Form()] = None,
        reference_template: Annotated[UploadFile | None, File()] = None,
        files: Annotated[list[UploadFile], File()] = [],
    ) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        store.prepare_workspace(record, instruction)
        uploads = await _collect_uploads(files, settings)
        attachments = store.save_uploads(record, uploads)
        reference_template_path = None
        if reference_template is not None and reference_template.filename:
            filename, content = await _read_single_upload(reference_template, settings, allowed={".pptx"})
            reference_template_path = store.save_reference_template(record, filename, content)
        resolved_template_path = _resolve_template_path(store, template_id, template)
        memory_profile = memory_profiles.get(memory_profile_id)
        if memory_profile.get("profile"):
            await store.save_global_user_profile(dict(memory_profile["profile"]), user_id=PUBLIC_USER_ID)
        effective_profile_id = _resolve_service_profile_id(store, service_profile_id or api_profile_id)
        payload = {
            "instruction": instruction,
            "num_pages": num_pages,
            "language": language,
            "memory_intent": str(memory_profile.get("intent") or memory_intent or ""),
            "memory_profile_id": str(memory_profile.get("memory_profile_id") or ""),
            "template": str(reference_template_path or resolved_template_path or ""),
            "template_id": "" if reference_template_path else (template_id if resolved_template_path else ""),
            "template_as_reference": bool(reference_template_path or resolved_template_path) or bool(template_as_reference),
            "api_profile_id": effective_profile_id,
            "service_profile_id": effective_profile_id,
            "memory_enabled": _coerce_bool(memory_enabled, record.memory_enabled),
            "attachments": [str(path) for path in attachments],
        }
        try:
            await store.generate(session_id, payload, attachments)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"accepted": True, **store.refresh_session_metadata(record)}

    @app.post("/api/sessions/{session_id}/revise")
    async def revise(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        feedback = str(payload.get("feedback") or "").strip()
        if not feedback:
            raise HTTPException(status_code=400, detail="feedback is required")
        memory_profile = memory_profiles.get(str(payload.get("memory_profile_id") or ""))
        effective_profile_id = _resolve_service_profile_id(store, str(payload.get("service_profile_id") or payload.get("api_profile_id") or record.api_profile_id or ""))
        try:
            await store.revise(
                session_id,
                feedback,
                str(memory_profile.get("intent") or payload.get("memory_intent") or record.memory_intent or ""),
                str(memory_profile.get("memory_profile_id") or ""),
                effective_profile_id,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"accepted": True, **store.refresh_session_metadata(record)}

    @app.post("/api/sessions/{session_id}/memory/save")
    async def save_memory(session_id: str, background: bool = True) -> dict[str, Any]:
        try:
            return await store.start_memory_save(session_id) if background else await store.save_memory(session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/memory/working")
    async def get_working_memory(session_id: str) -> dict[str, Any]:
        await _get_record(store, session_id)
        try:
            return await store.get_working_memory(session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/sessions/{session_id}/memory/working")
    async def update_working_memory(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        await _get_record(store, session_id)
        try:
            return await store.update_working_memory(session_id, payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/memory/profile")
    async def get_global_memory_profile(intent: Annotated[str, Query()] = "") -> dict[str, Any]:
        try:
            return await store.get_global_user_profile(user_id=PUBLIC_USER_ID, intent=intent)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/memory/profile")
    async def save_global_memory_profile(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return await store.save_global_user_profile(payload, user_id=PUBLIC_USER_ID)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/memory-profiles")
    async def list_memory_profiles() -> dict[str, Any]:
        return {"profiles": memory_profiles.list()}

    @app.post("/api/memory-profiles")
    async def create_memory_profile(payload: dict[str, Any]) -> dict[str, Any]:
        return memory_profiles.upsert(payload)

    @app.patch("/api/memory-profiles/{memory_profile_id}")
    async def update_memory_profile(memory_profile_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return memory_profiles.upsert({**payload, "memory_profile_id": memory_profile_id})

    @app.delete("/api/memory-profiles/{memory_profile_id}")
    async def delete_memory_profile(memory_profile_id: str) -> dict[str, Any]:
        deleted = memory_profiles.delete(memory_profile_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Unknown memory profile: {memory_profile_id}")
        return {"deleted": True, "memory_profile_id": memory_profile_id}

    @app.get("/api/sessions/{session_id}/memory/profile")
    async def get_memory_profile(session_id: str, intent: Annotated[str, Query()] = "") -> dict[str, Any]:
        await _get_record(store, session_id)
        try:
            return await store.get_user_profile(session_id, intent=intent)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/sessions/{session_id}/memory/profile")
    async def save_memory_profile(session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        await _get_record(store, session_id)
        try:
            return await store.save_user_profile(session_id, payload)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/status")
    async def status(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        return store.refresh_session_metadata(record)

    @app.get("/api/sessions/{session_id}/slides")
    async def slides(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        return {"slides": list_slides(record.workspace, session_id)}

    @app.get("/api/sessions/{session_id}/preview")
    async def preview(session_id: str, path: Annotated[str, Query()]) -> Response:
        record = await _get_record(store, session_id)
        try:
            return HTMLResponse(render_preview_html(record.workspace, session_id, path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/artifacts")
    async def artifacts(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        store.refresh_session_metadata(record)
        return list_artifacts(record.workspace, session_id)

    @app.get("/api/sessions/{session_id}/files/tree")
    async def files_tree(session_id: str, root: Annotated[str, Query()], path: Annotated[str, Query()] = "") -> dict[str, Any]:
        record = await _get_record(store, session_id)
        try:
            return list_files_tree(record.workspace, session_id, root=root, rel_path=path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}/events")
    async def events(session_id: str) -> StreamingResponse:
        record = await _get_record(store, session_id)
        return StreamingResponse(stream_workspace_events(record.workspace), media_type="text/event-stream")

    @app.get("/api/sessions/{session_id}/events/recent")
    async def recent_events(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        return {"events": read_recent_events(record.workspace)}

    @app.get("/api/sessions/{session_id}/events/full")
    async def full_events(session_id: str, limit: Annotated[int, Query()] = 400, include_filtered: Annotated[bool, Query()] = False) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        bounded_limit = max(50, min(limit, 2000))
        return {"events": read_full_events(record.workspace, limit=bounded_limit, include_filtered=include_filtered)}

    @app.get("/api/sessions/{session_id}/summary")
    async def session_summary(session_id: str) -> dict[str, Any]:
        record = await _get_record(store, session_id)
        return store.refresh_session_metadata(record)

    @app.get("/api/templates")
    async def templates() -> dict[str, Any]:
        return {"templates": [_public_template_item(item) for item in list_templates(store.cache_root)]}

    @app.post("/api/templates/induct")
    async def template_induct(template_file: Annotated[UploadFile, File()], output_name: Annotated[str, Form()] = "") -> dict[str, Any]:
        filename, content = await _read_single_upload(template_file, settings, allowed={".pptx"})
        uploaded = store.save_template_upload(filename, content)
        template_name = output_name.strip() or uploaded.stem
        template_id = _slugify_template_id(template_name)
        output_dir = store.cache_root / "templates" / PUBLIC_USER_ID / template_id
        result = await store.induct_template(uploaded, output_dir=output_dir, replace_existing=True)
        return {
            "template_dir": str(output_dir),
            "template_id": template_id,
            "name": template_name,
            "summary": template_summary(output_dir),
            "analysis": result,
        }

    @app.get("/api/files")
    async def files(session_id: Annotated[str, Query()], path: Annotated[str, Query()], download: Annotated[bool, Query()] = False) -> FileResponse:
        record = await _get_record(store, session_id)
        try:
            served = resolve_workspace_file(record.workspace, path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return FileResponse(served.path, media_type=served.media_type, filename=served.path.name if download else None)

    return app


class LocalMemoryProfileStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[dict[str, Any]]:
        return sorted(self._read(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def get(self, memory_profile_id: str) -> dict[str, Any]:
        profile_id = str(memory_profile_id or "").strip()
        if not profile_id:
            return {}
        return next((item for item in self._read() if str(item.get("memory_profile_id") or "") == profile_id), {})

    def upsert(self, payload: dict[str, Any]) -> dict[str, Any]:
        import datetime
        import re
        import uuid

        raw = dict(payload or {})
        intent = str(raw.get("intent") or "default").strip() or "default"
        profile_id = str(raw.get("memory_profile_id") or raw.get("id") or "").strip()
        if not profile_id:
            profile_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", intent).strip("._-") or uuid.uuid4().hex[:10]
        now = datetime.datetime.now().astimezone().isoformat()
        item = {
            "memory_profile_id": profile_id,
            "name": str(raw.get("name") or intent or "Memory profile"),
            "intent": intent,
            "profile": dict(raw.get("profile") or raw.get("payload") or {}),
            "is_default": bool(raw.get("is_default", False)),
            "updated_at": now,
            "created_at": str(raw.get("created_at") or now),
        }
        items = [existing for existing in self._read() if str(existing.get("memory_profile_id") or "") != profile_id]
        if item["is_default"]:
            for existing in items:
                existing["is_default"] = False
        items.append(item)
        self._write(items)
        return item

    def delete(self, memory_profile_id: str) -> bool:
        profile_id = str(memory_profile_id or "").strip()
        items = self._read()
        next_items = [item for item in items if str(item.get("memory_profile_id") or "") != profile_id]
        if len(next_items) == len(items):
            return False
        self._write(next_items)
        return True

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            data = data.get("profiles", [])
        return [dict(item) for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _write(self, items: list[dict[str, Any]]) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"profiles": items}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)


async def _open_record(store: WebSessionStore, item: dict[str, Any]):
    session_id = str(item.get("session_id") or "").strip()
    if session_id and session_id in store.records:
        return store.records[session_id]
    return await store.open_session(session_id=session_id or None, workspace=Path(str(item.get("workspace"))).expanduser() if item.get("workspace") else None)


async def _get_record(store: WebSessionStore, session_id: str):
    try:
        return store.get(session_id)
    except KeyError:
        try:
            return await store.open_session(session_id=session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


def _resolve_service_profile_id(store: WebSessionStore, requested_profile_id: str) -> str:
    requested = str(requested_profile_id or "").strip()
    profiles = store.api_profiles.list_profiles(user_id=PUBLIC_USER_ID)
    if requested:
        if not any(str(item.get("profile_id") or "") == requested for item in profiles):
            raise HTTPException(status_code=404, detail="API key configuration not found.")
        return requested
    preferred = next((item for item in profiles if item.get("is_default")), None) or (profiles[0] if profiles else None)
    return str((preferred or {}).get("profile_id") or "")


def _resolve_template_path(store: WebSessionStore, template_id: str, template: str) -> Path | None:
    if template.strip():
        path = Path(template).expanduser()
        return path if path.exists() else None
    wanted = str(template_id or "").strip()
    if not wanted:
        return None
    for item in list_templates(store.cache_root):
        if str(item.get("id") or item.get("template_id") or "") == wanted:
            path = Path(str(item.get("path") or "")).expanduser()
            return path if path.exists() else None
    return None


def _public_template_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        **item,
        "template_id": str(item.get("template_id") or item.get("id") or ""),
        "name": str(item.get("name") or item.get("id") or "Template"),
    }


async def _collect_uploads(files: list[UploadFile], settings: WebAppSettings) -> list[tuple[str, bytes]]:
    allowed = {f".{item.strip().lower().lstrip('.')}" for item in settings.allowed_upload_extensions if item.strip()}
    if len(files) > settings.upload_max_files:
        raise HTTPException(status_code=400, detail=f"At most {settings.upload_max_files} files are allowed per request")
    total = 0
    uploads: list[tuple[str, bytes]] = []
    for file in files:
        if not file.filename:
            continue
        filename, content = await _read_single_upload(file, settings, allowed=allowed)
        total += len(content)
        if total > settings.upload_max_total_bytes:
            raise HTTPException(status_code=400, detail="Uploaded files exceed the total size limit")
        uploads.append((filename, content))
    return uploads


async def _read_single_upload(file: UploadFile, settings: WebAppSettings, *, allowed: set[str]) -> tuple[str, bytes]:
    filename = Path(file.filename or "upload").name
    suffix = Path(filename).suffix.lower()
    if allowed and suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported upload type: {suffix}")
    content = await file.read()
    if len(content) > settings.upload_max_file_bytes:
        raise HTTPException(status_code=400, detail=f"{filename} exceeds the per-file size limit")
    return filename, content


def _slugify_template_id(value: str) -> str:
    import re
    import uuid

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-").lower()
    return slug or f"template_{uuid.uuid4().hex[:8]}"


def run_web(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    workspace_base: Path | None = None,
    data_root: Path | None = None,
    config_file: Path | None = None,
    log_level: str = "summary",
    log_file: Path | None = None,
    max_active_operations: int | None = None,
) -> None:
    import uvicorn

    resolved_workspace = resolve_web_cache_root(workspace_base)
    settings = load_web_settings(
        data_root=data_root,
        workspace_base=resolved_workspace,
        max_active_operations=max_active_operations,
    )
    effective_log_file = configure_web_logging(settings.data_root, log_level=log_level, log_file=log_file)
    configure_web_warnings(log_level)
    app = create_app(
        workspace_base=settings.workspace_base,
        data_root=settings.data_root,
        config_file=config_file,
        max_active_operations=settings.max_active_operations,
    )
    if log_level == "summary":
        print(f"MemSlides Local Studio: http://{host}:{port}")
        print(f"Logs: {effective_log_file}")
        print(f"Data root: {settings.data_root}")
        print("Mode: single-user local")
    uvicorn.run(
        app,
        host=host,
        port=port,
        access_log=(log_level == "debug"),
        log_level="debug" if log_level == "debug" else "warning" if log_level == "summary" else "info",
        log_config=None,
    )


def configure_web_logging(log_root: Path, *, log_level: str = "summary", log_file: Path | None = None) -> Path:
    logs_dir = log_root / ".history"
    logs_dir.mkdir(parents=True, exist_ok=True)
    target = log_file or logs_dir / "web.log"
    target.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if log_level == "debug" else logging.INFO
    formatter = logging.Formatter("%(levelname)-4s %(asctime)s [%(name)s] %(message)s")
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    for name in ("memslides.web", "uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers = [handler for handler in logger.handlers if not isinstance(handler, logging.StreamHandler)]
        logger.addHandler(file_handler)
        logger.propagate = False
    logging.getLogger("uvicorn.access").disabled = log_level != "debug"
    return target


def configure_web_warnings(log_level: str) -> None:
    if log_level == "debug":
        return
    warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets\..*")
    warnings.filterwarnings("ignore", message=r".*websockets\.legacy is deprecated.*", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=r".*WebSocketServerProtocol is deprecated.*", category=DeprecationWarning)
