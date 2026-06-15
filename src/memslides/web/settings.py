from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class WebAppSettings:
    data_root: Path
    workspace_base: Path
    memory_root: Path
    service_profile_encryption_key: str
    upload_max_files: int
    upload_max_file_bytes: int
    upload_max_total_bytes: int
    allowed_upload_extensions: tuple[str, ...]
    max_active_operations: int
    operation_heartbeat_interval_seconds: int
    operation_stale_timeout_seconds: int
    export_strict_repair_rounds: int
    export_allow_relaxed_fallback: bool
    export_page_timeout_seconds: int
    show_external_api_settings: bool

    @property
    def user_workspace_root(self) -> Path:
        return self.workspace_base / "users"


def load_web_settings(
    *,
    data_root: Path | None = None,
    workspace_base: Path | None = None,
    max_active_operations: int | None = None,
) -> WebAppSettings:
    default_data_root = Path.home() / ".cache" / "memslides"
    resolved_data_root = (
        data_root.expanduser()
        if data_root is not None
        else Path(os.getenv("MEMSLIDES_DATA_ROOT", str(default_data_root))).expanduser()
    )
    resolved_workspace_base = (
        workspace_base.expanduser()
        if workspace_base is not None
        else Path(os.getenv("MEMSLIDES_WORKSPACE_BASE", str(resolved_data_root / "web"))).expanduser()
    )
    resolved_memory_root = Path(
        os.getenv(
            "MEMSLIDES_MEMORY_ROOT",
            str(Path(os.getenv("MEMSLIDES_DEFAULT_CACHE_ROOT", str(resolved_data_root))).expanduser() / ".memory"),
        )
    ).expanduser()
    extensions = tuple(
        item.strip().lower()
        for item in os.getenv("MEMSLIDES_ALLOWED_UPLOAD_EXTENSIONS", "pptx,pdf,png,jpg,jpeg,md,txt").split(",")
        if item.strip()
    )
    return WebAppSettings(
        data_root=resolved_data_root,
        workspace_base=resolved_workspace_base,
        memory_root=resolved_memory_root,
        service_profile_encryption_key=os.getenv("MEMSLIDES_SERVICE_PROFILE_ENCRYPTION_KEY", "").strip(),
        upload_max_files=_env_int("MEMSLIDES_UPLOAD_MAX_FILES", 5),
        upload_max_file_bytes=_env_int("MEMSLIDES_UPLOAD_MAX_FILE_BYTES", 50 * 1024 * 1024),
        upload_max_total_bytes=_env_int("MEMSLIDES_UPLOAD_MAX_TOTAL_BYTES", 150 * 1024 * 1024),
        allowed_upload_extensions=extensions,
        max_active_operations=max_active_operations or _env_int("MEMSLIDES_WEB_MAX_ACTIVE_OPERATIONS", 2),
        operation_heartbeat_interval_seconds=_env_int("MEMSLIDES_OPERATION_HEARTBEAT_INTERVAL_SEC", 30),
        operation_stale_timeout_seconds=_env_int("MEMSLIDES_OPERATION_STALE_TIMEOUT_SEC", 600),
        export_strict_repair_rounds=_env_int("MEMSLIDES_EXPORT_STRICT_REPAIR_ROUNDS", 2),
        export_allow_relaxed_fallback=_env_bool("MEMSLIDES_EXPORT_ALLOW_RELAXED_FALLBACK", True),
        export_page_timeout_seconds=_env_int("MEMSLIDES_EXPORT_PAGE_TIMEOUT_SEC", 60),
        show_external_api_settings=True,
    )
