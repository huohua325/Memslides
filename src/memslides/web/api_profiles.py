from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet
from openai import AsyncOpenAI
from pydantic import BaseModel, Field


API_PROFILE_ROLE_KEYS = [
    "research_agent",
    "design_agent",
    "modify_agent",
    "reviewer_agent",
    "fast_model",
    "balanced_model",
    "long_context_model",
    "vision_model",
]

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_API_PROFILE_CONCURRENCY = 2
REQUIRED_SERVICE_KEYS = ("llm", "pdf")
OPTIONAL_SERVICE_KEYS = ("embedding", "search", "image_generation")
SERVICE_KEYS = (*REQUIRED_SERVICE_KEYS, *OPTIONAL_SERVICE_KEYS)


class ApiProfileError(RuntimeError):
    pass


class LLMServicePayload(BaseModel):
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    base_url: str = DEFAULT_OPENAI_BASE_URL
    model: str = ""
    api_key: str = ""
    enabled: bool = True
    vision_capable: bool = True


class EmbeddingServicePayload(BaseModel):
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    base_url: str = DEFAULT_OPENAI_BASE_URL
    model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str = ""
    dim: int = DEFAULT_EMBEDDING_DIM
    enabled: bool = False


class PdfServicePayload(BaseModel):
    provider: Literal["pdf_parser_official", "pdf_parser_compatible"] = "pdf_parser_official"
    api_key: str = ""
    api_url: str = ""
    backend: Literal["pdf_parser"] = "pdf_parser"
    request_timeout_sec: int = 180
    poll_timeout_sec: int = 180
    enabled: bool = True


class SearchServicePayload(BaseModel):
    provider: Literal["tavily"] = "tavily"
    api_key: str = ""
    enabled: bool = False


class ImageGenerationServicePayload(BaseModel):
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    base_url: str = DEFAULT_OPENAI_BASE_URL
    model: str = ""
    api_key: str = ""
    enabled: bool = False
    min_image_size: int | None = None


class ApiProfilePayload(BaseModel):
    profile_id: str = ""
    user_id: str = "web-demo"
    display_name: str = ""
    provider: Literal["openai", "openai_compatible"] = "openai_compatible"
    base_url: str = ""
    model: str = ""
    api_key: str = ""
    max_concurrent: int = DEFAULT_API_PROFILE_CONCURRENCY
    enabled: bool = True
    is_default: bool = False
    llm: LLMServicePayload | None = None
    embedding: EmbeddingServicePayload | None = None
    pdf: PdfServicePayload | None = None
    search: SearchServicePayload | None = None
    image_generation: ImageGenerationServicePayload | None = None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "")).strip("._-")[:80]


def _normalize_user_id(user_id: str) -> str:
    return str(user_id or "web-demo").strip() or "web-demo"


def _mask_api_key(api_key: str) -> str:
    key = str(api_key or "")
    if len(key) <= 8:
        return f"{key[:2]}...{key[-2:]}" if key else ""
    return f"{key[:4]}...{key[-4:]}"


def _fingerprint(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _fernet_key_from_secret(secret: str) -> bytes:
    raw = secret.strip().encode("utf-8")
    try:
        Fernet(raw)
        return raw
    except Exception:
        return base64.urlsafe_b64encode(hashlib.sha256(raw).digest())


def _default_role_mapping(base_url: str, model: str) -> dict[str, dict[str, str]]:
    return {role: {"base_url": base_url, "model": model} for role in API_PROFILE_ROLE_KEYS}


def _blank_service(service_name: str) -> dict[str, Any]:
    if service_name == "llm":
        return LLMServicePayload().model_dump(mode="python")
    if service_name == "embedding":
        return EmbeddingServicePayload().model_dump(mode="python")
    if service_name == "pdf":
        return PdfServicePayload().model_dump(mode="python")
    if service_name == "search":
        return SearchServicePayload().model_dump(mode="python")
    if service_name == "image_generation":
        return ImageGenerationServicePayload().model_dump(mode="python")
    return {}


def _has_service_payload(raw: dict[str, Any]) -> bool:
    return any(isinstance(raw.get(key), dict) for key in SERVICE_KEYS)


def _item_services(item: dict[str, Any]) -> dict[str, dict[str, Any]]:
    services = item.get("services")
    if isinstance(services, dict):
        return {key: dict(value) for key, value in services.items() if isinstance(value, dict)}
    return {
        "llm": {
            "provider": str(item.get("provider") or "openai_compatible"),
            "base_url": str(item.get("base_url") or DEFAULT_OPENAI_BASE_URL),
            "model": str(item.get("model") or ""),
            "api_key_encrypted": str(item.get("api_key_encrypted") or ""),
            "api_key_masked": str(item.get("api_key_masked") or ""),
            "api_key_fingerprint": str(item.get("api_key_fingerprint") or ""),
            "enabled": bool(item.get("enabled", True)),
            "vision_capable": True,
        }
    }


def _public_service(service: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in service.items()
        if key not in {"api_key", "api_key_encrypted", "api_key_fingerprint"}
    }
    public.setdefault("api_key_masked", "")
    return public


class _FileProfileBackend:
    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir).expanduser()
        self.store_path = self.memory_dir / "api_profiles.json"
        self._lock = threading.RLock()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def list_profiles(self, user_id: str) -> list[dict[str, Any]]:
        normalized_user = _normalize_user_id(user_id)
        with self._lock:
            return [
                dict(item)
                for item in self._read_store().get("profiles", [])
                if item.get("user_id") == normalized_user
            ]

    def get_profile(self, profile_id: str, *, user_id: str) -> dict[str, Any]:
        normalized_user = _normalize_user_id(user_id)
        normalized_profile = _safe_id(profile_id)
        with self._lock:
            for item in self._read_store().get("profiles", []):
                if item.get("user_id") == normalized_user and item.get("profile_id") == normalized_profile:
                    return dict(item)
        raise ApiProfileError(f"Unknown API profile: {profile_id}")

    def save_profile(self, item: dict[str, Any]) -> None:
        normalized_user = _normalize_user_id(str(item.get("user_id") or "web-demo"))
        profile_id = _safe_id(str(item.get("profile_id") or ""))
        with self._lock:
            data = self._read_store()
            profiles = [
                dict(existing)
                for existing in data.get("profiles", [])
                if not (existing.get("user_id") == normalized_user and existing.get("profile_id") == profile_id)
            ]
            if item.get("is_default"):
                for existing in profiles:
                    if existing.get("user_id") == normalized_user:
                        existing["is_default"] = False
            profiles.append(dict(item))
            data["version"] = 2
            data["profiles"] = profiles
            self._write_store(data)

    def delete_profile(self, profile_id: str, *, user_id: str) -> bool:
        normalized_user = _normalize_user_id(user_id)
        normalized_profile = _safe_id(profile_id)
        with self._lock:
            data = self._read_store()
            before = len(data.get("profiles", []))
            data["profiles"] = [
                item
                for item in data.get("profiles", [])
                if not (item.get("user_id") == normalized_user and item.get("profile_id") == normalized_profile)
            ]
            deleted = len(data["profiles"]) != before
            if deleted:
                self._write_store(data)
            return deleted

    def _read_store(self) -> dict[str, Any]:
        if not self.store_path.exists():
            return {"version": 2, "profiles": []}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 2, "profiles": []}
        if not isinstance(data, dict):
            return {"version": 2, "profiles": []}
        if not isinstance(data.get("profiles"), list):
            data["profiles"] = []
        data.setdefault("version", 2)
        return data

    def _write_store(self, data: dict[str, Any]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.store_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        _chmod_private(tmp_path)
        tmp_path.replace(self.store_path)
        _chmod_private(self.store_path)


class ApiProfileStore:
    """Encrypted local service profile store for the public Web Studio."""

    def __init__(
        self,
        memory_dir: Path,
        *,
        encryption_secret: str = "",
    ):
        self.memory_dir = Path(memory_dir).expanduser()
        self.cache_root = self.memory_dir.parent if self.memory_dir.name == ".memory" else self.memory_dir
        self.secret_path = self.cache_root / ".secrets" / "api_profiles.key"
        self._lock = threading.RLock()
        self._fernet = Fernet(self._load_or_create_fernet_key(encryption_secret))
        self.backend = _FileProfileBackend(self.memory_dir)

    def list_profiles(self, user_id: str = "web-demo") -> list[dict[str, Any]]:
        profiles = [self._public_profile(item) for item in self.backend.list_profiles(user_id)]
        return sorted(profiles, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def upsert_profile(self, payload: dict[str, Any], *, user_id: str = "web-demo") -> dict[str, Any]:
        raw = dict(payload or {})
        incoming = ApiProfilePayload.model_validate({**raw, "user_id": user_id or raw.get("user_id") or "web-demo"})
        normalized_user = _normalize_user_id(incoming.user_id)
        profile_id = _safe_id(incoming.profile_id) or uuid.uuid4().hex[:10]
        max_concurrent = max(1, min(20, int(incoming.max_concurrent or DEFAULT_API_PROFILE_CONCURRENCY)))
        service_payload = _has_service_payload(raw)
        with self._lock:
            try:
                existing = self.backend.get_profile(profile_id, user_id=normalized_user)
            except ApiProfileError:
                existing = {}
            services = self._build_services(raw, existing=existing, service_payload=service_payload)
            self._assert_required_services(services, service_payload=service_payload)
            llm = services.get("llm", {})
            display_name = incoming.display_name.strip() or str(existing.get("display_name") or "").strip()
            if not display_name:
                display_name = f"{llm.get('provider') or 'openai_compatible'}:{llm.get('model') or 'service-profile'}"
            now = _now_iso()
            item = {
                "schema_version": 2 if service_payload else int(existing.get("schema_version") or 1),
                "profile_id": profile_id,
                "user_id": normalized_user,
                "display_name": display_name,
                "provider": str(llm.get("provider") or "openai_compatible"),
                "base_url": str(llm.get("base_url") or DEFAULT_OPENAI_BASE_URL),
                "model": str(llm.get("model") or ""),
                "api_key_encrypted": str(llm.get("api_key_encrypted") or ""),
                "api_key_masked": str(llm.get("api_key_masked") or ""),
                "api_key_fingerprint": str(llm.get("api_key_fingerprint") or ""),
                "role_mapping": _default_role_mapping(str(llm.get("base_url") or DEFAULT_OPENAI_BASE_URL), str(llm.get("model") or "")),
                "services": services,
                "max_concurrent": max_concurrent,
                "enabled": bool(incoming.enabled),
                "is_default": bool(raw.get("is_default", existing.get("is_default", False))),
                "validation_status": str(existing.get("validation_status") or ""),
                "validation_message": str(existing.get("validation_message") or ""),
                "last_validated_at": str(existing.get("last_validated_at") or ""),
                "created_at": str(existing.get("created_at") or now),
                "updated_at": now,
            }
            self.backend.save_profile(item)
        return self._public_profile(item)

    def delete_profile(self, profile_id: str, *, user_id: str = "web-demo") -> bool:
        return self.backend.delete_profile(profile_id, user_id=user_id)

    def get_public_profile(self, profile_id: str, *, user_id: str = "web-demo") -> dict[str, Any]:
        return self._public_profile(self.backend.get_profile(profile_id, user_id=user_id))

    def get_runtime_profile(self, profile_id: str, *, user_id: str = "web-demo") -> dict[str, Any]:
        item = self.backend.get_profile(profile_id, user_id=user_id)
        if not item.get("enabled", True):
            raise ApiProfileError("API profile is disabled")
        public = self._public_profile(item)
        runtime_services: dict[str, dict[str, Any]] = {}
        for name, service in _item_services(item).items():
            runtime = dict(service)
            encrypted = str(runtime.pop("api_key_encrypted", "") or "")
            runtime.pop("api_key_fingerprint", None)
            runtime["api_key"] = self._decrypt(encrypted) if encrypted else ""
            runtime_services[name] = runtime
        llm = runtime_services.get("llm", {})
        return {
            **public,
            "services": runtime_services,
            **runtime_services,
            "api_key": str(llm.get("api_key") or ""),
        }

    async def validate_profile(self, profile_id: str, *, user_id: str = "web-demo") -> dict[str, Any]:
        runtime = self.get_runtime_profile(profile_id, user_id=user_id)
        result = await self._validate_runtime(runtime, service_payload=int(runtime.get("schema_version") or 1) >= 2)
        with self._lock:
            item = self.backend.get_profile(profile_id, user_id=user_id)
            services = _item_services(item)
            item["validation_status"] = "valid" if result.get("ok") else "failed"
            item["validation_message"] = str(result.get("message") or "")
            if result.get("ok"):
                item["last_validated_at"] = str(result.get("validated_at") or "")
            for name, service_result in result.get("services", {}).items():
                if name not in services:
                    continue
                services[name]["validation_status"] = "valid" if service_result.get("ok") else "failed"
                services[name]["validation_message"] = str(service_result.get("message") or service_result.get("error") or "")
                if service_result.get("ok") and not service_result.get("skipped"):
                    services[name]["last_validated_at"] = str(result.get("validated_at") or "")
            item["services"] = services
            item["updated_at"] = _now_iso()
            self.backend.save_profile(item)
            result["profile"] = self._public_profile(item)
        return result

    async def validate_payload(self, payload: dict[str, Any], *, user_id: str = "web-demo") -> dict[str, Any]:
        raw = dict(payload or {})
        service_payload = _has_service_payload(raw)
        profile_id = _safe_id(str(raw.get("profile_id") or ""))
        existing = {}
        if profile_id:
            try:
                existing = self.backend.get_profile(profile_id, user_id=user_id)
            except ApiProfileError:
                existing = {}
        services = self._build_services(raw, existing=existing, service_payload=service_payload)
        self._assert_required_services(services, service_payload=service_payload)
        runtime_services: dict[str, dict[str, Any]] = {}
        for name, service in services.items():
            runtime = dict(service)
            encrypted = str(runtime.pop("api_key_encrypted", "") or "")
            runtime.pop("api_key_fingerprint", None)
            runtime["api_key"] = self._decrypt(encrypted) if encrypted else ""
            runtime_services[name] = runtime
        public_item = {
            "profile_id": profile_id or "draft",
            "user_id": _normalize_user_id(user_id),
            "display_name": str(raw.get("display_name") or "Draft service profile"),
            "max_concurrent": int(raw.get("max_concurrent") or DEFAULT_API_PROFILE_CONCURRENCY),
            "enabled": bool(raw.get("enabled", True)),
            "is_default": bool(raw.get("is_default", False)),
            "services": services,
            "schema_version": 2 if service_payload else 1,
        }
        llm = runtime_services.get("llm", {})
        runtime = {
            **self._public_profile(public_item),
            "services": runtime_services,
            **runtime_services,
            "api_key": str(llm.get("api_key") or ""),
        }
        return await self._validate_runtime(runtime, service_payload=service_payload)

    def max_concurrent(self, profile_id: str, *, user_id: str = "web-demo") -> int:
        if not profile_id:
            return 0
        try:
            public = self.get_public_profile(profile_id, user_id=user_id)
            return max(1, int(public.get("max_concurrent") or DEFAULT_API_PROFILE_CONCURRENCY))
        except Exception:
            return DEFAULT_API_PROFILE_CONCURRENCY

    def _build_services(self, raw: dict[str, Any], *, existing: dict[str, Any], service_payload: bool) -> dict[str, dict[str, Any]]:
        existing_services = _item_services(existing)
        if not service_payload:
            llm_payload = {
                "provider": raw.get("provider") or existing.get("provider") or "openai_compatible",
                "base_url": raw.get("base_url") or existing.get("base_url") or DEFAULT_OPENAI_BASE_URL,
                "model": raw.get("model") or existing.get("model") or "",
                "api_key": raw.get("api_key") or "",
                "enabled": raw.get("enabled", existing.get("enabled", True)),
                "vision_capable": True,
            }
            return {"llm": self._prepare_service("llm", llm_payload, existing_services.get("llm", {}))}
        services: dict[str, dict[str, Any]] = {}
        for name in SERVICE_KEYS:
            incoming = dict(raw.get(name) or {})
            if not incoming and name in existing_services:
                services[name] = dict(existing_services[name])
                continue
            if not incoming:
                incoming = _blank_service(name)
            services[name] = self._prepare_service(name, incoming, existing_services.get(name, {}))
        return services

    def _prepare_service(self, service_name: str, incoming: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
        data = {**_blank_service(service_name), **dict(existing or {}), **dict(incoming or {})}
        api_key = str(data.pop("api_key", "") or "").strip()
        encrypted_key = str(data.get("api_key_encrypted") or "")
        if api_key:
            encrypted_key = self._encrypt(api_key)
            data["api_key_masked"] = _mask_api_key(api_key)
            data["api_key_fingerprint"] = _fingerprint(api_key)
        else:
            data["api_key_masked"] = str(data.get("api_key_masked") or "")
            data["api_key_fingerprint"] = str(data.get("api_key_fingerprint") or "")
        data["api_key_encrypted"] = encrypted_key
        if service_name in {"llm", "embedding", "image_generation"}:
            data["base_url"] = str(data.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip()
            data["model"] = str(data.get("model") or "").strip()
        if service_name == "embedding":
            try:
                data["dim"] = max(1, int(data.get("dim") or DEFAULT_EMBEDDING_DIM))
            except (TypeError, ValueError):
                data["dim"] = DEFAULT_EMBEDDING_DIM
        if service_name == "pdf":
            data["api_url"] = str(data.get("api_url") or "").strip()
            try:
                data["request_timeout_sec"] = max(30, min(1800, int(data.get("request_timeout_sec") or 180)))
            except (TypeError, ValueError):
                data["request_timeout_sec"] = 180
            try:
                data["poll_timeout_sec"] = max(30, min(3600, int(data.get("poll_timeout_sec") or 180)))
            except (TypeError, ValueError):
                data["poll_timeout_sec"] = 180
        if service_name == "image_generation":
            min_size = data.get("min_image_size")
            try:
                data["min_image_size"] = int(min_size) if min_size not in (None, "") else None
            except (TypeError, ValueError):
                data["min_image_size"] = None
        data["enabled"] = bool(data.get("enabled", service_name in REQUIRED_SERVICE_KEYS))
        return data

    def _assert_required_services(self, services: dict[str, dict[str, Any]], *, service_payload: bool) -> None:
        llm = services.get("llm", {})
        if not str(llm.get("model") or "").strip():
            raise ApiProfileError("llm.model is required")
        if not str(llm.get("api_key_encrypted") or "").strip():
            raise ApiProfileError("llm.api_key is required")
        if not service_payload:
            return
        pdf = services.get("pdf", {})
        if not str(pdf.get("api_key_encrypted") or "").strip() and not str(pdf.get("api_url") or "").strip():
            raise ApiProfileError("pdf.api_key or pdf.api_url is required")

    async def _validate_runtime(self, runtime: dict[str, Any], *, service_payload: bool) -> dict[str, Any]:
        services = runtime.get("services") if isinstance(runtime.get("services"), dict) else {}
        if not services:
            services = {"llm": runtime.get("llm") or runtime}
        service_results: dict[str, dict[str, Any]] = {}
        validated_at = _now_iso()
        required = REQUIRED_SERVICE_KEYS if service_payload else ("llm",)
        service_results["llm"] = await self._validate_llm_service(dict(services.get("llm") or {}))
        if service_payload:
            embedding = dict(services.get("embedding") or {})
            if embedding.get("enabled") and embedding.get("api_key"):
                service_results["embedding"] = await self._validate_embedding_service(embedding)
            else:
                service_results["embedding"] = {"ok": True, "skipped": True, "service": "embedding", "message": "Embedding is optional."}
            service_results["pdf"] = self._validate_pdf_service(dict(services.get("pdf") or {}))
            search = dict(services.get("search") or {})
            service_results["search"] = self._validate_search_service(search) if (search.get("enabled") or search.get("api_key")) else {"ok": True, "skipped": True, "message": "Web search is optional."}
            image = dict(services.get("image_generation") or {})
            service_results["image_generation"] = self._validate_image_service(image) if (image.get("enabled") or image.get("api_key") or image.get("model")) else {"ok": True, "skipped": True, "message": "Image generation is optional."}
        required_ok = all(bool(service_results.get(name, {}).get("ok")) for name in required)
        optional_ok = all(
            bool(payload.get("ok"))
            for name, payload in service_results.items()
            if name in OPTIONAL_SERVICE_KEYS and not payload.get("skipped")
        )
        ok = required_ok and optional_ok
        llm = dict(services.get("llm") or {})
        llm_result = service_results.get("llm", {})
        return {
            "ok": ok,
            "validated_at": validated_at,
            "services": service_results,
            "required_services": list(required),
            "message": self._validation_message(service_results, required),
            "profile": self._public_profile(runtime),
            "model": str(llm_result.get("model") or llm.get("model") or ""),
            "base_url": str(llm_result.get("base_url") or llm.get("base_url") or DEFAULT_OPENAI_BASE_URL),
        }

    async def _validate_llm_service(self, service: dict[str, Any]) -> dict[str, Any]:
        model = str(service.get("model") or "").strip()
        api_key = str(service.get("api_key") or "").strip()
        base_url = str(service.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip()
        if not model:
            return {"ok": False, "service": "llm", "error": "model is required"}
        if not api_key:
            return {"ok": False, "service": "llm", "error": "api_key is required"}
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                max_tokens=4,
            )
            content = response.choices[0].message.content if response.choices else ""
            return {"ok": True, "service": "llm", "model": model, "base_url": base_url, "message": str(content or "").strip()[:80] or "LLM connection OK"}
        finally:
            await self._close_client(client)

    async def _validate_embedding_service(self, service: dict[str, Any]) -> dict[str, Any]:
        model = str(service.get("model") or "").strip()
        api_key = str(service.get("api_key") or "").strip()
        base_url = str(service.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip()
        if not model:
            return {"ok": False, "service": "embedding", "error": "model is required"}
        if not api_key:
            return {"ok": False, "service": "embedding", "error": "api_key is required"}
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)
        try:
            response = await client.embeddings.create(model=model, input=["MemSlides embedding check"])
            embedding = response.data[0].embedding if response.data else []
            return {"ok": True, "service": "embedding", "model": model, "base_url": base_url, "dim": len(embedding), "message": f"Embedding connection OK ({len(embedding)} dimensions)"}
        finally:
            await self._close_client(client)

    def _validate_pdf_service(self, service: dict[str, Any]) -> dict[str, Any]:
        provider = str(service.get("provider") or "pdf_parser_official")
        has_key = bool(str(service.get("api_key") or "").strip())
        api_url = str(service.get("api_url") or "").strip()
        if provider == "pdf_parser_compatible" and not api_url:
            return {"ok": False, "service": "pdf", "error": "api_url is required for a compatible PDF parser endpoint"}
        if provider == "pdf_parser_official" and not has_key:
            return {"ok": False, "service": "pdf", "error": "api_key is required for official PDF parser"}
        return {"ok": True, "service": "pdf", "provider": provider, "message": "PDF parser is configured."}

    def _validate_search_service(self, service: dict[str, Any]) -> dict[str, Any]:
        if not str(service.get("api_key") or "").strip():
            return {"ok": False, "service": "search", "error": "api_key is required when search is enabled"}
        return {"ok": True, "service": "search", "provider": "tavily", "message": "Search key is configured."}

    def _validate_image_service(self, service: dict[str, Any]) -> dict[str, Any]:
        model = str(service.get("model") or "").strip()
        api_key = str(service.get("api_key") or "").strip()
        base_url = str(service.get("base_url") or DEFAULT_OPENAI_BASE_URL).strip()
        if not model:
            return {"ok": False, "service": "image_generation", "error": "model is required when image generation is enabled"}
        if not api_key:
            return {"ok": False, "service": "image_generation", "error": "api_key is required when image generation is enabled"}
        return {"ok": True, "service": "image_generation", "model": model, "base_url": base_url, "message": "Image generation endpoint is configured."}

    @staticmethod
    async def _close_client(client: AsyncOpenAI) -> None:
        close = getattr(client, "close", None)
        if close:
            result = close()
            if asyncio.iscoroutine(result):
                await result

    @staticmethod
    def _validation_message(service_results: dict[str, dict[str, Any]], required: tuple[str, ...]) -> str:
        failures = [
            f"{name}: {payload.get('error') or payload.get('message')}"
            for name, payload in service_results.items()
            if ((name in required or (name in OPTIONAL_SERVICE_KEYS and not payload.get("skipped"))) and not payload.get("ok"))
        ]
        return "; ".join(failures) if failures else "Required services are ready."

    def _public_profile(self, item: dict[str, Any]) -> dict[str, Any]:
        services = _item_services(item)
        public_services = {name: _public_service(service) for name, service in services.items()}
        llm = public_services.get("llm", {})
        public = {
            "schema_version": int(item.get("schema_version") or (2 if "services" in item else 1)),
            "profile_id": str(item.get("profile_id") or ""),
            "user_id": str(item.get("user_id") or "web-demo"),
            "display_name": str(item.get("display_name") or ""),
            "provider": str(llm.get("provider") or item.get("provider") or "openai_compatible"),
            "base_url": str(llm.get("base_url") or item.get("base_url") or ""),
            "model": str(llm.get("model") or item.get("model") or ""),
            "api_key_masked": str(llm.get("api_key_masked") or item.get("api_key_masked") or ""),
            "role_mapping": item.get("role_mapping") if isinstance(item.get("role_mapping"), dict) else {},
            "max_concurrent": int(item.get("max_concurrent") or DEFAULT_API_PROFILE_CONCURRENCY),
            "enabled": bool(item.get("enabled", True)),
            "is_default": bool(item.get("is_default", False)),
            "last_validated_at": str(item.get("last_validated_at") or ""),
            "validation_status": str(item.get("validation_status") or ""),
            "validation_message": str(item.get("validation_message") or ""),
            "created_at": str(item.get("created_at") or ""),
            "updated_at": str(item.get("updated_at") or ""),
            "services": public_services,
            **public_services,
        }
        public["required_ready"] = all(
            bool(public_services.get(name, {}).get("last_validated_at"))
            or bool(public_services.get(name, {}).get("validation_status") == "valid")
            for name in REQUIRED_SERVICE_KEYS
        )
        return public

    def _encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, value: str) -> str:
        if not value:
            return ""
        return self._fernet.decrypt(value.encode("ascii")).decode("utf-8")

    def _load_or_create_fernet_key(self, encryption_secret: str) -> bytes:
        if encryption_secret.strip():
            return _fernet_key_from_secret(encryption_secret)
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        if self.secret_path.exists():
            return self.secret_path.read_bytes().strip()
        key = Fernet.generate_key()
        self.secret_path.write_bytes(key)
        _chmod_private(self.secret_path)
        return key


ServiceProfileStore = ApiProfileStore
ServiceProfileError = ApiProfileError
