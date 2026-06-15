from __future__ import annotations

import uuid
import json
import os
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

from memslides.contracts import DeckRequest, DeckResult, SessionOptions, RevisionRequest
from memslides.memory.core.models import (
    DEFAULT_INTENT,
    UserCoreProfile,
    UserProfile,
    normalize_intent_label,
)
from memslides.memory.runtime import MemoryRuntime
from memslides.pipelines import GenerationPipeline, RevisionPipeline
from memslides.runtime import AgentLoop
from memslides.utils.config import GLOBAL_CONFIG, LLM, EmbeddingConfig, MemSlidesConfig
from memslides.utils.constants import WORKSPACE_BASE


RUNTIME_LLM_PROFILE_KEYS = [
    "research_agent",
    "design_agent",
    "modify_agent",
    "reviewer_agent",
    "fast_model",
    "balanced_model",
    "long_context_model",
    "vision_model",
]


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _apply_runtime_llm_profile(
    config: MemSlidesConfig,
    profile: dict[str, Any],
    workspace: Path,
) -> MemSlidesConfig:
    """Override Web-facing service config with one user-provided profile.

    Some MCP tools reload configuration through MEMSLIDES_CONFIG_FILE, so this
    function writes a workspace-local runtime config and points config.file_path
    at it instead of only mutating the in-memory config object.
    """
    services = profile.get("services") if isinstance(profile.get("services"), dict) else {}
    llm_profile = dict(services.get("llm") or profile.get("llm") or profile)
    base_url = str(llm_profile.get("base_url") or "").strip()
    model = str(llm_profile.get("model") or "").strip()
    api_key = str(llm_profile.get("api_key") or "").strip()
    if not base_url or not model or not api_key:
        return config

    try:
        max_concurrent = int(profile.get("max_concurrent") or 0) or None
    except (TypeError, ValueError):
        max_concurrent = None

    for key in RUNTIME_LLM_PROFILE_KEYS:
        existing = config.get_optional_llm(key)
        payload = existing.model_dump(mode="python") if existing is not None else {}
        payload.update({
            "base_url": base_url,
            "model": model,
            "api_key": api_key,
            "endpoints": [],
        })
        if max_concurrent is not None:
            payload["max_concurrent"] = max(1, max_concurrent)
        if key in {"design_agent", "modify_agent", "reviewer_agent", "vision_model"}:
            payload["is_multimodal"] = bool(payload.get("is_multimodal", True))
        payload.setdefault("client_kwargs", {"timeout": 300.0, "max_retries": 1})
        payload.setdefault("sampling_parameters", {})
        llm = LLM(**payload)
        if key in type(config).model_fields and isinstance(getattr(config, key, None), LLM):
            setattr(config, key, llm)
        else:
            config.extra_llms[key] = llm

    is_service_profile = int(profile.get("schema_version") or 1) >= 2
    if is_service_profile:
        embedding_profile = dict(services.get("embedding") or profile.get("embedding") or {})
        _apply_runtime_embedding_config(config, embedding_profile)

        image_profile = dict(services.get("image_generation") or profile.get("image_generation") or {})
        _apply_runtime_image_config(config, image_profile, max_concurrent=max_concurrent)

    runtime_dir = workspace / ".runtime_secrets"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / "runtime_config.yaml"
    runtime_mcp_config = runtime_dir / "runtime_mcp.json"
    _write_runtime_mcp_config(
        config,
        runtime_mcp_config,
        pdf_profile=dict(services.get("pdf") or profile.get("pdf") or {}) if is_service_profile else {},
        search_profile=dict(services.get("search") or profile.get("search") or {}) if is_service_profile else {},
    )
    if runtime_mcp_config.exists():
        config.mcp_config_file = str(runtime_mcp_config)
    data = json.loads(config.model_dump_json())
    extra_llms = data.pop("extra_llms", {}) or {}
    for key, value in extra_llms.items():
        data[key] = value
    data["file_path"] = str(runtime_config)
    runtime_config.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _chmod_private(runtime_config)
    config.file_path = str(runtime_config)
    return config


def _apply_runtime_embedding_config(config: MemSlidesConfig, embedding_profile: dict[str, Any]) -> None:
    if not embedding_profile or embedding_profile.get("enabled") is False:
        return
    base_url = str(embedding_profile.get("base_url") or "").strip()
    model = str(embedding_profile.get("model") or "").strip()
    api_key = str(embedding_profile.get("api_key") or "").strip()
    if not base_url or not model or not api_key:
        return
    try:
        dim = int(embedding_profile.get("dim") or 1536)
    except (TypeError, ValueError):
        dim = 1536
    existing = config.memory.embedding.model_dump(mode="python") if config.memory.embedding else {}
    existing.update(
        {
            "provider": "openai-compatible",
            "model": model,
            "dim": max(1, dim),
            "base_url": base_url,
            "api_key": api_key,
            "api_model": model,
            "api_base_url": base_url,
            "cache_enabled": bool(existing.get("cache_enabled", True)),
            "cache_size": int(existing.get("cache_size", 512) or 512),
        }
    )
    config.memory.embedding = EmbeddingConfig(**existing)


def _apply_runtime_image_config(
    config: MemSlidesConfig,
    image_profile: dict[str, Any],
    *,
    max_concurrent: int | None = None,
) -> None:
    if not image_profile or image_profile.get("enabled") is False:
        config.t2i_model = None
        return
    base_url = str(image_profile.get("base_url") or "").strip()
    model = str(image_profile.get("model") or "").strip()
    api_key = str(image_profile.get("api_key") or "").strip()
    if not base_url or not model or not api_key:
        config.t2i_model = None
        return
    payload: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "api_key": api_key,
        "endpoints": [],
        "client_kwargs": {"timeout": 300.0, "max_retries": 1},
        "sampling_parameters": {},
    }
    if max_concurrent is not None:
        payload["max_concurrent"] = max(1, max_concurrent)
    if image_profile.get("min_image_size") not in (None, ""):
        try:
            payload["min_image_size"] = max(1, int(image_profile["min_image_size"]))
        except (TypeError, ValueError):
            pass
    config.t2i_model = LLM(**payload)


def _write_runtime_mcp_config(
    config: MemSlidesConfig,
    target: Path,
    *,
    pdf_profile: dict[str, Any],
    search_profile: dict[str, Any],
) -> None:
    source = Path(str(config.mcp_config_file or "")).expanduser()
    if not source.exists():
        return
    try:
        mcp_data = json.loads(source.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(mcp_data, list):
        return

    pdf_enabled = bool(pdf_profile and pdf_profile.get("enabled", True))
    pdf_provider = str(pdf_profile.get("provider") or "pdf_parser_official")
    pdf_key = str(pdf_profile.get("api_key") or "").strip()
    pdf_url = str(pdf_profile.get("api_url") or "").strip()
    allow_local_fallback = os.getenv("MEMSLIDES_ALLOW_LOCAL_PDF_FALLBACK", "").strip().lower() in {"1", "true", "yes"}
    search_key = str(search_profile.get("api_key") or "").strip() if search_profile.get("enabled") else ""

    for server in mcp_data:
        if not isinstance(server, dict):
            continue
        name = str(server.get("name") or "")
        env = dict(server.get("env") or {})
        if name == "memslides_document_tools":
            env.pop("MEMSLIDES_PDF_PARSER_API_KEY", None)
            env.pop("MEMSLIDES_MINERU_API_URL", None)
            if pdf_enabled and pdf_provider == "pdf_parser_official" and pdf_key:
                env["MEMSLIDES_PDF_PARSER_API_KEY"] = pdf_key
                env["MEMSLIDES_PDF_CONVERSION_BACKEND"] = "auto" if allow_local_fallback else "pdf_parser"
            elif pdf_enabled and pdf_provider == "pdf_parser_compatible" and pdf_url:
                env["MEMSLIDES_MINERU_API_URL"] = pdf_url
                env["MEMSLIDES_PDF_CONVERSION_BACKEND"] = "auto" if allow_local_fallback else "pdf_parser"
            env["MEMSLIDES_MINERU_REQUEST_TIMEOUT_SEC"] = str(pdf_profile.get("request_timeout_sec") or 180)
            env["MEMSLIDES_MINERU_POLL_TIMEOUT_SEC"] = str(pdf_profile.get("poll_timeout_sec") or 180)
        elif name == "memslides_search_tools":
            env.pop("MEMSLIDES_TAVILY_API_KEY", None)
            if search_key:
                env["MEMSLIDES_TAVILY_API_KEY"] = search_key
        elif name in {"memslides_deck_tools", "memslides_template_tools", "memslides_asset_tools", "memslides_memory_tools"}:
            env["MEMSLIDES_CONFIG_FILE"] = "$MEMSLIDES_CONFIG_FILE"
        server["env"] = env

    target.write_text(json.dumps(mcp_data, ensure_ascii=False, indent=2), encoding="utf-8")
    _chmod_private(target)


class MemSlidesSession:
    """Public session API for MemSlides."""

    def __init__(self, options: SessionOptions | None = None):
        self.options = options or SessionOptions()
        self.session_id = self.options.session_id or str(uuid.uuid4())[:8]
        self.workspace = self.options.workspace or self._default_workspace(self.session_id)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = self._load_config(self.options, self.workspace)
        self._runtime = AgentLoop(
            config=self.config,
            session_id=self.session_id,
            workspace=self.workspace,
            language=self.options.language,
            user_id=self.options.memory.user_id,
        )
        self._memory_runtime: MemoryRuntime | None = None
        self._generation = GenerationPipeline(self._runtime)
        self._revision = RevisionPipeline(self._runtime)

    @staticmethod
    def _load_config(options: SessionOptions, workspace: Path | None = None) -> MemSlidesConfig:
        if options.config_file:
            config = MemSlidesConfig.load_from_file(str(options.config_file))
        else:
            config = MemSlidesConfig.model_validate(GLOBAL_CONFIG.model_dump(mode="python"))
        runtime_profile = dict(options.runtime_service_profile or options.runtime_llm_profile or {})
        if runtime_profile and workspace is not None:
            config = _apply_runtime_llm_profile(config, runtime_profile, workspace)
        return config

    @staticmethod
    def _default_workspace(session_id: str) -> Path:
        date_part = datetime.now().astimezone().strftime("%Y%m%d")
        return WORKSPACE_BASE / "sessions" / date_part / session_id

    @property
    def runtime(self) -> AgentLoop:
        return self._runtime

    @property
    def memory_runtime(self) -> MemoryRuntime | None:
        return self._memory_runtime

    async def generate(self, request: DeckRequest) -> DeckResult:
        self._apply_session_defaults(request)
        await self._ensure_memory_runtime()
        return await self._generation.run(request, check_llms=self.options.check_llms)

    async def revise(self, feedback: RevisionRequest | str) -> DeckResult:
        request = feedback if isinstance(feedback, RevisionRequest) else RevisionRequest(feedback=feedback)
        self._apply_request_memory_overrides(request)
        await self._ensure_memory_runtime()
        return await self._revision.run(request)

    async def resume(self) -> str | None:
        """Restore a workspace-backed session so reopened decks can be revised."""
        await self._ensure_memory_runtime()
        return await self._runtime.resume()

    async def save_memory(self, progress_callback: Any = None) -> dict[str, Any]:
        """Persist current working memory without closing the live session."""
        if self.options.memory.global_db_dir:
            self.config.memory.global_db_dir = str(self.options.memory.global_db_dir)
        return await self._runtime.flush_memory(
            reason="manual_web_save",
            progress_callback=progress_callback,
        )

    async def get_working_memory_snapshot(self) -> dict[str, Any]:
        """Return the current live WM state for inspection in Web Studio."""
        await self._ensure_memory_runtime()
        return self._runtime.get_working_memory_snapshot()

    async def update_working_memory_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Replace current session WM entries with user-edited data."""
        await self._ensure_memory_runtime()
        return await self._runtime.update_working_memory_snapshot(payload)

    async def get_user_profile(self, intent: str = "") -> dict[str, Any]:
        """Load the persisted core profile and the selected intent profile."""
        await self._ensure_memory_runtime()
        system = self._memory_runtime.system if self._memory_runtime else None
        if system is None:
            raise RuntimeError("Memory runtime is not available for this session.")
        core_store = getattr(system, "core_profile_store", None)
        profile_store = getattr(system, "profile_store", None)
        if core_store is None or profile_store is None:
            raise RuntimeError("User profile stores are not available for this session.")

        user_id = str(self.options.memory.user_id or self._runtime.user_id or "default")
        resolved_intent = self._resolve_profile_intent(intent)
        core_profile = await core_store.get(user_id)
        intent_profile = await profile_store.get(user_id, intent=resolved_intent)
        available_intents = []
        if hasattr(profile_store, "list_intents"):
            available_intents = await profile_store.list_intents(user_id)
        if resolved_intent not in available_intents:
            available_intents.append(resolved_intent)
        available_intents = sorted({str(item or DEFAULT_INTENT) for item in available_intents})

        return {
            "user_id": user_id,
            "intent": resolved_intent,
            "available_intents": available_intents,
            "version": {
                "core": getattr(core_profile, "version", 1),
                "intent": getattr(intent_profile, "version", 1),
            },
            "last_updated": {
                "core": getattr(core_profile, "last_updated", ""),
                "intent": getattr(intent_profile, "last_updated", ""),
            },
            "core_profile": _core_profile_to_editor(core_profile),
            "intent_profile": _intent_profile_to_editor(intent_profile),
        }

    async def save_user_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Persist a hand-edited core profile and intent profile."""
        await self._ensure_memory_runtime()
        system = self._memory_runtime.system if self._memory_runtime else None
        if system is None:
            raise RuntimeError("Memory runtime is not available for this session.")
        core_store = getattr(system, "core_profile_store", None)
        profile_store = getattr(system, "profile_store", None)
        if core_store is None or profile_store is None:
            raise RuntimeError("User profile stores are not available for this session.")

        user_id = str(self.options.memory.user_id or self._runtime.user_id or "default")
        resolved_intent = self._resolve_profile_intent(str(payload.get("intent") or ""))
        core_data = _editor_core_to_profile_data(payload.get("core_profile") or {})
        intent_data = _editor_intent_to_profile_data(payload.get("intent_profile") or {})
        version = payload.get("version") if isinstance(payload.get("version"), dict) else {}
        last_updated = payload.get("last_updated") if isinstance(payload.get("last_updated"), dict) else {}
        core_data["user_id"] = user_id
        intent_data["user_id"] = user_id
        core_data["version"] = _as_int(version.get("core"), 1)
        intent_data["version"] = _as_int(version.get("intent"), 1)
        core_data["last_updated"] = str(last_updated.get("core") or "")
        intent_data["last_updated"] = str(last_updated.get("intent") or "")

        core_profile = UserCoreProfile.from_dict(core_data)
        intent_profile = UserProfile.from_dict(intent_data)
        await core_store.save(core_profile)
        await profile_store.save(intent_profile, intent=resolved_intent)
        return await self.get_user_profile(resolved_intent)

    async def close(self) -> None:
        await self._runtime.close_env()
        if self._memory_runtime is not None:
            await self._memory_runtime.close()
            self._memory_runtime = None
            self._runtime.memory_system = None
            self._runtime.memory_runtime = None

    def _resolve_profile_intent(self, intent: str = "") -> str:
        normalized = normalize_intent_label(intent) or normalize_intent_label(
            getattr(self._runtime, "_resolved_request_intent", "")
        )
        return normalized or DEFAULT_INTENT

    async def _ensure_memory_runtime(self) -> None:
        if self._memory_runtime is not None:
            return
        if self.options.memory.global_db_dir:
            self.config.memory.global_db_dir = str(self.options.memory.global_db_dir)
        self.config.memory.enabled = bool(self.options.memory.enabled)
        self._memory_runtime = await MemoryRuntime.from_config(self.config, workspace=self.workspace)
        self._runtime.memory_system = self._memory_runtime.system if self._memory_runtime else None
        self._runtime.memory_runtime = self._memory_runtime

    def _apply_session_defaults(self, request: DeckRequest) -> None:
        self._apply_request_memory_overrides(request)
        if request.language is None:
            request.language = self.options.language
        template_options = self.options.template
        if request.template is None and template_options.template is not None:
            request.template = template_options.template
        if request.template_id is None and template_options.template_id:
            request.template_id = template_options.template_id
        if request.template_as_reference is None:
            request.template_as_reference = template_options.template_as_reference

    def _apply_request_memory_overrides(self, request: DeckRequest) -> None:
        extra = request.extra_info or {}
        overrides = extra.get("memory_module_overrides")
        if not isinstance(overrides, dict):
            return

        modules = getattr(self.config.memory, "modules", None)
        if modules is None:
            return

        for key, value in overrides.items():
            if not hasattr(modules, key):
                continue
            normalized = self._coerce_bool(value)
            if normalized is None:
                continue
            setattr(modules, key, normalized)

    @staticmethod
    def _coerce_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        return None


__all__ = ["MemSlidesSession"]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.splitlines() if "\n" in value else value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    items: list[str] = []
    for raw in raw_items:
        text = str(raw or "").strip()
        if text and text not in items:
            items.append(text)
    return items


def _as_confidence(value: Any, default: float = 0.5) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return min(1.0, max(0.0, numeric))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_font_size_range(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        first = int(value[0])
        second = int(value[1])
    except (TypeError, ValueError):
        return None
    if first <= 0 or second <= 0:
        return None
    return (min(first, second), max(first, second))


def _theme_to_editor(theme: Any) -> dict[str, Any]:
    return {
        "primary_colors": list(getattr(theme, "primary_colors", []) or []),
        "accent_colors": list(getattr(theme, "accent_colors", []) or []),
        "font_family": str(getattr(theme, "font_family", "") or ""),
        "font_size_range": list(getattr(theme, "font_size_range", None) or []) or None,
        "background_style": str(getattr(theme, "background_style", "") or ""),
        "confidence": float(getattr(theme, "confidence", 0.5) or 0.5),
        "notes": list(getattr(theme, "notes", []) or []),
    }


def _visual_to_editor(visual: Any) -> dict[str, Any]:
    return {
        "image_style": str(getattr(visual, "image_style", "") or ""),
        "chart_type_priority": list(getattr(visual, "chart_type_priority", []) or []),
        "icon_usage": str(getattr(visual, "icon_usage", "") or ""),
        "animation_preference": str(getattr(visual, "animation_preference", "") or ""),
        "confidence": float(getattr(visual, "confidence", 0.5) or 0.5),
        "notes": list(getattr(visual, "notes", []) or []),
    }


def _layout_to_editor(layout: Any) -> dict[str, Any]:
    return {
        "content_density": str(getattr(layout, "content_density", "") or ""),
        "alignment_style": str(getattr(layout, "alignment_style", "") or ""),
        "spacing_preference": str(getattr(layout, "spacing_preference", "") or ""),
        "slide_structure": str(getattr(layout, "slide_structure", "") or ""),
        "confidence": float(getattr(layout, "confidence", 0.5) or 0.5),
        "notes": list(getattr(layout, "notes", []) or []),
    }


def _content_to_editor(content: Any) -> dict[str, Any]:
    return {
        "text_density": str(getattr(content, "text_density", "") or ""),
        "language_style": str(getattr(content, "language_style", "") or ""),
        "bullet_point_style": str(getattr(content, "bullet_point_style", "") or ""),
        "title_length": str(getattr(content, "title_length", "") or ""),
        "confidence": float(getattr(content, "confidence", 0.5) or 0.5),
        "notes": list(getattr(content, "notes", []) or []),
    }


def _template_to_editor(template: Any) -> dict[str, Any]:
    return {
        "preferred_templates": list(getattr(template, "preferred_templates", []) or []),
        "avoid_templates": list(getattr(template, "avoid_templates", []) or []),
        "selection_criteria": str(getattr(template, "selection_criteria", "") or ""),
        "history_preferred_templates": list(getattr(template, "history_preferred_templates", []) or []),
        "history_preferred_template_ids": list(getattr(template, "history_preferred_template_ids", []) or []),
        "history_reuse_scenarios": list(getattr(template, "history_reuse_scenarios", []) or []),
        "history_supported_aspect_ratios": list(getattr(template, "history_supported_aspect_ratios", []) or []),
        "last_successful_template_id": str(getattr(template, "last_successful_template_id", "") or ""),
        "last_successful_template_name": str(getattr(template, "last_successful_template_name", "") or ""),
        "last_successful_at": str(getattr(template, "last_successful_at", "") or ""),
        "history_supporting_usage_count": int(getattr(template, "history_supporting_usage_count", 0) or 0),
        "confidence": float(getattr(template, "confidence", 0.5) or 0.5),
        "notes": list(getattr(template, "notes", []) or []),
    }


def _general_to_editor(general: Any) -> dict[str, Any]:
    return {
        "preferences": list(getattr(general, "preferences", []) or []),
        "confidence": float(getattr(general, "confidence", 0.5) or 0.5),
        "notes": list(getattr(general, "notes", []) or []),
    }


def _core_profile_to_editor(profile: UserCoreProfile) -> dict[str, Any]:
    return {
        "core_persona": str(profile.core_persona or ""),
        "theme": _theme_to_editor(profile.theme),
        "visual": _visual_to_editor(profile.visual),
        "layout": _layout_to_editor(profile.layout),
        "content": _content_to_editor(profile.content),
        "general": _general_to_editor(profile.general),
    }


def _intent_profile_to_editor(profile: UserProfile) -> dict[str, Any]:
    return {
        "theme": _theme_to_editor(profile.theme),
        "visual": _visual_to_editor(profile.visual),
        "layout": _layout_to_editor(profile.layout),
        "content": _content_to_editor(profile.content),
        "template": _template_to_editor(profile.template),
        "general": _general_to_editor(profile.general),
    }


def _theme_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "primary_colors": _as_string_list(data.get("primary_colors")),
        "accent_colors": _as_string_list(data.get("accent_colors")),
        "font_family": str(data.get("font_family") or "").strip(),
        "font_size_range": _as_font_size_range(data.get("font_size_range")),
        "background_style": str(data.get("background_style") or "").strip(),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _visual_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "image_style": str(data.get("image_style") or "").strip(),
        "chart_type_priority": _as_string_list(data.get("chart_type_priority")),
        "icon_usage": str(data.get("icon_usage") or "").strip(),
        "animation_preference": str(data.get("animation_preference") or "").strip(),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _layout_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "content_density": str(data.get("content_density") or "").strip(),
        "alignment_style": str(data.get("alignment_style") or "").strip(),
        "spacing_preference": str(data.get("spacing_preference") or "").strip(),
        "slide_structure": str(data.get("slide_structure") or "").strip(),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _content_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "text_density": str(data.get("text_density") or "").strip(),
        "language_style": str(data.get("language_style") or "").strip(),
        "bullet_point_style": str(data.get("bullet_point_style") or "").strip(),
        "title_length": str(data.get("title_length") or "").strip(),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _template_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "preferred_templates": _as_string_list(data.get("preferred_templates")),
        "avoid_templates": _as_string_list(data.get("avoid_templates")),
        "selection_criteria": str(data.get("selection_criteria") or "").strip(),
        "history_preferred_templates": _as_string_list(data.get("history_preferred_templates")),
        "history_preferred_template_ids": _as_string_list(data.get("history_preferred_template_ids")),
        "history_reuse_scenarios": _as_string_list(data.get("history_reuse_scenarios")),
        "history_supported_aspect_ratios": _as_string_list(data.get("history_supported_aspect_ratios")),
        "last_successful_template_id": str(data.get("last_successful_template_id") or "").strip(),
        "last_successful_template_name": str(data.get("last_successful_template_name") or "").strip(),
        "last_successful_at": str(data.get("last_successful_at") or "").strip(),
        "history_supporting_usage_count": int(data.get("history_supporting_usage_count") or 0),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _general_from_editor(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "preferences": _as_string_list(data.get("preferences")),
        "confidence": _as_confidence(data.get("confidence")),
        "notes": _as_string_list(data.get("notes")),
    }


def _editor_core_to_profile_data(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "core_persona": str(data.get("core_persona") or "").strip(),
        "theme": _theme_from_editor(data.get("theme")),
        "visual": _visual_from_editor(data.get("visual")),
        "layout": _layout_from_editor(data.get("layout")),
        "content": _content_from_editor(data.get("content")),
        "general": _general_from_editor(data.get("general")),
    }


def _editor_intent_to_profile_data(data: Any) -> dict[str, Any]:
    data = data if isinstance(data, dict) else {}
    return {
        "theme": _theme_from_editor(data.get("theme")),
        "visual": _visual_from_editor(data.get("visual")),
        "layout": _layout_from_editor(data.get("layout")),
        "content": _content_from_editor(data.get("content")),
        "template": _template_from_editor(data.get("template")),
        "general": _general_from_editor(data.get("general")),
    }
