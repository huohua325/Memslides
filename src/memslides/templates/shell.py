from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


NATIVE_BACKGROUND = "native_background"
LAYERED_SHELL = "layered_shell"
RASTER_SHELL = "raster_shell"
NO_SHELL = "no_shell"


@dataclass(frozen=True)
class TemplateShellView:
    shell_id: str
    shell_mode: str
    source_layout_name: str = ""
    sample_slide_indices: list[int] = field(default_factory=list)
    background_assets: list[dict[str, Any]] = field(default_factory=list)
    decorative_layers: list[dict[str, Any]] = field(default_factory=list)
    protected_regions: list[dict[str, Any]] = field(default_factory=list)
    safe_content_regions: list[dict[str, Any]] = field(default_factory=list)
    surface_policy: str = "adaptive_surface_allowed"
    reuse_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


def _semantic_model(template_profile: Any) -> Any | None:
    semantic_model = getattr(template_profile, "semantic_model", None)
    return semantic_model


def template_shell_model(template_profile: Any) -> dict[str, Any]:
    semantic_model = _semantic_model(template_profile)
    if semantic_model is None:
        return {}
    payload = getattr(semantic_model, "template_shell", None)
    return payload if isinstance(payload, dict) else {}


def template_shell_by_layout(template_profile: Any) -> dict[str, TemplateShellView]:
    shell_model = template_shell_model(template_profile)
    layouts = shell_model.get("layouts", {}) if isinstance(shell_model, dict) else {}
    if not isinstance(layouts, dict):
        return {}

    result: dict[str, TemplateShellView] = {}
    for layout_name, payload in layouts.items():
        if not isinstance(payload, dict):
            continue
        shell_id = str(payload.get("shell_id", "") or layout_name).strip()
        result[str(layout_name)] = TemplateShellView(
            shell_id=shell_id,
            shell_mode=str(payload.get("shell_mode", "") or NO_SHELL),
            source_layout_name=str(payload.get("source_layout_name", "") or ""),
            sample_slide_indices=[
                int(idx) for idx in (payload.get("sample_slide_indices", []) or [])
                if str(idx).strip()
            ],
            background_assets=[
                item for item in (payload.get("background_assets", []) or [])
                if isinstance(item, dict)
            ],
            decorative_layers=[
                item for item in (payload.get("decorative_layers", []) or [])
                if isinstance(item, dict)
            ],
            protected_regions=[
                item for item in (payload.get("protected_regions", []) or [])
                if isinstance(item, dict)
            ],
            safe_content_regions=[
                item for item in (payload.get("safe_content_regions", []) or [])
                if isinstance(item, dict)
            ],
            surface_policy=str(payload.get("surface_policy", "") or "adaptive_surface_allowed"),
            reuse_confidence=float(payload.get("reuse_confidence", 0.0) or 0.0),
            notes=[str(item) for item in (payload.get("notes", []) or []) if str(item).strip()],
        )
    return result


def resolve_template_shell(template_profile: Any, layout_name: str) -> TemplateShellView | None:
    shells = template_shell_by_layout(template_profile)
    if not shells:
        return None
    if layout_name in shells:
        return shells[layout_name]

    lowered = str(layout_name or "").strip().lower()
    for name, shell in shells.items():
        if name.lower() == lowered or shell.source_layout_name.lower() == lowered:
            return shell
    return None


def shell_summary(template_profile: Any) -> dict[str, Any]:
    shell_model = template_shell_model(template_profile)
    summary = shell_model.get("summary", {}) if isinstance(shell_model, dict) else {}
    return summary if isinstance(summary, dict) else {}


__all__ = [
    "NATIVE_BACKGROUND",
    "LAYERED_SHELL",
    "RASTER_SHELL",
    "NO_SHELL",
    "TemplateShellView",
    "template_shell_model",
    "template_shell_by_layout",
    "resolve_template_shell",
    "shell_summary",
]
