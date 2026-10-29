from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memslides.memory.core.template_models import CanonicalLayout, CanonicalSlot
_INDUCTION_META_KEYS = {"functional_keys", "language", "layout_capabilities"}


@dataclass(frozen=True)
class CanonicalLayoutView:
    name: str
    aliases: list[str]
    source_layout_name: str
    layout_archetype: str
    visual_pattern: str
    slots: list[CanonicalSlot]
    sample_content: list[CanonicalSlot]
    sample_slide_indices: list[int]
    capability_signature: dict[str, Any]
    surface_mode: str
    allowed_text_roles: list[str]
    allowed_visual_roles: list[str]
    supports_visual_asset: bool
    confidence: float
    shell_id: str = ""
    shell_mode: str = ""
    protected_regions: list[dict[str, Any]] | None = None
    surface_policy: str = ""
    raw: CanonicalLayout | None = None


def _profile_semantic_model(template_profile: Any) -> Any | None:
    semantic_model = getattr(template_profile, "semantic_model", None)
    if semantic_model is None:
        return None
    layouts = getattr(semantic_model, "canonical_layouts", None)
    if layouts:
        return semantic_model
    return None


def canonical_layout_views(template_profile: Any) -> list[CanonicalLayoutView]:
    semantic_model = _profile_semantic_model(template_profile)
    if semantic_model is not None:
        views: list[CanonicalLayoutView] = []
        for layout in getattr(semantic_model, "canonical_layouts", []) or []:
            if not isinstance(layout, CanonicalLayout):
                continue
            views.append(
                CanonicalLayoutView(
                    name=str(layout.name or layout.id or "").strip(),
                    aliases=[str(alias).strip() for alias in (layout.aliases or []) if str(alias).strip()],
                    source_layout_name=str(layout.source_layout_name or "").strip(),
                    layout_archetype=str(layout.layout_archetype or "").strip(),
                    visual_pattern=str(layout.visual_pattern or "").strip(),
                    slots=list(layout.slots or []),
                    sample_content=list(layout.sample_content or []),
                    sample_slide_indices=[int(idx) for idx in (layout.sample_slide_indices or [])],
                    capability_signature=dict(layout.capability_signature or {}),
                    surface_mode=_reference_surface_mode(str(layout.surface_mode or "").strip()),
                    shell_id="",
                    shell_mode="",
                    protected_regions=[],
                    surface_policy=_reference_surface_policy(str(getattr(layout, "surface_policy", "") or "")),
                    allowed_text_roles=[str(role).strip() for role in (layout.allowed_text_roles or []) if str(role).strip()],
                    allowed_visual_roles=[str(role).strip() for role in (layout.allowed_visual_roles or []) if str(role).strip()],
                    supports_visual_asset=bool(layout.supports_visual_asset),
                    confidence=float(layout.confidence or 0.0),
                    raw=layout,
                )
            )
        if views:
            return views

    induction = getattr(template_profile, "slide_induction", {}) or {}
    if not isinstance(induction, dict):
        return []

    fallback_views: list[CanonicalLayoutView] = []
    for key, value in induction.items():
        if key in _INDUCTION_META_KEYS or not isinstance(value, dict):
            continue
        slots = [
            CanonicalSlot(
                name=str(element.get("name", "") or ""),
                type=str(element.get("type", "text") or "text"),
                role=str(element.get("semantic_role", "") or ""),
                geometry={
                    "left_pct": float(element.get("left_pct", 0) or 0),
                    "top_pct": float(element.get("top_pct", 0) or 0),
                    "width_pct": float(element.get("width_pct", 0) or 0),
                    "height_pct": float(element.get("height_pct", 0) or 0),
                },
                capacity_hint={
                    "sample_char_count": max((len(str(item or "")) for item in element.get("data", []) or []), default=0)
                },
                sample_values=[str(item or "") for item in element.get("data", []) or [] if str(item or "")],
                source=str(element.get("source", "legacy_induction") or "legacy_induction"),
                notes=["fallback_from_slide_induction"],
            )
            for element in value.get("elements", []) or []
        ]
        fallback_views.append(
            CanonicalLayoutView(
                name=str(key),
                aliases=[],
                source_layout_name=str(value.get("source_layout_name", "") or ""),
                layout_archetype="legacy",
                visual_pattern="legacy",
                slots=slots,
                sample_content=[],
                sample_slide_indices=[int(idx) for idx in value.get("slides", []) or []],
                capability_signature={
                    "supports_title": any(slot.role == "title" for slot in slots),
                    "supports_short_body": any(slot.role == "body" for slot in slots),
                    "supports_long_body": any(slot.role == "body" for slot in slots),
                    "supports_figure": any(slot.type == "image" for slot in slots),
                    "supports_chart": any(slot.type == "chart" for slot in slots),
                    "supports_table": any(slot.type == "table" for slot in slots),
                    "supports_multi_column": sum(1 for slot in slots if slot.role == "body") >= 2,
                    "body_slot_count": sum(1 for slot in slots if slot.role == "body"),
                    "image_slot_count": sum(1 for slot in slots if slot.type in {"image", "chart", "table"}),
                    "text_slot_count": sum(1 for slot in slots if slot.type == "text"),
                },
                surface_mode="light_surface" if any(slot.type == "text" for slot in slots) else "visual_surface",
                shell_id="",
                shell_mode="",
                protected_regions=[],
                surface_policy="readable_surface_required",
                allowed_text_roles=[slot.role for slot in slots if slot.type == "text" and slot.role],
                allowed_visual_roles=[slot.role for slot in slots if slot.type in {"image", "chart", "table"} and slot.role],
                supports_visual_asset=any(slot.type in {"image", "chart", "table"} for slot in slots),
                confidence=0.35,
                raw=None,
            )
        )
    return fallback_views


def _reference_surface_mode(surface_mode: str) -> str:
    mode = (surface_mode or "").strip()
    if mode in {"", "shell_text"}:
        return "light_surface"
    return mode


def _reference_surface_policy(surface_policy: str) -> str:
    policy = (surface_policy or "").strip()
    if policy in {"", "shell_text_only", "native_background", "layered_shell", "raster_shell"}:
        return "readable_surface_required"
    if "shell" in policy.lower():
        return "readable_surface_required"
    return policy


def canonical_layout_names(template_profile: Any) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for layout in canonical_layout_views(template_profile):
        name = str(layout.name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def resolve_canonical_layout(template_profile: Any, requested_name: str) -> CanonicalLayoutView | None:
    needle = str(requested_name or "").strip()
    if not needle:
        return None
    lowered = needle.lower()
    views = canonical_layout_views(template_profile)
    for layout in views:
        if layout.name == needle or layout.name.lower() == lowered:
            return layout
    for layout in views:
        haystack = [layout.name, layout.source_layout_name, *layout.aliases]
        for alias in haystack:
            alias_text = str(alias or "").strip()
            if alias_text and alias_text.lower() == lowered:
                return layout
    return None


def semantic_style_system(template_profile: Any) -> dict[str, Any]:
    semantic_model = _profile_semantic_model(template_profile)
    if semantic_model is None:
        return {}
    style_system = getattr(semantic_model, "style_system", None)
    return style_system if isinstance(style_system, dict) else {}


def semantic_decorative_system(template_profile: Any) -> dict[str, Any]:
    semantic_model = _profile_semantic_model(template_profile)
    if semantic_model is None:
        return {}
    decorative_system = getattr(semantic_model, "decorative_system", None)
    return decorative_system if isinstance(decorative_system, dict) else {}


def semantic_sample_content(template_profile: Any) -> dict[str, Any]:
    semantic_model = _profile_semantic_model(template_profile)
    if semantic_model is None:
        return {}
    sample_content = getattr(semantic_model, "sample_content", None)
    return sample_content if isinstance(sample_content, dict) else {}
