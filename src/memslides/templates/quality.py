from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from memslides.templates.semantic_access import canonical_layout_views


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    color = str(color or "").strip().lower()
    if color.startswith("#") and len(color) == 4:
        color = f"#{color[1]*2}{color[2]*2}{color[3]*2}"
    if color.startswith("#") and len(color) == 7:
        try:
            return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        except Exception:
            return None
    return None


def _relative_luminance(color: str) -> float | None:
    rgb = _hex_to_rgb(color)
    if rgb is None:
        return None

    def channel(v: int) -> float:
        x = v / 255.0
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _same_hue_family(c1: str, c2: str) -> bool:
    rgb1 = _hex_to_rgb(c1)
    rgb2 = _hex_to_rgb(c2)
    if rgb1 is None or rgb2 is None:
        return False
    # This is intentionally coarse. It flags generated pale surfaces that are
    # clearly derived from a dark shell color, which should keep the template in
    # adaptive mode instead of pretending the original PPT had safe content areas.
    dominant1 = max(range(3), key=lambda idx: rgb1[idx])
    dominant2 = max(range(3), key=lambda idx: rgb2[idx])
    return dominant1 == dominant2


def _contrast_ratio(c1: str, c2: str) -> float:
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    if l1 is None or l2 is None:
        return 0.0
    light = max(l1, l2)
    dark = min(l1, l2)
    return (light + 0.05) / (dark + 0.05)


STRUCTURAL_TEMPLATE = "structural_template"
STRICT_STRUCTURAL_TEMPLATE = "strict_structural"
ADAPTIVE_STRUCTURAL_TEMPLATE = "adaptive_structural"
STYLE_REFERENCE = "style_reference"
DISABLED = "disabled"


@dataclass
class TemplateQualityReport:
    mode: str
    structure_score: float
    visual_safety_score: float
    reason: str
    issues: list[str] = field(default_factory=list)
    layout_count: int = 0
    content_layout_count: int = 0
    text_slot_count: int = 0
    image_slot_count: int = 0
    zero_text_layout_count: int = 0
    body_size: int = 0
    title_size: int = 0
    background_color: str = ""
    text_color: str = ""
    surface_color: str = ""
    primary_text_color: str = ""
    inverse_text_color: str = ""
    visual_asset_layout_count: int = 0
    effective_surface_color: str = ""
    effective_primary_text_color: str = ""
    effective_muted_text_color: str = ""

    @property
    def is_structural(self) -> bool:
        return self.mode in {STRUCTURAL_TEMPLATE, STRICT_STRUCTURAL_TEMPLATE, ADAPTIVE_STRUCTURAL_TEMPLATE}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iter_layouts(profile: Any) -> list[tuple[str, dict[str, Any]]]:
    layouts: list[tuple[str, dict[str, Any]]] = []
    for layout in canonical_layout_views(profile):
        layouts.append(
            (
                layout.name,
                {
                    "elements": [
                        {
                            "name": slot.name,
                            "type": slot.type,
                            "semantic_role": slot.role,
                        }
                        for slot in layout.slots
                    ],
                    "layout_archetype": layout.layout_archetype,
                    "sample_slide_indices": layout.sample_slide_indices,
                },
            )
        )
    return layouts


def _functional_keys(profile: Any) -> set[str]:
    induction = getattr(profile, "slide_induction", {}) or {}
    keys = induction.get("functional_keys", []) if isinstance(induction, dict) else []
    return {str(key).lower() for key in keys or []}


def assess_template_quality(profile: Any | None) -> TemplateQualityReport:
    if profile is None:
        return TemplateQualityReport(
            mode=DISABLED,
            structure_score=0.0,
            visual_safety_score=0.0,
            reason="No template profile is active.",
        )

    layouts = _iter_layouts(profile)
    functional = _functional_keys(profile)
    content_layouts = [
        (name, data) for name, data in layouts if name.lower() not in functional
    ]

    text_slot_count = 0
    image_slot_count = 0
    zero_text_layout_count = 0
    visual_asset_layout_count = 0
    for _name, data in layouts:
        elements = data.get("elements", []) or []
        text_n = sum(1 for el in elements if el.get("type") == "text")
        img_n = sum(1 for el in elements if el.get("type") in {"image", "chart", "table"})
        text_slot_count += text_n
        image_slot_count += img_n
        if img_n > 0:
            visual_asset_layout_count += 1
        if text_n == 0:
            zero_text_layout_count += 1

    dc = getattr(profile, "design_constraints", None)
    tp = getattr(dc, "typography", None) if dc is not None else None
    cp = getattr(dc, "color_palette", None) if dc is not None else None
    body_size = int(getattr(tp, "body_size", 0) or 0) if tp is not None else 0
    title_size = int(getattr(tp, "title_size", 0) or 0) if tp is not None else 0
    background_color = str(getattr(cp, "background", "") or "") if cp is not None else ""
    text_color = str(getattr(cp, "text", "") or "") if cp is not None else ""
    surface_color = str(getattr(cp, "surface", "") or "") if cp is not None else ""
    primary_text_color = str(getattr(cp, "primary_text", "") or "") if cp is not None else ""
    inverse_text_color = str(getattr(cp, "inverse_text", "") or "") if cp is not None else ""

    issues: list[str] = []
    content_text_layouts = 0
    for name, data in content_layouts:
        elements = data.get("elements", []) or []
        if any(el.get("type") == "text" for el in elements):
            content_text_layouts += 1
        elif ":0t:" in name or name.endswith(":0t"):
            issues.append(f"content layout has no text slot: {name}")

    if not layouts:
        issues.append("no usable layouts found")
    if content_layouts and content_text_layouts == 0:
        issues.append("content layouts contain no text slots")
    if body_size <= 0:
        issues.append("body font size is missing or zero")
    if zero_text_layout_count >= max(1, len(layouts) - 1):
        issues.append("almost all layouts have zero text slots")
    if background_color and not text_color:
        issues.append("background color detected without text color")
    if background_color and not inverse_text_color:
        issues.append("background color detected without inverse text color")
    if surface_color and not primary_text_color:
        issues.append("surface color detected without primary text color")
    if visual_asset_layout_count == 0 and image_slot_count == 0:
        issues.append("template exposes no visual asset layouts")

    score = 1.0
    if not layouts:
        score = 0.0
    else:
        text_layout_ratio = content_text_layouts / max(len(content_layouts), 1)
        text_slot_ratio = text_slot_count / max(text_slot_count + image_slot_count, 1)
        score = 0.45 * text_layout_ratio + 0.35 * min(1.0, text_slot_ratio * 2) + 0.2
        if body_size <= 0:
            score -= 0.25
        if zero_text_layout_count >= max(1, len(layouts) - 1):
            score -= 0.2
        score = max(0.0, min(1.0, score))

    surface_is_likely_derived = False
    bg_lum = _relative_luminance(background_color)
    surface_lum = _relative_luminance(surface_color)
    if (
        background_color
        and surface_color
        and bg_lum is not None
        and surface_lum is not None
        and surface_lum - bg_lum >= 0.35
        and _same_hue_family(background_color, surface_color)
    ):
        surface_is_likely_derived = True
        issues.append("surface appears derived rather than evidenced in template")

    visual_score = 0.0
    if background_color and inverse_text_color:
        visual_score += 0.4
    if surface_color and primary_text_color:
        visual_score += 0.22 if surface_is_likely_derived else 0.4
    elif text_color and surface_color:
        visual_score += 0.15
    if visual_asset_layout_count > 0:
        visual_score += 0.2
    visual_score = max(0.0, min(1.0, visual_score))

    structural_ok = score >= 0.58 and not any(
        issue in issues
        for issue in (
            "content layouts contain no text slots",
            "body font size is missing or zero",
            "almost all layouts have zero text slots",
        )
    )
    visual_safe = visual_score >= 0.72 and not surface_is_likely_derived
    if structural_ok and visual_safe:
        mode = STRICT_STRUCTURAL_TEMPLATE
    elif structural_ok:
        mode = ADAPTIVE_STRUCTURAL_TEMPLATE
    else:
        mode = STYLE_REFERENCE

    reason = (
        "Template has enough content structure and visual safety for strict layout guidance."
        if mode == STRICT_STRUCTURAL_TEMPLATE
        else "Template structure is usable but content pages need adaptive rendering."
        if mode == ADAPTIVE_STRUCTURAL_TEMPLATE
        else "Template structure is weak; use it as style reference only."
    )

    report = TemplateQualityReport(
        mode=mode,
        structure_score=round(score, 3),
        visual_safety_score=round(visual_score, 3),
        reason=reason,
        issues=issues,
        layout_count=len(layouts),
        content_layout_count=len(content_layouts),
        text_slot_count=text_slot_count,
        image_slot_count=image_slot_count,
        zero_text_layout_count=zero_text_layout_count,
        body_size=body_size,
        title_size=title_size,
        background_color=background_color,
        text_color=text_color,
        surface_color=surface_color,
        primary_text_color=primary_text_color,
        inverse_text_color=inverse_text_color,
        visual_asset_layout_count=visual_asset_layout_count,
    )
    effective_palette = effective_template_palette(profile, mode=mode, report=report)
    report.effective_surface_color = effective_palette.surface
    report.effective_primary_text_color = effective_palette.primary_text
    report.effective_muted_text_color = effective_palette.muted_text
    return report


def effective_template_palette(
    profile: Any | None,
    *,
    mode: str = "",
    report: TemplateQualityReport | None = None,
):
    from memslides.memory.core.template_models import ColorPalette

    if profile is None:
        return ColorPalette()

    dc = getattr(profile, "design_constraints", None)
    cp = getattr(dc, "color_palette", None) if dc is not None else None
    if cp is None:
        return ColorPalette()

    resolved_report = report
    if resolved_report is None and profile is not None:
        try:
            resolved_report = assess_template_quality(profile)
        except Exception:
            resolved_report = None

    if hasattr(cp, "to_dict"):
        effective = ColorPalette.from_dict(cp.to_dict())
    else:
        effective = ColorPalette(
            primary=str(getattr(cp, "primary", "") or ""),
            secondary=str(getattr(cp, "secondary", "") or ""),
            accent=str(getattr(cp, "accent", "") or ""),
            background=str(getattr(cp, "background", "") or ""),
            text=str(getattr(cp, "text", "") or ""),
            surface=str(getattr(cp, "surface", "") or ""),
            primary_text=str(getattr(cp, "primary_text", "") or ""),
            inverse_text=str(getattr(cp, "inverse_text", "") or ""),
            muted_text=str(getattr(cp, "muted_text", "") or ""),
            border=str(getattr(cp, "border", "") or ""),
            additional=list(getattr(cp, "additional", None) or []),
        )

    resolved_mode = str(mode or getattr(resolved_report, "mode", "") or "")
    issues = set(getattr(resolved_report, "issues", []) or [])
    surface_is_unsafe = (
        resolved_mode == ADAPTIVE_STRUCTURAL_TEMPLATE
        and (
            "surface appears derived rather than evidenced in template" in issues
            or not effective.surface
            or not effective.primary_text
            or _contrast_ratio(effective.surface or "#ffffff", effective.primary_text or "#000000") < 5.5
        )
    )

    if surface_is_unsafe:
        effective.surface = "#ffffff"
        text_candidates = [
            str(getattr(cp, "primary_text", "") or ""),
            str(getattr(cp, "text", "") or ""),
            "#1f2937",
            "#111827",
        ]
        chosen_primary_text = ""
        for candidate in text_candidates:
            if candidate and _contrast_ratio(candidate, effective.surface) >= 7.0:
                chosen_primary_text = candidate
                break
        if not chosen_primary_text:
            chosen_primary_text = "#1f2937"
        effective.primary_text = chosen_primary_text
        if not effective.text or _contrast_ratio(effective.text, effective.surface) < 4.5:
            effective.text = chosen_primary_text
        if not effective.muted_text or _contrast_ratio(effective.muted_text, effective.surface) < 3.8:
            effective.muted_text = "#475569"
        if not effective.border:
            effective.border = effective.primary or effective.accent or "#d0d7de"

    if not effective.inverse_text and effective.background:
        effective.inverse_text = "#ffffff"

    return effective


def write_template_quality_report(
    workspace: Path | str,
    report: TemplateQualityReport,
) -> Path:
    path = Path(workspace) / "template_quality.json"
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


__all__ = [
    "ADAPTIVE_STRUCTURAL_TEMPLATE",
    "DISABLED",
    "STRICT_STRUCTURAL_TEMPLATE",
    "STRUCTURAL_TEMPLATE",
    "STYLE_REFERENCE",
    "TemplateQualityReport",
    "assess_template_quality",
    "write_template_quality_report",
]
