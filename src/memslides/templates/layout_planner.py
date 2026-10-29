from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from memslides.templates.page_asset_planner import PageAssetPlanner
from memslides.templates.semantic_access import canonical_layout_views, semantic_style_system


@dataclass
class PageBrief:
    page_index: int
    title: str
    body: str
    raw_markdown: str
    page_purpose: str
    content_shape: str
    asset_kinds: list[str] = field(default_factory=list)
    formula_count: int = 0
    table_count: int = 0
    image_count: int = 0
    bound_asset_kind: str = "none"
    bound_asset_path: str = ""
    visual_requirement: str = "none"
    formula_snippet: str = ""


@dataclass
class LayoutRecommendation:
    selected_layout: str
    alternatives: list[str]
    fit_score: float
    slot_assignment_preview: dict[str, str]
    rejection_reasons: list[str]
    why_selected: str


class TemplateLayoutPlanner:
    """Deterministic page-to-layout planner for template-driven generation."""

    def __init__(self, template_profile: Any):
        self.template_profile = template_profile
        self.layouts = canonical_layout_views(template_profile)

    def build_layout_mapping(
        self,
        *,
        manuscript_text: str,
        asset_manifest: dict[str, Any] | None = None,
        page_count_hint: int | None = None,
    ) -> dict[str, Any]:
        asset_manifest_payload = asset_manifest or {}
        page_briefs = self._extract_page_briefs(
            manuscript_text,
            asset_manifest=asset_manifest_payload,
            page_count_hint=page_count_hint,
        )
        asset_plan = PageAssetPlanner(Path(asset_manifest_payload.get("workspace", "."))).build(
            manuscript_text=manuscript_text,
            page_briefs=page_briefs,
            asset_manifest=asset_manifest_payload,
        )
        bindings_by_page = {
            int(item.get("page", 0) or 0): item
            for item in asset_plan.get("bindings", []) or []
            if int(item.get("page", 0) or 0) > 0
        }
        slides: list[dict[str, Any]] = []
        for brief in page_briefs:
            binding = bindings_by_page.get(brief.page_index, {})
            brief.bound_asset_kind = str(binding.get("bound_asset_kind", "none") or "none")
            brief.bound_asset_path = str(binding.get("bound_asset_path", "") or "")
            brief.visual_requirement = str(binding.get("visual_requirement", "none") or "none")
            brief.formula_snippet = str(binding.get("formula_snippet", "") or "")
            rec = self.recommend_layout(brief)
            surface_mode = self._layout_surface_mode(rec.selected_layout)
            selected_layout = self._layout_by_name(rec.selected_layout)
            density_hint = self._density_hint(selected_layout, brief)
            safe_surface_policy = self._safe_surface_policy(brief, selected_layout)
            slides.append(
                {
                    "page": brief.page_index,
                    "title": brief.title,
                    "page_purpose": brief.page_purpose,
                    "content_shape": brief.content_shape,
                    "selected_layout": rec.selected_layout,
                    "reference_archetype": getattr(selected_layout, "layout_archetype", "") or brief.content_shape,
                    "reference_use": "outline+layout+density+style",
                    "density_hint": density_hint,
                    "candidate_layouts": rec.alternatives,
                    "fit_score": round(rec.fit_score, 3),
                    "slot_fill_plan": rec.slot_assignment_preview,
                    "visual_requirement": brief.visual_requirement,
                    "bound_asset_kind": brief.bound_asset_kind,
                    "bound_asset_path": brief.bound_asset_path,
                    "surface_mode": surface_mode,
                    "safe_surface_policy": safe_surface_policy,
                    "style_reference": self._style_reference(selected_layout, brief),
                    "why_selected": rec.why_selected,
                }
            )
        return {
            "template_name": getattr(self.template_profile, "name", "") or "",
            "reference_mode": "layout_reference",
            "slides": slides,
            "page_asset_plan": asset_plan,
            "template_reference_profile": self.build_template_reference_profile(),
        }

    def build_template_reference_profile(self) -> dict[str, Any]:
        """Build a generation-facing reference profile without shell replay data."""
        style = semantic_style_system(self.template_profile)
        style_tokens = self._style_tokens(style, self.template_profile)
        layouts: list[dict[str, Any]] = []
        for layout in self.layouts:
            text_slots = [slot for slot in getattr(layout, "slots", []) or [] if slot.type == "text"]
            visual_slots = [
                slot
                for slot in getattr(layout, "slots", []) or []
                if slot.type in {"image", "chart", "table"}
            ]
            layouts.append(
                {
                    "layout_name": layout.name,
                    "page_type": layout.layout_archetype or "content",
                    "reference_archetype": layout.layout_archetype or "content",
                    "visual_pattern": layout.visual_pattern or "balanced",
                    "density_hint": self._density_hint(layout, PageBrief(0, "", "", "", "content", "title_body")),
                    "visual_asset_support": bool(getattr(layout, "supports_visual_asset", False) or visual_slots),
                    "slot_regions": [
                        {
                            "name": slot.name,
                            "role": slot.role,
                            "type": slot.type,
                            "geometry": slot.geometry,
                            "capacity_hint": slot.capacity_hint,
                        }
                        for slot in getattr(layout, "slots", []) or []
                    ],
                    "title_treatment": self._title_treatment(layout),
                    "body_capacity": self._body_capacity(text_slots),
                    "style_tokens": style_tokens,
                    "safe_surface_policy": self._safe_surface_policy(
                        PageBrief(0, "", "", "", "content", "title_body"),
                        layout,
                    ),
                    "do_not_copy_sample_content": True,
                }
            )
        return {
            "template_name": getattr(self.template_profile, "name", "") or "",
            "reference_mode": "layout_reference",
            "style_tokens": style_tokens,
            "layouts": layouts,
        }

    @staticmethod
    def strip_legacy_layout_mapping(manuscript_text: str) -> str:
        marker = "# === LAYOUT_MAPPING ==="
        text = manuscript_text or ""
        idx = text.find(marker)
        if idx >= 0:
            return text[:idx].rstrip()
        return text

    def recommend_layout(self, page_brief: PageBrief) -> LayoutRecommendation:
        scored: list[tuple[float, Any, dict[str, str], list[str]]] = []
        for layout in self.layouts:
            score, slot_preview, rejection = self._score_layout(layout, page_brief)
            scored.append((score, layout, slot_preview, rejection))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_layout, best_slots, best_rejections = scored[0]
        alternatives = [item[1].name for item in scored[1:4] if item[0] > 0]
        return LayoutRecommendation(
            selected_layout=best_layout.name,
            alternatives=alternatives,
            fit_score=best_score,
            slot_assignment_preview=best_slots,
            rejection_reasons=best_rejections,
            why_selected=self._build_selection_reason(best_layout, page_brief, best_score),
        )

    def dump_layout_mapping(self, mapping: dict[str, Any], output_path: Path) -> Path:
        output_path.write_text(
            yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return output_path

    def _extract_page_briefs(
        self,
        manuscript_text: str,
        *,
        asset_manifest: dict[str, Any],
        page_count_hint: int | None,
    ) -> list[PageBrief]:
        segments = [
            segment.strip()
            for segment in re.split(
                r"\n\s*---+\s*\n",
                self.strip_legacy_layout_mapping(manuscript_text),
            )
            if segment.strip()
        ]
        if not segments and manuscript_text.strip():
            segments = [manuscript_text.strip()]
        briefs: list[PageBrief] = []
        for idx, segment in enumerate(segments, start=1):
            title = self._extract_title(segment) or f"Slide {idx}"
            body = self._strip_heading_lines(segment)
            image_count = len(re.findall(r"!\[[^\]]*\]\([^)]+\)|<img\b", segment, flags=re.IGNORECASE))
            table_count = len(re.findall(r"(?m)^\|.*\|\s*$", segment))
            formula_count = len(re.findall(r"\$[^\$]{2,160}\$|\\\(|\\\)|softmax|QK\^?T|√d", segment, flags=re.IGNORECASE))
            asset_kinds = self._infer_asset_kinds(segment, asset_manifest)
            page_purpose = self._infer_page_purpose(idx, title, body, len(segments))
            content_shape = self._infer_content_shape(
                body=body,
                image_count=image_count,
                table_count=table_count,
                formula_count=formula_count,
                page_purpose=page_purpose,
            )
            briefs.append(
                PageBrief(
                    page_index=idx,
                    title=title,
                    body=body,
                    raw_markdown=segment,
                    page_purpose=page_purpose,
                    content_shape=content_shape,
                    asset_kinds=asset_kinds,
                    formula_count=formula_count,
                    table_count=table_count,
                    image_count=image_count,
                )
            )
        if page_count_hint and briefs and len(briefs) > page_count_hint:
            briefs = briefs[:page_count_hint]
        return briefs

    def _score_layout(
        self,
        layout: Any,
        page_brief: PageBrief,
    ) -> tuple[float, dict[str, str], list[str]]:
        signature = getattr(layout, "capability_signature", {}) or {}
        score = 0.15
        rejection: list[str] = []

        if page_brief.page_purpose == "opening":
            if layout.name == "opening" or signature.get("supports_ending") is False and signature.get("supports_title"):
                score += 0.45
            else:
                rejection.append("not an opening-style layout")
        elif page_brief.page_purpose == "ending":
            if layout.name == "ending" or signature.get("supports_ending"):
                score += 0.45
            else:
                rejection.append("not an ending-style layout")
        elif page_brief.page_purpose == "table_of_contents":
            if layout.name == "table of contents" or signature.get("supports_toc"):
                score += 0.5
            else:
                rejection.append("not a toc-style layout")
        elif page_brief.page_purpose == "section_divider":
            if signature.get("supports_section_divider") or layout.visual_pattern == "section_divider":
                score += 0.45
            else:
                rejection.append("not a section-divider layout")

        if page_brief.content_shape in {"title_body_figure", "figure_focus"}:
            if signature.get("supports_figure"):
                score += 0.36
            else:
                score -= 0.2
                rejection.append("missing figure slot")
        if page_brief.content_shape in {"table_focus"}:
            if signature.get("supports_table") or signature.get("supports_chart"):
                score += 0.36
            else:
                score -= 0.2
                rejection.append("missing table/chart slot")
        if page_brief.visual_requirement == "required":
            if page_brief.bound_asset_kind in {"figure"} and signature.get("supports_figure"):
                score += 0.75
            elif page_brief.bound_asset_kind in {"table", "chart"} and (
                signature.get("supports_table") or signature.get("supports_chart")
            ):
                score += 0.75
            elif page_brief.bound_asset_kind in {"table", "chart"} and signature.get("supports_figure"):
                # Many PPT templates expose generic image slots that can safely
                # carry a rendered table/chart image.
                score += 0.58
            elif page_brief.bound_asset_kind == "formula" and signature.get("supports_long_body"):
                score += 0.18
            else:
                score -= 0.6
                rejection.append("required visual/formula asset is unsupported by this layout")
        if page_brief.content_shape in {"two_column", "compare"}:
            if signature.get("supports_multi_column"):
                score += 0.22
            else:
                rejection.append("missing multi-column support")
        if page_brief.content_shape in {"title_body", "title_body_formula"}:
            if signature.get("supports_short_body") or signature.get("supports_long_body"):
                score += 0.18
            else:
                rejection.append("missing body text slot")
        if page_brief.formula_count > 0 and signature.get("supports_long_body"):
            score += 0.08
        if page_brief.page_purpose == "content" and getattr(layout, "layout_archetype", "") in {
            "title_content",
            "title_content_visual",
            "figure_explanation",
            "outline",
        }:
            score += 0.12
        if getattr(layout, "visual_pattern", "") == "vertical_title" and page_brief.content_shape == "title_body":
            score += 0.06

        slot_preview = self._build_slot_preview(layout, page_brief)
        return score, slot_preview, rejection

    def _build_slot_preview(self, layout: Any, page_brief: PageBrief) -> dict[str, str]:
        preview: dict[str, str] = {}
        for slot in getattr(layout, "slots", []) or []:
            role = str(getattr(slot, "role", "") or "")
            if role == "title" and "title" not in preview:
                preview[slot.name] = "page_title"
            elif role in {"subtitle", "section_label"} and "title" not in preview.values():
                preview[slot.name] = "page_title_support"
            elif role == "body" and "body_main" not in preview.values():
                preview[slot.name] = "body_main"
            elif role == "body":
                preview[slot.name] = "body_secondary"
            elif role in {"image", "chart", "table"}:
                if page_brief.bound_asset_kind in {"table", "chart"} and role in {"table", "chart", "image"}:
                    preview[slot.name] = "table_or_chart_asset"
                elif page_brief.bound_asset_kind == "formula" and role == "image":
                    preview[slot.name] = "formula_visualization_asset"
                else:
                    preview[slot.name] = "figure_asset"
            elif role == "caption":
                preview[slot.name] = "caption_or_formula"
        return preview

    def _build_selection_reason(self, layout: Any, page_brief: PageBrief, score: float) -> str:
        return (
            f"Matched page purpose `{page_brief.page_purpose}` and content shape "
            f"`{page_brief.content_shape}` to layout archetype "
            f"`{getattr(layout, 'layout_archetype', 'content')}` "
            f"with pattern `{getattr(layout, 'visual_pattern', 'balanced')}` "
            f"(score={score:.2f})."
        )

    @staticmethod
    def _extract_title(segment: str) -> str:
        for line in (segment or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return re.sub(r"^#+\s*", "", stripped).strip()
            if stripped and not stripped.startswith(("-", "*", "+")) and not re.match(r"^\d+[.)]\s", stripped):
                return stripped
        return ""

    @staticmethod
    def _strip_heading_lines(segment: str) -> str:
        lines = []
        for line in (segment or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    @staticmethod
    def _infer_asset_kinds(segment: str, asset_manifest: dict[str, Any]) -> list[str]:
        kinds: list[str] = []
        segment_lower = segment.lower()
        for asset in asset_manifest.get("assets", []) or []:
            path = str(asset.get("path", "") or "")
            if path and Path(path).name.lower() in segment_lower:
                kind = str(asset.get("kind", "") or "")
                if kind and kind not in kinds:
                    kinds.append(kind)
        return kinds

    @staticmethod
    def _infer_page_purpose(page_index: int, title: str, body: str, total_pages: int) -> str:
        title_lower = title.lower()
        body_lower = body.lower()
        if page_index == 1:
            return "opening"
        if page_index == total_pages and any(
            token in f"{title_lower}\n{body_lower}"
            for token in ("thanks", "thank you", "q&a", "questions", "结束", "总结", "致谢")
        ):
            return "ending"
        if any(token in title_lower for token in ("contents", "目录", "agenda", "outline")):
            return "table_of_contents"
        if any(token in title_lower for token in ("section", "part", "研究背景", "方法概述")) and len(body_lower.strip()) < 80:
            return "section_divider"
        return "content"

    @staticmethod
    def _infer_content_shape(
        *,
        body: str,
        image_count: int,
        table_count: int,
        formula_count: int,
        page_purpose: str,
    ) -> str:
        bullet_lines = len(
            [
                line for line in body.splitlines()
                if line.strip().startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s", line.strip())
            ]
        )
        if page_purpose in {"opening", "ending", "table_of_contents", "section_divider"}:
            return page_purpose
        if table_count > 0:
            return "table_focus"
        if image_count > 0 and bullet_lines >= 2:
            return "title_body_figure"
        if image_count > 0 and bullet_lines <= 1:
            return "figure_focus"
        if formula_count > 0 and bullet_lines >= 2:
            return "title_body_formula"
        if formula_count > 0:
            return "title_body_formula"
        if bullet_lines >= 5:
            return "two_column"
        return "title_body"

    def _layout_surface_mode(self, layout_name: str) -> str:
        for layout in self.layouts:
            if layout.name == layout_name:
                mode = getattr(layout, "surface_mode", "") or "light_surface"
                return "light_surface" if mode == "shell_text" else mode
        return "light_surface"

    @staticmethod
    def _surface_policy_for_mode(surface_mode: str) -> str:
        return "readable_surface_required"

    def _layout_by_name(self, layout_name: str) -> Any | None:
        for layout in self.layouts:
            if layout.name == layout_name:
                return layout
        return None

    @staticmethod
    def _density_hint(layout: Any | None, page_brief: PageBrief) -> str:
        if page_brief.page_purpose in {"opening", "ending", "section_divider"}:
            return "low"
        text_slots = [
            slot for slot in (getattr(layout, "slots", []) or []) if getattr(slot, "type", "") == "text"
        ]
        body_len = len(page_brief.body or "")
        body_slots = [
            slot for slot in text_slots if getattr(slot, "role", "") in {"body", "caption", "subtitle"}
        ]
        if body_len > 520 or len(body_slots) >= 3:
            return "high"
        if body_len < 180 and len(body_slots) <= 1:
            return "low"
        return "medium"

    @staticmethod
    def _safe_surface_policy(page_brief: PageBrief, layout: Any | None) -> str:
        if page_brief.visual_requirement == "required":
            return "light_visual_panel_required"
        if page_brief.page_purpose in {"opening", "section_divider"}:
            return "readable_title_surface"
        if page_brief.page_purpose == "table_of_contents":
            return "readable_toc_surface"
        return "readable_surface_required"

    @staticmethod
    def _style_tokens(style_system: dict[str, Any], profile: Any | None = None) -> dict[str, Any]:
        colors = style_system.get("colors", {}) if isinstance(style_system, dict) else {}
        typography = style_system.get("typography", {}) if isinstance(style_system, dict) else {}
        typography_defaults = (
            style_system.get("typography_defaults", {}) if isinstance(style_system, dict) else {}
        )
        design_constraints = getattr(profile, "design_constraints", None) if profile is not None else None
        color_palette = getattr(design_constraints, "color_palette", None) if design_constraints is not None else None
        design_typography = getattr(design_constraints, "typography", None) if design_constraints is not None else None
        title_defaults = typography_defaults.get("title", {}) if isinstance(typography_defaults, dict) else {}
        body_defaults = typography_defaults.get("body", {}) if isinstance(typography_defaults, dict) else {}
        caption_defaults = {}
        if isinstance(typography_defaults, dict):
            caption_defaults = (
                typography_defaults.get("caption")
                or typography_defaults.get("toc_item")
                or typography_defaults.get("label")
                or {}
            )

        def pick(*values: Any) -> Any:
            for value in values:
                if value is None:
                    continue
                if isinstance(value, str):
                    stripped = value.strip()
                    if stripped:
                        return stripped
                    continue
                if isinstance(value, (int, float)):
                    if value:
                        return value
                    continue
                if value:
                    return value
            return ""

        return {
            "colors": {
                key: value
                for key, value in {
                    "background": pick(
                        getattr(color_palette, "background", ""),
                        colors.get("background"),
                    ),
                    "surface": pick(
                        getattr(color_palette, "surface", ""),
                        colors.get("surface"),
                    ),
                    "primary_text": pick(
                        getattr(color_palette, "primary_text", ""),
                        getattr(color_palette, "text", ""),
                        colors.get("primary_text"),
                        colors.get("text"),
                    ),
                    "inverse_text": pick(
                        getattr(color_palette, "inverse_text", ""),
                        colors.get("inverse_text"),
                    ),
                    "muted_text": pick(
                        getattr(color_palette, "muted_text", ""),
                        colors.get("muted_text"),
                    ),
                    "accent": pick(
                        getattr(color_palette, "accent", ""),
                        getattr(color_palette, "secondary", ""),
                        colors.get("accent"),
                        colors.get("primary"),
                    ),
                    "border": pick(
                        getattr(color_palette, "border", ""),
                        colors.get("border"),
                    ),
                    "primary": pick(
                        getattr(color_palette, "primary", ""),
                        colors.get("primary"),
                    ),
                    "secondary": pick(
                        getattr(color_palette, "secondary", ""),
                        colors.get("secondary"),
                    ),
                }.items()
                if value
            },
            "typography": {
                key: value
                for key, value in {
                    "title_font": pick(
                        getattr(design_typography, "display_title_font", ""),
                        getattr(design_typography, "title_font", ""),
                        typography.get("display_title_font"),
                        typography.get("title_font"),
                        title_defaults.get("font_name"),
                    ),
                    "title_size": pick(
                        getattr(design_typography, "title_size", 0),
                        typography.get("title_size"),
                        title_defaults.get("font_size"),
                    ),
                    "body_font": pick(
                        getattr(design_typography, "body_font", ""),
                        typography.get("body_font"),
                        body_defaults.get("font_name"),
                    ),
                    "body_size": pick(
                        getattr(design_typography, "body_size", 0),
                        typography.get("body_size"),
                        body_defaults.get("font_size"),
                    ),
                    "caption_font": pick(
                        getattr(design_typography, "caption_font", ""),
                        typography.get("caption_font"),
                        caption_defaults.get("font_name"),
                    ),
                    "caption_size": pick(
                        getattr(design_typography, "caption_size", 0),
                        typography.get("caption_size"),
                        caption_defaults.get("font_size"),
                    ),
                    "latin_fallback_font": pick(
                        getattr(design_typography, "latin_fallback_font", ""),
                        typography.get("latin_fallback_font"),
                    ),
                }.items()
                if value
            },
        }

    @staticmethod
    def _title_treatment(layout: Any | None) -> str:
        pattern = str(getattr(layout, "visual_pattern", "") or "")
        archetype = str(getattr(layout, "layout_archetype", "") or "")
        if "hero" in pattern or "title_page" in archetype:
            return "large readable title, short subtitle, template-inspired accent geometry"
        if "vertical" in pattern:
            return "top-left title with a short accent rule and restrained spacing"
        return "clear title hierarchy with template-inspired accent color"

    @staticmethod
    def _body_capacity(text_slots: list[Any]) -> str:
        if len(text_slots) >= 4:
            return "high"
        if len(text_slots) <= 1:
            return "low"
        return "medium"

    def _style_reference(self, layout: Any | None, page_brief: PageBrief) -> dict[str, Any]:
        tokens = self._style_tokens(semantic_style_system(self.template_profile), self.template_profile)
        palette_roles = tokens.get("colors", {}) if isinstance(tokens, dict) else {}
        typography_roles = tokens.get("typography", {}) if isinstance(tokens, dict) else {}
        return {
            "title_treatment": self._title_treatment(layout),
            "safe_surface_policy": self._safe_surface_policy(page_brief, layout),
            "visual_pattern": str(getattr(layout, "visual_pattern", "") or page_brief.content_shape),
            "palette_roles": {
                key: palette_roles[key]
                for key in ("background", "surface", "primary_text", "inverse_text", "accent", "border")
                if palette_roles.get(key)
            },
            "typography_roles": {
                key: typography_roles[key]
                for key in ("title_font", "title_size", "body_font", "body_size", "caption_font")
                if typography_roles.get(key)
            },
            "do_not_copy_sample_content": True,
        }


def load_layout_mapping(path: Path | str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def dump_layout_mapping_json(mapping: dict[str, Any]) -> str:
    return json.dumps(mapping, ensure_ascii=False, indent=2)
