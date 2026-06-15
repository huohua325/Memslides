"""TemplateProfile 数据模型 (Stage 5)

定义模板档案相关的数据结构：
- DesignConstraints: 设计约束
- ContentPatterns: 内容模式
- TemplateProfile: 综合模板档案

布局信息直接由 slide_induction 承载，
不再使用冗余的 LayoutPattern / ContentMapping 中间层。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ═══════════════════════════════════════════════
# 样式相关模型
# ═══════════════════════════════════════════════

@dataclass
class ColorPalette:
    """配色方案"""
    primary: str = ""  # 主色
    secondary: str = ""  # 辅助色
    accent: str = ""  # 强调色
    background: str = ""  # 背景色
    text: str = ""  # 文字色
    surface: str = ""  # 内容承载面/卡片面
    primary_text: str = ""  # 浅背景主文本
    inverse_text: str = ""  # 深背景反白文本
    muted_text: str = ""  # 次级文本
    border: str = ""  # 线条/边框
    additional: list[str] = field(default_factory=list)  # 其他颜色

    def to_dict(self) -> dict:
        return {
            "primary": self.primary,
            "secondary": self.secondary,
            "accent": self.accent,
            "background": self.background,
            "text": self.text,
            "surface": self.surface,
            "primary_text": self.primary_text,
            "inverse_text": self.inverse_text,
            "muted_text": self.muted_text,
            "border": self.border,
            "additional": self.additional,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ColorPalette":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Typography:
    """字体排版"""
    title_font: str = ""
    title_size: int = 0
    body_font: str = ""
    body_size: int = 0
    display_title_font: str = ""
    caption_font: str = ""
    caption_size: int = 0
    latin_fallback_font: str = ""
    line_height: float = 1.5

    def to_dict(self) -> dict:
        return {
            "title_font": self.title_font,
            "title_size": self.title_size,
            "body_font": self.body_font,
            "body_size": self.body_size,
            "display_title_font": self.display_title_font,
            "caption_font": self.caption_font,
            "caption_size": self.caption_size,
            "latin_fallback_font": self.latin_fallback_font,
            "line_height": self.line_height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Typography":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ChartStyle:
    """图表风格"""
    default_type: str = ""  # bar, line, pie, etc.
    color_scheme: list[str] = field(default_factory=list)
    show_legend: bool = True
    show_grid: bool = True

    def to_dict(self) -> dict:
        return {
            "default_type": self.default_type,
            "color_scheme": self.color_scheme,
            "show_legend": self.show_legend,
            "show_grid": self.show_grid,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChartStyle":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DesignConstraints:
    """设计约束"""
    color_palette: ColorPalette = field(default_factory=ColorPalette)
    typography: Typography = field(default_factory=Typography)
    spacing: dict[str, Any] = field(default_factory=dict)  # margin, padding
    chart_style: ChartStyle = field(default_factory=ChartStyle)

    def to_dict(self) -> dict:
        return {
            "color_palette": self.color_palette.to_dict(),
            "typography": self.typography.to_dict(),
            "spacing": self.spacing,
            "chart_style": self.chart_style.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DesignConstraints":
        return cls(
            color_palette=ColorPalette.from_dict(data.get("color_palette", {})),
            typography=Typography.from_dict(data.get("typography", {})),
            spacing=data.get("spacing", {}),
            chart_style=ChartStyle.from_dict(data.get("chart_style", {})),
        )


# ═══════════════════════════════════════════════
# 内容模式
# ═══════════════════════════════════════════════

@dataclass
class ContentPatterns:
    """内容模式"""
    info_density: str = "medium"  # low, medium, high
    narrative_style: str = ""  # business, academic, casual
    typical_sections: list[str] = field(default_factory=list)  # opening, content, ending
    bullet_style: str = ""  # numbered, bulleted, icon
    max_bullets_per_slide: int = 5

    def to_dict(self) -> dict:
        return {
            "info_density": self.info_density,
            "narrative_style": self.narrative_style,
            "typical_sections": self.typical_sections,
            "bullet_style": self.bullet_style,
            "max_bullets_per_slide": self.max_bullets_per_slide,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentPatterns":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CanonicalSlot:
    """Normalized editable slot or sample content region."""

    name: str = ""
    type: str = "text"  # text | image | chart | table
    role: str = ""  # title | subtitle | body | caption | toc_item | number_badge | ...
    geometry: dict[str, float] = field(default_factory=dict)
    capacity_hint: dict[str, Any] = field(default_factory=dict)
    sample_values: list[str] = field(default_factory=list)
    font_name: str = ""
    font_size: int = 0
    source: str = ""  # layout_capability | sample_content
    placeholder_type: str = ""
    placeholder_idx: int | None = None
    shape_name: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "role": self.role,
            "geometry": self.geometry,
            "capacity_hint": self.capacity_hint,
            "sample_values": self.sample_values,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "source": self.source,
            "placeholder_type": self.placeholder_type,
            "placeholder_idx": self.placeholder_idx,
            "shape_name": self.shape_name,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalSlot":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CanonicalLayout:
    """Canonical template layout consumable by generation-time tools."""

    id: str = ""
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    source_layout_name: str = ""
    layout_archetype: str = ""
    visual_pattern: str = ""
    slot_count: int = 0
    slot_roles: list[str] = field(default_factory=list)
    slots: list[CanonicalSlot] = field(default_factory=list)
    sample_content: list[CanonicalSlot] = field(default_factory=list)
    sample_slide_indices: list[int] = field(default_factory=list)
    style_notes: list[str] = field(default_factory=list)
    capability_signature: dict[str, Any] = field(default_factory=dict)
    surface_mode: str = ""  # shell_text | light_surface | figure_panel
    shell_id: str = ""
    shell_mode: str = ""  # native_background | layered_shell | raster_shell | no_shell
    protected_regions: list[dict[str, Any]] = field(default_factory=list)
    surface_policy: str = ""
    allowed_text_roles: list[str] = field(default_factory=list)
    allowed_visual_roles: list[str] = field(default_factory=list)
    supports_visual_asset: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "source_layout_name": self.source_layout_name,
            "layout_archetype": self.layout_archetype,
            "visual_pattern": self.visual_pattern,
            "slot_count": self.slot_count,
            "slot_roles": self.slot_roles,
            "slots": [slot.to_dict() for slot in self.slots],
            "sample_content": [slot.to_dict() for slot in self.sample_content],
            "sample_slide_indices": self.sample_slide_indices,
            "style_notes": self.style_notes,
            "capability_signature": self.capability_signature,
            "surface_mode": self.surface_mode,
            "shell_id": self.shell_id,
            "shell_mode": self.shell_mode,
            "protected_regions": self.protected_regions,
            "surface_policy": self.surface_policy,
            "allowed_text_roles": self.allowed_text_roles,
            "allowed_visual_roles": self.allowed_visual_roles,
            "supports_visual_asset": self.supports_visual_asset,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CanonicalLayout":
        payload = dict(data)
        payload["slots"] = [
            slot if isinstance(slot, CanonicalSlot) else CanonicalSlot.from_dict(slot)
            for slot in payload.get("slots", []) or []
        ]
        payload["sample_content"] = [
            slot if isinstance(slot, CanonicalSlot) else CanonicalSlot.from_dict(slot)
            for slot in payload.get("sample_content", []) or []
        ]
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass
class TemplateSemanticModel:
    """Normalized semantic template representation for active runtime use."""

    canonical_layouts: list[CanonicalLayout] = field(default_factory=list)
    layout_capability: dict[str, Any] = field(default_factory=dict)
    template_shell: dict[str, Any] = field(default_factory=dict)
    sample_content: dict[str, Any] = field(default_factory=dict)
    decorative_system: dict[str, Any] = field(default_factory=dict)
    style_system: dict[str, Any] = field(default_factory=dict)
    suspicious_ambiguities: list[str] = field(default_factory=list)
    low_confidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_layouts": [layout.to_dict() for layout in self.canonical_layouts],
            "layout_capability": self.layout_capability,
            "template_shell": self.template_shell,
            "sample_content": self.sample_content,
            "decorative_system": self.decorative_system,
            "style_system": self.style_system,
            "suspicious_ambiguities": self.suspicious_ambiguities,
            "low_confidence": self.low_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateSemanticModel":
        payload = dict(data)
        payload["canonical_layouts"] = [
            layout if isinstance(layout, CanonicalLayout) else CanonicalLayout.from_dict(layout)
            for layout in payload.get("canonical_layouts", []) or []
        ]
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════
# 综合模板技能
# ═══════════════════════════════════════════════

@dataclass
class TemplateProfile:
    """模板档案 — 综合所有提取的模板知识"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""

    # 来源
    template_source: str = ""  # 原始 PPT 文件路径
    slide_count: int = 0
    aspect_ratio: str = "16:9"

    # slide_induction 原始布局数据（新增）
    slide_induction: dict[str, Any] = field(default_factory=dict)  # 完整的 slide_induction.json
    semantic_model: TemplateSemanticModel = field(default_factory=TemplateSemanticModel)
    template_dir: str = ""  # 模板存储目录（含 source.pptx, images/, slide_images/）
    image_stats: dict[str, Any] = field(default_factory=dict)  # image_stats.json

    # 提取的技能（memslides 增量）
    design_constraints: DesignConstraints = field(default_factory=DesignConstraints)
    content_patterns: ContentPatterns = field(default_factory=ContentPatterns)

    # 元数据
    confidence: float = 0.8
    shape_geometry_path: str = ""  # Task 12: 指向 shape_geometry.json 的路径
    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "template_source": self.template_source,
            "slide_count": self.slide_count,
            "aspect_ratio": self.aspect_ratio,
            # slide_induction 原始布局数据
            "slide_induction": self.slide_induction,
            "semantic_model": self.semantic_model.to_dict(),
            "template_dir": self.template_dir,
            "image_stats": self.image_stats,
            # memslides 增量
            "design_constraints": self.design_constraints.to_dict(),
            "content_patterns": self.content_patterns.to_dict(),
            "confidence": self.confidence,
            "shape_geometry_path": self.shape_geometry_path,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateProfile":
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            template_source=data.get("template_source", ""),
            slide_count=data.get("slide_count", 0),
            aspect_ratio=data.get("aspect_ratio", "16:9"),
            # slide_induction 原始布局数据
            slide_induction=data.get("slide_induction", {}),
            semantic_model=TemplateSemanticModel.from_dict(
                data.get("semantic_model", {})
            ),
            template_dir=data.get("template_dir", ""),
            image_stats=data.get("image_stats", {}),
            # memslides 增量
            design_constraints=DesignConstraints.from_dict(
                data.get("design_constraints", {})
            ),
            content_patterns=ContentPatterns.from_dict(
                data.get("content_patterns", {})
            ),
            confidence=data.get("confidence", 0.8),
            shape_geometry_path=data.get("shape_geometry_path", ""),
            user_id=data.get("user_id", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            status=data.get("status", "active"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TemplateProfile":
        return cls.from_dict(json.loads(json_str))
