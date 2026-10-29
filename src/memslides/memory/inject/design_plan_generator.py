"""
DesignPlanGenerator — 确定性 design_plan.md 生成器

从 TemplateProfile 数据生成 design_plan.md，无 LLM，纯代码拼装。
此模块是模板驱动 PPT 生成功能的一部分，与 TemplateGuideBuilder 配合使用。

设计原则：
- 按数据可用性动态生成章节（空字段跳过）
- 生成可直接复制的 CSS 变量块
- 从 slide_induction 提取每个布局的元素规格
- 所有内容来自 TemplateProfile 数据，零 LLM 创意
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from memslides.templates.quality import (
    ADAPTIVE_STRUCTURAL_TEMPLATE,
    STRICT_STRUCTURAL_TEMPLATE,
    STRUCTURAL_TEMPLATE,
    STYLE_REFERENCE,
    effective_template_palette,
)
from memslides.templates.semantic_access import (
    canonical_layout_views,
    semantic_decorative_system,
    semantic_style_system,
)

if TYPE_CHECKING:
    from ..core.template_models import (
        ColorPalette, ContentPatterns,
        DesignConstraints, TemplateProfile, Typography,
    )

logger = logging.getLogger(__name__)


def _classify_layout_type(
    key: str,
    elements: list[dict],
    functional_keys: list[str],
) -> str:
    """统一的布局类型分类逻辑

    Args:
        key: 布局名称
        elements: 布局元素列表
        functional_keys: 功能页键列表

    Returns:
        布局类型标签: "功能页" | "图文" | "纯文本"
    """
    if key in functional_keys or key.lower() in [fk.lower() for fk in functional_keys]:
        return "功能页"
    elif key.endswith(":image"):
        return "图文"
    elif key.endswith(":text"):
        return "纯文本"
    else:
        img_n = sum(1 for e in elements if e.get("type") == "image")
        return "图文" if img_n > 0 else "纯文本"


class DesignPlanGenerator:
    """确定性 design_plan.md 生成器"""

    def __init__(self, profile: "TemplateProfile", mode: str = "structural_template"):
        self.profile = profile
        self.mode = mode or "structural_template"

    @property
    def is_style_reference(self) -> bool:
        return self.mode == STYLE_REFERENCE

    @property
    def is_adaptive_structural(self) -> bool:
        return self.mode == ADAPTIVE_STRUCTURAL_TEMPLATE

    @property
    def is_strict_structural(self) -> bool:
        return self.mode in {STRUCTURAL_TEMPLATE, STRICT_STRUCTURAL_TEMPLATE}

    def generate(self, output_path: Path) -> Path:
        """生成 design_plan.md

        Args:
            output_path: 输出文件路径

        Returns:
            写入的文件路径
        """
        from ..core.template_models import (
            ColorPalette, ContentPatterns,
            DesignConstraints, Typography,
        )

        dc = self.profile.design_constraints or DesignConstraints()
        cp = effective_template_palette(self.profile, mode=self.mode)
        tp = dc.typography or Typography()
        spacing = dc.spacing or {}
        cp_patterns = self.profile.content_patterns or ContentPatterns()
        semantic_style = semantic_style_system(self.profile)

        # 收集各章节，空章节不输出
        section_builders = [
            ("header",      lambda: self._header()),
            ("css_vars",    lambda: self._css_variables(cp, tp, spacing)),
            ("colors",      lambda: self._color_table(cp)),
            ("typography",  lambda: self._typography(tp)),
            ("spacing",     lambda: self._spacing(spacing)),
            ("content",     lambda: self._content_style(cp_patterns)),
            ("reference",   lambda: self._template_reference_contract()),
            ("layouts",     lambda: self._layouts()),
            ("decorative",  lambda: self._decorative()),
            ("style_contract", lambda: self._style_contract(semantic_style, cp, tp)),
            ("rules",       lambda: self._rules(cp, tp)),
        ]

        all_sections: list[str] = []
        included_sections: list[str] = []
        for name, builder_fn in section_builders:
            text = builder_fn()
            if text:
                all_sections.append(text)
                included_sections.append(name)

        content = "\n\n".join(all_sections) + "\n"

        # 写入文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"Generated design_plan.md from template ({len(content)} chars, "
                     f"sections={included_sections}) -> {output_path}")
        return output_path

    # ── 各章节构建器 ──

    def _header(self) -> str:
        """标题 + 元信息"""
        lines = [
            f"# Design Plan — {self.profile.name}",
            "",
            (
                "> 本文件由模板自动生成。当前模板仅作为视觉参考，页面结构必须由稿件内容驱动。"
                if self.is_style_reference
                else "> 本文件由模板自动生成。当前模板结构可用，但内容页必须优先保证可读性，可使用自适应内容承载面。"
                if self.is_adaptive_structural
                else "> 本文件由模板自动生成，**禁止修改**。所有设计决策必须严格遵循以下规范。"
            ),
            (
                "> Auto-generated from template as style reference. Preserve readability and content assets."
                if self.is_style_reference
                else "> Auto-generated from template in adaptive structural mode. Use template structure as reference, but render content on safe readable surfaces."
                if self.is_adaptive_structural
                else "> Auto-generated from template. **Do NOT modify**. Follow these specs exactly."
            ),
            "",
            f"- 模板: {self.profile.name}",
            f"- 比例: {self.profile.aspect_ratio}",
            f"- 页数: {self.profile.slide_count}",
            f"- 模式: {self.mode}",
        ]
        return "\n".join(lines)

    def _css_variables(self, cp: "ColorPalette", tp: "Typography", spacing: dict) -> str:
        """生成可直接复制到 <style> 的 CSS 变量块"""
        has_colors = any([
            cp.primary,
            cp.secondary,
            cp.accent,
            cp.text,
            cp.background,
            cp.surface,
            cp.primary_text,
            cp.inverse_text,
            cp.muted_text,
            cp.border,
        ])
        has_fonts = any([tp.title_font, tp.body_font])
        if not has_colors and not has_fonts:
            return ""

        lines = [
            "## CSS 变量（直接复制到每页 `<style>` 中使用）",
            "",
            "```css",
            ":root {",
        ]
        if cp.primary:
            lines.append(f"  --color-primary: {cp.primary};")
        if cp.secondary:
            lines.append(f"  --color-secondary: {cp.secondary};")
        if cp.accent:
            lines.append(f"  --color-accent: {cp.accent};")
        if cp.text:
            lines.append(f"  --color-text: {cp.text};")
        if cp.background:
            lines.append(f"  --color-bg: {cp.background};")
        if cp.surface:
            lines.append(f"  --color-surface: {cp.surface};")
        if cp.primary_text:
            lines.append(f"  --color-primary-text: {cp.primary_text};")
        if cp.inverse_text:
            lines.append(f"  --color-inverse-text: {cp.inverse_text};")
        if cp.muted_text:
            lines.append(f"  --color-muted-text: {cp.muted_text};")
        if cp.border:
            lines.append(f"  --color-border: {cp.border};")
        additional = getattr(cp, 'additional', None) or []
        for i, c in enumerate(additional[:4]):
            lines.append(f"  --color-aux-{i+1}: {c};")
        if tp.title_font:
            lines.append(f"  --font-title: '{tp.title_font}', 'Noto Sans CJK SC', sans-serif;")
            if tp.title_size:
                lines.append(f"  --size-title: {tp.title_size}px;")
        if tp.body_font:
            lines.append(f"  --font-body: '{tp.body_font}', 'Noto Sans CJK SC', sans-serif;")
            if tp.body_size and tp.body_size > 0:
                lines.append(f"  --size-body: {tp.body_size}px;")
        if spacing.get("margin"):
            lines.append(f"  --spacing-margin: {spacing['margin']};")
        if spacing.get("padding"):
            lines.append(f"  --spacing-padding: {spacing['padding']};")
        lines.append("}")
        lines.append("```")
        lines.append("")
        if self.is_style_reference:
            lines.append("**参考原则**：这些变量是模板观察值，不是硬限制；优先保证正文、公式、图片和表格清晰可读。")
        elif self.is_adaptive_structural:
            lines.append("**自适应原则**：参考模板颜色与标题节奏，正文/公式/表格必须落在安全 `surface + primary_text` 或等价高对比组合上。")
        else:
            lines.append("**⚠️ 使用模板语义颜色角色，不要直接把深色背景色用于正文承载面。**")
        return "\n".join(lines)

    def _color_table(self, cp: "ColorPalette") -> str:
        """配色方案表格（仅在有颜色数据时生成）"""
        rows = []
        if cp.primary:
            rows.append(f"| 标题/重点 primary | `{cp.primary}` | h1, h2, 强调文字 |")
        if cp.secondary:
            rows.append(f"| 次要 secondary | `{cp.secondary}` | 副标题, 辅助装饰 |")
        if cp.accent:
            rows.append(f"| 高亮 accent | `{cp.accent}` | 图标, 徽标, 高亮 |")
        if cp.text:
            rows.append(f"| 兼容 text | `{cp.text}` | 旧字段兼容；新链路优先使用 primary_text/inverse_text |")
        if cp.background:
            rows.append(f"| 页面参考 background | `{cp.background}` | 页面底色参考、标题带、装饰底色 |")
        if cp.surface:
            rows.append(f"| 内容承载 surface | `{cp.surface}` | 正文卡片、图表底、公式面板 |")
        if cp.primary_text:
            rows.append(f"| 浅面正文 primary_text | `{cp.primary_text}` | surface 上的正文、公式、caption |")
        if cp.inverse_text:
            rows.append(f"| 反白 inverse_text | `{cp.inverse_text}` | 深色标题带或强调区上的标题/短标签 |")
        if cp.muted_text:
            rows.append(f"| 次级文本 muted_text | `{cp.muted_text}` | 注释、图注、辅助说明 |")
        if cp.border:
            rows.append(f"| 线条 border | `{cp.border}` | 分隔线、卡片边框、表格线 |")
        additional = getattr(cp, 'additional', None) or []
        for i, c in enumerate(additional[:6]):
            rows.append(f"| 辅助色 {i+1} | `{c}` | 图表, 装饰 |")

        if not rows:
            return ""

        lines = [
            "## 配色方案",
            "",
            "| 用途 | 颜色值 | 用于 |",
            "|------|--------|------|",
        ] + rows
        return "\n".join(lines)

    def _typography(self, tp: "Typography") -> str:
        """字体规范（仅在有字体数据时生成）"""
        if not tp.title_font and not tp.body_font:
            return ""
        lines = ["## 字体规范", ""]
        if tp.title_font:
            lines.append(f"- **标题**: `{tp.title_font}` {tp.title_size}pt")
            if tp.title_size:
                lines.append(f"- **副标题**: {int(tp.title_size * 0.7)}pt")
        if tp.body_font:
            if tp.body_size and tp.body_size > 0:
                lines.append(f"- **正文**: `{tp.body_font}` {tp.body_size}pt")
            else:
                lines.append(f"- **正文**: `{tp.body_font}`，字号由内容可读性决定（不得使用 0pt）")
            if tp.body_size:
                lines.append(f"- **注释/标签**: {int(tp.body_size * 0.75)}pt")
        if tp.line_height and tp.line_height != 1.5:
            lines.append(f"- **行高**: {tp.line_height}")
        return "\n".join(lines)

    def _spacing(self, spacing: dict) -> str:
        """间距规范（仅在有间距数据时生成）"""
        items = []
        if spacing.get("margin"):
            items.append(f"- **页边距**: {spacing['margin']}")
        if spacing.get("padding"):
            items.append(f"- **内边距**: {spacing['padding']}")
        if spacing.get("element_gap"):
            items.append(f"- **元素间距**: {spacing['element_gap']}")
        if not items:
            return ""
        return "\n".join(["## 间距规范", ""] + items)

    def _content_style(self, cp: "ContentPatterns") -> str:
        """内容风格（仅在有非默认数据时生成）"""
        items = []
        density_map = {
            "low": "精简 — 每页 2-3 要点",
            "medium": "平衡 — 每页 4-5 要点",
            "high": "密集 — 每页 6+ 要点",
        }
        if cp.info_density and cp.info_density != "medium":
            items.append(f"- **密度**: {density_map.get(cp.info_density, cp.info_density)}")
        style_map = {
            "academic": "学术", "business": "商务",
            "creative": "创意", "educational": "教育", "technical": "技术",
        }
        if cp.narrative_style:
            items.append(f"- **风格**: {style_map.get(cp.narrative_style, cp.narrative_style)}")
        bullet_map = {"bulleted": "圆点", "numbered": "数字", "icon": "图标"}
        if cp.bullet_style:
            items.append(f"- **列表**: {bullet_map.get(cp.bullet_style, cp.bullet_style)}")
        if cp.max_bullets_per_slide and cp.max_bullets_per_slide != 5:
            items.append(f"- **每页最多要点**: {cp.max_bullets_per_slide}")
        if not items:
            return ""
        return "\n".join(["## 内容风格", ""] + items)

    def _layouts(self) -> str:
        """布局 contract：只输出 canonical layout 语义"""
        layouts = canonical_layout_views(self.profile)
        if not layouts:
            return ""
        lines = [
            "## Layout Contract",
            "",
            (
                "以下 canonical layouts 仅作结构参考；页面内容和层次必须以稿件为先。若系统另行提供 `layout_mapping.yaml`，优先执行其中的逐页布局选择。"
                if self.is_style_reference
                else "若系统提供 `layout_mapping.yaml`，先按其中的 `selected_layout` 执行；生成每页前 **必须** 调用 `query_slide_layout(layout_name)` 获取 canonical slot contract。"
            ),
            "",
        ]

        for idx, layout in enumerate(layouts, 1):
            text_n = sum(1 for slot in layout.slots if slot.type == "text")
            img_n = sum(1 for slot in layout.slots if slot.type in {"image", "chart", "table"})

            display = layout.name if len(layout.name) <= 50 else layout.name[:47] + "..."
            lines.append(
                f"### {idx}. `{display}` [{layout.layout_archetype or 'content'} / {layout.visual_pattern or 'balanced'}]"
            )
            lines.append("")

            if layout.slots:
                lines.append("| 槽位 | 角色 | 类型 | 建议容量 |")
                lines.append("|------|------|------|----------|")
                for slot in layout.slots:
                    hint = slot.capacity_hint or {}
                    char_hint = hint.get("suggested_max_chars") or hint.get("sample_char_count") or "—"
                    lines.append(
                        f"| {slot.name or '?'} | {slot.role or 'slot'} | {slot.type or 'text'} | {char_hint} |"
                    )
                lines.append("")
            if self.is_style_reference:
                lines.append(f"观察到的元素: {text_n}文 / {img_n}图 — 仅供参考，可按稿件增减。")
            elif self.is_adaptive_structural:
                lines.append(
                    f"结构锚点: {text_n}文 / {img_n}图。优先保留主要区域、密度和图文比例，但可增加安全 surface 以承载真实正文/图表。"
                )
            else:
                lines.append(f"结构锚点: {text_n}文 / {img_n}图 — 保持主要 layout 意图，避免破坏模板节奏。")
            if getattr(layout, "surface_mode", ""):
                lines.append(f"- Surface mode: `{layout.surface_mode}`")
            if getattr(layout, "surface_policy", ""):
                lines.append(f"- Readable surface policy: `{layout.surface_policy}`")
            lines.append(
                f"- Supports visual asset: `{bool(getattr(layout, 'supports_visual_asset', False))}`"
            )
            if getattr(layout, "allowed_text_roles", None):
                lines.append(
                    "- Text roles: "
                    + ", ".join(f"`{role}`" for role in layout.allowed_text_roles[:8])
                )
            if getattr(layout, "allowed_visual_roles", None):
                lines.append(
                    "- Visual roles: "
                    + ", ".join(f"`{role}`" for role in layout.allowed_visual_roles[:8])
                )
            lines.append("")

        return "\n".join(lines)

    def _template_reference_contract(self) -> str:
        layouts = canonical_layout_views(self.profile)
        if not layouts:
            return ""

        lines = [
            "## Template Reference Contract",
            "",
            "- Use the template as a reference for outline structure, page archetypes, slot geometry, density, text/visual balance, title rhythm, palette, and typography.",
            "- Do not copy raw template backgrounds, original decorative image assets, shared template caches, or old sample content.",
            "- Body text, formulas, tables, and captions must sit on high-contrast readable surfaces. Required real assets take priority over exact template imitation.",
        ]
        lines.append("")
        lines.append("| Layout | Archetype | Density Ref | Visual Asset | Notes |")
        lines.append("|--------|-----------|-------------|--------------|-------|")
        for layout in layouts:
            text_n = sum(1 for slot in layout.slots if slot.type == "text")
            density = "high" if text_n >= 4 else "low" if text_n <= 1 else "medium"
            visual = bool(getattr(layout, "supports_visual_asset", False))
            notes = "reference geometry and density; regenerate clean readable HTML"
            lines.append(
                f"| `{layout.name}` | `{layout.layout_archetype or 'content'}` | `{density}` | `{visual}` | {notes} |"
            )
        return "\n".join(lines)

    def _decorative(self) -> str:
        """装饰图片（仅在有装饰图时生成）"""
        decorative = semantic_decorative_system(self.profile)
        if not decorative:
            return ""

        lines = ["## Decorative System", ""]
        background_count = decorative.get("background_count", 0)
        decorative_count = decorative.get("decorative_count", 0)
        lines.append(f"- 背景元素: {background_count}")
        lines.append(f"- 装饰元素: {decorative_count}")
        for item in decorative.get("repeated_positions", [])[:8]:
            lines.append(f"- 高频装饰位置: `{item.get('position', '')}` × {item.get('count', 0)}")
        return "\n".join(lines)

    def _style_contract(
        self,
        semantic_style: dict,
        cp: "ColorPalette",
        tp: "Typography",
    ) -> str:
        lines = [
            "## Page Reference Contract",
            "",
            "- Preserve the template's page rhythm, title bands, recurring accent marks, geometric balance, and aspect ratio as newly generated HTML/CSS.",
            "- Title or accent text on dark frame areas must use `inverse_text` or another high-contrast text role.",
            "",
            "## Content Surface Contract",
            "",
            "- Body paragraphs, formulas, tables, captions, and detailed evidence must render on `surface + primary_text` or an equivalent readable high-contrast pair.",
            "- Figure/table pages with a bound asset must allocate a visible asset panel before writing bullets.",
        ]
        if self.is_adaptive_structural:
            lines.append(
                "- Adaptive mode explicitly allows adding a light content card/panel when the template reference is too dark or too decorative for dense content."
            )
        elif self.is_strict_structural:
            lines.append(
                "- Strict mode should use existing template content regions; still keep body text on readable semantic color pairs."
            )
        lines.extend(["", "## Typography Contract", ""])
        if semantic_style.get("aspect_ratio"):
            lines.append(f"- 画幅比例: `{semantic_style['aspect_ratio']}`")
        language = semantic_style.get("language") or {}
        if language.get("lid"):
            lines.append(f"- 语言信号: `{language['lid']}`")
        if cp.background or cp.text or cp.surface or cp.primary_text or cp.inverse_text:
            lines.append(
                "- 语义配色: "
                f"frame={cp.background or cp.primary or 'n/a'} + inverse_text={cp.inverse_text or 'n/a'}; "
                f"surface={cp.surface or 'n/a'} + primary_text={cp.primary_text or cp.text or 'n/a'}"
            )
        if tp.title_font or tp.body_font:
            lines.append(
                f"- 字体系统: title=`{tp.title_font or tp.display_title_font or 'auto'}`, body=`{tp.body_font or 'auto'}`"
            )
        if len(lines) == 2:
            return ""
        return "\n".join(lines)

    def _rules(self, cp: "ColorPalette", tp: "Typography") -> str:
        """关键规则（根据数据可用性动态调整）"""
        rules = []
        has_colors = any([
            cp.primary,
            cp.secondary,
            cp.accent,
            cp.text,
            cp.background,
            cp.surface,
            cp.primary_text,
            cp.inverse_text,
            cp.muted_text,
            cp.border,
        ])
        has_fonts = any([tp.title_font, tp.body_font])

        if has_colors:
            if self.is_style_reference:
                rules.append("**参考模板配色** — 可以扩展颜色，但必须保持对比度和可读性")
            elif self.is_adaptive_structural:
                rules.append("**使用语义配色** — 背景参考、内容面、正文、图注分别使用 background/surface/primary_text/inverse_text 等角色")
                rules.append("**允许安全派生** — 若模板没有可读正文面，可从 surface 派生浅色卡片，不得使用低对比正文组合")
            else:
                rules.append("**遵守模板语义颜色角色** — 不要把 background 当作正文底色的唯一选择")
        if has_fonts:
            if self.is_style_reference:
                rules.append("**参考模板字体** — 正文字号不得为 0，优先投影可读")
            else:
                rules.append("**沿用模板字体** — 优先使用上方字体规范；字号以投影可读和不溢出为准")
        if self.is_style_reference:
            rules.append("**页面结构由稿件决定** — 真实文本、公式、论文图片和表格优先")
            rules.append("**不要用 SVG 文本图伪装正文** — 正文必须作为可读 HTML 文本呈现")
            rules.append("**不要强制纯蓝背景** — 背景色只作参考，可用浅色内容区提升可读性")
        elif self.is_adaptive_structural:
            rules.append("**先执行 `page_asset_plan.json`** — required figure/table/formula 页面必须显示绑定资产")
            rules.append("**先执行 `layout_mapping.yaml`** — 用 selected_layout 作为 reference 槽位锚点，不要自行改选 layout")
            rules.append("**生成每页前必须调用 `query_slide_layout`** — 获取 canonical slot contract")
            rules.append("**可读性优先** — 如果模板槽位无法安全承载正文，加入 adaptive surface/panel")
            rules.append("**禁止纯文本退化** — required visual 页面不能只输出 bullet list")
            rules.append("**每页 HTML 开头** — `<!-- template-layout: {layout_name} -->`")
        else:
            rules.append("**先执行 `layout_mapping.yaml`** — 若系统已生成逐页布局映射，不要自行改选 layout")
            rules.append("**生成每页前必须调用 `query_slide_layout`** — 获取元素清单和字符限制")
            rules.append("**保持 layout 主要意图** — 元素数量是结构锚点，不得为了偷懒退化为纯文本页")
            rules.append("**遵守字符限制** — 标题 ±20%，正文不超过 150%")
            rules.append("**装饰图保持不变** — 内容图可替换")
            rules.append("**每页 HTML 开头** — `<!-- template-layout: {layout_name} -->`")

        numbered = [f"{i+1}. {r}" for i, r in enumerate(rules)]
        return "\n".join(["## 关键规则", ""] + numbered)
