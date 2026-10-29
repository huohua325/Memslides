"""
TemplateGuideBuilder — Guide 驱动的模板 Prompt 构建器 (Stage 7)

替代 TemplatePromptBuilder，实现：
- 按阶段注入（Research / Design 各自获取所需子集）
- 按需查询（布局详情通过 MCP Tool 按页获取，不在启动时全量注入）
- Guide 驱动（6 个 Guide 文档提供结构化指导）
- 降级友好（Guide 缺失时仍可工作）
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from memslides.templates.semantic_access import (
    canonical_layout_views,
    resolve_canonical_layout,
    semantic_decorative_system,
)
from memslides.templates.layout_planner import TemplateLayoutPlanner
from memslides.templates.quality import assess_template_quality, effective_template_palette

logger = logging.getLogger(__name__)

try:
    from ..core.template_models import (
        ColorPalette,
        ContentPatterns,
        DesignConstraints,
        TemplateProfile,
        Typography,
    )
except ImportError:
    import sys
    import pathlib
    _core = str(pathlib.Path(__file__).resolve().parent.parent / "core")
    if _core not in sys.path:
        sys.path.insert(0, _core)
    from template_models import (  # type: ignore[no-redef]
        ColorPalette,
        ContentPatterns,
        DesignConstraints,
        TemplateProfile,
        Typography,
    )

PACKAGE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_GUIDES_DIR = PACKAGE_DIR / "template_guides"
DEFAULT_PROMPTS_DIR = DEFAULT_GUIDES_DIR / "prompts"

# Guide 文档与阶段的映射
RESEARCH_GUIDES = ["00_field_map", "01_content_planning"]
DESIGN_GUIDES = ["00_field_map", "02_layout_selection", "04_visual_constraints", "05_image_handling"]
PER_PAGE_GUIDES = ["03_element_filling"]


class TemplateGuideBuilder:
    """Guide 驱动的模板 Prompt 构建器"""

    def __init__(
        self,
        template_profile: TemplateProfile,
        guides_dir: Path | None = None,
        workspace: Path | None = None,
    ):
        self.skill = template_profile
        self.guides_dir = Path(guides_dir) if guides_dir else DEFAULT_GUIDES_DIR
        self.prompts_dir = self.guides_dir / "prompts"
        self._guides_cache: dict[str, str] = {}
        self._token_stats: dict[str, int] = {}
        self._last_queried_layout: str = ""
        self._workspace: Path | None = Path(workspace) if workspace else None

        # Jinja2 环境初始化
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # 注入记录系统
        self._injection_log: dict[str, Any] = {
            "template_name": template_profile.name if template_profile else "unknown",
            "created_at": datetime.now().isoformat(),
            "guides": {},
            "injections": [],
            "mcp_queries": [],       # MCP Tool 查询记录
            "compliance_feedbacks": [],  # 合规检查反馈记录
        }
        self._modify_count: int = 0

    # ══════════════════════════════════════════════
    # 阶段注入方法（Jinja2 模板驱动）
    # ══════════════════════════════════════════════

    def build_for_research(self) -> str:
        """构建 Research Agent 注入 Prompt (~1,000 tokens)

        使用 Jinja2 模板驱动，数据从 TemplateProfile 提取
        """
        context = self._build_research_context()
        result = self._render_template("research_prompt.md.j2", context)
        self._token_stats["research_prompt"] = self._estimate_tokens(result)

        # 缓存并记录注入
        self._cache_prompt("research", result)
        self._record_injection("research", result, {
            "guides_used": ["00_field_map", "01_content_planning"],
            "sections": ["任务说明", "内容规划指南", "模板内容模式", "可用功能页", "布局选择提示"],
        })
        return result

    def build_for_design(self) -> str:
        """构建 DeckDesigner 注入 Prompt (~2,100 tokens)

        使用 Jinja2 模板驱动，数据从 TemplateProfile 提取
        """
        context = self._build_design_context()
        result = self._render_template("design_prompt.md.j2", context)

        # Task 10 (F1): 低置信度警告
        quality_score = self.skill.confidence if self.skill else 0.8
        if 0.3 <= quality_score < 0.6:
            result += (
                "\n\n> ⚠️ **模板分析置信度较低** (score={:.2f})，"
                "部分约束可能不准确，请结合自身判断。\n".format(quality_score)
            )

        self._token_stats["design_prompt"] = self._estimate_tokens(result)

        # 缓存并记录注入
        self._cache_prompt("design", result)
        self._record_injection("design", result, {
            "guides_used": ["00_field_map", "02_layout_selection", "04_visual_constraints", "05_image_handling"],
            "sections": ["布局选择规则", "可用布局摘要", "视觉约束", "品牌元素", "图片处理规则", "装饰图片"],
        })
        return result

    def build_for_modify(self, current_page_layout: str = "") -> str:
        """构建 RevisionEditor 注入 Prompt (~800 tokens)

        使用 Jinja2 模板驱动，比 build_for_design() 更精简

        Args:
            current_page_layout: 当前被修改页面的布局名称（可选）
        """
        context = self._build_modify_context(current_page_layout=current_page_layout)
        result = self._render_template("modify_prompt.md.j2", context)
        self._token_stats["modify_prompt"] = self._estimate_tokens(result)

        # 缓存并记录注入
        self._modify_count += 1
        stage_name = f"modify_{self._modify_count:02d}"
        self._cache_prompt(stage_name, result)
        self._record_injection(stage_name, result, {
            "guides_used": ["04_visual_constraints"],
            "sections": ["颜色规范", "字体规范", "可用布局"],
        })
        return result

    # ══════════════════════════════════════════════
    # 确定性 design_plan.md 生成（委托给 DesignPlanGenerator）
    # ══════════════════════════════════════════════

    def generate_design_plan(self, output_path: Path, mode: str = "structural_template") -> Path:
        """从 TemplateProfile 确定性生成 design_plan.md（委托给 DesignPlanGenerator）

        Args:
            output_path: 输出文件路径 (e.g., workspace / "design_plan.md")
            mode: structural_template uses strict layout guidance; style_reference
                keeps only visual inspiration and lets manuscript content drive layout.

        Returns:
            写入的文件路径
        """
        from .design_plan_generator import DesignPlanGenerator

        generator = DesignPlanGenerator(self.skill, mode=mode)
        result = generator.generate(output_path)

        # 记录注入
        content = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        self._record_injection("design_plan", content, {"generator": "DesignPlanGenerator"})

        return result

    # ══════════════════════════════════════════════
    # Jinja2 模板渲染辅助方法
    # ══════════════════════════════════════════════

    def _render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """渲染 Jinja2 模板

        Args:
            template_name: 模板文件名 (e.g., "research_prompt.md.j2")
            context: 模板变量字典

        Returns:
            渲染后的字符串
        """
        try:
            template = self._jinja_env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound:
            logger.warning(f"Template not found: {template_name}, falling back to empty")
            return ""
        except Exception as e:
            logger.warning(f"Template render failed: {e}")
            return ""

    def _build_research_context(self) -> dict[str, Any]:
        """构建 Research Agent 模板上下文"""
        cp = self.skill.content_patterns or ContentPatterns()
        si = self.skill.slide_induction or {}
        functional_keys = si.get("functional_keys", [])

        # 信息密度映射
        density_map = {
            "low": ("精简风格", "每页 2-3 个要点，大量留白"),
            "medium": ("平衡风格", "每页 4-5 个要点，图文混排"),
            "high": ("信息密集", "每页 6+ 个要点"),
        }
        current_density = cp.info_density or "medium"
        density_name, density_desc = density_map.get(current_density, (current_density, ""))

        # 叙事风格映射
        style_map = {
            "academic": ("学术风格", "引用严谨、术语专业、逻辑递进"),
            "business": ("商务风格", "数据驱动、简洁直接、先结论后论据"),
            "creative": ("创意风格", "故事叙述、视觉优先"),
            "educational": ("教育风格", "循序渐进、解释清晰"),
            "technical": ("技术风格", "架构图、代码片段、实现细节"),
        }
        style_name, style_desc = "", ""
        if cp.narrative_style:
            style_name, style_desc = style_map.get(cp.narrative_style, (cp.narrative_style, ""))

        # 章节结构
        section_names = {
            "opening": "封面/开场", "table_of_contents": "目录页",
            "section_divider": "章节分隔页", "content": "内容页",
            "ending": "结尾/致谢页",
        }
        typical_sections_display = ""
        if cp.typical_sections:
            named = [f"`{s}` ({section_names.get(s, s)})" for s in cp.typical_sections]
            typical_sections_display = " → ".join(named)

        # 列表风格
        bullet_style_display = ""
        if cp.bullet_style:
            bullet_desc = {"bulleted": "圆点列表", "numbered": "数字编号", "icon": "图标列表"}
            bullet_style_display = bullet_desc.get(cp.bullet_style, cp.bullet_style)

        # 功能页布局
        fk_names = {
            "opening": "封面", "table of contents": "目录",
            "ending": "结尾/致谢", "section_divider": "章节分隔",
        }
        functional_layouts = []
        for fk in functional_keys:
            if fk in si and isinstance(si[fk], dict):
                layout_data = si[fk]
                elements = layout_data.get("elements", [])
                functional_layouts.append({
                    "name": fk,
                    "display": fk_names.get(fk.lower(), fk),
                    "text_count": sum(1 for e in elements if e.get("type") == "text"),
                    "image_count": sum(1 for e in elements if e.get("type") == "image"),
                })

        # 内容页布局
        content_layout_keys = [
            k for k in si.keys()
            if k not in ("functional_keys", "language", "layout_capabilities") and isinstance(si[k], dict)
            and k.lower() not in [fk.lower() for fk in functional_keys]
        ]
        content_layouts = []
        for layout_name in content_layout_keys:
            layout_data = si[layout_name]
            elements = layout_data.get("elements", [])
            # 截断长名称但保留后缀
            if len(layout_name) > 70:
                suffix = ""
                if ":image" in layout_name:
                    suffix = ":image"
                elif ":text" in layout_name:
                    suffix = ":text"
                short_name = layout_name[:65] + "..." + suffix
            else:
                short_name = layout_name
            content_layouts.append({
                "name": layout_name,
                "short_name": short_name,
                "slides": layout_data.get("slides", []),
                "text_count": sum(1 for e in elements if e.get("type") == "text"),
                "image_count": sum(1 for e in elements if e.get("type") == "image"),
            })

        # 任务说明
        task_section = ""
        guide_00 = self._load_guide("00_field_map")
        if guide_00:
            task_sec = self._extract_section(guide_00, "阶段分工")
            if task_sec:
                research_part = self._extract_section(task_sec, "Research 阶段")
                if research_part:
                    task_section = research_part.strip()

        # Task 14 (D1): 图片需求汇总 + 内容容量估算
        total_image_demand = 0
        total_text_capacity = 0
        for layout in content_layouts:
            total_image_demand += layout["image_count"]
            # 计算文本容量：各 text element 的 max(len(data)) 之和
            layout_name_key = layout["name"]
            ld = si.get(layout_name_key, {})
            for el in ld.get("elements", []):
                if el.get("type") == "text":
                    el_data = el.get("data", [])
                    if el_data:
                        total_text_capacity += max(len(d) for d in el_data if isinstance(d, str))

        template_language = si.get("language", {}).get("lid", "zh")
        lang_display = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文"}.get(
            template_language, template_language
        )

        return {
            "template_name": self.skill.name,
            "aspect_ratio": self.skill.aspect_ratio,
            "slide_count": self.skill.slide_count,
            "task_section": task_section,
            "info_density": current_density,
            "density_name": density_name,
            "density_desc": density_desc,
            "narrative_style": cp.narrative_style,
            "style_name": style_name,
            "style_desc": style_desc,
            "typical_sections": cp.typical_sections,
            "typical_sections_display": typical_sections_display,
            "max_bullets": cp.max_bullets_per_slide,
            "bullet_style": cp.bullet_style,
            "bullet_style_display": bullet_style_display,
            "language": template_language,
            "lang_display": lang_display,
            "functional_layouts": functional_layouts,
            "content_layouts": content_layouts,
            "total_image_demand": total_image_demand,
            "total_text_capacity": total_text_capacity,
        }

    def _build_design_context(self) -> dict[str, Any]:
        """构建 DeckDesigner 模板上下文"""
        dc = self.skill.design_constraints or DesignConstraints()
        raw_cp = dc.color_palette or ColorPalette()
        tp = dc.typography or Typography()
        spacing = dc.spacing or {}
        cs = dc.chart_style if hasattr(dc, 'chart_style') and dc.chart_style else None
        state_mode = ""
        try:
            from memslides.templates.runtime_state import load_template_runtime_state

            if self._workspace:
                state = load_template_runtime_state(self._workspace)
                state_mode = state.mode if state else ""
        except Exception:
            state_mode = ""
        if not state_mode:
            state_mode = assess_template_quality(self.skill).mode
        cp = effective_template_palette(self.skill, mode=state_mode)

        # 布局选择规则
        layout_rules = ""
        guide_02 = self._load_guide("02_layout_selection")
        if guide_02:
            rules = self._extract_section(guide_02, "使用规则")
            if rules:
                layout_rules = rules.strip()

        # 图片处理规则
        image_rules = ""
        guide_05 = self._load_guide("05_image_handling")
        if guide_05:
            img_rules = self._extract_section(guide_05, "区分装饰图 vs 内容图")
            if img_rules:
                image_rules = img_rules.strip()

        # 辅助色格式化
        additional = getattr(cp, 'additional', None) or []
        color_additional = ", ".join(f"`{c}`" for c in additional[:4]) if additional else ""
        adaptive_surface_note = ""
        if state_mode == "adaptive_structural" and (
            cp.surface != raw_cp.surface or cp.primary_text != raw_cp.primary_text
        ):
            adaptive_surface_note = (
                f"正文/公式/表格优先落在安全内容面 `{cp.surface or '#ffffff'}` 上，"
                f"正文颜色使用 `{cp.primary_text or cp.text or '#1f2937'}`；"
                "标题节奏和强调区域可保留模板主色，但不要复刻模板背景或旧样例内容。"
            )

        return {
            "template_name": self.skill.name,
            "aspect_ratio": self.skill.aspect_ratio,
            "slide_count": self.skill.slide_count,
            "template_mode": state_mode,
            "layout_rules": layout_rules,
            "layout_summary_table": self.get_layout_summary_table(),
            "template_reference_summary": self.get_reference_summary(),
            "color_primary": cp.primary,
            "color_secondary": cp.secondary,
            "color_accent": cp.accent,
            "color_text": cp.text,
            "color_background": cp.background,
            "color_surface": cp.surface,
            "color_primary_text": cp.primary_text,
            "color_inverse_text": cp.inverse_text,
            "color_muted_text": cp.muted_text,
            "color_border": cp.border,
            "color_additional": color_additional,
            "adaptive_surface_note": adaptive_surface_note,
            "title_font": tp.title_font,
            "title_size": tp.title_size,
            "subtitle_size": int(tp.title_size * 0.7) if tp.title_size else 0,
            "body_font": tp.body_font,
            "body_size": tp.body_size,
            "caption_size": int(tp.body_size * 0.75) if tp.body_size else 0,
            "line_height": tp.line_height,
            "spacing_margin": spacing.get("margin"),
            "spacing_padding": spacing.get("padding"),
            "spacing_gap": spacing.get("element_gap"),
            "image_rules": image_rules,
            "decorative_images": self.get_image_summary(),
            # 图表风格约束
            "chart_default_type": cs.default_type if cs else "",
            "chart_color_scheme": ", ".join(f"`{c}`" for c in (cs.color_scheme or [])[:6]) if cs else "",
        }

    def _build_modify_context(self, current_page_layout: str = "") -> dict[str, Any]:
        """构建 RevisionEditor 模板上下文"""
        dc = self.skill.design_constraints or DesignConstraints()
        cp = dc.color_palette or ColorPalette()
        tp = dc.typography or Typography()

        # 当前页面布局详情
        current_layout_detail = ""
        if current_page_layout:
            current_layout_detail = self.get_layout_detail(current_page_layout)

        return {
            "template_name": self.skill.name,
            "color_primary": cp.primary,
            "color_secondary": cp.secondary,
            "color_accent": cp.accent,
            "color_text": cp.text,
            "color_background": cp.background,
            "color_surface": cp.surface,
            "color_primary_text": cp.primary_text,
            "color_inverse_text": cp.inverse_text,
            "color_muted_text": cp.muted_text,
            "color_border": cp.border,
            "title_font": tp.title_font,
            "title_size": tp.title_size,
            "body_font": tp.body_font,
            "body_size": tp.body_size,
            "layout_summary_table": self.get_layout_summary_table(),
            "current_layout_detail": current_layout_detail,
        }

    # ══════════════════════════════════════════════
    # MCP Tool 支撑方法
    # ══════════════════════════════════════════════

    def get_layout_detail(self, layout_name: str) -> str:
        """获取布局详情 (~600 tokens/页)，供 query_slide_layout MCP Tool 使用"""
        layout = resolve_canonical_layout(self.skill, layout_name)
        if layout is None:
            available = self._get_available_layout_names()
            return (
                f"❌ 未找到布局 `{layout_name}`\n\n"
                f"可用布局：\n" + "\n".join(f"- `{n}`" for n in available)
            )

        self._last_queried_layout = layout.name

        # 记录 MCP 查询
        self._record_mcp_query("query_slide_layout", layout.name)

        sections: list[str] = []
        sections.append(f"## 布局详情：{layout.name}")
        sections.append(
            f"类型：{layout.layout_archetype or 'content'} / pattern={layout.visual_pattern or 'balanced'}"
        )
        if getattr(layout, "surface_mode", ""):
            sections.append(f"推荐承载面：`{layout.surface_mode}`")
        sections.append(
            "Reference-only：此布局只提供 slot 几何、文字密度、图文比例、标题节奏和风格参考；"
            "不要复刻模板背景、壳层图片或旧样例页。"
        )
        if getattr(layout, "supports_visual_asset", False):
            sections.append("视觉资产：支持真实 figure / table / chart 绑定")
        if getattr(layout, "allowed_text_roles", None):
            sections.append(
                "文本角色：" + ", ".join(f"`{role}`" for role in layout.allowed_text_roles[:8])
            )
        if getattr(layout, "allowed_visual_roles", None):
            sections.append(
                "视觉角色：" + ", ".join(f"`{role}`" for role in layout.allowed_visual_roles[:8])
            )
        if layout.aliases:
            sections.append(f"别名：{', '.join(f'`{alias}`' for alias in layout.aliases[:4])}")
        sections.append("")

        sections.append("### Editable Slots")
        sections.append("| # | 槽位 | 角色 | 类型 | 容量提示 | 样例内容 |")
        sections.append("|---|------|------|------|----------|----------|")

        text_count = 0
        image_count = 0
        for i, slot in enumerate(layout.slots, 1):
            if slot.type == "text":
                text_count += 1
            elif slot.type in {"image", "chart", "table"}:
                image_count += 1
            hint = slot.capacity_hint or {}
            char_hint = hint.get("suggested_max_chars") or hint.get("sample_char_count") or "—"
            sample_display = ""
            if slot.sample_values:
                sample = slot.sample_values[0]
                sample_display = sample[:20] + "..." if len(sample) > 20 else sample
            sections.append(
                f"| {i} | {slot.name or '?'} | {slot.role or 'slot'} | {slot.type or 'text'} | "
                f"{char_hint} | {sample_display or '—'} |"
            )

        sections.append("")
        if layout.sample_content:
            sections.append("### Sample Content Signals")
            for slot in layout.sample_content[:6]:
                if not slot.sample_values:
                    continue
                sample = slot.sample_values[0]
                sections.append(f"- `{slot.name}` ({slot.role or slot.type}): {sample[:80]}")
            sections.append("")

        # 元素填充指南（从 Guide 03 的「字符限制规则」和「元素数量规则」）
        guide_03 = self._load_guide("03_element_filling")
        if guide_03:
            char_rules = self._extract_section(guide_03, "字符限制规则")
            if char_rules:
                sections.append("### 字符限制规则")
                sections.append(char_rules.strip())
                sections.append("")

        # 元素数量规则
        state_mode = ""
        try:
            from memslides.templates.runtime_state import load_template_runtime_state

            if self._workspace:
                state_mode = load_template_runtime_state(self._workspace).mode
        except Exception:
            state_mode = ""

        sections.append("### 元素数量规则")
        if state_mode == "adaptive_structural":
            sections.append(f"- 结构锚点：文本元素 {text_count} 个，图片元素 {image_count} 个。")
            sections.append("- 优先保留主要槽位意图；可增加安全 surface/panel 承载正文、公式、表格或真实视觉资产。")
        elif state_mode == "style_reference":
            sections.append(f"- 观察到的元素：文本元素 {text_count} 个，图片元素 {image_count} 个。")
            sections.append("- 仅作为风格参考，页面结构由稿件与真实资产需求决定。")
        else:
            sections.append(f"- 文本元素：{text_count} 个（结构锚点）")
            sections.append(f"- 图片元素：{image_count} 个（结构锚点）")
            sections.append("- 保持 canonical layout 的主要意图；可为可读性和真实资产承载做小幅自适应。")

        result = "\n".join(sections)
        self._token_stats.setdefault("layout_queries", 0)
        self._token_stats["layout_queries"] += self._estimate_tokens(result)
        return result

    def get_layout_geometry(self, layout_name: str) -> str:
        """获取布局几何信息 (~200 tokens)，供 query_layout_geometry MCP Tool 使用

        优先从 shape_geometry.json 读取精确坐标，无文件时返回提示。
        """
        # 记录 MCP 查询
        self._record_mcp_query("query_layout_geometry", layout_name)
        layout = resolve_canonical_layout(self.skill, layout_name)
        if layout is None:
            return f"❌ 未找到布局 `{layout_name}` 的几何信息"

        sections = [f"## 几何信息：{layout.name}", ""]
        sections.append("| # | 槽位 | 角色 | 左% | 上% | 宽% | 高% | 字号 |")
        sections.append("|---|------|------|-----|-----|-----|-----|------|")
        for i, slot in enumerate(layout.slots, 1):
            geo = slot.geometry or {}
            sections.append(
                f"| {i} | {slot.name or '?'} | {slot.role or 'slot'} "
                f"| {geo.get('left_pct', 0):.1f} | {geo.get('top_pct', 0):.1f} "
                f"| {geo.get('width_pct', 0):.1f} | {geo.get('height_pct', 0):.1f} "
                f"| {slot.font_size or '—'} |"
            )
        return "\n".join(sections)

    def get_image_summary(self) -> str:
        """获取装饰图片摘要，供 query_image_info MCP Tool 使用"""
        image_stats = self.skill.image_stats or {}
        if not image_stats:
            return ""

        decorative: list[str] = []
        for img_name, stats in image_stats.items():
            if not isinstance(stats, dict):
                continue
            appear = stats.get("appear_times", 1)
            area = stats.get("relative_area", 0)

            # 装饰图判断：多次出现+小面积，或多次出现+大面积
            if (appear >= 3 and area < 5) or (appear >= 2 and area > 20):
                short_name = img_name[:16] + "..." if len(img_name) > 16 else img_name
                decorative.append(f"- `{short_name}` (出现{appear}次, 面积{area:.1f}%)")

        if not decorative:
            return ""

        return (
            "\n".join(decorative[:10])
            + "\n- 这些是历史解析名，仅说明模板的装饰节奏。生成 HTML 时不要引用共享模板缓存路径、原始背景资源或旧样例图片。"
        )

    def get_reference_summary(self) -> str:
        layouts = canonical_layout_views(self.skill)
        if not layouts:
            return ""
        visual_layouts = sum(1 for layout in layouts if getattr(layout, "supports_visual_asset", False))
        return (
            f"layouts={len(layouts)}, visual_asset_layouts={visual_layouts}, "
            "mode=reference_only"
        )

    # ══════════════════════════════════════════════
    # 辅助方法
    # ══════════════════════════════════════════════

    def get_layout_summary_table(self) -> str:
        """生成布局摘要表格"""
        layouts = canonical_layout_views(self.skill)
        if not layouts:
            return ""
        rows: list[str] = []
        rows.append("| # | Canonical Layout | Archetype | Density Ref | Slots | 文/图 |")
        rows.append("|---|------------------|-----------|-------------|-------|-------|")
        for idx, layout in enumerate(layouts, 1):
            text_n = sum(1 for slot in layout.slots if slot.type == "text")
            img_n = sum(1 for slot in layout.slots if slot.type in {"image", "chart", "table"})
            display_name = layout.name if len(layout.name) <= 40 else layout.name[:37] + "..."
            density = "high" if text_n >= 4 else "low" if text_n <= 1 else "medium"
            rows.append(
                f"| {idx} | `{display_name}` | {layout.layout_archetype or 'content'} "
                f"| `{density}` | {len(layout.slots)} | {text_n}文/{img_n}图 |"
            )
        return "\n".join(rows)

    def detect_conflicts(self, user_instruction: str) -> list[dict]:
        """检测模板约束与用户指令的潜在冲突

        Args:
            user_instruction: 用户指令

        Returns:
            冲突列表，每项包含 {type, template_value, user_hint}
        """
        conflicts: list[dict] = []
        skill = self.skill

        if not user_instruction or not user_instruction.strip():
            return conflicts

        instruction_lower = user_instruction.lower()

        # ── 颜色冲突 ──
        color_keywords = {
            "蓝色": "blue", "blue": "blue",
            "红色": "red", "red": "red",
            "绿色": "green", "green": "green",
            "黄色": "yellow", "yellow": "yellow",
            "橙色": "orange", "orange": "orange",
            "紫色": "purple", "purple": "purple",
            "黑色": "black", "black": "black",
            "白色": "white", "white": "white",
            "深色": "dark", "浅色": "light",
            "暗色": "dark", "亮色": "light",
        }

        template_primary = ""
        try:
            if skill.design_constraints and skill.design_constraints.color_palette:
                template_primary = (skill.design_constraints.color_palette.primary or "").lower()
        except (AttributeError, TypeError):
            pass

        if template_primary:
            for keyword, color_name in color_keywords.items():
                if keyword in instruction_lower:
                    if color_name not in template_primary and color_name not in ("dark", "light"):
                        conflicts.append({
                            "type": "color",
                            "template_value": skill.design_constraints.color_palette.primary,
                            "user_hint": f"User mentioned '{keyword}'",
                        })
                        break

        # ── 布局冲突 ──
        layout_keywords = {
            "单栏": "single_column", "双栏": "two_column",
            "三栏": "three_column", "全图": "full_image",
            "single column": "single_column", "two column": "two_column",
            "full image": "full_image",
        }

        # 从 slide_induction 提取可用布局
        si = skill.slide_induction or {}
        available_layouts = set()
        for key in si.keys():
            if key not in ("functional_keys", "language", "layout_capabilities") and isinstance(si.get(key), dict):
                available_layouts.add(key)

        if available_layouts:
            for keyword, layout in layout_keywords.items():
                if keyword in instruction_lower:
                    # 检查是否有匹配的布局
                    has_layout = any(
                        layout in layout_name.lower()
                        for layout_name in available_layouts
                    )
                    if not has_layout:
                        conflicts.append({
                            "type": "layout",
                            "template_value": list(available_layouts)[:5],
                            "user_hint": f"User requested '{keyword}' layout",
                        })
                        break

        # ── 风格冲突 ──
        style_keywords = {
            "简约": "minimal", "极简": "minimal",
            "复杂": "complex", "华丽": "ornate",
            "minimal": "minimal", "complex": "complex",
        }

        template_style = ""
        try:
            if skill.content_patterns and skill.content_patterns.narrative_style:
                template_style = skill.content_patterns.narrative_style.lower()
        except (AttributeError, TypeError):
            pass

        if template_style:
            for keyword, style in style_keywords.items():
                if keyword in instruction_lower:
                    if style not in template_style:
                        conflicts.append({
                            "type": "style",
                            "template_value": skill.content_patterns.narrative_style,
                            "user_hint": f"User mentioned '{keyword}' style",
                        })
                        break

        # ── 宽高比冲突 ──
        aspect_keywords = {
            "4:3": "4:3", "4：3": "4:3",
            "16:9": "16:9", "16：9": "16:9",
            "竖版": "portrait", "portrait": "portrait",
            "横版": "landscape", "landscape": "landscape",
        }
        template_ratio = getattr(skill, "aspect_ratio", "") or ""
        if template_ratio:
            for keyword, ratio_hint in aspect_keywords.items():
                if keyword in instruction_lower:
                    mismatch = False
                    if ratio_hint in ("4:3", "16:9") and ratio_hint != template_ratio:
                        mismatch = True
                    elif ratio_hint == "portrait" and template_ratio in ("16:9", "4:3"):
                        mismatch = True
                    elif ratio_hint == "landscape" and template_ratio not in ("16:9", "4:3"):
                        mismatch = True
                    if mismatch:
                        conflicts.append({
                            "type": "aspect_ratio",
                            "template_value": template_ratio,
                            "user_hint": f"User requested '{keyword}' but template is {template_ratio}",
                        })
                        break

        # ── 页数冲突 ──
        si = skill.slide_induction or {}
        content_layout_count = sum(
            1 for k, v in si.items()
            if k not in ("functional_keys", "language", "layout_capabilities") and isinstance(v, dict)
            and k.lower() not in [fk.lower() for fk in si.get("functional_keys", [])]
        )
        if content_layout_count > 0:
            page_match = re.search(r'(\d+)\s*(?:页|slides?|pages?)', instruction_lower)
            if page_match:
                requested_pages = int(page_match.group(1))
                if requested_pages > content_layout_count * 3:
                    conflicts.append({
                        "type": "page_count",
                        "template_value": f"{content_layout_count} content layouts",
                        "user_hint": (
                            f"User requested {requested_pages} pages but template has only "
                            f"{content_layout_count} content layout types — layouts will repeat heavily"
                        ),
                    })

        # ── 语言冲突 ──
        template_lang = si.get("language", {}).get("lid", "")
        if template_lang:
            # 简单判断：中文字符占比
            cn_chars = sum(1 for c in user_instruction if '\u4e00' <= c <= '\u9fff')
            cn_ratio = cn_chars / max(len(user_instruction), 1)
            if template_lang == "en" and cn_ratio > 0.3:
                conflicts.append({
                    "type": "language",
                    "template_value": f"template language: {template_lang}",
                    "user_hint": "User instruction is in Chinese but template is English — font compatibility may be affected",
                })
            elif template_lang in ("zh", "zh-cn", "zh-tw") and cn_ratio < 0.1 and len(user_instruction) > 20:
                conflicts.append({
                    "type": "language",
                    "template_value": f"template language: {template_lang}",
                    "user_hint": "User instruction is in English but template is Chinese — font compatibility may be affected",
                })

        return conflicts

    def format_conflicts_for_prompt(self, conflicts: list[dict]) -> str:
        """格式化冲突信息用于 prompt 注入

        Args:
            conflicts: 冲突列表

        Returns:
            格式化的冲突描述
        """
        if not conflicts:
            return ""

        lines = ["Detected potential conflicts between template constraints and your instructions:"]

        type_icons = {
            "color": "🎨", "layout": "📐", "style": "✨",
            "aspect_ratio": "📏", "page_count": "📄", "language": "🌐",
        }

        for i, c in enumerate(conflicts, 1):
            ctype = c.get("type", "unknown")
            template_val = c.get("template_value", "")
            user_hint = c.get("user_hint", "")
            icon = type_icons.get(ctype, "⚠️")

            if isinstance(template_val, list):
                template_val = ", ".join(template_val[:3])

            lines.append(f"{i}. {icon} [{ctype.upper()}] Template uses '{template_val}', but {user_hint}")

        lines.append("")
        lines.append("How would you like to proceed?")

        return "\n".join(lines)

    def get_last_queried_layout(self) -> str:
        """获取最后一次 get_layout_detail 查询的布局名称"""
        return self._last_queried_layout

    def build_layout_mapping(
        self,
        *,
        manuscript_content: str,
        asset_manifest: dict[str, Any] | None = None,
        page_count_hint: int | None = None,
    ) -> dict[str, Any]:
        planner = TemplateLayoutPlanner(self.skill)
        return planner.build_layout_mapping(
            manuscript_text=manuscript_content,
            asset_manifest=asset_manifest or {},
            page_count_hint=page_count_hint,
        )

    # ══════════════════════════════════════════════
    # 内部方法
    # ══════════════════════════════════════════════

    def _load_guide(self, guide_id: str) -> str:
        """加载 Guide 文档，缓存结果，移除 YAML frontmatter"""
        if guide_id in self._guides_cache:
            return self._guides_cache[guide_id]

        guide_path = self.guides_dir / f"{guide_id}.guide.md"
        if not guide_path.exists():
            logger.debug(f"Guide not found: {guide_path}")
            self._guides_cache[guide_id] = ""
            self._record_guide_load(guide_id, "", success=False)
            return ""

        content = guide_path.read_text(encoding="utf-8")
        # 移除 YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                content = parts[2].strip()

        self._guides_cache[guide_id] = content
        self._record_guide_load(guide_id, content, success=True)
        return content

    @staticmethod
    def parse_layout_mapping(manuscript_content: str) -> list[dict[str, Any]]:
        """从 manuscript 末尾解析 LAYOUT_MAPPING

        Args:
            manuscript_content: manuscript 文件内容

        Returns:
            布局映射列表，每个元素包含 page, title, layout 字段
            如果解析失败返回空列表

        Example:
            >>> content = '''
            ... # Slide Content
            ... ...
            ... ```yaml
            ... # === LAYOUT_MAPPING ===
            ... slides:
            ...   - page: 1
            ...     title: "Title"
            ...     layout: "opening"
            ... ```
            ... '''
            >>> TemplateGuideBuilder.parse_layout_mapping(content)
            [{'page': 1, 'title': 'Title', 'layout': 'opening'}]
        """
        # 查找 LAYOUT_MAPPING 分隔符
        marker = "# === LAYOUT_MAPPING ==="
        if marker not in manuscript_content:
            return []

        # 提取 YAML 内容
        # 支持两种格式：
        # 1. ```yaml\n# === LAYOUT_MAPPING ===\n...\n``` 代码块
        # 2. 直接的 # === LAYOUT_MAPPING ===\n... YAML 内容

        # 首先尝试在代码块内查找
        code_block_pattern = r'```ya?ml\s*\n(.*?)```'
        for match in re.finditer(code_block_pattern, manuscript_content, re.DOTALL):
            block_content = match.group(1)
            if marker in block_content:
                # 提取 marker 之后的 YAML 内容
                idx = block_content.find(marker)
                yaml_content = block_content[idx + len(marker):].strip()
                break
        else:
            # 没有在代码块中找到，直接提取
            idx = manuscript_content.find(marker)
            after_marker = manuscript_content[idx + len(marker):]
            # 截取到下一个代码块结束符或文件末尾
            end_match = re.search(r'\n```', after_marker)
            if end_match:
                yaml_content = after_marker[:end_match.start()].strip()
            else:
                yaml_content = after_marker.strip()

        try:
            data = yaml.safe_load(yaml_content)
            if isinstance(data, dict) and "slides" in data:
                return data["slides"]
            return []
        except Exception as e:
            logger.warning(f"Failed to parse LAYOUT_MAPPING: {e}")
            return []

    def get_layout_for_page(self, manuscript_content: str, page_number: int) -> str | None:
        """根据 manuscript 获取指定页的布局名称

        Args:
            manuscript_content: manuscript 文件内容
            page_number: 页码（1-indexed）

        Returns:
            布局名称，如果未找到返回 None
        """
        mapping = self.parse_layout_mapping(manuscript_content)
        for item in mapping:
            if item.get("page") == page_number:
                return item.get("layout")
        return None

    def _extract_section(self, content: str, section_title: str) -> str:
        """从 Markdown 内容中提取指定标题的 section"""
        lines = content.split("\n")
        result_lines: list[str] = []
        in_section = False
        section_level = 0

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)

            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                if section_title.lower() in title.lower():
                    in_section = True
                    section_level = level
                    continue
                elif in_section and level <= section_level:
                    break
            elif in_section:
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def _get_available_layout_names(self) -> list[str]:
        """获取所有可用布局名称"""
        return [layout.name for layout in canonical_layout_views(self.skill)]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗估 token 数（中英混合约 0.5-0.7 token/char）"""
        return int(len(text) * 0.6)

    # ══════════════════════════════════════════════
    # 注入记录系统
    # ══════════════════════════════════════════════

    def _record_injection(self, stage: str, prompt: str, metadata: dict[str, Any]) -> None:
        """记录一次注入（内存 + JSONL 增量写入）"""
        entry = {
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "char_count": len(prompt),
            "token_estimate": self._estimate_tokens(prompt),
            "guides_used": metadata.get("guides_used", []),
            "sections_included": metadata.get("sections", []),
        }
        self._injection_log["injections"].append(entry)
        self._append_jsonl({"type": "injection", **entry})

    def _record_guide_load(self, guide_id: str, content: str, success: bool) -> None:
        """记录 guide 加载状态"""
        self._injection_log["guides"][guide_id] = {
            "loaded": success,
            "char_count": len(content) if success else 0,
            "load_time": datetime.now().isoformat(),
        }

    def _record_mcp_query(self, tool_name: str, layout_name: str) -> None:
        """记录 MCP Tool 查询（内存 + JSONL 增量写入）"""
        entry = {
            "tool": tool_name,
            "layout": layout_name,
            "timestamp": datetime.now().isoformat(),
        }
        self._injection_log["mcp_queries"].append(entry)
        self._append_jsonl({"type": "mcp_query", **entry})

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        """Task 11 (F2): 增量 append 一行 JSON 到 workspace/.history/injection_events.jsonl"""
        if not self._workspace:
            return
        try:
            jsonl_dir = self._workspace / ".history"
            jsonl_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = jsonl_dir / "injection_events.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # non-fatal

    def record_compliance_feedback(self, slide_file: str, issues_count: int, feedback: str) -> None:
        """记录合规检查反馈（供 task.py inspect_slide 调用）"""
        self._injection_log["compliance_feedbacks"].append({
            "slide": slide_file,
            "issues_count": issues_count,
            "feedback_chars": len(feedback),
            "timestamp": datetime.now().isoformat(),
        })
        # 缓存反馈内容用于保存
        self._cache_prompt(f"compliance_{len(self._injection_log['compliance_feedbacks']):02d}", feedback)

    def save_injection_logs(self, output_dir: Path) -> Path:
        """保存注入日志到指定目录

        Task 11 (F2): 优先从 JSONL 文件汇总，fallback 到内存 dict。

        Args:
            output_dir: 输出目录 (e.g., workspace/.history)

        Returns:
            保存的目录路径
        """
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Task 11: 尝试从 JSONL 汇总补充内存 dict
        jsonl_path = log_dir / "injection_events.jsonl"
        if jsonl_path.exists():
            try:
                jsonl_injections = []
                jsonl_mcp = []
                for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    rtype = record.pop("type", "")
                    if rtype == "injection":
                        jsonl_injections.append(record)
                    elif rtype == "mcp_query":
                        jsonl_mcp.append(record)
                # 如果 JSONL 有更多记录（进程 crash 后恢复），用 JSONL 数据
                if len(jsonl_injections) > len(self._injection_log["injections"]):
                    self._injection_log["injections"] = jsonl_injections
                if len(jsonl_mcp) > len(self._injection_log["mcp_queries"]):
                    self._injection_log["mcp_queries"] = jsonl_mcp
            except Exception:
                pass  # fallback to in-memory dict

        # 1. 保存摘要 JSON
        summary_file = log_dir / "injection_summary.json"
        summary_file.write_text(
            json.dumps(self._injection_log, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 2. 保存各阶段的 prompt 文件
        for inj in self._injection_log["injections"]:
            stage = inj["stage"]
            prompt_content = self._get_cached_prompt(stage)
            if prompt_content:
                filename = f"template_prompt_{stage}.md"
                prompt_file = log_dir / filename
                header = f"""<!--
Injection Log
Stage: {stage}
Timestamp: {inj['timestamp']}
Chars: {inj['char_count']}
Tokens (estimated): {inj['token_estimate']}
Guides Used: {', '.join(inj.get('guides_used', []))}
Sections Included: {', '.join(inj.get('sections_included', []))}
-->

"""
                prompt_file.write_text(header + prompt_content, encoding="utf-8")

        # 3. 保存 MCP 查询日志
        if self._injection_log["mcp_queries"]:
            mcp_file = log_dir / "mcp_queries.json"
            mcp_file.write_text(
                json.dumps(self._injection_log["mcp_queries"], ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        # 4. 保存合规检查反馈
        for fb in self._injection_log["compliance_feedbacks"]:
            idx = self._injection_log["compliance_feedbacks"].index(fb) + 1
            feedback_content = self._get_cached_prompt(f"compliance_{idx:02d}")
            if feedback_content:
                fb_file = log_dir / f"compliance_feedback_{idx:02d}.md"
                header = f"""<!--
Compliance Feedback
Slide: {fb['slide']}
Issues: {fb['issues_count']}
Timestamp: {fb['timestamp']}
-->

"""
                fb_file.write_text(header + feedback_content, encoding="utf-8")

        # 5. 保存 guide 加载详情
        guides_file = log_dir / "guide_loads.json"
        guides_file.write_text(
            json.dumps(self._injection_log["guides"], ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        logger.info(f"Injection logs saved to {log_dir}")
        return log_dir

    def _cache_prompt(self, stage: str, prompt: str) -> None:
        """缓存 prompt 内容"""
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache: dict[str, str] = {}
        self._prompt_cache[stage] = prompt

    def _get_cached_prompt(self, stage: str) -> str:
        """获取缓存的 prompt"""
        if not hasattr(self, "_prompt_cache"):
            self._prompt_cache: dict[str, str] = {}
        return self._prompt_cache.get(stage, "")
