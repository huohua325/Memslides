"""
TemplateComplianceChecker — 验证生成的 HTML 是否符合模板约束 (Stage 7)

检查项：颜色、字体、元素数量、字符长度
可自动修复：颜色违规、字体违规
不可修复（反馈给模型重试）：元素数量不匹配
"""

from __future__ import annotations

import logging
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from memslides.templates.quality import effective_template_palette
from memslides.templates.runtime_state import load_template_runtime_state

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.template_models import TemplateProfile


# ═══════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════

@dataclass
class ComplianceIssue:
    """合规检查问题"""
    type: Literal["color", "font", "element_count", "char_limit", "image", "visual_asset"]
    severity: Literal["error", "warning"]
    message: str
    location: str  # CSS selector 或元素描述
    auto_fixable: bool
    fix_suggestion: str = ""


# ═══════════════════════════════════════════════
# 颜色格式工具函数
# ═══════════════════════════════════════════════

def rgb_to_hex(rgb_str: str) -> str:
    """将 rgb(r,g,b) 或 rgba(r,g,b,a) 转换为 #rrggbb 格式"""
    match = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)', rgb_str)
    if match:
        r, g, b = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"#{r:02x}{g:02x}{b:02x}"
    return rgb_str


def normalize_hex(color: str) -> str:
    """标准化 hex 颜色为 6 位小写格式"""
    color = color.strip().lower()
    if color.startswith("#"):
        hex_part = color[1:]
        if len(hex_part) == 3:
            hex_part = "".join(c * 2 for c in hex_part)
        return f"#{hex_part}"
    return color


def normalize_color(color: str) -> str:
    """将任意颜色格式标准化为 #rrggbb"""
    color = color.strip()
    if color.startswith("rgb"):
        return rgb_to_hex(color)
    return normalize_hex(color)


def _hex_to_rgb(color: str) -> tuple[int, int, int] | None:
    normalized = normalize_color(color)
    if not (normalized.startswith("#") and len(normalized) == 7):
        return None
    try:
        return (
            int(normalized[1:3], 16),
            int(normalized[3:5], 16),
            int(normalized[5:7], 16),
        )
    except ValueError:
        return None


def _relative_luminance(color: str) -> float | None:
    rgb = _hex_to_rgb(color)
    if rgb is None:
        return None
    channels = []
    for value in rgb:
        c = value / 255.0
        channels.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(c1: str, c2: str) -> float:
    l1 = _relative_luminance(c1)
    l2 = _relative_luminance(c2)
    if l1 is None or l2 is None:
        return 21.0
    light = max(l1, l2)
    dark = min(l1, l2)
    return (light + 0.05) / (dark + 0.05)


# ═══════════════════════════════════════════════
# 安全颜色集合
# ═══════════════════════════════════════════════

SAFE_COLORS = {
    "#ffffff", "#fff", "white", "transparent", "inherit", "initial",
    "currentcolor", "#000000", "#000", "black",
}

SAFE_FONTS = {
    "arial", "sans-serif", "serif", "monospace",
    "noto sans cjk sc", "microsoft yahei", "微软雅黑",
    "simhei", "simsun", "kaiti", "fangsong",
    "helvetica", "times new roman", "courier new",
}


# ═══════════════════════════════════════════════
# 正则模式
# ═══════════════════════════════════════════════

# hex 颜色
_HEX_PATTERNS = [
    re.compile(r"(?:^|;)\s*color\s*:\s*(#[0-9a-fA-F]{3,6})", re.IGNORECASE),
    re.compile(r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6})", re.IGNORECASE),
    re.compile(r"border(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6})", re.IGNORECASE),
]

# rgb/rgba 颜色
_RGB_PATTERNS = [
    re.compile(r"(?:^|;)\s*color\s*:\s*(rgba?\([^)]+\))", re.IGNORECASE),
    re.compile(r"background(?:-color)?\s*:\s*(rgba?\([^)]+\))", re.IGNORECASE),
    re.compile(r"border(?:-color)?\s*:\s*(rgba?\([^)]+\))", re.IGNORECASE),
]

# 字体
_FONT_PATTERN = re.compile(r"font-family\s*:\s*([^;]+)", re.IGNORECASE)


def _color_hex_distance(c1: str, c2: str) -> int:
    """计算两个 #rrggbb 颜色的 RGB 通道差值之和（0~765）。

    用于判断两个颜色是否"足够接近"，距离 < 30 时可跳过修复。
    """
    try:
        c1 = normalize_color(c1)
        c2 = normalize_color(c2)
        if not (c1.startswith("#") and len(c1) == 7 and c2.startswith("#") and len(c2) == 7):
            return 999
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        return abs(r1 - r2) + abs(g1 - g2) + abs(b1 - b2)
    except (ValueError, IndexError):
        return 999


def _infer_css_role(location: str) -> str:
    """从 ComplianceIssue.location 推断颜色的 CSS 语义角色。

    Returns: "background" | "text" | "border" | "unknown"
    """
    loc = location.lower()
    if "background" in loc:
        return "background"
    # "color:" 但不是 "background-color:" / "border-color:"
    if re.search(r'(?<!-)color\s*:', loc):
        return "text"
    if "border" in loc:
        return "border"
    return "unknown"


def _extract_colors_from_text(text: str) -> set[str]:
    colors: set[str] = set()
    for match in re.finditer(r"#[0-9a-fA-F]{3,6}\b", str(text or "")):
        colors.add(normalize_color(match.group(0)))
    for match in re.finditer(r"rgba?\([^)]+\)", str(text or ""), flags=re.IGNORECASE):
        colors.add(normalize_color(match.group(0)))
    return colors


def _iter_nested_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(_iter_nested_strings(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            strings.extend(_iter_nested_strings(item))
    return strings


def _normalize_css_properties(values: Any) -> set[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = []
    return {
        str(item or "").strip().lower().replace("_", "-")
        for item in raw_values
        if str(item or "").strip()
    }


def _roles_from_rule_spec(spec: dict[str, Any]) -> set[str]:
    dimension = str(spec.get("dimension", "") or "").strip().lower()
    action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
    css_props = _normalize_css_properties(action.get("css_properties") if isinstance(action, dict) else [])

    roles: set[str] = set()
    if any(prop in {"color", "text-color", "font-color"} for prop in css_props):
        roles.add("text")
    if any(prop in {"background", "background-color"} for prop in css_props):
        roles.add("background")
    if any(prop in {"border", "border-color"} for prop in css_props):
        roles.add("border")

    if not roles:
        if "text_color" in dimension or "font_color" in dimension or dimension.endswith(".color"):
            roles.add("text")
        elif "background" in dimension:
            roles.add("background")
        elif "border" in dimension:
            roles.add("border")

    return roles or {"any"}


def _merge_color_roles(
    target: dict[str, set[str]],
    colors: set[str],
    roles: set[str],
) -> None:
    for color in colors:
        if not color:
            continue
        target.setdefault(color, set()).update(roles or {"any"})


class TemplateComplianceChecker:
    """模板合规检查器"""

    def __init__(
        self,
        skill: "TemplateProfile",
        *,
        user_preference_colors: list[str] | set[str] | tuple[str, ...] | None = None,
        user_preference_rule_specs: list[dict[str, Any]] | None = None,
    ):
        self.skill = skill
        self._template_mode = self._load_template_mode()
        self.dc = deepcopy(skill.design_constraints)
        if self.dc and getattr(self.dc, "color_palette", None):
            self.dc.color_palette = effective_template_palette(
                skill,
                mode=self._template_mode,
            )
        self.si = skill.slide_induction or {}
        self._allowed_colors = self._build_allowed_colors()
        self._allowed_fonts = self._build_allowed_fonts()
        self._page_asset_plan = self._load_page_asset_plan()
        self._user_preference_color_roles = self._build_user_preference_color_roles(
            user_preference_colors=user_preference_colors,
            user_preference_rule_specs=user_preference_rule_specs,
        )

    def _build_allowed_colors(self) -> set[str]:
        """构建允许的颜色集合（全部标准化为 #rrggbb）"""
        colors = set(SAFE_COLORS)
        if not self.dc or not self.dc.color_palette:
            return colors

        cp = self.dc.color_palette
        for attr in (
            "primary",
            "secondary",
            "accent",
            "text",
            "background",
            "surface",
            "primary_text",
            "inverse_text",
            "muted_text",
            "border",
        ):
            val = getattr(cp, attr, None)
            if val:
                colors.add(normalize_color(val))
        additional = getattr(cp, "additional", None) or []
        for c in additional:
            if c:
                colors.add(normalize_color(c))
        return colors

    def _build_allowed_fonts(self) -> set[str]:
        """构建允许的字体集合（全部小写）"""
        fonts = set(SAFE_FONTS)
        if not self.dc or not self.dc.typography:
            return fonts

        tp = self.dc.typography
        if tp.title_font:
            fonts.add(tp.title_font.lower())
        if tp.body_font:
            fonts.add(tp.body_font.lower())
        return fonts

    def _load_template_mode(self) -> str:
        workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
        if not workspace:
            return ""
        state = load_template_runtime_state(workspace)
        return state.mode if state else ""

    def _load_current_modify_context(self) -> dict[str, Any]:
        workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
        if not workspace:
            return {}
        try:
            path = Path(workspace) / ".current_modify_context.json"
            if not path.exists():
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.debug("Failed to load current modify context for template compliance", exc_info=True)
            return {}

    def _build_user_preference_color_roles(
        self,
        *,
        user_preference_colors: list[str] | set[str] | tuple[str, ...] | None,
        user_preference_rule_specs: list[dict[str, Any]] | None,
    ) -> dict[str, set[str]]:
        """Colors explicitly requested by the user outrank template palette membership.

        This waiver is intentionally narrow: it only suppresses palette-whitelist
        replacement/warnings for matching CSS roles. Visual contrast and render
        validation still run normally.
        """
        roles_by_color: dict[str, set[str]] = {}

        explicit_colors = {
            normalize_color(str(color))
            for color in (user_preference_colors or [])
            if str(color or "").strip()
        }
        _merge_color_roles(roles_by_color, explicit_colors, {"any"})

        context = self._load_current_modify_context()
        raw_user_message = str(context.get("raw_user_message", "") or "")
        _merge_color_roles(roles_by_color, _extract_colors_from_text(raw_user_message), {"any"})

        context_colors = context.get("user_preference_colors")
        if isinstance(context_colors, (list, tuple, set)):
            _merge_color_roles(
                roles_by_color,
                {normalize_color(str(color)) for color in context_colors if str(color or "").strip()},
                {"any"},
            )

        specs: list[dict[str, Any]] = []
        for source in (
            user_preference_rule_specs or [],
            context.get("user_preference_rule_specs") or [],
            context.get("active_rule_specs") or [],
        ):
            if isinstance(source, dict):
                specs.append(source)
            elif isinstance(source, list):
                specs.extend(item for item in source if isinstance(item, dict))

        for spec in specs:
            roles = _roles_from_rule_spec(spec)
            texts: list[str] = []
            action = spec.get("action") if isinstance(spec.get("action"), dict) else {}
            if isinstance(action, dict):
                texts.extend(_iter_nested_strings(action))
            for key in (
                "preference",
                "content",
                "normalized_sentence",
                "general_sentence",
                "natural_language",
                "dimension",
            ):
                texts.extend(_iter_nested_strings(spec.get(key)))
            if isinstance(spec.get("semantic_compiler"), dict):
                texts.extend(_iter_nested_strings(spec.get("semantic_compiler")))
            _merge_color_roles(
                roles_by_color,
                {color for text in texts for color in _extract_colors_from_text(text)},
                roles,
            )

        return roles_by_color

    def _is_user_preference_color(self, raw_color: str, context: str) -> bool:
        normalized = normalize_color(raw_color)
        roles = self._user_preference_color_roles.get(normalized)
        if not roles:
            return False
        css_role = _infer_css_role(context)
        return "any" in roles or css_role in roles or css_role == "unknown"

    # ══════════════════════════════════════════════
    # 主入口
    # ══════════════════════════════════════════════

    def check(
        self,
        html_content: str,
        layout_name: str = "",
        file_path: str | Path | None = None,
    ) -> list[ComplianceIssue]:
        """运行全部检查"""
        issues: list[ComplianceIssue] = []
        issues.extend(self._check_colors(html_content))
        issues.extend(self._check_semantic_contrast(html_content))
        issues.extend(self._check_bound_visual_asset(html_content, file_path=file_path))
        issues.extend(self._check_fonts(html_content))
        if layout_name:
            issues.extend(self._check_element_count(html_content, layout_name))
            issues.extend(self._check_char_limits(html_content, layout_name))
        return issues

    # ══════════════════════════════════════════════
    # 自动修复
    # ══════════════════════════════════════════════

    def auto_fix(
        self, html_content: str, issues: list[ComplianceIssue]
    ) -> tuple[str, list[ComplianceIssue]]:
        """修复可自动修复的问题

        Returns:
            tuple[str, list[ComplianceIssue]]: (修复后的 HTML, 已修复的问题列表)
        """
        fixed_html = html_content
        fixed_issues: list[ComplianceIssue] = []

        for issue in issues:
            if not issue.auto_fixable:
                continue

            if issue.type == "color":
                fixed_html = self._fix_color(fixed_html, issue)
                fixed_issues.append(issue)
            elif issue.type == "font":
                fixed_html = self._fix_font(fixed_html, issue)
                fixed_issues.append(issue)

        return fixed_html, fixed_issues

    # ══════════════════════════════════════════════
    # 格式化输出
    # ══════════════════════════════════════════════

    def format_issues_for_retry(self, issues: list[ComplianceIssue]) -> str:
        """格式化问题供模型重试"""
        if not issues:
            return ""

        lines = ["## ⚠️ 模板合规问题\n"]
        for i, issue in enumerate(issues, 1):
            severity_icon = "🔴" if issue.severity == "error" else "🟡"
            lines.append(f"{i}. {severity_icon} **{issue.type}**: {issue.message}")
            if issue.location:
                lines.append(f"   位置: `{issue.location}`")
            if issue.fix_suggestion:
                lines.append(f"   建议: {issue.fix_suggestion}")
        return "\n".join(lines)

    # ══════════════════════════════════════════════
    # 内部检查方法
    # ══════════════════════════════════════════════

    def _check_colors(self, html_content: str) -> list[ComplianceIssue]:
        """检查颜色是否在允许集合中。

        location 字段保存 CSS 属性上下文（如 "background-color: #xxx"），
        供 _fix_color 推断语义角色。
        """
        if not self._allowed_colors or not self.dc or not self.dc.color_palette:
            return []

        issues: list[ComplianceIssue] = []
        # (normalized_color, css_property_context) — 去重
        found_colors: set[tuple[str, str]] = set()

        # 带属性名的颜色正则：捕获 (属性名, 颜色值)
        _prop_color_re = re.compile(
            r'(background(?:-color)?|(?<!-)color|border(?:-color)?)'
            r'\s*:\s*'
            r'(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))',
            re.IGNORECASE,
        )

        # 1. inline style
        style_pattern = re.compile(r'style\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
        for style_match in style_pattern.finditer(html_content):
            style_val = style_match.group(1)
            for m in _prop_color_re.finditer(style_val):
                prop_name = m.group(1).lower()
                raw_color = m.group(2)
                found_colors.add((raw_color, f"{prop_name}: {raw_color}"))

        # 2. <style> 块
        css_block_pattern = re.compile(r'<style[^>]*>(.*?)</style>', re.IGNORECASE | re.DOTALL)
        for css_match in css_block_pattern.finditer(html_content):
            css_content = css_match.group(1)
            for m in _prop_color_re.finditer(css_content):
                prop_name = m.group(1).lower()
                raw_color = m.group(2)
                found_colors.add((raw_color, f"{prop_name}: {raw_color}"))

            # linear-gradient 中的颜色
            gradient_pattern = re.compile(
                r'linear-gradient\s*\([^)]*?(#[0-9a-fA-F]{3,6})[^)]*?\)', re.IGNORECASE
            )
            for grad_match in gradient_pattern.finditer(css_content):
                color = grad_match.group(1)
                found_colors.add((color, f"background: gradient {color}"))

        # 找最近的 palette 颜色（用于 fix_suggestion）
        cp = self.dc.color_palette

        for raw_color, context in found_colors:
            normalized = normalize_color(raw_color)

            if normalized in self._allowed_colors or raw_color.lower() in SAFE_COLORS:
                continue
            if self._is_user_preference_color(raw_color, context):
                continue

            # 颜色相似度跳过：与任一 palette 颜色距离 < 30 时不报错
            skip = False
            for attr in (
                "primary",
                "secondary",
                "accent",
                "text",
                "background",
                "surface",
                "primary_text",
                "inverse_text",
                "muted_text",
                "border",
            ):
                palette_val = getattr(cp, attr, None)
                if palette_val and _color_hex_distance(normalized, palette_val) < 30:
                    skip = True
                    break
            if not skip:
                for add_c in (getattr(cp, "additional", None) or []):
                    if add_c and _color_hex_distance(normalized, add_c) < 30:
                        skip = True
                        break
            if skip:
                continue

            # 根据 CSS 属性推断建议替换色
            role = _infer_css_role(context)
            if role == "background":
                suggestion = cp.surface or cp.background or cp.primary or ""
            elif role == "text":
                suggestion = cp.primary_text or cp.text or cp.inverse_text or cp.primary or ""
            elif role == "border":
                suggestion = cp.border or cp.accent or cp.secondary or cp.primary or ""
            else:
                suggestion = cp.primary or ""

            issues.append(ComplianceIssue(
                type="color",
                severity="error",
                message=f"使用了未定义的颜色 {raw_color}",
                location=context,
                auto_fixable=True,
                fix_suggestion=f"替换为 {suggestion}",
            ))

        return issues

    def _check_semantic_contrast(self, html_content: str) -> list[ComplianceIssue]:
        """Hard-fail obviously unreadable text/background pairs.

        This is intentionally semantic rather than palette-whitelist based: a color
        can be in the template palette and still be wrong when used as body text on
        a similar background.
        """
        issues: list[ComplianceIssue] = []

        def _extract_pair(style_text: str) -> tuple[str, str] | None:
            bg_match = re.search(
                r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))",
                style_text,
                re.IGNORECASE,
            )
            fg_match = re.search(
                r"(?<!-)color\s*:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))",
                style_text,
                re.IGNORECASE,
            )
            if not bg_match or not fg_match:
                return None
            return normalize_color(bg_match.group(1)), normalize_color(fg_match.group(1))

        checked: set[tuple[str, str, str]] = set()
        style_pattern = re.compile(r'style\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
        for style_match in style_pattern.finditer(html_content):
            pair = _extract_pair(style_match.group(1))
            if pair:
                checked.add((pair[0], pair[1], style_match.group(1)[:80]))

        css_rule_pattern = re.compile(r"([^{}]+)\{([^{}]+)\}", re.IGNORECASE | re.DOTALL)
        for block_match in re.finditer(r"<style[^>]*>(.*?)</style>", html_content, re.IGNORECASE | re.DOTALL):
            css_content = block_match.group(1)
            for rule_match in css_rule_pattern.finditer(css_content):
                selector = " ".join(rule_match.group(1).split())[:80]
                pair = _extract_pair(rule_match.group(2))
                if pair:
                    checked.add((pair[0], pair[1], selector))

        for bg, fg, location in checked:
            ratio = _contrast_ratio(bg, fg)
            if ratio < 3.8:
                issues.append(
                    ComplianceIssue(
                        type="color",
                        severity="error",
                        message=(
                            f"正文/背景对比度过低：background {bg} 与 text {fg} "
                            f"contrast={ratio:.2f}"
                        ),
                        location=location,
                        auto_fixable=False,
                        fix_suggestion=(
                            "将正文放在 surface + primary_text 上，或改用 inverse_text / primary_text 的高对比组合。"
                        ),
                    )
                )
        return issues

    def _load_page_asset_plan(self) -> dict:
        workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
        if not workspace:
            return {}
        path = Path(workspace) / "page_asset_plan.json"
        try:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {}
        except Exception:
            logger.debug("Failed to load page_asset_plan.json", exc_info=True)
        return {}

    def _infer_page_index(
        self,
        html_content: str,
        file_path: str | Path | None = None,
    ) -> int | None:
        if file_path is not None:
            path_text = str(file_path)
            for pattern in (r"slide[_-](\d+)\.html", r"slide[_-](\d+)"):
                match = re.search(pattern, path_text, re.IGNORECASE)
                if match:
                    try:
                        return int(match.group(1))
                    except Exception:
                        pass
        for pattern in (
            r"slide[_-](\d+)\.html",
            r"data-slide-index=[\"']?(\d+)",
            r"page[_ -]?(\d+)",
        ):
            match = re.search(pattern, html_content, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    pass
        return None

    def _binding_for_html(
        self,
        html_content: str,
        file_path: str | Path | None = None,
    ) -> dict:
        page_index = self._infer_page_index(html_content, file_path=file_path)
        bindings = self._page_asset_plan.get("bindings", []) or []
        if page_index is not None:
            for binding in bindings:
                try:
                    if int(binding.get("page", 0) or 0) == page_index:
                        return binding
                except Exception:
                    continue
        # Fallback: if there is exactly one required page, apply it. This keeps the
        # checker useful in unit tests without creating false positives for whole decks.
        required = [
            item for item in bindings
            if str(item.get("visual_requirement", "") or "") == "required"
        ]
        return required[0] if len(required) == 1 else {}

    def _check_bound_visual_asset(
        self,
        html_content: str,
        *,
        file_path: str | Path | None = None,
    ) -> list[ComplianceIssue]:
        binding = self._binding_for_html(html_content, file_path=file_path)
        if not binding:
            return []
        if str(binding.get("visual_requirement", "") or "") != "required":
            return []

        kind = str(binding.get("bound_asset_kind", "") or "none")
        asset_path = str(binding.get("bound_asset_path", "") or "")
        formula = str(binding.get("formula_snippet", "") or "")
        issues: list[ComplianceIssue] = []

        if kind in {"figure", "table", "chart"}:
            basename = Path(asset_path).name if asset_path else ""
            has_asset = bool(
                asset_path and (asset_path in html_content or (basename and basename in html_content))
            )
            has_img = bool(re.search(r"<img\b", html_content, re.IGNORECASE))
            if not has_asset or not has_img:
                issues.append(
                    ComplianceIssue(
                        type="visual_asset",
                        severity="error",
                        message=(
                            f"required {kind} page did not render the bound asset "
                            f"`{asset_path}`"
                        ),
                        location=f"page {binding.get('page', '?')}",
                        auto_fixable=False,
                        fix_suggestion="Render the bound asset visibly before adding bullets.",
                    )
                )
        elif kind == "formula":
            normalized_formula = re.sub(r"\s+", "", formula.lower())
            normalized_html = re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html_content).lower())
            has_formula = bool(
                normalized_formula and normalized_formula[:20] in normalized_html
            ) or bool(re.search(r"softmax|qk\^?t|sqrt|√d|attention", html_content, re.IGNORECASE))
            if not has_formula:
                issues.append(
                    ComplianceIssue(
                        type="visual_asset",
                        severity="error",
                        message="required formula page did not render a visible formula anchor",
                        location=f"page {binding.get('page', '?')}",
                        auto_fixable=False,
                        fix_suggestion="Render the formula as visible HTML/math text on a readable surface.",
                    )
                )
        return issues

    def _check_fonts(self, html_content: str) -> list[ComplianceIssue]:
        """检查字体是否在允许集合中"""
        if not self._allowed_fonts or not self.dc or not self.dc.typography:
            return []

        issues: list[ComplianceIssue] = []

        style_pattern = re.compile(r'style\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
        for style_match in style_pattern.finditer(html_content):
            style_val = style_match.group(1)

            for font_match in _FONT_PATTERN.finditer(style_val):
                font_value = font_match.group(1).strip().rstrip(";")
                # 检查字体列表中是否包含任何允许字体
                font_lower = font_value.lower()
                if not any(allowed in font_lower for allowed in self._allowed_fonts):
                    title_font = self.dc.typography.title_font or ""
                    issues.append(ComplianceIssue(
                        type="font",
                        severity="warning",
                        message=f"使用了未定义的字体: {font_value}",
                        location=style_val[:60],
                        auto_fixable=True,
                        fix_suggestion=f"替换为 {title_font}",
                    ))

        return issues

    def _check_element_count(self, html_content: str, layout_name: str) -> list[ComplianceIssue]:
        """检查元素数量是否匹配模板。

        优先统计 <div data-element="..."> 容器（与 design_prompt.md.j2 约定一致）。
        如果 HTML 中没有 data-element 标记，fallback 到统计顶层 <section> 或
        直接子 <div> 中包含文本/图片的数量。
        同时检查偏少和偏多。
        """
        layout_data = self.si.get(layout_name)
        if not layout_data or not isinstance(layout_data, dict):
            return []

        elements = layout_data.get("elements", [])
        expected_text = sum(1 for e in elements if e.get("type") == "text")
        expected_image = sum(1 for e in elements if e.get("type") == "image")

        if expected_text == 0 and expected_image == 0:
            return []

        issues: list[ComplianceIssue] = []

        # 策略 1: data-element 容器（推荐）
        text_elements = re.findall(
            r'<div[^>]*data-element=["\'][^"\']*["\'][^>]*>(?:(?!<div\b).)*?(?:<p|<h[1-6]|<li|<span)',
            html_content, re.IGNORECASE | re.DOTALL,
        )
        image_elements = re.findall(
            r'<div[^>]*data-element=["\'][^"\']*["\'][^>]*>(?:(?!<div\b).)*?<img\b',
            html_content, re.IGNORECASE | re.DOTALL,
        )
        has_data_element = bool(text_elements or image_elements)

        if not has_data_element:
            # 策略 2 fallback: 统计 <section> 子元素
            sections = re.findall(r'<section\b[^>]*>(.*?)</section>', html_content, re.IGNORECASE | re.DOTALL)
            if sections:
                # 每个 section 算一个内容容器
                actual_text = sum(
                    1 for s in sections
                    if re.search(r'<(?:p|h[1-6]|li)\b', s, re.IGNORECASE)
                )
                actual_image = sum(
                    1 for s in sections
                    if re.search(r'<img\b', s, re.IGNORECASE)
                )
            else:
                # 策略 3 fallback: 粗略统计（保留旧逻辑但放宽阈值）
                actual_text = len(re.findall(r"<(?:p|h[1-6])\b[^>]*>", html_content, re.IGNORECASE))
                actual_image = len(re.findall(r"<img\b[^>]*>", html_content, re.IGNORECASE))
        else:
            actual_text = len(text_elements)
            actual_image = len(image_elements)

        # 文本元素偏少检查
        if expected_text > 0 and actual_text < max(1, int(expected_text * 0.5)):
            issues.append(ComplianceIssue(
                type="element_count",
                severity="warning",
                message=f"文本元素偏少：期望 {expected_text}，实际 {actual_text}",
                location="document",
                auto_fixable=False,
                fix_suggestion=f"请确保生成 {expected_text} 个文本元素",
            ))

        # 文本元素偏多检查（仅在有 data-element 标记时检查，避免 fallback 误报）
        if has_data_element and expected_text > 0 and actual_text > expected_text * 2:
            issues.append(ComplianceIssue(
                type="element_count",
                severity="warning",
                message=f"文本元素偏多：期望 {expected_text}，实际 {actual_text}",
                location="document",
                auto_fixable=False,
                fix_suggestion=f"请精简到 {expected_text} 个文本元素",
            ))

        # 图片元素偏少检查
        if expected_image > 0 and actual_image < max(1, int(expected_image * 0.5)):
            issues.append(ComplianceIssue(
                type="element_count",
                severity="warning",
                message=f"图片元素偏少：期望 {expected_image}，实际 {actual_image}",
                location="document",
                auto_fixable=False,
                fix_suggestion=f"请确保生成 {expected_image} 个图片元素",
            ))

        # 图片元素偏多检查
        if has_data_element and expected_image > 0 and actual_image > expected_image * 2:
            issues.append(ComplianceIssue(
                type="element_count",
                severity="warning",
                message=f"图片元素偏多：期望 {expected_image}，实际 {actual_image}",
                location="document",
                auto_fixable=False,
                fix_suggestion=f"请精简到 {expected_image} 个图片元素",
            ))

        return issues

    def _check_char_limits(self, html_content: str, layout_name: str) -> list[ComplianceIssue]:
        """检查字符长度是否符合模板限制。

        Task 13 (C3): 使用 P75 百分位数作为基准，放宽阈值，增加最小基准值和中英文宽度系数。
        """
        if self._template_mode == "style_reference":
            return []

        layout_data = self.si.get(layout_name)
        if not layout_data or not isinstance(layout_data, dict):
            return []

        elements = layout_data.get("elements", [])
        issues: list[ComplianceIssue] = []

        # 计算 P75 百分位数
        def _p75(lengths: list[int]) -> int:
            if not lengths:
                return 0
            s = sorted(lengths)
            idx = int(len(s) * 0.75)
            return s[min(idx, len(s) - 1)]

        # 中英文宽度系数：模板语言 vs 生成内容语言
        lang_coeff = 1.0
        template_lang = self.si.get("language", {}).get("lid", "")
        if template_lang in ("zh", "zh-cn", "zh-tw"):
            # 检测 HTML 中英文字符占比
            text_only = re.sub(r"<[^>]+>", "", html_content)
            cn_chars = sum(1 for c in text_only if '\u4e00' <= c <= '\u9fff')
            total = max(len(text_only.strip()), 1)
            if cn_chars / total < 0.1:
                lang_coeff = 1.5  # 英文字符窄，允许更多字符

        # 提取标题和正文元素的字符基准
        title_limits: list[int] = []
        body_limits: list[int] = []
        for el in elements:
            el_name = el.get("name", "").lower()
            el_data = el.get("data", [])
            if not el_data or el.get("type") != "text":
                continue
            lengths = [len(d) for d in el_data if isinstance(d, str)]
            if not lengths:
                continue
            baseline = _p75(lengths)
            if "title" in el_name:
                title_limits.append(max(baseline, 10))  # 最小基准 10 字
            elif any(k in el_name for k in ("body", "text", "content", "desc")):
                body_limits.append(max(baseline, 30))  # 最小基准 30 字

        title_multiplier = 1.5
        body_multiplier = 2.0
        min_title_baseline = 10
        min_body_baseline = 30
        skip_secondary_title_checks = False

        if self._template_mode == "adaptive_structural":
            # Adaptive mode keeps layout rhythm, but sample-title length should not
            # dominate readable cover/content copy.
            title_multiplier = 2.6
            body_multiplier = 3.0
            min_title_baseline = 18
            min_body_baseline = 48
            skip_secondary_title_checks = True

        # 检查 h1/h2 标题长度
        if title_limits:
            title_limit = int(max(max(title_limits), min_title_baseline) * title_multiplier * lang_coeff)
            for tag_match in re.finditer(r"<(h[12])[^>]*>(.*?)</h[12]>", html_content, re.DOTALL | re.IGNORECASE):
                tag_name = tag_match.group(1).lower()
                if skip_secondary_title_checks and tag_name == "h2":
                    continue
                text = re.sub(r"<[^>]+>", "", tag_match.group(2)).strip()
                if len(text) > title_limit:
                    issues.append(ComplianceIssue(
                        type="char_limit",
                        severity="warning",
                        message=f"标题字符超限：{len(text)}字 > 限制{title_limit}字",
                        location=text[:30],
                        auto_fixable=False,
                        fix_suggestion=f"请将标题缩短到 {title_limit} 字以内",
                    ))

        # 检查 p 正文长度
        if body_limits:
            body_limit = int(max(max(body_limits), min_body_baseline) * body_multiplier * lang_coeff)
            for tag_match in re.finditer(r"<p[^>]*>(.*?)</p>", html_content, re.DOTALL | re.IGNORECASE):
                text = re.sub(r"<[^>]+>", "", tag_match.group(1)).strip()
                if len(text) > body_limit:
                    issues.append(ComplianceIssue(
                        type="char_limit",
                        severity="warning",
                        message=f"正文字符超限：{len(text)}字 > 限制{body_limit}字",
                        location=text[:30],
                        auto_fixable=False,
                        fix_suggestion=f"请将正文缩短到 {body_limit} 字以内",
                    ))

        return issues

    # ══════════════════════════════════════════════
    # 修复方法
    # ══════════════════════════════════════════════

    def _fix_color(self, html: str, issue: ComplianceIssue) -> str:
        """修复颜色违规 — 根据 CSS 语义角色选择替换色。

        background 上下文 → palette.background
        text 上下文       → palette.text
        border 上下文     → palette.accent / secondary
        其他              → palette.primary
        """
        match = re.search(r"(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))", issue.message)
        if not match:
            return html

        old_color = match.group(1)
        if not self.dc or not self.dc.color_palette:
            return html

        cp = self.dc.color_palette
        role = _infer_css_role(issue.location)

        if role == "background":
            new_color = cp.background or cp.primary or ""
        elif role == "text":
            new_color = cp.text or cp.primary or ""
        elif role == "border":
            new_color = cp.accent or cp.secondary or cp.primary or ""
        else:
            new_color = cp.primary or ""

        if not new_color:
            return html

        return html.replace(old_color, new_color)

    def _fix_font(self, html: str, issue: ComplianceIssue) -> str:
        """修复字体违规 — 替换为模板字体"""
        if not self.dc or not self.dc.typography:
            return html

        tp = self.dc.typography
        title_font = tp.title_font or "sans-serif"
        replacement = f"font-family: '{title_font}', 'Noto Sans CJK SC', sans-serif"
        return re.sub(r"font-family\s*:\s*[^;\"']+", replacement, html)
