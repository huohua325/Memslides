"""Implementation details."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from ..compliance.template_checker import (
    SAFE_COLORS,
    SAFE_FONTS,
    _FONT_PATTERN,
    _HEX_PATTERNS,
    _RGB_PATTERNS,
    normalize_color,
)
from ..core.models import SlideVisualFingerprint

logger = logging.getLogger(__name__)


_SLIDE_AREA_WAN_PX = 92.16

# ═══════════════════════════════════════════════
# Layer 1: SlideVisualFingerprintExtractor
# ═══════════════════════════════════════════════


_GRADIENT_PATTERN = re.compile(r"linear-gradient|radial-gradient", re.IGNORECASE)
_BG_IMAGE_PATTERN = re.compile(r"background(?:-image)?\s*:\s*url\(", re.IGNORECASE)
_BG_COLOR_PATTERN = re.compile(
    r"background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6}|rgba?\([^)]+\))",
    re.IGNORECASE,
)


_FONT_SIZE_PATTERN = re.compile(r"font-size\s*:\s*(\d+(?:\.\d+)?)\s*(?:px|pt|em|rem)", re.IGNORECASE)


_WIDTH_PATTERN = re.compile(r"width\s*:\s*(\d+(?:\.\d+)?)\s*(?:px|%)", re.IGNORECASE)
_HEIGHT_PATTERN = re.compile(r"height\s*:\s*(\d+(?:\.\d+)?)\s*(?:px|%)", re.IGNORECASE)


_STYLE_ATTR_PATTERN = re.compile(r'style\s*=\s*"([^"]*)"', re.IGNORECASE)
_STYLE_TAG_PATTERN = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)


class SlideVisualFingerprintExtractor:
    """Implementation details."""

    def extract(self, html_path: str | Path, layout_name: str = "") -> SlideVisualFingerprint:
        """Implementation details."""
        html_path = Path(html_path)
        content = html_path.read_text(encoding="utf-8")

        fp = SlideVisualFingerprint(
            html_path=str(html_path),
            layout_name=layout_name,
        )


        all_styles = self._collect_all_styles(content)


        fp.colors_used, fp.background_color, fp.background_type = self._extract_colors(
            content, all_styles,
        )


        fp.fonts_used, fp.title_font, fp.title_size, fp.body_font, fp.body_size = (
            self._extract_fonts(content, all_styles)
        )


        fp.element_counts = self._count_elements(content)
        fp.content_area_ratio, fp.image_text_ratio = self._estimate_area_ratios(content)
        fp.has_bullet_points = "<li" in content.lower()
        fp.bullet_point_count = content.lower().count("<li")


        fp.title_char_count, fp.body_char_count = self._count_text_chars(content)
        total_chars = fp.title_char_count + fp.body_char_count
        fp.total_text_density = total_chars / _SLIDE_AREA_WAN_PX if _SLIDE_AREA_WAN_PX > 0 else 0

        return fp



    def _extract_colors(
        self, html: str, all_styles: str,
    ) -> tuple[list[str], str, str]:
        """Implementation details."""
        colors: set[str] = set()


        for pattern in _HEX_PATTERNS:
            for m in pattern.finditer(all_styles):
                normalized = normalize_color(m.group(1))
                if normalized not in SAFE_COLORS:
                    colors.add(normalized)

        for pattern in _RGB_PATTERNS:
            for m in pattern.finditer(all_styles):
                normalized = normalize_color(m.group(1))
                if normalized not in SAFE_COLORS:
                    colors.add(normalized)


        bg_color = ""
        bg_match = _BG_COLOR_PATTERN.search(all_styles)
        if bg_match:
            bg_color = normalize_color(bg_match.group(1))


        bg_type = "none"
        if _GRADIENT_PATTERN.search(all_styles):
            bg_type = "gradient"
        elif _BG_IMAGE_PATTERN.search(html):
            bg_type = "image"
        elif bg_color:
            bg_type = "solid"

        return sorted(colors), bg_color, bg_type



    def _extract_fonts(
        self, html: str, all_styles: str,
    ) -> tuple[list[tuple[str, float]], str, float, str, float]:
        """Implementation details."""
        fonts: list[tuple[str, float]] = []
        title_font, title_size = "", 0.0
        body_font, body_size = "", 0.0



        for style_match in _STYLE_ATTR_PATTERN.finditer(html):
            style_val = style_match.group(1)
            font_match = _FONT_PATTERN.search(style_val)
            size_match = _FONT_SIZE_PATTERN.search(style_val)
            if font_match:
                font_name = font_match.group(1).strip().strip("'\"").split(",")[0].strip().strip("'\"")
                font_size = float(size_match.group(1)) if size_match else 0.0
                if font_name.lower() not in SAFE_FONTS:
                    fonts.append((font_name, font_size))


        for style_tag_match in _STYLE_TAG_PATTERN.finditer(html):
            css_content = style_tag_match.group(1)

            for rule_match in re.finditer(r"([^{}]+)\{([^{}]+)\}", css_content):
                selector = rule_match.group(1).strip()
                rule_body = rule_match.group(2)

                font_match = _FONT_PATTERN.search(rule_body)
                size_match = _FONT_SIZE_PATTERN.search(rule_body)
                if font_match:
                    font_name = font_match.group(1).strip().strip("'\"").split(",")[0].strip().strip("'\"")
                    font_size = float(size_match.group(1)) if size_match else 0.0

                    if font_name.lower() not in SAFE_FONTS:
                        fonts.append((font_name, font_size))


                    sel_lower = selector.lower()
                    if any(kw in sel_lower for kw in (".title", "h1", "h2", ".heading")):
                        title_font = font_name
                        title_size = font_size
                    elif any(kw in sel_lower for kw in (".content", ".body", "p", ".text")):
                        body_font = font_name
                        body_size = font_size


        if not title_font and fonts:

            largest = max(fonts, key=lambda x: x[1], default=("", 0))
            if largest[1] > 0:
                title_font, title_size = largest

        return fonts, title_font, title_size, body_font, body_size



    @staticmethod
    def _count_elements(html: str) -> dict[str, int]:
        """Implementation details."""
        html_lower = html.lower()
        return {
            "text": (
                html_lower.count("<p") + html_lower.count("<h1")
                + html_lower.count("<h2") + html_lower.count("<h3")
                + html_lower.count("<h4") + html_lower.count("<h5")
                + html_lower.count("<h6") + html_lower.count("<span")
                + html_lower.count("<li")
            ),
            "image": html_lower.count("<img"),
            "table": html_lower.count("<table"),
            "chart": html_lower.count("<svg") + html_lower.count("<canvas"),
        }



    @staticmethod
    def _estimate_area_ratios(html: str) -> tuple[float, float]:
        """Implementation details."""
        total_content_area = 0.0
        image_area = 0.0
        text_area = 0.0


        slide_w, slide_h = 1280.0, 720.0
        slide_area = slide_w * slide_h

        for style_match in _STYLE_ATTR_PATTERN.finditer(html):
            style_val = style_match.group(1)
            w_match = _WIDTH_PATTERN.search(style_val)
            h_match = _HEIGHT_PATTERN.search(style_val)

            if w_match and h_match:
                w_str, h_str = w_match.group(0), h_match.group(0)
                w_val = float(w_match.group(1))
                h_val = float(h_match.group(1))


                if "%" in w_str:
                    w_val = w_val / 100 * slide_w
                if "%" in h_str:
                    h_val = h_val / 100 * slide_h

                elem_area = w_val * h_val
                total_content_area += elem_area


                pos = style_match.start()
                preceding = html[max(0, pos - 100):pos].lower()
                if "<img" in preceding:
                    image_area += elem_area
                else:
                    text_area += elem_area

        content_ratio = min(1.0, total_content_area / slide_area) if slide_area > 0 else 0.0
        img_text_ratio = image_area / text_area if text_area > 0 else 0.0

        return content_ratio, img_text_ratio



    @staticmethod
    def _count_text_chars(html: str) -> tuple[int, int]:
        """Implementation details."""

        clean = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        clean = re.sub(r"<script[^>]*>.*?</script>", "", clean, flags=re.DOTALL | re.IGNORECASE)

        title_chars = 0
        body_chars = 0


        for m in re.finditer(r"<(h[1-6]|[^>]*class\s*=\s*\"[^\"]*title[^\"]*\"[^>]*)>(.*?)</", clean, re.DOTALL | re.IGNORECASE):
            text = re.sub(r"<[^>]+>", "", m.group(2))
            title_chars += len(text.strip())


        for m in re.finditer(r"<(p|li|span)[^>]*>(.*?)</\1>", clean, re.DOTALL | re.IGNORECASE):
            text = re.sub(r"<[^>]+>", "", m.group(2))
            body_chars += len(text.strip())

        return title_chars, body_chars



    @staticmethod
    def _collect_all_styles(html: str) -> str:
        """Implementation details."""
        parts: list[str] = []
        for m in _STYLE_ATTR_PATTERN.finditer(html):
            parts.append(m.group(1))
        for m in _STYLE_TAG_PATTERN.finditer(html):
            parts.append(m.group(1))
        return "\n".join(parts)








# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

VLM_VISUAL_PREFERENCE_PROMPT = """你是一位 PPT 视觉设计分析专家。请分析以下 {n} 张用户最近制作的 PPT 幻灯片截图，提取其视觉偏好模式。

## 已知的确定性数据（从 HTML 结构提取，无需重复分析）
{statistical_summary}

## 你需要分析的主观维度（HTML 无法表达的信息）

请**仅**关注以下 PPT 特有的主观视觉维度：

1. **整体视觉风格**：商务简约/科技感/学术严谨/创意活泼/其他（描述关键词）
2. **配色和谐度**：颜色搭配是否和谐？是否有明确的色彩层次（主色/辅色/点缀色）？
3. **信息层次**：标题-正文-辅助信息的视觉层次是否清晰？字号/颜色/位置的层次感如何？
4. **视觉重心**：内容偏左上/居中/分散？图片和文字的排列方式？
5. **留白与呼吸感**：留白多还是内容填满？整体紧凑还是宽松？

## 输出格式
返回 JSON 数组，每个元素：
```json
[{{"dimension": "visual.style|theme.color_harmony|layout.hierarchy|layout.visual_center|layout.spacing_preference",
   "observation": "观察到的一致模式（<30字）",
   "confidence": 0.3-0.8}}]
```

重要：
- 只输出跨多张幻灯片**一致**的模式，单张的特殊情况不算偏好
- confidence 不要超过 0.8（视觉分析天然比确定性数据低）
- 如果没有明显一致模式，返回空数组 []
"""
