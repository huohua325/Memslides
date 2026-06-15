"""PPT State Extractor: extract quantifiable parameters from slides.

# DEPRECATED: This module is superseded by MemoryCollector (Task 04) + tool_segment.py
# for Layer 1 deterministic structural filtering. Retained because main.py,
# config_helper.py, compliance.py, and pipeline.py still reference it.
# Remove once all callers are migrated.

Provides:
- ExtractionResult: dataclass for extraction output with diff support
- PPTStateExtractor: HTML/PPTX dual-path parameter extraction (9+2 params)
"""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ExtractionResult:
    """参数提取结果"""

    slide_id: str = ""
    slide_type: Literal["html", "pptx"] = "html"
    params: dict[str, float | int | str] = field(default_factory=dict)
    confidence: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "slide_id": self.slide_id,
            "slide_type": self.slide_type,
            "params": dict(self.params),
            "confidence": dict(self.confidence),
            "errors": list(self.errors),
        }

    def diff(self, other: ExtractionResult) -> dict:
        """计算两个提取结果的差异

        Returns:
            {"param_name": {"before": val, "after": val, "delta": val}, ...}
        """
        diffs = {}
        all_params = set(list(self.params.keys()) + list(other.params.keys()))
        for param in sorted(all_params):
            b = self.params.get(param)
            a = other.params.get(param)
            if b != a:
                delta = None
                if isinstance(b, (int, float)) and isinstance(a, (int, float)):
                    delta = a - b
                diffs[param] = {"before": b, "after": a, "delta": delta}
        return diffs


# Named color map (subset)
_NAMED_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "transparent": (255, 255, 255),
}


class PPTStateExtractor:
    """幻灯片参数提取器

    支持11个可量化参数（9个已实现，2个待补充）
    """

    SUPPORTED_PARAMS = {
        "background_brightness",
        "avg_font_size",
        "title_font_size",
        "text_count",
        "image_count",
        "element_count",
        "content_density",
        "dominant_color_hue",
        "text_image_ratio",
    }
    PENDING_PARAMS = {"color_saturation", "color_contrast_ratio"}

    def extract_from_html(self, html_path: Path) -> ExtractionResult:
        """从HTML幻灯片提取参数"""
        from bs4 import BeautifulSoup

        html_content = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_content, "html.parser")
        result = ExtractionResult(
            slide_id=html_path.stem,
            slide_type="html",
        )

        # 1. background_brightness
        try:
            bg_color = self._extract_background_color(soup)
            if bg_color:
                brightness = self._color_to_brightness(bg_color)
                if brightness is not None:
                    result.params["background_brightness"] = round(brightness, 4)
                    result.confidence["background_brightness"] = "high"
        except Exception as e:
            result.errors.append(f"background_brightness: {e}")

        # 2-3. font sizes
        try:
            font_sizes = self._extract_font_sizes(soup)
            if font_sizes:
                result.params["avg_font_size"] = round(
                    sum(font_sizes) / len(font_sizes), 1
                )
                result.confidence["avg_font_size"] = "high"
            title_size = self._extract_title_font_size(soup)
            if title_size:
                result.params["title_font_size"] = round(title_size, 1)
                result.confidence["title_font_size"] = "high"
        except Exception as e:
            result.errors.append(f"font_size: {e}")

        # 4. text_count
        try:
            text = soup.get_text(strip=True)
            result.params["text_count"] = len(text)
            result.confidence["text_count"] = "high"
        except Exception as e:
            result.errors.append(f"text_count: {e}")

        # 5. image_count
        try:
            result.params["image_count"] = len(soup.find_all("img"))
            result.confidence["image_count"] = "high"
        except Exception as e:
            result.errors.append(f"image_count: {e}")

        # 6. element_count
        try:
            safe_div = soup.select_one(".safe")
            if safe_div:
                result.params["element_count"] = len(
                    safe_div.find_all(recursive=False)
                )
                result.confidence["element_count"] = "high"
            else:
                # Fallback: count body direct children
                body = soup.find("body")
                if body:
                    result.params["element_count"] = len(
                        body.find_all(recursive=False)
                    )
                    result.confidence["element_count"] = "medium"
        except Exception as e:
            result.errors.append(f"element_count: {e}")

        # 7. content_density
        slide_area = 1280 * 720
        if "text_count" in result.params:
            result.params["content_density"] = round(
                result.params["text_count"] / slide_area, 6
            )
            result.confidence["content_density"] = "medium"

        # 8. dominant_color_hue
        try:
            hue = self._extract_dominant_color_hue(soup)
            if hue is not None:
                result.params["dominant_color_hue"] = hue
                result.confidence["dominant_color_hue"] = "medium"
        except Exception as e:
            result.errors.append(f"dominant_color_hue: {e}")

        # 9. text_image_ratio
        if "text_count" in result.params and "image_count" in result.params:
            result.params["text_image_ratio"] = round(
                result.params["text_count"] / max(result.params["image_count"], 1),
                1,
            )
            result.confidence["text_image_ratio"] = "medium"

        return result

    def extract_from_pptx(self, slide_page) -> ExtractionResult:
        """从PPTX SlidePage对象提取参数（TemplatePlanner路径）"""
        result = ExtractionResult(slide_type="pptx")
        # Placeholder — will be implemented when the new template analyzer pipeline is ready
        return result

    def extract(self, source, slide_type: str = "html") -> ExtractionResult:
        """统一提取入口"""
        if slide_type == "html":
            return self.extract_from_html(Path(source))
        elif slide_type == "pptx":
            return self.extract_from_pptx(source)
        raise ValueError(f"Unknown slide_type: {slide_type}")

    # === 内部方法 ===

    def _extract_background_color(self, soup) -> str | None:
        """提取背景色（CSS解析）"""
        # Check inline style on .slide or body
        for selector in [".slide", "body"]:
            elem = soup.select_one(selector)
            if elem and elem.get("style"):
                bg = self._parse_css_property(elem["style"], "background-color")
                if bg:
                    return bg
                bg = self._parse_css_property(elem["style"], "background")
                if bg:
                    return bg
        # Check <style> tags
        for style in soup.find_all("style"):
            text = style.string or ""
            match = re.search(
                r"(?:body|\.slide)\s*\{[^}]*background(?:-color)?\s*:\s*([^;]+)",
                text,
            )
            if match:
                return match.group(1).strip()
        return None

    @staticmethod
    def _color_to_brightness(color_str: str) -> float | None:
        """颜色字符串转亮度(0-1)

        支持: #RRGGBB, #RGB, rgb(r,g,b), rgba(r,g,b,a), 颜色名
        使用相对亮度公式: L = 0.2126*R + 0.7152*G + 0.0722*B
        """
        color_str = color_str.strip().lower()

        # Named color
        if color_str in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[color_str]
            return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)

        # #RRGGBB or #RGB
        hex_match = re.match(r"#([0-9a-f]{3,8})", color_str)
        if hex_match:
            hex_val = hex_match.group(1)
            if len(hex_val) == 3:
                r = int(hex_val[0] * 2, 16)
                g = int(hex_val[1] * 2, 16)
                b = int(hex_val[2] * 2, 16)
            elif len(hex_val) >= 6:
                r = int(hex_val[0:2], 16)
                g = int(hex_val[2:4], 16)
                b = int(hex_val[4:6], 16)
            else:
                return None
            return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)

        # rgb(r, g, b) or rgba(r, g, b, a)
        rgb_match = re.match(r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color_str)
        if rgb_match:
            r = int(rgb_match.group(1))
            g = int(rgb_match.group(2))
            b = int(rgb_match.group(3))
            return 0.2126 * (r / 255) + 0.7152 * (g / 255) + 0.0722 * (b / 255)

        return None

    @staticmethod
    def _color_to_hue(color_str: str) -> int | None:
        """颜色字符串转色相(0-360)"""
        color_str = color_str.strip().lower()

        r = g = b = None

        if color_str in _NAMED_COLORS:
            r, g, b = _NAMED_COLORS[color_str]
        else:
            hex_match = re.match(r"#([0-9a-f]{3,8})", color_str)
            if hex_match:
                hv = hex_match.group(1)
                if len(hv) == 3:
                    r, g, b = int(hv[0] * 2, 16), int(hv[1] * 2, 16), int(hv[2] * 2, 16)
                elif len(hv) >= 6:
                    r, g, b = int(hv[0:2], 16), int(hv[2:4], 16), int(hv[4:6], 16)
            else:
                rgb_match = re.match(
                    r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)", color_str
                )
                if rgb_match:
                    r = int(rgb_match.group(1))
                    g = int(rgb_match.group(2))
                    b = int(rgb_match.group(3))

        if r is None:
            return None

        h, _, _ = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        return round(h * 360)

    def _extract_font_sizes(self, soup) -> list[float]:
        """提取所有字号"""
        sizes = []
        for elem in soup.find_all(style=True):
            fs = self._parse_css_property(elem["style"], "font-size")
            if fs:
                num = self._parse_px_value(fs)
                if num:
                    sizes.append(num)
        # Also check <style> tags for font-size declarations
        for style in soup.find_all("style"):
            text = style.string or ""
            for match in re.finditer(r"font-size\s*:\s*([^;]+)", text):
                num = self._parse_px_value(match.group(1).strip())
                if num:
                    sizes.append(num)
        return sizes

    def _extract_title_font_size(self, soup) -> float | None:
        """提取标题字号（.title类或h1/h2或最大字号）"""
        # Try .title class
        title_elem = soup.select_one(".title")
        if title_elem and title_elem.get("style"):
            fs = self._parse_css_property(title_elem["style"], "font-size")
            if fs:
                return self._parse_px_value(fs)

        # Try h1, h2
        for tag in ["h1", "h2"]:
            elem = soup.find(tag)
            if elem and elem.get("style"):
                fs = self._parse_css_property(elem["style"], "font-size")
                if fs:
                    return self._parse_px_value(fs)

        # Fallback: max font size
        sizes = self._extract_font_sizes(soup)
        return max(sizes) if sizes else None

    def _extract_dominant_color_hue(self, soup) -> int | None:
        """提取主色调色相"""
        colors = []
        # Collect all color mentions
        for elem in soup.find_all(style=True):
            for prop in ["color", "background-color", "background"]:
                val = self._parse_css_property(elem["style"], prop)
                if val:
                    hue = self._color_to_hue(val)
                    if hue is not None:
                        colors.append(hue)
        # Also from <style> tags
        for style in soup.find_all("style"):
            text = style.string or ""
            for match in re.finditer(r"(?:color|background(?:-color)?)\s*:\s*([^;]+)", text):
                hue = self._color_to_hue(match.group(1).strip())
                if hue is not None:
                    colors.append(hue)

        if not colors:
            return None
        # Most common hue (bucket by 30-degree intervals)
        from collections import Counter

        buckets = [h // 30 * 30 for h in colors]
        most_common = Counter(buckets).most_common(1)
        return most_common[0][0] if most_common else None

    @staticmethod
    def _parse_css_property(style_str: str, prop: str) -> str | None:
        """从inline style中提取CSS属性值"""
        pattern = re.compile(
            rf"(?:^|;)\s*{re.escape(prop)}\s*:\s*([^;]+)", re.IGNORECASE
        )
        match = pattern.search(style_str)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_px_value(value: str) -> float | None:
        """解析像素值（如 '24px', '1.5em', '16pt'）"""
        match = re.match(r"([\d.]+)\s*(px|pt|em|rem|%)?", value.strip())
        if match:
            num = float(match.group(1))
            unit = match.group(2) or "px"
            if unit == "pt":
                return num * 1.333  # pt → px approximate
            elif unit in ("em", "rem"):
                return num * 16  # assume base 16px
            elif unit == "%":
                return num * 0.16  # rough approximation
            return num
        return None
