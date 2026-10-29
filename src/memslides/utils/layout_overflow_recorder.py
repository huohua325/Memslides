"""Implementation details."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlideLayoutStats:
    """Implementation details."""
    slide_name: str           # "slide_06"
    char_count: int
    image_count: int
    image_total_area_px: int
    font_size_max: float
    font_size_min: float
    font_size_avg: float
    element_count: int
    has_table: bool
    has_code_block: bool
    shrink_attempts: int
    final_scale: float


def extract_layout_stats(html_path: Path, shrink_attempts: int = 3, scale_factor: float = 0.9) -> SlideLayoutStats:
    """Implementation details."""
    content = html_path.read_text(encoding='utf-8')


    text_only = re.sub(r'<[^>]+>', ' ', content)
    text_only = re.sub(r'\s+', ' ', text_only).strip()
    char_count = len(text_only)


    img_matches = re.findall(r'<img[^>]+>', content, re.IGNORECASE)
    image_count = len(img_matches)
    image_total_area = 0
    for img in img_matches:

        w_match = re.search(r'width[:\s]*(\d+)', img)
        h_match = re.search(r'height[:\s]*(\d+)', img)
        if w_match and h_match:
            image_total_area += int(w_match.group(1)) * int(h_match.group(1))


    font_sizes = []
    for match in re.finditer(r'font-size\s*:\s*([\d.]+)(pt|px)', content):
        size = float(match.group(1))
        unit = match.group(2)
        if unit == 'px':
            size = size * 0.75  # px to pt
        font_sizes.append(size)


    if font_sizes:
        font_size_max = max(font_sizes)
        font_size_min = min(font_sizes)
        font_size_avg = sum(font_sizes) / len(font_sizes)
    else:
        font_size_max = font_size_min = font_size_avg = 0.0


    element_count = len(re.findall(r'<\w+', content))


    has_table = '<table' in content.lower()
    has_code_block = '<pre' in content.lower() or '<code' in content.lower()

    return SlideLayoutStats(
        slide_name=html_path.stem,
        char_count=char_count,
        image_count=image_count,
        image_total_area_px=image_total_area,
        font_size_max=round(font_size_max, 1),
        font_size_min=round(font_size_min, 1),
        font_size_avg=round(font_size_avg, 1),
        element_count=element_count,
        has_table=has_table,
        has_code_block=has_code_block,
        shrink_attempts=shrink_attempts,
        final_scale=round(scale_factor ** shrink_attempts, 3),
    )


def format_overflow_lesson(stats: SlideLayoutStats) -> str:
    """Implementation details."""
    parts = [f"{stats.slide_name} content exceeds the slide boundary:"]


    if stats.char_count > 500:
        parts.append(f"{stats.char_count} characters (recommended <500)")
    else:
        parts.append(f"{stats.char_count} characters")


    if stats.image_count > 0:
        parts.append(f"{stats.image_count} images")


    if stats.font_size_max > 0:
        parts.append(f"font size {stats.font_size_min:.0f}-{stats.font_size_max:.0f}pt")


    extras = []
    if stats.has_table:
        extras.append("contains a table")
    if stats.has_code_block:
        extras.append("contains a code block")
    if extras:
        parts.append(",".join(extras))


    parts.append(f"{stats.shrink_attempts} shrink attempts (scale={stats.final_scale}) still overflow")

    return " | ".join(parts)


def format_workaround(stats: SlideLayoutStats) -> str:
    """Implementation details."""
    suggestions = []

    if stats.char_count > 600:
        suggestions.append("Reduce slide text to 500 characters or fewer")
    if stats.image_count > 2:
        suggestions.append("Reduce the image count or image dimensions")
    if stats.font_size_max > 24:
        suggestions.append(f"Reduce the maximum font size (currently {stats.font_size_max:.0f}pt; target 18pt)")
    if stats.has_table:
        suggestions.append("Simplify the table or split it across slides")
    if stats.has_code_block:
        suggestions.append("Use a smaller font for code blocks or collapse them")

    if not suggestions:
        suggestions.append("Check whether an absolutely positioned element exceeds the slide boundary")

    return "；".join(suggestions)


async def record_layout_overflow(
    html_path: Path,
    session_id: str,
    experience_writer: Any,
    shrink_attempts: int = 3,
    scale_factor: float = 0.9,
) -> Any | None:
    """Implementation details."""
    if not experience_writer:
        logger.debug("No experience_writer provided, skipping overflow recording")
        return None

    try:
        stats = extract_layout_stats(html_path, shrink_attempts, scale_factor)
        lesson = format_overflow_lesson(stats)
        workaround = format_workaround(stats)

        trace = await experience_writer._write_typed_trace(
            session_id=session_id,
            task=f"Tool limitation: html_to_pptx — {stats.slide_name} layout overflow",
            tools=["html_to_pptx", "write_html_file"],
            outcome="partial",
            lesson=f"{lesson}\nWorkaround: {workaround}",
            scenarios=["tool_limitation", "tool_html_to_pptx", "layout_overflow", f"slide_{stats.slide_name}"],
            experience_type="tool_limitation",
            confidence=0.9,
        )

        logger.info(
            "Recorded layout overflow for %s: %d chars, %d images, scale=%.3f",
            stats.slide_name, stats.char_count, stats.image_count, stats.final_scale,
        )
        return trace

    except Exception as e:
        logger.warning("Failed to record layout overflow (non-fatal): %s", e)
        return None


async def record_dimension_mismatch(
    html_path: Path,
    session_id: str,
    experience_writer: Any,
    error_msg: str,
) -> Any | None:
    """Implementation details."""
    if not experience_writer:
        logger.debug("No experience_writer provided, skipping dimension mismatch recording")
        return None

    try:
        slide_name = html_path.stem


        match = re.search(
            r'HTML dimensions \(([\d.]+)" × ([\d.]+)"\).*?presentation layout \(([\d.]+)" × ([\d.]+)"\)',
            error_msg
        )
        if match:
            html_w, html_h = float(match.group(1)), float(match.group(2))
            layout_w, layout_h = float(match.group(3)), float(match.group(4))
            lesson = (
                f"{slide_name} uses the wrong aspect ratio: "
                f"generated {html_w:.1f}\"x{html_h:.1f}\" ({html_w/html_h:.2f}:1), "
                f"expected {layout_w:.1f}\"x{layout_h:.1f}\" ({layout_w/layout_h:.2f}:1)."
                f"\nWorkaround: regenerate the slide with an HTML body matching the target aspect ratio."
            )
        else:
            lesson = (
                f"{slide_name} dimensions do not match the target layout."
                f"\nWorkaround: regenerate the slide with an HTML body matching the target aspect ratio."
            )

        trace = await experience_writer._write_typed_trace(
            session_id=session_id,
            task=f"Tool limitation: html_to_pptx — {slide_name} aspect ratio mismatch",
            tools=["html_to_pptx", "write_html_file"],
            outcome="failed",
            lesson=lesson,
            scenarios=["tool_limitation", "tool_html_to_pptx", "aspect_ratio_mismatch"],
            experience_type="tool_limitation",
            confidence=0.95,
        )

        logger.info("Recorded dimension mismatch for %s", slide_name)
        return trace

    except Exception as e:
        logger.warning("Failed to record dimension mismatch (non-fatal): %s", e)
        return None
