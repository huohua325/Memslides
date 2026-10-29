"""LayoutOverflowRecorder — 记录布局溢出到 ExperienceTrace

当 html_to_pptx 触发自动缩放后仍失败时，提取 HTML 参数并写入 tool_limitations 类型经验。
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SlideLayoutStats:
    """从溢出 HTML 提取的布局参数"""
    slide_name: str           # "slide_06"
    char_count: int           # 文字总数
    image_count: int          # 图片数量
    image_total_area_px: int  # 图片总面积 (px²)
    font_size_max: float      # 最大字号 (pt)
    font_size_min: float      # 最小字号 (pt)
    font_size_avg: float      # 平均字号 (pt)
    element_count: int        # DOM 元素数
    has_table: bool           # 是否含表格
    has_code_block: bool      # 是否含代码块
    shrink_attempts: int      # 缩放尝试次数
    final_scale: float        # 最终缩放比例 (e.g., 0.729 = 0.9^3)


def extract_layout_stats(html_path: Path, shrink_attempts: int = 3, scale_factor: float = 0.9) -> SlideLayoutStats:
    """从 HTML 文件提取布局统计信息
    
    Args:
        html_path: HTML 文件路径
        shrink_attempts: 已执行的缩放次数
        scale_factor: 每次缩放比例
    """
    content = html_path.read_text(encoding='utf-8')
    
    # 提取文字（去除 HTML 标签）
    text_only = re.sub(r'<[^>]+>', ' ', content)
    text_only = re.sub(r'\s+', ' ', text_only).strip()
    char_count = len(text_only)
    
    # 提取图片数量和尺寸
    img_matches = re.findall(r'<img[^>]+>', content, re.IGNORECASE)
    image_count = len(img_matches)
    image_total_area = 0
    for img in img_matches:
        # 尝试从 style 或 width/height 属性提取尺寸
        w_match = re.search(r'width[:\s]*(\d+)', img)
        h_match = re.search(r'height[:\s]*(\d+)', img)
        if w_match and h_match:
            image_total_area += int(w_match.group(1)) * int(h_match.group(1))
    
    # 提取字号
    font_sizes = []
    for match in re.finditer(r'font-size\s*:\s*([\d.]+)(pt|px)', content):
        size = float(match.group(1))
        unit = match.group(2)
        if unit == 'px':
            size = size * 0.75  # px to pt
        font_sizes.append(size)
    
    # 计算字号统计
    if font_sizes:
        font_size_max = max(font_sizes)
        font_size_min = min(font_sizes)
        font_size_avg = sum(font_sizes) / len(font_sizes)
    else:
        font_size_max = font_size_min = font_size_avg = 0.0
    
    # 统计元素数量
    element_count = len(re.findall(r'<\w+', content))
    
    # 检测特殊元素
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
    """格式化溢出教训，供 LLM/用户理解"""
    parts = [f"{stats.slide_name} 内容超出边界:"]
    
    # 文字量
    if stats.char_count > 500:
        parts.append(f"文字{stats.char_count}字(建议<500)")
    else:
        parts.append(f"文字{stats.char_count}字")
    
    # 图片
    if stats.image_count > 0:
        parts.append(f"{stats.image_count}张图")
    
    # 字号
    if stats.font_size_max > 0:
        parts.append(f"字号{stats.font_size_min:.0f}-{stats.font_size_max:.0f}pt")
    
    # 特殊元素
    extras = []
    if stats.has_table:
        extras.append("含表格")
    if stats.has_code_block:
        extras.append("含代码块")
    if extras:
        parts.append(",".join(extras))
    
    # 缩放结果
    parts.append(f"缩放{stats.shrink_attempts}次(scale={stats.final_scale})仍溢出")
    
    return " | ".join(parts)


def format_workaround(stats: SlideLayoutStats) -> str:
    """根据统计数据生成针对性的 workaround 建议"""
    suggestions = []
    
    if stats.char_count > 600:
        suggestions.append("减少单页文字量至500字以内")
    if stats.image_count > 2:
        suggestions.append("减少图片数量或缩小图片尺寸")
    if stats.font_size_max > 24:
        suggestions.append(f"降低最大字号(当前{stats.font_size_max:.0f}pt→18pt)")
    if stats.has_table:
        suggestions.append("简化表格或分页展示")
    if stats.has_code_block:
        suggestions.append("代码块使用更小字号或折叠")
    
    if not suggestions:
        suggestions.append("检查是否有绝对定位元素超出边界")
    
    return "；".join(suggestions)


async def record_layout_overflow(
    html_path: Path,
    session_id: str,
    experience_writer: Any,
    shrink_attempts: int = 3,
    scale_factor: float = 0.9,
) -> Any | None:
    """记录布局溢出到 ExperienceTrace
    
    Args:
        html_path: 溢出的 HTML 文件路径
        session_id: 当前 session ID
        experience_writer: ExperienceTraceWriter 实例
        shrink_attempts: 已执行的缩放次数
        scale_factor: 每次缩放比例
        
    Returns:
        创建的 ExperienceTrace 或 None
    """
    if not experience_writer:
        logger.debug("No experience_writer provided, skipping overflow recording")
        return None
    
    try:
        stats = extract_layout_stats(html_path, shrink_attempts, scale_factor)
        lesson = format_overflow_lesson(stats)
        workaround = format_workaround(stats)
        
        trace = await experience_writer._write_typed_trace(
            session_id=session_id,
            task=f"Tool limitation: html_to_pptx — {stats.slide_name} 布局溢出",
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
    """记录尺寸不匹配错误到 ExperienceTrace
    
    尺寸不匹配是 Agent 生成了错误 aspect ratio 的 HTML，需要重新生成该页。
    
    Args:
        html_path: 出错的 HTML 文件路径
        session_id: 当前 session ID
        experience_writer: ExperienceTraceWriter 实例
        error_msg: 原始错误消息
        
    Returns:
        创建的 ExperienceTrace 或 None
    """
    if not experience_writer:
        logger.debug("No experience_writer provided, skipping dimension mismatch recording")
        return None
    
    try:
        slide_name = html_path.stem
        
        # 解析尺寸信息
        import re
        match = re.search(
            r'HTML dimensions \(([\d.]+)" × ([\d.]+)"\).*?presentation layout \(([\d.]+)" × ([\d.]+)"\)',
            error_msg
        )
        if match:
            html_w, html_h = float(match.group(1)), float(match.group(2))
            layout_w, layout_h = float(match.group(3)), float(match.group(4))
            lesson = (
                f"{slide_name} 使用了错误的 aspect ratio: "
                f"生成 {html_w:.1f}\"×{html_h:.1f}\" (≈{html_w/html_h:.2f}:1), "
                f"期望 {layout_w:.1f}\"×{layout_h:.1f}\" (≈{layout_w/layout_h:.2f}:1)。"
                f"\nWorkaround: 重新生成该页，确保 HTML body 尺寸与目标 aspect ratio 匹配。"
            )
        else:
            lesson = (
                f"{slide_name} 尺寸不匹配目标布局。"
                f"\nWorkaround: 重新生成该页，确保 HTML body 尺寸与目标 aspect ratio 匹配。"
            )
        
        trace = await experience_writer._write_typed_trace(
            session_id=session_id,
            task=f"Tool limitation: html_to_pptx — {slide_name} aspect ratio 不匹配",
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
