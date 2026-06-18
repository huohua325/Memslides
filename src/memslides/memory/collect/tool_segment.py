"""
tool_segment.py — 记忆采集阶段的数据结构与压缩逻辑

定义了 collect 阶段的三层数据结构，负责将 Agent 原始交互数据逐步压缩为
LLM 友好的紧凑表示，供下游 EpisodeExtractor / ExperienceWriter 消费。

三层数据结构:
  ToolCallSegment  最底层。将单次工具调用（可能带几 KB 的 HTML 内容）压缩为
                   属性级摘要（如 "font-size=32px; color=navy"）。针对
                   write_html_file、inspect_slide、image_generation 等不同
                   工具类型有专门的提取策略。

  ModifyRound      中间层。暂存一轮交互的所有原始数据（tool_calls_log、
                   agent_reasoning 等），是 MemoryCollector
                   在 begin_round → end_round 期间的工作缓冲区。

  AgentRound       最终输出（别名 CompactRound）。一轮交互的完整压缩记录，
                   目标 ~300-500 token。是 EpisodeExtractor.extract()、
                   ExperienceTraceWriter.from_agent_round()、以及
                   RoundArtifactWriter 落盘的统一输入源。

压缩路径:
  ModifyRound.to_compact()
    → 对每个 tool call 调用 ToolCallSegment.from_raw_tool_call()
    → 输出 AgentRound

辅助函数:
  _extract_html_changes   从 write_html_file 的 HTML 中提取文字内容 + CSS 属性
  _extract_html_diff      对比前后两版 HTML 的 CSS 属性变化 (before→after)
  _extract_inspect_slide_summary  从 inspect_slide 结果中提取 slide 状态摘要
  _extract_image_action   从图片生成/搜索工具中提取 prompt/query
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Any


# ═══════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════


def _parse_result_payload(result: Any) -> dict[str, Any] | None:
    """尽力解析 tool result 中的 JSON object。"""
    if isinstance(result, dict):
        return result
    if not isinstance(result, str):
        return None
    text = result.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _looks_like_soft_error_result(result: Any) -> bool:
    """识别 MCP 未显式标错、但语义上已失败的 tool result。"""
    payload = _parse_result_payload(result)
    if isinstance(payload, dict):
        if payload.get("success") is False:
            return True
        if payload.get("error_code"):
            return True
        if payload.get("error") and payload.get("success") is not True:
            return True

    text = str(result or "").strip()
    if not text:
        return False
    lower = text.lower()
    return lower.startswith("error:") or text.startswith("❌")


def _summarize_error_result(result: Any, max_len: int = 100) -> str:
    """为软/硬错误生成更稳定的 error_msg。"""
    payload = _parse_result_payload(result)
    if isinstance(payload, dict):
        code = str(payload.get("error_code", "") or "").strip()
        message = str(
            payload.get("error", "")
            or payload.get("message", "")
            or ""
        ).strip()
        parts = [part for part in (code, message) if part]
        if parts:
            return ": ".join(parts)[:max_len]
    return str(result or "").strip()[:max_len]

def _unwrap_raw_args(args: dict | str) -> dict:
    """解包 {'raw': '{"key": "val"}'} 嵌套格式为平坦 dict。

    MCP 工具调用经常将 args 序列化为 {'raw': '<json_string>'}，
    导致下游提取函数拿不到实际参数。此函数统一处理。
    """
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(args, dict):
        return {}
    # 解包 {'raw': '...'} 嵌套
    if "raw" in args and len(args) == 1 and isinstance(args["raw"], str):
        try:
            inner = json.loads(args["raw"])
            if isinstance(inner, dict):
                return inner
        except (json.JSONDecodeError, TypeError):
            pass
    return args


def _extract_target(args: dict | str) -> str:
    """从 tool args 中提取 target_file"""
    args = _unwrap_raw_args(args)
    return (
        args.get("target_file", "")
        or args.get("file_path", "")
        or args.get("html_file", "")
        or args.get("path", "")
    )


def _extract_html_changes(args: dict | str) -> str:
    """从 write_html_file args 中提取变更摘要：可读文字内容 + CSS 属性"""
    args = _unwrap_raw_args(args)
    content = args.get("content", "") or args.get("html_content", "")
    if not content:
        target = _extract_target(args)
        return f"write {target}" if target else "write html"

    changes = []

    # ① 提取可读文字内容（去除 HTML 标签，取前 ~15 个词）
    body_text = re.sub(r'<style[^>]*>.*?</style>', ' ', content, flags=re.DOTALL)
    body_text = re.sub(r'<[^>]+>', ' ', body_text)
    body_text = re.sub(r'\s+', ' ', body_text).strip()
    words = [w for w in body_text.split() if len(w) > 1]
    if words:
        text_summary = ' '.join(words[:15])
        changes.append(f'text="{text_summary[:80]}"')

    # ② 提取关键 CSS 属性（仅 layout/color 类，跳过重复）
    _LAYOUT_PROPS = {
        'font-size', 'font-weight', 'color', 'background', 'background-color',
        'padding', 'margin', 'line-height', 'text-align', 'border',
    }
    style_matches = re.findall(r'([\w-]+)\s*:\s*([^;"{}\n]+)', content)
    seen: set[str] = set()
    for prop, val in style_matches:
        key = prop.strip().lower()
        if key in _LAYOUT_PROPS and key not in seen:
            seen.add(key)
            changes.append(f"{key}={val.strip()[:25]}")
            if len(changes) >= 5:
                break

    if not changes:
        target = _extract_target(args)
        return f"write {target}" if target else "write html"
    return "; ".join(changes)


def _extract_image_action(args: dict | str) -> str:
    """从 image_generation / search_image args 中提取关键参数"""
    args = _unwrap_raw_args(args)
    prompt = args.get("prompt", "") or args.get("query", "")
    if prompt:
        return prompt[:60]
    return "image action"


def _extract_inspect_slide_summary(result: str) -> str:
    """从 inspect_slide result 中提取 slide 属性摘要（状态 / title / elements / images / screenshot）"""
    if not result:
        return "render preview"
    if _looks_like_soft_error_result(result):
        cleaned = re.sub(r"\s+", " ", result).strip()
        return cleaned[:120] if cleaned else "render preview"
    result = result.replace("\\n", "\n")
    parts = []

    # ① 捕获修改状态（优先）
    if 'UNCHANGED' in result or '⚠️' in result:
        parts.append("UNCHANGED")
    elif 'CHANGED' in result or '⚡' in result:
        parts.append("CHANGED")

    # ② Title
    title_m = re.search(r'Title:\s*"([^"]{1,80})"', result)
    if title_m:
        parts.append(f"title={title_m.group(1)[:40]}")

    # ③ Content elements count
    content_m = re.search(r'Content elements:\s*(\d+)', result)
    if content_m:
        parts.append(f"elements={content_m.group(1)}")

    # ④ Images — only when actually present
    img_m = re.search(r'Images:\s*(\d+\s+found)', result)
    if img_m:
        parts.append(f"images={img_m.group(1)}")

    # ⑤ 提取截图路径（供后续视觉分析）
    screenshot_m = re.search(r'screenshot[:\s]+([^\s,\]]+\.png)', result, re.IGNORECASE)
    if not screenshot_m:
        screenshot_m = re.search(r'(\.slide_images[^\s,\]]+\.png)', result)
    if not screenshot_m:
        screenshot_m = re.search(r'(workspace/[^\s,\]]+\.png)', result)
    if screenshot_m:
        parts.append(f"screenshot={screenshot_m.group(1)}")

    return "; ".join(parts) if parts else "render preview"


def _truncate_args(args: dict | str, max_len: int = 80) -> str:
    """通用 args 截断"""
    if isinstance(args, dict):
        text = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str):
        text = args
    else:
        text = str(args)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ── 新增：非 HTML 工具的压缩辅助函数 ──

_STATE_SUMMARY_KEYWORDS = {
    "state_summary", "continuity", "continuation", "handoff",
    "session_state", "working_state", "context_handoff",
    "project_state", "canonical_state",
}


def _is_state_summary_file(file_path: str) -> bool:
    """判断 write_markdown_file 的目标是否为 Agent 内部状态续传文件（噪声）"""
    lower = file_path.lower().replace("/", "_").replace("\\", "_")
    return any(kw in lower for kw in _STATE_SUMMARY_KEYWORDS)


def _extract_markdown_write_summary(args: dict | str) -> str:
    """从 write_markdown_file 中提取摘要

    - 状态续传文件 → 标记为 [state] 噪声
    - manuscript → 提取前几行标题/模板信息
    - plan/action → 提取文件名 + 首行
    """
    args = _unwrap_raw_args(args)

    file_path = args.get("file_path", "")
    content = args.get("content", "")

    # 状态续传文件 → 噪声标记
    if _is_state_summary_file(file_path):
        return f"[state] {file_path.rsplit('/', 1)[-1]}"

    # 提取首行有意义内容（跳过 % 注释行）
    first_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("%"):
            first_lines.append(stripped)
            if len(first_lines) >= 2:
                break

    summary = "; ".join(first_lines)[:100] if first_lines else ""
    fname = file_path.rsplit("/", 1)[-1] if file_path else "file.md"
    return f"write {fname}: {summary}" if summary else f"write {fname}"


def _extract_image_caption_summary(args: dict | str, result: str) -> str:
    """从 image_caption 中提取图片路径 + caption 文本"""
    args = _unwrap_raw_args(args)
    image_path = args.get("image_path", "")
    fname = image_path.rsplit("/", 1)[-1] if image_path else ""

    # 从 result JSON 中提取 caption
    caption = ""
    if result:
        try:
            r = json.loads(result)
            caption = r.get("caption", "")
        except (json.JSONDecodeError, TypeError):
            pass

    if caption:
        return f"{fname}: {caption.strip()[:80]}"
    return f"caption {fname}" if fname else "caption image"


def _extract_query_layout_summary(args: dict | str, result: str) -> str:
    """从 query_slide_layout 中提取 layout_name + 元素数量"""
    args = _unwrap_raw_args(args)
    layout = args.get("layout_name", "")

    # 从 result 中提取元素数量
    element_count = 0
    if result:
        element_count = result.count("| text |") + result.count("| image |")

    if element_count:
        return f"layout={layout} ({element_count} elements)"
    return f"layout={layout}" if layout else "query layout"


def _extract_convert_markdown_summary(args: dict | str, result: str) -> str:
    """从 convert_to_markdown 中提取源文件 + 图片数"""
    args = _unwrap_raw_args(args)
    file_path = args.get("file_path", "")
    fname = file_path.rsplit("/", 1)[-1] if file_path else ""

    # 从 result 中提取图片数（兼容截断的 JSON）
    img_count = ""
    if result:
        try:
            r = json.loads(result)
            images_str = r.get("images", "")
            if "Found" in images_str:
                img_count = images_str.split("\n")[0]  # "Found 15 images"
        except (json.JSONDecodeError, TypeError, ValueError):
            import re as _re
            m = _re.search(r'Found\s+(\d+)\s+images', result)
            if m:
                img_count = f"Found {m.group(1)} images"

    parts = [f"convert {fname}"]
    if img_count:
        parts.append(img_count)
    return "; ".join(parts)


def _extract_list_figures_summary(args: dict | str, result: str) -> str:
    """从 list_document_figures 中提取 total + 文件名列表"""
    total = 0
    filenames: list[str] = []
    if result:
        try:
            r = json.loads(result)
            total = r.get("total", 0)
            for fig in r.get("figures", [])[:5]:
                filenames.append(fig.get("filename", ""))
        except (json.JSONDecodeError, TypeError, ValueError):
            # result 可能被截断导致 JSON 不完整，尝试正则提取
            import re as _re
            total_m = _re.search(r'"total"\s*:\s*(\d+)', result)
            if total_m:
                total = int(total_m.group(1))
            filenames = _re.findall(r'"filename"\s*:\s*"([^"]+)"', result)[:5]

    if total:
        names = ", ".join(f for f in filenames if f)[:80]
        return f"{total} figures: {names}" if names else f"{total} figures"
    return "list figures"


def _extract_inspect_manuscript_summary(args: dict | str, result: str) -> str:
    """从 inspect_manuscript 中提取 num_pages + language + warnings"""
    args = _unwrap_raw_args(args)
    md_file = args.get("md_file", "")
    fname = md_file.rsplit("/", 1)[-1] if md_file else ""

    parts = [f"inspect {fname}"] if fname else ["inspect manuscript"]
    if result:
        try:
            r = json.loads(result)
            if "error" in r:
                return f"inspect {fname} [ERROR: {r['error'][:60]}]"
            pages = r.get("num_pages", 0)
            lang = r.get("language", "")
            warnings = r.get("warnings", [])
            if pages:
                parts.append(f"{pages} pages")
            if lang:
                parts.append(f"lang={lang}")
            if warnings:
                parts.append(f"{len(warnings)} warnings")
        except (json.JSONDecodeError, TypeError):
            pass
    return "; ".join(parts)


def _extract_remember_lesson_summary(args: dict | str) -> str:
    """从 remember_lesson 中提取 tool_name + content 摘要"""
    args = _unwrap_raw_args(args)
    tool_name = args.get("tool_name", "")
    content = args.get("content", "")
    # 取 content 前 80 字符作为摘要
    summary = content[:80].replace("\n", " ").strip()
    if tool_name:
        return f"[lesson:{tool_name}] {summary}"
    return f"[lesson] {summary}" if summary else "[lesson]"


def _extract_read_file_summary(args: dict | str) -> str:
    """从 read_file / read_html_file 中提取 file_path + offset/limit 信息"""
    args = _unwrap_raw_args(args)

    file_path = (
        args.get("file_path", "")
        or args.get("target_file", "")
        or args.get("html_file", "")
        or args.get("path", "")
    )
    fname = file_path.rsplit("/", 1)[-1] if file_path else "file"
    offset = args.get("offset", 0)
    limit = args.get("limit", 0)

    if offset or limit:
        return f"read {fname} [L{offset}:{offset + limit}]"
    return f"read {fname}"




# ═══════════════════════════════════════════════
# Observation 摘要提取（从 tool result 中提取关键信息）
# ═══════════════════════════════════════════════

# 每个 observation 摘要的最大字符数
_OBS_MAX_CHARS = 200


def _extract_observation(name: str, args: dict | str, result: str) -> str:
    """从 tool result 中提取 observation 摘要（≤200 chars）。

    不同工具有不同的提取策略，目标是保留对经验提取有价值的信息：
    - 工具返回了什么？Agent 看到了什么？
    - 有没有错误/警告？
    - 关键数据点（文件名、数量、结构信息）
    """
    if not result:
        return ""
    result_str = str(result)

    # inspect_slide — 已在 action_summary 中充分提取，observation 补充 warnings
    if name == "inspect_slide":
        return _obs_inspect_slide(result_str)

    # read_file / read_html_file — 提取文件头部内容摘要
    if name in ("read_file", "read_html_file"):
        return _obs_read_file(result_str)

    # list_files — 提取目录结构
    if name == "list_files":
        return _obs_list_files(result_str)

    # explore_workspace_images — 提取图片清单
    if name == "explore_workspace_images":
        return _obs_explore_images(result_str)

    # write_html_file / write_markdown_file — 提取写入确认
    if name in ("write_html_file", "write_markdown_file"):
        return _obs_write_file(result_str)

    # inspect_manuscript — 已在 action_summary 中提取，补充 success/warnings
    if name == "inspect_manuscript":
        return _obs_inspect_manuscript(result_str)

    # convert_to_markdown — 提取转换结果
    if name == "convert_to_markdown":
        return _obs_convert_markdown(result_str)

    # list_document_figures — 提取图片列表
    if name == "list_document_figures":
        return _obs_list_figures(result_str)

    # image_generation / search_image — 提取生成结果
    if name in ("image_generation", "search_image"):
        return _obs_image_result(result_str)

    # image_caption — 提取 caption
    if name == "image_caption":
        return _obs_image_caption(result_str)

    # query_slide_layout — 提取布局详情
    if name == "query_slide_layout":
        return _obs_query_layout(result_str)

    # finalize — 通常很短，直接返回
    if name == "finalize":
        return result_str[:_OBS_MAX_CHARS]

    # 通用降级：截断
    return _obs_generic(result_str)


def _obs_inspect_slide(result: str) -> str:
    """inspect_slide: 提取 warnings/issues 和关键状态"""
    if _looks_like_soft_error_result(result):
        return _obs_generic(result)

    parts = []
    # 提取 warnings
    for pattern in (r'⚠️\s*(.+)', r'Warning:\s*(.+)', r'Issue:\s*(.+)'):
        for m in re.finditer(pattern, result):
            parts.append(m.group(1).strip()[:60])
    # 提取 overflow/overlap 信息
    if "overflow" in result.lower():
        parts.append("overflow detected")
    if "overlap" in result.lower():
        parts.append("overlap detected")
    # 如果没有问题，标记 OK
    if not parts:
        if "UNCHANGED" in result:
            return "no changes detected"
        return "OK"
    return "; ".join(parts)[:_OBS_MAX_CHARS]


def _obs_read_file(result: str) -> str:
    """read_file: 提取文件名 + 前几行内容摘要"""
    lines = result.split("\n")
    # 第一行通常是 "File: xxx (N lines total)"
    header = lines[0] if lines else ""
    # 提取有意义的内容行（跳过行号前缀）
    content_lines = []
    for line in lines[1:]:
        # 去掉行号前缀 "   1 | "
        cleaned = re.sub(r'^\s*\d+\s*\|\s*', '', line).strip()
        if cleaned and not cleaned.startswith("[..."):
            content_lines.append(cleaned)
        if len(content_lines) >= 3:
            break
    summary = "; ".join(content_lines)[:150]
    # 组合 header + content preview
    if summary:
        return f"{header[:50]}; content: {summary}"[:_OBS_MAX_CHARS]
    return header[:_OBS_MAX_CHARS]


def _obs_list_files(result: str) -> str:
    """list_files: 提取目录结构摘要"""
    # 直接截断，目录结构本身就是有价值的信息
    return result.replace("\n", "; ")[:_OBS_MAX_CHARS]


def _obs_explore_images(result: str) -> str:
    """explore_workspace_images: 提取图片数量和文件名列表"""
    try:
        r = json.loads(result)
        total = r.get("total_found", 0)
        images = r.get("images", [])
        names = [img.get("filename", "") for img in images[:6]]
        names_str = ", ".join(n for n in names if n)
        return f"{total} images: {names_str}"[:_OBS_MAX_CHARS]
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_write_file(result: str) -> str:
    """write_html_file / write_markdown_file: 提取写入确认"""
    # 通常是 "Successfully wrote ... to: path\nFile size: N bytes"
    if "Successfully" in result or "success" in result.lower():
        # 提取路径
        path_m = re.search(r'to:\s*(\S+)', result)
        size_m = re.search(r'size:\s*(\d+\s*\w+)', result)
        parts = []
        if path_m:
            parts.append(path_m.group(1))
        if size_m:
            parts.append(size_m.group(1))
        return "OK: " + ", ".join(parts) if parts else "OK"
    return result[:_OBS_MAX_CHARS]


def _obs_inspect_manuscript(result: str) -> str:
    """inspect_manuscript: 补充 success/warnings 详情"""
    try:
        r = json.loads(result)
        if "error" in r:
            return f"ERROR: {r['error'][:150]}"
        parts = []
        for s in r.get("success", []):
            parts.append(s[:60])
        for w in r.get("warnings", []):
            parts.append(f"⚠ {w[:60]}")
        return "; ".join(parts)[:_OBS_MAX_CHARS] if parts else "OK"
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_convert_markdown(result: str) -> str:
    """convert_to_markdown: 提取转换结果"""
    try:
        r = json.loads(result)
        parts = []
        if r.get("markdown_file"):
            parts.append(f"→ {r['markdown_file'].rsplit('/', 1)[-1]}")
        images = r.get("images", "")
        if images:
            parts.append(images.split("\n")[0][:60])
        return "; ".join(parts)[:_OBS_MAX_CHARS] if parts else "OK"
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_list_figures(result: str) -> str:
    """list_document_figures: 提取图片列表"""
    try:
        r = json.loads(result)
        total = r.get("total", 0)
        figs = r.get("figures", [])
        names = [f.get("filename", "") for f in figs[:5]]
        return f"{total} figures: {', '.join(n for n in names if n)}"[:_OBS_MAX_CHARS]
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_image_result(result: str) -> str:
    """image_generation / search_image: 提取结果路径"""
    try:
        r = json.loads(result)
        path = r.get("image_path", "") or r.get("path", "")
        if path:
            return f"→ {path.rsplit('/', 1)[-1]}"
        return result[:_OBS_MAX_CHARS]
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_image_caption(result: str) -> str:
    """image_caption: 提取 caption 文本"""
    try:
        r = json.loads(result)
        caption = r.get("caption", "")
        return caption[:_OBS_MAX_CHARS] if caption else "OK"
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_query_layout(result: str) -> str:
    """query_slide_layout: 提取布局元素摘要"""
    # 通常是表格格式，提取元素类型列表
    elements = re.findall(r'\|\s*(text|image|shape)\s*\|', result, re.IGNORECASE)
    if elements:
        return f"elements: {', '.join(elements)}"[:_OBS_MAX_CHARS]
    return result[:_OBS_MAX_CHARS]


def _obs_generic(result: str) -> str:
    """通用降级：去除多余空白后截断"""
    cleaned = re.sub(r'\s+', ' ', result).strip()
    return cleaned[:_OBS_MAX_CHARS]


def _extract_html_diff(prev_content: str, new_content: str) -> str:
    """对比两个 HTML 的关键 CSS 属性变化，返回 before→after diff 摘要

    例如: "font-size: 24px→32px; color: #000→#F00"
    """
    _DIFF_PROPS = {
        'font-size', 'font-weight', 'color', 'background', 'background-color',
        'padding', 'margin', 'line-height', 'text-align', 'border', 'width', 'height'
    }

    def _extract_props(html: str) -> dict[str, str]:
        """从 HTML 中提取关键 CSS 属性的最后一次出现值"""
        matches = re.findall(r'([\w-]+)\s*:\s*([^;"{}\n]+)', html)
        props: dict[str, str] = {}
        for prop, val in matches:
            key = prop.strip().lower()
            if key in _DIFF_PROPS:
                props[key] = val.strip()[:25]
        return props

    old_props = _extract_props(prev_content)
    new_props = _extract_props(new_content)

    changes = []
    for key in _DIFF_PROPS:
        old_val = old_props.get(key)
        new_val = new_props.get(key)
        if old_val and new_val and old_val != new_val:
            changes.append(f"{key}: {old_val}→{new_val}")
        elif not old_val and new_val:
            changes.append(f"+{key}={new_val}")
        elif old_val and not new_val:
            changes.append(f"-{key}")

    return "; ".join(changes[:4]) if changes else ""


# ═══════════════════════════════════════════════
# ToolCallSegment
# ═══════════════════════════════════════════════

@dataclass
class ToolCallSegment:
    """单次 tool call 的压缩摘要 — Layer 1 输出"""

    tool_name: str
    target_file: str = ""
    action_summary: str = ""
    observation_summary: str = ""                 # 工具返回结果摘要（≤200 chars）
    is_error: bool = False
    error_msg: str = ""

    @classmethod
    def from_raw_tool_call(
        cls,
        name: str,
        args: dict | str,
        result: str = "",
        is_error: bool = False,
        prev_content: str = "",
    ) -> ToolCallSegment:
        parsed_args = _unwrap_raw_args(args)
        soft_error = _looks_like_soft_error_result(result)
        seg = cls(tool_name=name, is_error=is_error or soft_error)
        seg.target_file = _extract_target(args)
        if name == "write_html_file":
            base_summary = _extract_html_changes(args)
            if prev_content:
                new_content = parsed_args.get("content", "")
                diff_summary = _extract_html_diff(prev_content, new_content)
                if diff_summary:
                    seg.action_summary = f"{base_summary} | Δ {diff_summary}"
                else:
                    seg.action_summary = base_summary
            else:
                seg.action_summary = base_summary
        elif name == "write_markdown_file":
            seg.action_summary = _extract_markdown_write_summary(args)
        elif name in ("read_html_file", "read_file"):
            seg.action_summary = _extract_read_file_summary(args)
        elif name == "list_files":
            directory = parsed_args.get("directory", ".")
            seg.action_summary = f"list {directory}"
        elif name == "inspect_slide":
            seg.action_summary = _extract_inspect_slide_summary(result)
        elif name == "image_caption":
            seg.action_summary = _extract_image_caption_summary(args, result)
        elif name == "query_slide_layout":
            seg.action_summary = _extract_query_layout_summary(args, result)
        elif name == "convert_to_markdown":
            seg.action_summary = _extract_convert_markdown_summary(args, result)
        elif name == "list_document_figures":
            seg.action_summary = _extract_list_figures_summary(args, result)
        elif name == "inspect_manuscript":
            seg.action_summary = _extract_inspect_manuscript_summary(args, result)
        elif name == "remember_lesson":
            seg.action_summary = _extract_remember_lesson_summary(args)
        elif name in ("image_generation", "search_image"):
            seg.action_summary = _extract_image_action(args)
        elif name == "thinking":
            thought = parsed_args.get("thought", str(args))[:120]
            seg.action_summary = f"[think] {thought}"
        elif name in ("todo_create", "todo_update", "todo_delete"):
            intent = (
                parsed_args.get("todo_content", "")
                or parsed_args.get("content", "")
                or parsed_args.get("task", "")
            )[:80]
            seg.action_summary = f"[plan] {intent}" if intent else "[plan]"
        elif name == "finalize":
            outcome = parsed_args.get("outcome", "")[:40]
            seg.action_summary = f"→ {outcome}" if outcome else "finalize"
        else:
            seg.action_summary = _truncate_args(args, max_len=80)
        # observation 摘要（从 result 提取）
        seg.observation_summary = _extract_observation(name, args, result)
        if seg.is_error and result:
            seg.error_msg = _summarize_error_result(result)
        return seg

    def to_text(self) -> str:
        parts = [self.tool_name]
        if self.target_file:
            parts.append(f"({self.target_file})")
        if self.action_summary:
            parts.append(f": {self.action_summary}")
        if self.is_error:
            parts.append(f" [ERROR: {self.error_msg}]")
        if self.observation_summary:
            parts.append(f" → {self.observation_summary}")
        return " ".join(parts)


# ═══════════════════════════════════════════════
# CompactRound
# ═══════════════════════════════════════════════

@dataclass
class AgentRound:
    """一轮 Agent 执行的完整记录 — 统一输入源

    扩展自原 CompactRound，新增以下字段：
    - agent_name: 执行此轮的 Agent 名称
    - memory_injection: 注入的 memory prompt（压缩后）
    - agent_reasoning: Agent 的推理过程（可选）

    用途：
    1. ExperienceTraceWriter.from_agent_round() 的输入
    2. EpisodeExtractor.extract() 的输入
    3. 中间产物 rounds/round_xxx/agent_round.json 的内容
    """

    # ── 基础信息 ──
    round_id: int = 0
    agent_name: str = ""                          # "design" | "research" | "modify"
    session_id: str = ""                          # 会话 ID
    user_id: str = ""                             # 用户 ID

    # ── 输入上下文 ──
    user_message: str = ""                        # 用户原始消息
    memory_injection: str = ""                    # 注入的 memory prompt (≤500 字符压缩版)
    memory_injection_full: str = ""               # 完整 memory prompt (用于调试)
    memory_injection_tokens: int = 0              # 注入 token 数

    # ── 执行过程 ──
    segments: list[ToolCallSegment] = field(default_factory=list)
    agent_reasoning: str = ""                     # Agent 的推理过程 (≤300 字符)
    reasoning_traces: list[str] = field(default_factory=list)  # 每个 turn 的 reasoning 摘要
    agent_reply: str = ""                         # Agent 最终回复

    # ── 统计信息 ──
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    total_tool_calls: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转换为 dict，用于 JSON 序列化"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRound":
        """从 dict 重建"""
        valid_fields = {f.name for f in fields(cls)}
        # 处理 segments 字段的反序列化
        if "segments" in data and data["segments"]:
            data = dict(data)  # 复制避免修改原始数据
            data["segments"] = [
                ToolCallSegment(**s) if isinstance(s, dict) else s
                for s in data["segments"]
            ]
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    @classmethod
    def from_compact_round(
        cls,
        compact: "AgentRound",
        agent_name: str = "",
        memory_injection: str = "",
        memory_injection_full: str = "",
        agent_reasoning: str = "",
        session_id: str = "",
        user_id: str = "",
    ) -> "AgentRound":
        """从旧 CompactRound 升级为 AgentRound（向后兼容）"""
        return cls(
            round_id=compact.round_id,
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            user_message=compact.user_message,
            memory_injection=memory_injection[:500] if memory_injection else "",
            memory_injection_full=memory_injection_full or memory_injection,
            memory_injection_tokens=len(memory_injection) // 4 if memory_injection else 0,
            segments=compact.segments,
            agent_reasoning=agent_reasoning[:300] if agent_reasoning else "",
            reasoning_traces=compact.reasoning_traces if compact.reasoning_traces else [],
            agent_reply=compact.agent_reply,
            timestamp=compact.timestamp,
            duration_seconds=compact.duration_seconds,
            total_tool_calls=len(compact.segments),
            error_count=sum(1 for s in compact.segments if s.is_error),
        )

    def to_extraction_text(self) -> str:
        """转为 EpisodeExtractor / ChainSegmenter 的输入文本

        格式化为紧凑、LLM 友好的文本表示。
        包含 observation 和 reasoning 以提供完整的决策上下文。

        注意：不包含 memory_injection，因为 Episode 提取关注的是：
        - 用户的真实意图
        - Agent 的理解偏差
        - 设计洞察
        这些都从用户消息 + 工具执行 + 用户反馈中推断，不需要 memory 上下文。
        """
        lines = []
        # Agent context (如果有)
        context = f" [{self.agent_name}]" if self.agent_name else ""
        lines.append(f"[Round {self.round_id}]{context} User: {self.user_message}")

        # Reasoning traces（Agent 的决策思路）
        if self.reasoning_traces:
            for i, trace in enumerate(self.reasoning_traces):
                if trace:
                    lines.append(f"  [Reasoning {i}] {trace}")

        # tool chain 摘要（含 observation）
        if self.segments:
            chain_lines = []
            for s in self.segments:
                chain_lines.append(f"  {s.to_text()}")
            lines.append("Tool chain:")
            lines.extend(chain_lines)

        # Agent 回复
        if self.agent_reply:
            lines.append(f"Agent reply: {self.agent_reply}")

        return "\n".join(lines)

    def estimate_tokens(self) -> int:
        """粗略估算 token 数（1 token ≈ 4 chars for English, ≈ 2 chars for Chinese）"""
        text = self.to_extraction_text()
        return len(text) // 3  # 中英文混合的粗略估算


# 向后兼容别名
CompactRound = AgentRound


# ═══════════════════════════════════════════════
# ModifyRound — 原始数据暂存
# ═══════════════════════════════════════════════

@dataclass
class ModifyRound:
    """原始单轮数据暂存 → to_compact() 执行 Layer 1 压缩

    Stage 4 扩展: 新增 agent_name, memory_injection 等字段，
    支持生成包含完整上下文的 AgentRound。
    """

    round_id: int = 0
    user_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_calls_log: list[dict[str, Any]] = field(default_factory=list)
    agent_responses: list[str] = field(default_factory=list)
    target_slides: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    # ── Stage 4 新增字段 ──
    agent_name: str = ""                          # "design" | "research" | "modify"
    session_id: str = ""                          # 会话 ID
    user_id: str = ""                             # 用户 ID
    memory_injection: str = ""                    # 注入的 memory prompt (压缩版)
    memory_injection_full: str = ""               # 完整 memory prompt
    agent_reasoning: str = ""                     # Agent 推理过程
    reasoning_traces: list[str] = field(default_factory=list)  # 每个 turn 的 reasoning 摘要

    def to_compact(self) -> AgentRound:
        """Layer 1 压缩: ModifyRound → AgentRound

        Stage 4 改造: 返回包含完整上下文的 AgentRound，
        包括 agent_name, memory_injection 等新字段。
        """
        segments = []
        for tc in self.tool_calls_log:
            seg = ToolCallSegment.from_raw_tool_call(
                name=tc.get("name", "unknown"),
                args=tc.get("args", {}),
                result=tc.get("result", ""),
                is_error=tc.get("is_error", False),
                prev_content=tc.get("prev_content", ""),  # Stage 5 新增
            )
            segments.append(seg)

        # agent_reply: DeckDesigner 主要通过工具调用（finalize）完成任务，不输出自然语言回复
        agent_reply = ""

        return AgentRound(
            # 基础信息
            round_id=self.round_id,
            agent_name=self.agent_name,
            session_id=self.session_id,
            user_id=self.user_id,
            # 输入上下文
            user_message=self.user_message,
            memory_injection=self.memory_injection[:500] if self.memory_injection else "",
            memory_injection_full=self.memory_injection_full or self.memory_injection,
            memory_injection_tokens=len(self.memory_injection) // 4 if self.memory_injection else 0,
            # 执行过程
            segments=segments,
            agent_reasoning=self.agent_reasoning[:300] if self.agent_reasoning else "",
            reasoning_traces=[t[:300] for t in self.reasoning_traces] if self.reasoning_traces else [],
            agent_reply=agent_reply,
            # 统计信息
            timestamp=self.timestamp,
            duration_seconds=self.duration_seconds,
            total_tool_calls=len(segments),
            error_count=sum(1 for s in segments if s.is_error),
        )
