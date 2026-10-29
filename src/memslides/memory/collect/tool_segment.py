"""Implementation details."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Any


# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════


def _parse_result_payload(result: Any) -> dict[str, Any] | None:
    """Implementation details."""
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
    """Implementation details."""
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
    """Implementation details."""
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
    """Implementation details."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(args, dict):
        return {}

    if "raw" in args and len(args) == 1 and isinstance(args["raw"], str):
        try:
            inner = json.loads(args["raw"])
            if isinstance(inner, dict):
                return inner
        except (json.JSONDecodeError, TypeError):
            pass
    return args


def _extract_target(args: dict | str) -> str:
    """Implementation details."""
    args = _unwrap_raw_args(args)
    return (
        args.get("target_file", "")
        or args.get("file_path", "")
        or args.get("html_file", "")
        or args.get("path", "")
    )


def _extract_html_changes(args: dict | str) -> str:
    """Implementation details."""
    args = _unwrap_raw_args(args)
    content = args.get("content", "") or args.get("html_content", "")
    if not content:
        target = _extract_target(args)
        return f"write {target}" if target else "write html"

    changes = []


    body_text = re.sub(r'<style[^>]*>.*?</style>', ' ', content, flags=re.DOTALL)
    body_text = re.sub(r'<[^>]+>', ' ', body_text)
    body_text = re.sub(r'\s+', ' ', body_text).strip()
    words = [w for w in body_text.split() if len(w) > 1]
    if words:
        text_summary = ' '.join(words[:15])
        changes.append(f'text="{text_summary[:80]}"')


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
    """Implementation details."""
    args = _unwrap_raw_args(args)
    prompt = args.get("prompt", "") or args.get("query", "")
    if prompt:
        return prompt[:60]
    return "image action"


def _extract_inspect_slide_summary(result: str) -> str:
    """Implementation details."""
    if not result:
        return "render preview"
    if _looks_like_soft_error_result(result):
        cleaned = re.sub(r"\s+", " ", result).strip()
        return cleaned[:120] if cleaned else "render preview"
    result = result.replace("\\n", "\n")
    parts = []


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


    screenshot_m = re.search(r'screenshot[:\s]+([^\s,\]]+\.png)', result, re.IGNORECASE)
    if not screenshot_m:
        screenshot_m = re.search(r'(\.slide_images[^\s,\]]+\.png)', result)
    if not screenshot_m:
        screenshot_m = re.search(r'(workspace/[^\s,\]]+\.png)', result)
    if screenshot_m:
        parts.append(f"screenshot={screenshot_m.group(1)}")

    return "; ".join(parts) if parts else "render preview"


def _truncate_args(args: dict | str, max_len: int = 80) -> str:
    """Implementation details."""
    if isinstance(args, dict):
        text = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str):
        text = args
    else:
        text = str(args)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."




_STATE_SUMMARY_KEYWORDS = {
    "state_summary", "continuity", "continuation", "handoff",
    "session_state", "working_state", "context_handoff",
    "project_state", "canonical_state",
}


def _is_state_summary_file(file_path: str) -> bool:
    """Implementation details."""
    lower = file_path.lower().replace("/", "_").replace("\\", "_")
    return any(kw in lower for kw in _STATE_SUMMARY_KEYWORDS)


def _extract_markdown_write_summary(args: dict | str) -> str:
    """Implementation details."""
    args = _unwrap_raw_args(args)

    file_path = args.get("file_path", "")
    content = args.get("content", "")


    if _is_state_summary_file(file_path):
        return f"[state] {file_path.rsplit('/', 1)[-1]}"


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
    """Implementation details."""
    args = _unwrap_raw_args(args)
    image_path = args.get("image_path", "")
    fname = image_path.rsplit("/", 1)[-1] if image_path else ""


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
    """Implementation details."""
    args = _unwrap_raw_args(args)
    layout = args.get("layout_name", "")


    element_count = 0
    if result:
        element_count = result.count("| text |") + result.count("| image |")

    if element_count:
        return f"layout={layout} ({element_count} elements)"
    return f"layout={layout}" if layout else "query layout"


def _extract_convert_markdown_summary(args: dict | str, result: str) -> str:
    """Implementation details."""
    args = _unwrap_raw_args(args)
    file_path = args.get("file_path", "")
    fname = file_path.rsplit("/", 1)[-1] if file_path else ""


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
    """Implementation details."""
    total = 0
    filenames: list[str] = []
    if result:
        try:
            r = json.loads(result)
            total = r.get("total", 0)
            for fig in r.get("figures", [])[:5]:
                filenames.append(fig.get("filename", ""))
        except (json.JSONDecodeError, TypeError, ValueError):

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
    """Implementation details."""
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
    """Implementation details."""
    args = _unwrap_raw_args(args)
    tool_name = args.get("tool_name", "")
    content = args.get("content", "")

    summary = content[:80].replace("\n", " ").strip()
    if tool_name:
        return f"[lesson:{tool_name}] {summary}"
    return f"[lesson] {summary}" if summary else "[lesson]"


def _extract_read_file_summary(args: dict | str) -> str:
    """Implementation details."""
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

# ═══════════════════════════════════════════════


_OBS_MAX_CHARS = 200


def _extract_observation(name: str, args: dict | str, result: str) -> str:
    """Implementation details."""
    if not result:
        return ""
    result_str = str(result)


    if name == "inspect_slide":
        return _obs_inspect_slide(result_str)


    if name in ("read_file", "read_html_file"):
        return _obs_read_file(result_str)


    if name == "list_files":
        return _obs_list_files(result_str)


    if name == "explore_workspace_images":
        return _obs_explore_images(result_str)


    if name in ("write_html_file", "write_markdown_file"):
        return _obs_write_file(result_str)


    if name == "inspect_manuscript":
        return _obs_inspect_manuscript(result_str)


    if name == "convert_to_markdown":
        return _obs_convert_markdown(result_str)


    if name == "list_document_figures":
        return _obs_list_figures(result_str)


    if name in ("image_generation", "search_image"):
        return _obs_image_result(result_str)


    if name == "image_caption":
        return _obs_image_caption(result_str)


    if name == "query_slide_layout":
        return _obs_query_layout(result_str)


    if name == "finalize":
        return result_str[:_OBS_MAX_CHARS]


    return _obs_generic(result_str)


def _obs_inspect_slide(result: str) -> str:
    """Implementation details."""
    if _looks_like_soft_error_result(result):
        return _obs_generic(result)

    parts = []

    for pattern in (r'⚠️\s*(.+)', r'Warning:\s*(.+)', r'Issue:\s*(.+)'):
        for m in re.finditer(pattern, result):
            parts.append(m.group(1).strip()[:60])

    if "overflow" in result.lower():
        parts.append("overflow detected")
    if "overlap" in result.lower():
        parts.append("overlap detected")

    if not parts:
        if "UNCHANGED" in result:
            return "no changes detected"
        return "OK"
    return "; ".join(parts)[:_OBS_MAX_CHARS]


def _obs_read_file(result: str) -> str:
    """Implementation details."""
    lines = result.split("\n")

    header = lines[0] if lines else ""

    content_lines = []
    for line in lines[1:]:

        cleaned = re.sub(r'^\s*\d+\s*\|\s*', '', line).strip()
        if cleaned and not cleaned.startswith("[..."):
            content_lines.append(cleaned)
        if len(content_lines) >= 3:
            break
    summary = "; ".join(content_lines)[:150]

    if summary:
        return f"{header[:50]}; content: {summary}"[:_OBS_MAX_CHARS]
    return header[:_OBS_MAX_CHARS]


def _obs_list_files(result: str) -> str:
    """Implementation details."""

    return result.replace("\n", "; ")[:_OBS_MAX_CHARS]


def _obs_explore_images(result: str) -> str:
    """Implementation details."""
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
    """Implementation details."""

    if "Successfully" in result or "success" in result.lower():

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
    """Implementation details."""
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
    """Implementation details."""
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
    """Implementation details."""
    try:
        r = json.loads(result)
        total = r.get("total", 0)
        figs = r.get("figures", [])
        names = [f.get("filename", "") for f in figs[:5]]
        return f"{total} figures: {', '.join(n for n in names if n)}"[:_OBS_MAX_CHARS]
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_image_result(result: str) -> str:
    """Implementation details."""
    try:
        r = json.loads(result)
        path = r.get("image_path", "") or r.get("path", "")
        if path:
            return f"→ {path.rsplit('/', 1)[-1]}"
        return result[:_OBS_MAX_CHARS]
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_image_caption(result: str) -> str:
    """Implementation details."""
    try:
        r = json.loads(result)
        caption = r.get("caption", "")
        return caption[:_OBS_MAX_CHARS] if caption else "OK"
    except (json.JSONDecodeError, TypeError):
        return result[:_OBS_MAX_CHARS]


def _obs_query_layout(result: str) -> str:
    """Implementation details."""

    elements = re.findall(r'\|\s*(text|image|shape)\s*\|', result, re.IGNORECASE)
    if elements:
        return f"elements: {', '.join(elements)}"[:_OBS_MAX_CHARS]
    return result[:_OBS_MAX_CHARS]


def _obs_generic(result: str) -> str:
    """Implementation details."""
    cleaned = re.sub(r'\s+', ' ', result).strip()
    return cleaned[:_OBS_MAX_CHARS]


def _extract_html_diff(prev_content: str, new_content: str) -> str:
    """Implementation details."""
    _DIFF_PROPS = {
        'font-size', 'font-weight', 'color', 'background', 'background-color',
        'padding', 'margin', 'line-height', 'text-align', 'border', 'width', 'height'
    }

    def _extract_props(html: str) -> dict[str, str]:
        """Implementation details."""
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
    """Implementation details."""

    tool_name: str
    target_file: str = ""
    action_summary: str = ""
    observation_summary: str = ""
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
    """Implementation details."""


    round_id: int = 0
    agent_name: str = ""                          # "design" | "research" | "modify"
    session_id: str = ""
    user_id: str = ""


    user_message: str = ""
    memory_injection: str = ""
    memory_injection_full: str = ""
    memory_injection_tokens: int = 0


    segments: list[ToolCallSegment] = field(default_factory=list)
    agent_reasoning: str = ""
    reasoning_traces: list[str] = field(default_factory=list)
    agent_reply: str = ""


    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0
    total_tool_calls: int = 0
    error_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Implementation details."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRound":
        """Implementation details."""
        valid_fields = {f.name for f in fields(cls)}

        if "segments" in data and data["segments"]:
            data = dict(data)
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
        """Implementation details."""
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
        """Implementation details."""
        lines = []

        context = f" [{self.agent_name}]" if self.agent_name else ""
        lines.append(f"[Round {self.round_id}]{context} User: {self.user_message}")


        if self.reasoning_traces:
            for i, trace in enumerate(self.reasoning_traces):
                if trace:
                    lines.append(f"  [Reasoning {i}] {trace}")


        if self.segments:
            chain_lines = []
            for s in self.segments:
                chain_lines.append(f"  {s.to_text()}")
            lines.append("Tool chain:")
            lines.extend(chain_lines)


        if self.agent_reply:
            lines.append(f"Agent reply: {self.agent_reply}")

        return "\n".join(lines)

    def estimate_tokens(self) -> int:
        """Implementation details."""
        text = self.to_extraction_text()
        return len(text) // 3



CompactRound = AgentRound


# ═══════════════════════════════════════════════

# ═══════════════════════════════════════════════

@dataclass
class ModifyRound:
    """Implementation details."""

    round_id: int = 0
    user_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tool_calls_log: list[dict[str, Any]] = field(default_factory=list)
    agent_responses: list[str] = field(default_factory=list)
    target_slides: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0


    agent_name: str = ""                          # "design" | "research" | "modify"
    session_id: str = ""
    user_id: str = ""
    memory_injection: str = ""
    memory_injection_full: str = ""
    agent_reasoning: str = ""
    reasoning_traces: list[str] = field(default_factory=list)

    def to_compact(self) -> AgentRound:
        """Implementation details."""
        segments = []
        for tc in self.tool_calls_log:
            seg = ToolCallSegment.from_raw_tool_call(
                name=tc.get("name", "unknown"),
                args=tc.get("args", {}),
                result=tc.get("result", ""),
                is_error=tc.get("is_error", False),
                prev_content=tc.get("prev_content", ""),
            )
            segments.append(seg)


        agent_reply = ""

        return AgentRound(

            round_id=self.round_id,
            agent_name=self.agent_name,
            session_id=self.session_id,
            user_id=self.user_id,

            user_message=self.user_message,
            memory_injection=self.memory_injection[:500] if self.memory_injection else "",
            memory_injection_full=self.memory_injection_full or self.memory_injection,
            memory_injection_tokens=len(self.memory_injection) // 4 if self.memory_injection else 0,

            segments=segments,
            agent_reasoning=self.agent_reasoning[:300] if self.agent_reasoning else "",
            reasoning_traces=[t[:300] for t in self.reasoning_traces] if self.reasoning_traces else [],
            agent_reply=agent_reply,

            timestamp=self.timestamp,
            duration_seconds=self.duration_seconds,
            total_tool_calls=len(segments),
            error_count=sum(1 for s in segments if s.is_error),
        )
