import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from html import escape as html_escape
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from fastmcp import FastMCP
from filelock import FileLock
from mcp.types import ImageContent, TextContent
from PIL import Image

from pptx import Presentation
from pydantic import BaseModel, Field

from memslides.utils.config import MemSlidesConfig
from memslides.utils.log import error, info, set_logger, warning
from memslides.utils.webview import (
    _get_expected_playwright_binary,
    convert_html_to_pptx_with_retry,
    should_auto_install_playwright,
)
from memslides.memory.extract.llm_compat import (
    extract_response_text,
    resolve_llm_retry_times,
)

mcp = FastMCP(name="MemSlidesDeckTools")

# Stage 7: 立即导入 template_tools，确保模板工具在 MCP 服务器启动前注册
# 注意：必须在 mcp 实例创建后、任何工具列表获取之前导入
import memslides.tools.template_tools  # noqa: F401, E402

# Module-level variable to track current agent (replaces deprecated exclude_args)
_current_agent_name: str = ""


_AGENT_NAME_FILE = ".current_agent"
_AGENT_CONTEXT_FILE = ".current_agent_context.json"
_MODIFY_CONTEXT_FILE = ".current_modify_context.json"
_INTERMEDIATE_OUTPUT_FILE = "intermediate_output.json"
_TEMPLATE_SKILL_FILE = ".template_skill.json"
_DESIGN_PLAN_STATE_FILE = ".design_plan_state.json"
_CONTROL_DOCUMENT_STATE_FILE = ".control_document_state.json"
_WORKSPACE_ENV_KEYS = ("MEMSLIDES_WORKSPACE",)
_LOCAL_ABSOLUTE_PATH_RE = re.compile(
    r'''(?:src|href)=["'](\/[^"'<>]+)["']|url\((["']?)(\/[^"')]+)\2\)''',
    re.IGNORECASE,
)


def _detect_language_label(markdown: str) -> str:
    """Best-effort language label detection for manuscript inspection."""
    sample = (markdown or "")[:1000]
    cjk = sum(1 for ch in sample if "\u4e00" <= ch <= "\u9fff")
    latin = sum(1 for ch in sample if ch.isascii() and ch.isalpha())
    if cjk > latin:
        return "zh"
    if latin > 0:
        return "en"
    return "unknown"


def _iter_agent_context_base_dirs() -> list[Path]:
    """Return candidate directories that may contain agent context files.

    Tool subprocesses should usually run inside the workspace, but we still
    check the workspace env vars first so guardrails remain reliable even if
    the current working directory drifts.
    """
    candidates: list[Path] = []
    for env_key in _WORKSPACE_ENV_KEYS:
        env_value = os.environ.get(env_key, "").strip()
        if not env_value:
            continue
        try:
            base_dir = Path(env_value).resolve()
        except Exception:
            continue
        if base_dir not in candidates:
            candidates.append(base_dir)

    cwd = Path.cwd().resolve()
    if cwd not in candidates:
        candidates.append(cwd)
    return candidates


def _read_current_agent_context_from(base_dir: Path) -> dict[str, str]:
    """Read persisted current-agent context from a specific workspace-like dir."""
    try:
        raw = (base_dir / _AGENT_CONTEXT_FILE).read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return {
                "agent_name": str(data.get("agent_name", "") or ""),
                "model_ref": str(data.get("model_ref", "") or ""),
            }
    except Exception:
        pass
    return {}


def set_current_modify_context(
    *,
    workspace: Path | str | None = None,
    target_slide_paths: list[str] | None = None,
    operation_kind: str = "",
    coverage_required: bool = False,
    raw_user_message: str = "",
    user_preference_rule_specs: list[dict[str, Any]] | None = None,
    user_preference_colors: list[str] | None = None,
    diagram_contract: dict[str, Any] | None = None,
    rewrite_decision: dict[str, Any] | None = None,
    strict_visual_preference_eval: bool = False,
) -> None:
    """Persist the current RevisionEditor plan for cross-process tools."""
    try:
        base_dir = Path(workspace) if workspace else Path.cwd()
        payload = {
            "target_slide_paths": [str(path) for path in (target_slide_paths or [])],
            "operation_kind": str(operation_kind or ""),
            "coverage_required": bool(coverage_required),
            "raw_user_message": str(raw_user_message or ""),
            "user_preference_rule_specs": user_preference_rule_specs or [],
            "user_preference_colors": user_preference_colors or [],
            "diagram_contract": diagram_contract or {},
            "rewrite_decision": rewrite_decision or {},
            "strict_visual_preference_eval": bool(strict_visual_preference_eval),
        }
        (base_dir / _MODIFY_CONTEXT_FILE).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _read_current_modify_context() -> dict[str, Any]:
    candidates = _iter_agent_context_base_dirs()
    workspace_env = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
    if workspace_env:
        try:
            env_path = Path(workspace_env).resolve()
            if env_path not in candidates:
                candidates.insert(0, env_path)
        except Exception:
            pass
    for base_dir in candidates:
        try:
            raw = (base_dir / _MODIFY_CONTEXT_FILE).read_text(encoding="utf-8").strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _read_current_modify_context_for_path(path: Path | None) -> dict[str, Any]:
    candidates: list[Path] = []
    if path is not None:
        try:
            start = path.parent if path.suffix else path
            for candidate in [start, *list(start.parents)[:6]]:
                if candidate not in candidates:
                    candidates.append(candidate)
        except Exception:
            pass
    for candidate in [*_iter_agent_context_base_dirs(), Path.cwd().resolve()]:
        if candidate not in candidates:
            candidates.append(candidate)
    for base_dir in candidates:
        try:
            raw = (base_dir / _MODIFY_CONTEXT_FILE).read_text(encoding="utf-8").strip()
            if not raw:
                continue
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _path_matches_modify_target(path: Path, raw_targets: list[Any]) -> bool:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    for raw in raw_targets or []:
        text = str(raw or "").strip()
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            for base_dir in _iter_agent_context_base_dirs():
                try:
                    if (base_dir / candidate).resolve() == resolved:
                        return True
                except Exception:
                    continue
        try:
            if candidate.resolve() == resolved:
                return True
        except Exception:
            if candidate.as_posix() == path.as_posix() or candidate.name == path.name:
                return True
    return False


def _revision_inspect_completion_hint(html_path: Path) -> str | None:
    if get_current_agent() != "RevisionEditor":
        return None
    context = _read_current_modify_context()
    targets = context.get("target_slide_paths") or []
    operation_kind = str(context.get("operation_kind", "") or "").strip()
    if not isinstance(targets, list) or not targets:
        if operation_kind == "structural":
            return (
                "\n💡 Current slide passed for this RevisionEditor structural round. "
                "Continue completing and inspecting the inserted/affected slide files, then call `finalize` only after the structural request is fully satisfied."
            )
        return "\n💡 Current slide passed for this RevisionEditor round. If the requested change is correct, call `finalize` now."
    if _path_matches_modify_target(html_path, targets):
        if len(targets) > 1:
            return (
                "\n💡 This target slide passed for the current RevisionEditor round. "
                "Continue until every current target slide has been updated and inspected; then call `finalize`."
            )
        return "\n💡 Target slide passed for this RevisionEditor round. If all requested changes are correct, call `finalize` now."
    target_preview = ", ".join(str(item) for item in targets[:3])
    return (
        "\n💡 This slide passed, but it is not one of the current RevisionEditor target slides. "
        f"Continue with the current target slide(s): {target_preview}."
    )


def _ensure_template_context():
    """Ensure template context is available in MCP subprocess.

    Delegates to template_tools._ensure_context() which handles both
    contextvars and cross-process file fallback.
    Returns (builder, skill) or (None, None).
    """
    from memslides.tools.template_tools import _ensure_context
    builder = _ensure_context()
    if builder and builder.skill:
        return builder, builder.skill
    return None, None


def set_current_agent(
    agent_name: str,
    workspace: Path | str | None = None,
    model_ref: str | None = None,
) -> None:
    """Set the current agent name for finalize() behavior.

    Writes to both module-level variable (in-process) and .current_agent file
    (cross-process, for MCP subprocess to read).

    Args:
        agent_name: e.g. 'Researcher', 'DeckDesigner', 'TemplatePlanner', 'RevisionEditor'
        workspace: workspace directory (required when called from main.py;
                   omit when cwd is already the workspace)
        model_ref: actual resolved LLM config key used by the agent
    """
    global _current_agent_name
    _current_agent_name = agent_name
    try:
        base_dir = Path(workspace) if workspace else Path.cwd()
        (base_dir / _AGENT_NAME_FILE).write_text(agent_name, encoding="utf-8")
        (base_dir / _AGENT_CONTEXT_FILE).write_text(
            json.dumps(
                {"agent_name": agent_name, "model_ref": model_ref or ""},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _read_current_agent_context() -> dict[str, str]:
    """Read persisted current-agent context for cross-process tool execution."""
    for base_dir in _iter_agent_context_base_dirs():
        context = _read_current_agent_context_from(base_dir)
        if context.get("agent_name", "").strip() or context.get("model_ref", "").strip():
            return context
    return {}


def get_current_agent() -> str:
    """Get the current agent name.

    Reads from .current_agent file first (cross-process), falls back to
    module-level variable (in-process).
    """
    context = _read_current_agent_context()
    agent_name = context.get("agent_name", "").strip()
    if agent_name:
        return agent_name
    for base_dir in _iter_agent_context_base_dirs():
        try:
            name = (base_dir / _AGENT_NAME_FILE).read_text(encoding="utf-8").strip()
            if name:
                return name
        except Exception:
            pass
    return _current_agent_name


def get_current_agent_model_ref() -> str:
    """Get the resolved model-ref associated with the current agent."""
    context = _read_current_agent_context()
    return context.get("model_ref", "").strip()


def _read_intermediate_output(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> dict:
    """Load intermediate_output.json from the actual workspace when available."""
    candidate_dirs: list[Path] = []
    seen: set[str] = set()

    def _add(base_dir: Path | str | None) -> None:
        if not base_dir:
            return
        try:
            resolved = Path(base_dir).resolve()
        except Exception:
            return
        key = str(resolved)
        if key in seen:
            return
        seen.add(key)
        candidate_dirs.append(resolved)

    _add(workspace)
    for base_dir in _iter_agent_context_base_dirs():
        _add(base_dir)
    inferred_root = _infer_workspace_root_from_anchor(anchor)
    if inferred_root is not None:
        _add(inferred_root)
    _add(Path.cwd())

    for base_dir in candidate_dirs:
        try:
            raw = (base_dir / _INTERMEDIATE_OUTPUT_FILE).read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _get_active_slide_dir(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> Path | None:
    """Return the current canonical slide directory for this workspace."""
    workspace_root = _resolve_workspace_root(workspace, anchor=anchor)
    intermediate_output = _read_intermediate_output(workspace_root, anchor=anchor)
    slide_html_dir = str(intermediate_output.get("slide_html_dir", "") or "").strip()
    if slide_html_dir:
        slide_dir = Path(slide_html_dir)
        if not slide_dir.is_absolute():
            slide_dir = workspace_root / slide_dir
        if slide_dir.exists() and slide_dir.is_dir():
            return slide_dir.resolve()

    candidates: list[Path] = []
    for rel in ("outputs", "outputs/slides", "slides"):
        candidate = workspace_root / rel
        if candidate.is_dir() and any(candidate.glob("slide_*.html")):
            candidates.append(candidate.resolve())
    return candidates[0] if candidates else None


def _resolve_slide_alias_path(path: Path) -> Path | None:
    """Map stale slide aliases to the current slide_html_dir when possible."""
    if path.exists():
        return None
    if path.suffix.lower() != ".html":
        return None
    if re.fullmatch(r"slide_.*\.html", path.name, re.IGNORECASE) is None:
        return None

    slide_dir = _get_active_slide_dir(anchor=path)
    if slide_dir is None:
        return None

    if path.parent in (Path("."), Path("")):
        return slide_dir / path.name

    parent_tokens = {part.lower() for part in path.parts[:-1]}
    if {"slides", "outputs"} & parent_tokens:
        return slide_dir / path.name

    if path.is_absolute() and not path.parent.exists():
        return slide_dir / path.name
    return None


def _normalize_path(file_path: str) -> Path:
    """Normalize a file path and remap stale slide aliases when possible.

    LLMs often generate '/workspace/...' as if running in a Docker container.
    Since the deck runtime does os.chdir(actual_workspace) at startup, we strip
    the /workspace prefix and resolve the remainder as a relative path.
    """
    p = file_path.strip()
    # Strip common hallucinated prefixes
    for prefix in ("/workspace/", "/workspace"):
        if p.startswith(prefix):
            p = p[len(prefix):]
            break
    path = Path(p)
    slide_alias = _resolve_slide_alias_path(path)
    if slide_alias is not None:
        return slide_alias
    # If the path is absolute:
    # 1. If it exists, use it as-is
    # 2. If it doesn't exist but parent exists, use it as-is (new file in existing dir)
    # 3. Otherwise, use only the filename under CWD (prevent nesting absolute paths)
    if path.is_absolute() and not path.exists():
        if path.parent.exists():
            # Parent dir exists — this is a new file, keep absolute path
            pass
        else:
            # Neither file nor parent exists — likely a hallucinated path
            # Only use the filename part to avoid recreating a caller's
            # machine-specific cache path inside the workspace.
            path = Path.cwd() / path.name
    return path


def _resolve_local_html_asset_path(
    raw_path: str,
    *,
    html_path: Path,
) -> Path | None:
    """Resolve a local HTML asset path against the slide file and workspace."""
    text = str(raw_path or "").strip().strip("\"'")
    if not text or re.match(r"^(?:https?://|data:)", text, re.IGNORECASE):
        return None

    html_file = html_path.resolve()
    workspace_root = _resolve_workspace_root(anchor=html_file)
    html_dir = html_file.parent

    variants: list[str] = [text]
    normalized = text
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    elif normalized == "/workspace":
        normalized = ""
    elif normalized.startswith("/workspace"):
        normalized = normalized[len("/workspace") :].lstrip("/")
    elif normalized.startswith("workspace/"):
        normalized = normalized[len("workspace/") :]
    elif normalized == "workspace":
        normalized = ""
    if normalized and normalized not in variants:
        variants.append(normalized)

    stripped = normalized or text
    while stripped.startswith("../"):
        stripped = stripped[3:]
    if stripped and stripped not in variants:
        variants.append(stripped)

    for variant in variants:
        if not variant:
            continue
        candidate = Path(variant)
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            try:
                rel_variant = candidate.resolve().relative_to(workspace_root.resolve()).as_posix()
            except Exception:
                continue
            if rel_variant not in variants:
                variants.append(rel_variant)
            continue

        for base_dir in (html_dir, workspace_root, Path.cwd().resolve()):
            resolved = (base_dir / candidate).resolve()
            if resolved.exists():
                return resolved
    return None


def _html_relative_asset_src(asset_path: Path, *, html_path: Path) -> str:
    """Return a browser-usable src from an HTML slide to a local asset."""
    try:
        return os.path.relpath(asset_path.resolve(), start=html_path.resolve().parent).replace(os.sep, "/")
    except Exception:
        return asset_path.resolve().as_posix()


def _normalize_html_image_sources(content: str, *, html_path: Path) -> str:
    """Rewrite local HTML asset paths to sources resolvable from the slide HTML."""
    html_file = html_path.resolve()

    def _normalized_src(src_val: str) -> str | None:
        resolved = _resolve_local_html_asset_path(src_val, html_path=html_file)
        if resolved is None:
            return None
        return _html_relative_asset_src(resolved, html_path=html_file)

    def _fix_html_img_src(match: re.Match[str]) -> str:
        prefix, src_val, suffix = match.group(1), match.group(2), match.group(3)
        rewritten = _normalized_src(src_val)
        if rewritten is None:
            return match.group(0)
        return f"{prefix}{rewritten}{suffix}"

    def _fix_css_url(match: re.Match[str]) -> str:
        raw = match.group(2)
        rewritten = _normalized_src(raw)
        if rewritten is None:
            return match.group(0)
        quote = match.group(1) or ""
        return f"url({quote}{rewritten}{quote})"

    content = re.sub(r'(src=["\'])([^"\']+)(["\'])', _fix_html_img_src, content)
    return re.sub(r'url\(\s*([\'"]?)([^\'")]+)\1\s*\)', _fix_css_url, content)


def _missing_local_html_assets(html_path: Path, content: str | None = None) -> list[str]:
    """Return local img/css-url references that cannot be resolved from a slide HTML."""
    html_file = html_path.resolve()
    text = content if content is not None else html_file.read_text(encoding="utf-8", errors="replace")
    refs: list[str] = []
    for match in re.finditer(r'(?:src=["\']([^"\']+)["\']|url\(\s*[\'"]?([^\'")]+)[\'"]?\s*\))', text, re.IGNORECASE):
        raw = (match.group(1) or match.group(2) or "").strip()
        if not raw or re.match(r"^(?:https?://|data:|#)", raw, re.IGNORECASE):
            continue
        if _resolve_local_html_asset_path(raw, html_path=html_file) is None and raw not in refs:
            refs.append(raw)
    return refs


def _format_missing_asset_error(html_path: Path, missing: list[str]) -> str:
    preview = ", ".join(missing[:5])
    suffix = f"; and {len(missing) - 5} more" if len(missing) > 5 else ""
    return (
        "Error: MISSING_LOCAL_ASSET. "
        f"{html_path.name} references local image/CSS assets that cannot be resolved from the HTML file: "
        f"{preview}{suffix}. "
        "Use paths that resolve from the HTML file location, for example `../converted/...` "
        "when writing files under `outputs/`."
    )



def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _extract_local_absolute_paths_from_html(content: str) -> list[str]:
    paths: list[str] = []
    for match in _LOCAL_ABSOLUTE_PATH_RE.finditer(content or ""):
        candidate = match.group(1) or match.group(3) or ""
        candidate = candidate.strip()
        if candidate and candidate not in paths:
            paths.append(candidate)
    return paths


def _workspace_contract_error(
    *,
    action: str,
    path: Path | None = None,
    html_content: str | None = None,
) -> str | None:
    env_workspace = os.environ.get("MEMSLIDES_WORKSPACE", "").strip()
    cwd_workspace = Path.cwd().resolve()
    workspace_root = Path(env_workspace).resolve() if env_workspace else cwd_workspace

    if env_workspace and cwd_workspace != workspace_root:
        return _structured_patch_error(
            "WORKSPACE_CONTEXT_MISMATCH",
            f"{action} rejected because MEMSLIDES_WORKSPACE and cwd point to different workspaces.",
            action=action,
            cwd=str(cwd_workspace),
            memslides_workspace=str(workspace_root),
        )

    if path is not None:
        resolved_path = path.resolve()
        if not _path_is_within(resolved_path, workspace_root):
            return _structured_patch_error(
                "WORKSPACE_CONTEXT_MISMATCH",
                f"{action} rejected because the target path is outside the active workspace.",
                action=action,
                cwd=str(cwd_workspace),
                memslides_workspace=str(workspace_root),
                target_path=str(resolved_path),
            )

    intermediate_output = _read_intermediate_output(workspace_root, anchor=path)
    slide_html_dir = str(intermediate_output.get("slide_html_dir", "") or "").strip()
    if slide_html_dir:
        slide_dir = Path(slide_html_dir)
        if not slide_dir.is_absolute():
            slide_dir = workspace_root / slide_dir
        slide_dir = slide_dir.resolve()
        if not _path_is_within(slide_dir, workspace_root):
            return _structured_patch_error(
                "WORKSPACE_CONTEXT_MISMATCH",
                f"{action} rejected because slide_html_dir points outside the active workspace.",
                action=action,
                cwd=str(cwd_workspace),
                memslides_workspace=str(workspace_root),
                slide_html_dir=str(slide_dir),
            )

    if html_content:
        bad_refs = []
        for raw in _extract_local_absolute_paths_from_html(html_content):
            try:
                candidate = Path(raw).resolve()
            except Exception:
                continue
            if not _path_is_within(candidate, workspace_root):
                bad_refs.append(str(candidate))
        if bad_refs:
            return _structured_patch_error(
                "WORKSPACE_CONTEXT_MISMATCH",
                f"{action} rejected because HTML references paths outside the active workspace.",
                action=action,
                cwd=str(cwd_workspace),
                memslides_workspace=str(workspace_root),
                offending_paths=bad_refs[:10],
            )
    return None


def _looks_like_workspace_root(path: Path) -> bool:
    return any(
        (path / marker).exists()
        for marker in (
            ".input_request.json",
            _AGENT_CONTEXT_FILE,
            _INTERMEDIATE_OUTPUT_FILE,
            ".history",
            "outputs",
            "attachments",
        )
    )


def _infer_workspace_root_from_anchor(anchor: Path | str | None) -> Path | None:
    if anchor is None:
        return None
    anchor_path = Path(anchor)
    if not anchor_path.is_absolute():
        return None

    current = anchor_path if anchor_path.is_dir() else anchor_path.parent
    for candidate in (current, *current.parents):
        if _looks_like_workspace_root(candidate):
            return candidate
    return current


def _resolve_workspace_root(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> Path:
    if workspace is not None:
        return Path(workspace).resolve()

    for env_key in _WORKSPACE_ENV_KEYS:
        env_value = os.environ.get(env_key, "").strip()
        if env_value:
            return Path(env_value).resolve()

    inferred = _infer_workspace_root_from_anchor(anchor)
    if inferred is not None:
        return inferred.resolve()

    return Path.cwd().resolve()


def _read_input_request(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> dict[str, Any]:
    root = _resolve_workspace_root(workspace, anchor=anchor)
    path = root / ".input_request.json"
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _coerce_positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return None
    return parsed if parsed > 0 else None


def _expected_num_pages(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> int | None:
    request = _read_input_request(workspace, anchor=anchor)
    return _coerce_positive_int(request.get("num_pages"))


def _strip_leading_markdown_title_preamble(markdown: str) -> str:
    """Ignore deck-level H1 title blocks before the first real slide separator."""
    normalized = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    first_nonempty = next((idx for idx, line in enumerate(lines) if line.strip()), None)
    if first_nonempty is None:
        return normalized

    second_nonempty = next(
        (idx for idx in range(first_nonempty + 1, len(lines)) if lines[idx].strip()),
        None,
    )
    if second_nonempty is None:
        return normalized

    first_line = lines[first_nonempty].strip()
    second_line = lines[second_nonempty].strip()
    if re.fullmatch(r"#\s+\S.*", first_line) and second_line == "---":
        return "\n".join(lines[second_nonempty + 1 :])
    return normalized


def _split_markdown_slide_pages(markdown: str) -> list[str]:
    normalized = _strip_leading_markdown_title_preamble(markdown)
    return [
        chunk.strip()
        for chunk in re.split(r"(?m)^\s*---\s*$", normalized)
        if chunk.strip()
    ]


def _find_design_plan_path(workspace: Path | str | None = None) -> Path | None:
    """Return the first existing design-plan path in the workspace, if any."""
    workspace_root = _resolve_workspace_root(workspace)
    for candidate in _DESIGN_PLAN_CANDIDATES:
        raw_path = Path(candidate)
        plan_path = raw_path if raw_path.is_absolute() else workspace_root / raw_path
        if plan_path.exists():
            return plan_path.resolve()
    return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_text_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _design_plan_history_dir(workspace: Path | str | None = None) -> Path:
    root = _resolve_workspace_root(workspace)
    return root / ".history" / "design_plan"


def _design_plan_state_path(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> Path:
    root = _resolve_workspace_root(workspace, anchor=anchor)
    return root / _DESIGN_PLAN_STATE_FILE


def _design_plan_relative_label(
    path: Path,
    workspace: Path | str | None = None,
) -> str:
    root = _resolve_workspace_root(workspace, anchor=path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _control_document_state_path(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> Path:
    root = _resolve_workspace_root(workspace, anchor=anchor)
    return root / _CONTROL_DOCUMENT_STATE_FILE


def _read_control_document_state(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> dict[str, Any]:
    path = _control_document_state_path(workspace, anchor=anchor)
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_control_document_state(
    state: dict[str, Any],
    workspace: Path | str | None = None,
) -> None:
    path = _control_document_state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state or {})
    payload.setdefault("version", 1)
    payload["updated_at"] = _utc_now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _control_document_relative_label(
    path: Path,
    workspace: Path | str | None = None,
) -> str:
    root = _resolve_workspace_root(workspace, anchor=path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _upsert_control_document_entry(
    state: dict[str, Any],
    *,
    workspace_root: Path,
    document_path: Path,
    content_hash: str,
    source: str,
    required_for_html_prewrite: bool,
) -> dict[str, Any]:
    documents = state.setdefault("documents", {})
    label = _control_document_relative_label(document_path, workspace_root)
    entry = documents.get(label)
    if not isinstance(entry, dict):
        entry = {
            "path": label,
            "name": document_path.name,
            "source": source,
            "required_for_html_prewrite": bool(required_for_html_prewrite),
            "current_hash": content_hash,
            "last_read_hash": "",
            "last_read_at": "",
            "read_after_last_change": False,
            "created_at": _utc_now_iso(),
        }
        documents[label] = entry
    else:
        previous_hash = str(entry.get("current_hash", "") or "")
        entry["path"] = label
        entry["name"] = document_path.name
        entry["source"] = source or str(entry.get("source", "") or "")
        entry["required_for_html_prewrite"] = bool(
            entry.get("required_for_html_prewrite") or required_for_html_prewrite
        )
        entry["current_hash"] = content_hash
        if previous_hash != content_hash:
            entry["read_after_last_change"] = bool(
                entry.get("last_read_hash")
                and entry.get("last_read_hash") == content_hash
                and entry.get("last_read_at")
            )
    return entry


def initialize_control_document_tracking(
    workspace: Path | str,
    document_path: Path | str,
    content: str,
    *,
    source: str,
    required_for_html_prewrite: bool = False,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(workspace, anchor=document_path)
    path = Path(document_path)
    content_hash = _compute_text_hash(content)
    state = _read_control_document_state(workspace_root)
    entry = _upsert_control_document_entry(
        state,
        workspace_root=workspace_root,
        document_path=path,
        content_hash=content_hash,
        source=source,
        required_for_html_prewrite=required_for_html_prewrite,
    )
    entry["current_hash"] = content_hash
    entry["read_after_last_change"] = False
    _write_control_document_state(state, workspace_root)
    return entry


def _ensure_control_document_tracking_from_disk(
    document_path: Path,
    *,
    required_for_html_prewrite: bool = False,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(anchor=document_path)
    state = _read_control_document_state(workspace_root)
    if not document_path.exists():
        return {}
    try:
        content_hash = _compute_text_hash(document_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    entry = _upsert_control_document_entry(
        state,
        workspace_root=workspace_root,
        document_path=document_path,
        content_hash=content_hash,
        source="existing_file",
        required_for_html_prewrite=required_for_html_prewrite,
    )
    _write_control_document_state(state, workspace_root)
    return entry


def _record_control_document_read(path: Path) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(anchor=path)
    state = _read_control_document_state(workspace_root)
    if not path.exists():
        return {}
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return {}
    entry = _upsert_control_document_entry(
        state,
        workspace_root=workspace_root,
        document_path=path,
        content_hash=_compute_text_hash(content),
        source="read_file",
        required_for_html_prewrite=False,
    )
    entry["last_read_hash"] = entry.get("current_hash", "")
    entry["last_read_at"] = _utc_now_iso()
    entry["read_after_last_change"] = True
    _write_control_document_state(state, workspace_root)
    return entry


def _is_page_execution_plan_target(path: Path) -> bool:
    return path.name.lower() == "page_execution_plan.md"


def _find_page_execution_plan_path(
    workspace: Path | str | None = None,
) -> Path | None:
    root = _resolve_workspace_root(workspace)
    candidate = root / "page_execution_plan.md"
    return candidate if candidate.exists() else None


def _read_design_plan_state(
    workspace: Path | str | None = None,
    *,
    anchor: Path | str | None = None,
) -> dict[str, Any]:
    path = _design_plan_state_path(workspace, anchor=anchor)
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return {}
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_design_plan_state(
    state: dict[str, Any],
    workspace: Path | str | None = None,
) -> None:
    path = _design_plan_state_path(
        workspace,
        anchor=state.get("current_path") if isinstance(state, dict) else None,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _normalize_design_heading_text(text: str) -> str:
    normalized = str(text or "").strip()
    normalized = normalized.replace("’", "'").replace("‘", "'")
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("don't", "dont").replace("don’t", "dont")
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized.lower())
    return " ".join(normalized.split())


def _design_heading_matches_alias(heading: str, alias: str) -> bool:
    heading = _normalize_design_heading_text(heading)
    alias = _normalize_design_heading_text(alias)
    if not heading or not alias:
        return False
    candidates = {heading}
    candidates.add(re.sub(r"^(?:section|part|chapter)?\s*\d+\s+", "", heading).strip())
    candidates.add(re.sub(r"^\d+\s+", "", heading).strip())
    for candidate in candidates:
        if candidate == alias or candidate.startswith(alias + " "):
            return True
    return False


def _design_plan_required_sections() -> list[tuple[str, str, set[str]]]:
    return [
        ("design_goal", "Design Goal", {"design goal"}),
        ("theme_keywords", "Theme Keywords", {"theme keywords"}),
        ("color_palette", "Color Palette", {"color palette"}),
        ("typography", "Typography", {"typography"}),
        ("spacing_grid", "Spacing & Grid", {"spacing and grid", "spacing grid"}),
        ("page_archetypes", "Page Archetypes", {"page archetypes"}),
        ("component_rules", "Component Rules", {"component rules"}),
        ("do_dont", "Do / Don't", {"do dont", "do don t"}),
    ]


def _validate_structured_design_plan_markdown(content: str) -> dict[str, Any]:
    lines = str(content or "").splitlines()
    headings: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line.strip())
        if not match:
            continue
        headings.append(
            {
                "line_index": idx,
                "level": len(match.group(1)),
                "raw": match.group(2).strip(),
                "normalized": _normalize_design_heading_text(match.group(2)),
            }
        )

    title_present = any(_design_heading_matches_alias(heading["normalized"], "design plan") for heading in headings)

    alias_to_key: dict[str, str] = {}
    key_to_label: dict[str, str] = {}
    for key, label, aliases in _design_plan_required_sections():
        key_to_label[key] = label
        alias_to_key[_normalize_design_heading_text(label)] = key
        for alias in aliases:
            alias_to_key[_normalize_design_heading_text(alias)] = key

    section_hits: dict[str, dict[str, Any]] = {}
    for idx, heading in enumerate(headings):
        normalized_heading = str(heading["normalized"])
        key = alias_to_key.get(normalized_heading)
        if not key:
            for alias, alias_key in alias_to_key.items():
                if _design_heading_matches_alias(normalized_heading, alias):
                    key = alias_key
                    break
        if not key or key in section_hits:
            continue
        next_line_index = len(lines)
        if idx + 1 < len(headings):
            next_line_index = int(headings[idx + 1]["line_index"])
        body_lines = [
            line
            for line in lines[int(heading["line_index"]) + 1 : next_line_index]
            if line.strip()
        ]
        has_body = any(not re.match(r"^#{1,6}\s+", line.strip()) for line in body_lines)
        section_hits[key] = {
            "label": key_to_label[key],
            "line_index": int(heading["line_index"]),
            "has_body": has_body,
        }

    missing_sections = [
        label for key, label, _ in _design_plan_required_sections()
        if key not in section_hits
    ]
    empty_sections = [
        hit["label"] for hit in section_hits.values()
        if not bool(hit["has_body"])
    ]

    return {
        "valid": bool(title_present and not missing_sections and not empty_sections),
        "title_present": title_present,
        "missing_sections": missing_sections,
        "empty_sections": empty_sections,
        "present_sections": [
            hit["label"]
            for key, _, _ in _design_plan_required_sections()
            if (hit := section_hits.get(key))
        ],
    }


def _validate_template_design_plan_markdown(content: str) -> dict[str, Any]:
    stripped = str(content or "").strip()
    return {
        "valid": bool(stripped),
        "title_present": "design plan" in _normalize_design_heading_text(stripped[:200]),
        "missing_sections": [],
        "empty_sections": [],
        "present_sections": [],
    }


def _validate_design_plan_markdown(
    content: str,
    *,
    mode: Literal["structured", "template"] = "structured",
) -> dict[str, Any]:
    if mode == "template":
        return _validate_template_design_plan_markdown(content)
    return _validate_structured_design_plan_markdown(content)


def _format_design_plan_validation_error(validation: dict[str, Any]) -> str:
    parts: list[str] = []
    if not validation.get("title_present"):
        parts.append("missing `# Design Plan`")
    missing_sections = validation.get("missing_sections") or []
    if missing_sections:
        parts.append("missing sections: " + ", ".join(str(item) for item in missing_sections))
    empty_sections = validation.get("empty_sections") or []
    if empty_sections:
        parts.append("empty sections: " + ", ".join(str(item) for item in empty_sections))
    return "; ".join(parts) or "invalid structure"


def _next_design_plan_snapshot_index(workspace: Path | str | None = None) -> int:
    history_dir = _design_plan_history_dir(workspace)
    if not history_dir.exists():
        return 1
    max_index = 0
    for path in history_dir.glob("design_plan_v*.md"):
        match = re.search(r"design_plan_v(\d+)\.md$", path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _snapshot_design_plan_version(
    *,
    content: str,
    validation: dict[str, Any],
    workspace: Path | str | None = None,
    source: str,
    plan_path: Path | None = None,
) -> dict[str, str]:
    history_dir = _design_plan_history_dir(workspace)
    history_dir.mkdir(parents=True, exist_ok=True)
    index = _next_design_plan_snapshot_index(workspace)
    md_path = history_dir / f"design_plan_v{index:02d}.md"
    meta_path = history_dir / f"design_plan_v{index:02d}.json"
    md_path.write_text(content, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "version": index,
                "created_at": _utc_now_iso(),
                "source": source,
                "plan_path": _design_plan_relative_label(plan_path, workspace)
                if plan_path is not None
                else "",
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "markdown_path": str(md_path),
        "metadata_path": str(meta_path),
    }


def initialize_design_plan_tracking(
    workspace: Path | str,
    plan_path: Path | str,
    content: str,
    *,
    source: str,
    requires_refinement: bool,
) -> dict[str, Any]:
    workspace_root = Path(workspace)
    path = Path(plan_path)
    validation_mode: Literal["structured", "template"] = (
        "template" if source == "template_generated" else "structured"
    )
    validation = _validate_design_plan_markdown(content, mode=validation_mode)
    content_hash = _compute_text_hash(content)
    snapshot_paths = _snapshot_design_plan_version(
        content=content,
        validation=validation,
        workspace=workspace_root,
        source=source,
        plan_path=path,
    )
    state = {
        "version": 1,
        "current_path": _design_plan_relative_label(path, workspace_root),
        "source": source,
        "requires_refinement": bool(requires_refinement),
        "scaffold_hash": content_hash if requires_refinement else "",
        "current_hash": content_hash,
        "last_write_hash": content_hash,
        "last_write_at": _utc_now_iso(),
        "last_read_hash": "",
        "last_read_at": "",
        "read_after_last_write": False,
        "status": (
            "scaffold_created"
            if requires_refinement
            else ("template_generated" if source == "template_generated" else "plan_written")
        ),
        "validation_mode": validation_mode,
        "validation": validation,
        "latest_snapshot": snapshot_paths,
    }
    _write_design_plan_state(state, workspace_root)
    initialize_control_document_tracking(
        workspace_root,
        path,
        content,
        source=source,
        required_for_html_prewrite=False,
    )
    return state


def _ensure_design_plan_state_from_disk(
    workspace: Path | str | None = None,
) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(workspace)
    state = _read_design_plan_state(workspace_root)
    plan_path = _find_design_plan_path(workspace_root)
    if plan_path is None or not plan_path.exists():
        return state

    content = plan_path.read_text(encoding="utf-8")
    content_hash = _compute_text_hash(content)
    validation_mode = str(state.get("validation_mode", "structured") or "structured")
    validation = _validate_design_plan_markdown(
        content,
        mode="template" if validation_mode == "template" else "structured",
    )
    plan_label = _design_plan_relative_label(plan_path, workspace_root)
    changed = False

    if not state:
        state = {
            "version": 1,
            "current_path": plan_label,
            "source": "existing_file",
            "requires_refinement": False,
            "scaffold_hash": "",
            "current_hash": content_hash,
            "last_write_hash": content_hash,
            "last_write_at": "",
            "last_read_hash": "",
            "last_read_at": "",
            "read_after_last_write": False,
            "status": "existing_detected" if validation.get("valid") else "invalid",
            "validation_mode": "structured",
            "validation": validation,
            "latest_snapshot": {},
        }
        changed = True
    else:
        if state.get("current_path") != plan_label:
            state["current_path"] = plan_label
            changed = True
        if state.get("current_hash") != content_hash:
            state["current_hash"] = content_hash
            state["validation"] = validation
            state["read_after_last_write"] = bool(
                state.get("last_read_hash")
                and state.get("last_read_hash") == content_hash
                and state.get("last_read_at")
            )
            if not state.get("read_after_last_write"):
                if state.get("requires_refinement") and content_hash == state.get("scaffold_hash"):
                    state["status"] = "scaffold_created"
                else:
                    state["status"] = "plan_written" if validation.get("valid") else "invalid"
            changed = True
        if state.get("validation") != validation:
            state["validation"] = validation
            changed = True

    if changed:
        _write_design_plan_state(state, workspace_root)
    return state


def _record_design_plan_write(path: Path, content: str) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(anchor=path)
    state = _ensure_design_plan_state_from_disk(workspace_root)
    validation_mode = str(state.get("validation_mode", "structured") or "structured")
    validation = _validate_design_plan_markdown(
        content,
        mode="template" if validation_mode == "template" else "structured",
    )
    content_hash = _compute_text_hash(content)
    source = "agent_write" if get_current_agent().strip().lower() == "design" else "manual_write"
    snapshot_paths = _snapshot_design_plan_version(
        content=content,
        validation=validation,
        workspace=workspace_root,
        source=source,
        plan_path=path,
    )
    state.update(
        {
            "version": 1,
            "current_path": _design_plan_relative_label(path, workspace_root),
            "source": source,
            "validation_mode": validation_mode,
            "current_hash": content_hash,
            "last_write_hash": content_hash,
            "last_write_at": _utc_now_iso(),
            "read_after_last_write": False,
            "status": "plan_written" if validation.get("valid") else "invalid",
            "validation": validation,
            "latest_snapshot": snapshot_paths,
        }
    )
    _write_design_plan_state(state, workspace_root)
    initialize_control_document_tracking(
        workspace_root,
        path,
        content,
        source=source,
        required_for_html_prewrite=False,
    )
    return state


def _record_design_plan_read(path: Path) -> dict[str, Any]:
    workspace_root = _resolve_workspace_root(anchor=path)
    state = _ensure_design_plan_state_from_disk(workspace_root)
    if not state:
        return {}

    content = path.read_text(encoding="utf-8")
    content_hash = _compute_text_hash(content)
    read_after_last_write = bool(state.get("current_hash") and state.get("current_hash") == content_hash)
    scaffold_hash = str(state.get("scaffold_hash", "") or "")
    requires_refinement = bool(state.get("requires_refinement"))
    validation_mode = str(state.get("validation_mode", "structured") or "structured")
    validation = (
        state.get("validation")
        if isinstance(state.get("validation"), dict)
        else _validate_design_plan_markdown(
            content,
            mode="template" if validation_mode == "template" else "structured",
        )
    )

    unlocked = bool(
        validation.get("valid")
        and read_after_last_write
        and (not requires_refinement or not scaffold_hash or content_hash != scaffold_hash)
    )

    state.update(
        {
            "last_read_hash": content_hash,
            "last_read_at": _utc_now_iso(),
            "read_after_last_write": read_after_last_write,
            "status": "unlocked" if unlocked else "plan_read_back",
            "validation": validation,
        }
    )
    _write_design_plan_state(state, workspace_root)
    _record_control_document_read(path)
    return state


def _is_design_plan_target(path: Path) -> bool:
    normalized = _design_plan_relative_label(path)
    if normalized in _DESIGN_PLAN_CANDIDATES:
        return True
    return path.name.lower() in {"design_plan.md", "design-plan.md"}


def _design_stage_markdown_target_error(path: Path) -> str | None:
    """Restrict DeckDesigner-stage markdown writes to design_plan.md only."""
    if get_current_agent().strip().lower() != "design":
        return None
    if _is_design_plan_target(path):
        return None

    target_label = _design_plan_relative_label(path)
    return (
        "Error: DeckDesigner stage treats the manuscript and all non-design-plan markdown files as read-only. "
        f"`write_markdown_file` may only target `design_plan.md` (got: `{target_label}`). "
        "Overwrite `design_plan.md` with a manuscript-specific structured design plan, "
        "read it back with `read_file`, and do not write any slide HTML until that is complete."
    )


def _design_html_precondition_error() -> str | None:
    """Block DeckDesigner-stage HTML generation until design_plan state is unlocked."""
    if get_current_agent().strip().lower() != "design":
        return None
    workspace_root = _resolve_workspace_root()
    plan_path = _find_design_plan_path(workspace_root)
    if plan_path is None:
        return (
            "Error: DeckDesigner stage is blocked because `design_plan.md` is missing. "
            "Return to design-plan setup: create `design_plan.md` with `write_markdown_file`, "
            "read it back with `read_file`, and do not write any slide HTML yet."
        )

    state = _ensure_design_plan_state_from_disk(workspace_root)
    if not state:
        return (
            "Error: DeckDesigner stage is blocked because `design_plan.md` exists but its execution state is unknown. "
            "Return to design-plan setup: read `design_plan.md` with `read_file` so the latest plan state is recorded, "
            "and do not write any slide HTML yet."
        )

    validation = state.get("validation") if isinstance(state.get("validation"), dict) else {}
    if not validation.get("valid"):
        return (
            "Error: DeckDesigner stage is blocked because `design_plan.md` is not a valid structured design plan yet "
            f"({_format_design_plan_validation_error(validation)}). "
            "Return to design-plan setup: rewrite it with `write_markdown_file`, read it back with `read_file`, "
            "and do not write any slide HTML yet."
        )

    scaffold_hash = str(state.get("scaffold_hash", "") or "")
    current_hash = str(state.get("current_hash", "") or "")
    if bool(state.get("requires_refinement")) and scaffold_hash and scaffold_hash == current_hash:
        return (
            "Error: DeckDesigner stage is blocked because `design_plan.md` still matches the system scaffold and has not been refined yet. "
            "Return to design-plan setup: overwrite it with a manuscript-specific structured design plan using "
            "`write_markdown_file`, read it back with `read_file`, and do not write any slide HTML yet."
        )

    if not bool(state.get("read_after_last_write")):
        current_label = str(state.get("current_path", "") or _design_plan_relative_label(plan_path))
        return (
            "Error: DeckDesigner sequencing guard: the latest `design_plan.md` write has not been read back yet. "
            f"Call `read_file` on `{current_label}` after the most recent write, then retry `write_html_file`."
        )
    return None


def _template_page_execution_plan_precondition_error() -> str | None:
    if get_current_agent().strip().lower() != "deckdesigner":
        return None
    plan_path = _find_page_execution_plan_path()
    if plan_path is None:
        return None
    entry = _ensure_control_document_tracking_from_disk(
        plan_path,
        required_for_html_prewrite=True,
    )
    if not entry:
        return (
            "Error: DeckDesigner stage detected `page_execution_plan.md`, but its read state is unknown. "
            "Read `page_execution_plan.md` with `read_file` before writing any slide HTML."
        )
    if not bool(entry.get("read_after_last_change")):
        label = str(entry.get("path", "") or _control_document_relative_label(plan_path))
        return (
            "Error: Template page work queue is not unlocked yet. "
            f"Read `{label}` with `read_file` before writing any slide HTML so the latest per-page contract is recorded."
        )
    return None


_SLIDE_CANONICAL_RE = re.compile(r"slide_(\d+)\.html$", re.IGNORECASE)
_SLIDE_ANY_HTML_RE = re.compile(r"slide_.*\.html$", re.IGNORECASE)
_TRUNCATED_HTML_MARKERS = (
    "[旧 HTML 已压缩]",
    "[... 旧 HTML 已压缩 ...]",
    "旧 HTML 已压缩",
    "正文与结构保持不变",
    "正文保持不变",
    "[截断，原始长度",
    "截断，原始长度",
)
_TRUNCATED_HTML_REGEXES = (
    re.compile(r"旧\s*HTML\s*已压缩", re.IGNORECASE),
    re.compile(r"正文(?:与结构)?保持不变"),
    re.compile(r"截断\s*[，,]?\s*原始长度"),
)
_PLACEHOLDER_SIGNAL_TOKENS = (
    "旧html已压缩",
    "html已压缩",
    "正文与结构保持不变",
    "正文保持不变",
    "截断原始长度",
)


def _canonical_slide_number_from_name(filename: str) -> int | None:
    """Return slide number for canonical name `slide_<num>.html`, else None."""
    m = _SLIDE_CANONICAL_RE.fullmatch(filename)
    if not m:
        return None
    return int(m.group(1))


def _find_truncated_html_marker(content: str) -> str | None:
    """Detect placeholder markers that indicate compressed/truncated HTML."""
    text = str(content or "")
    compact = text.replace(" ", "")
    for marker in _TRUNCATED_HTML_MARKERS:
        if marker in text:
            return marker
        compact_marker = marker.replace(" ", "")
        if compact_marker and compact_marker in compact:
            return marker
    for pattern in _TRUNCATED_HTML_REGEXES:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _normalize_placeholder_signal_text(text: str) -> str:
    normalized = str(text or "").lower().replace("…", "...")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[\[\]\(\)\{\}<>'\"`“”‘’.,，:：;；!！?？、/\\|+=_-]+", "", normalized)
    return normalized


def _placeholder_signal_hits(text: str) -> tuple[list[str], str]:
    normalized = _normalize_placeholder_signal_text(text)
    if not normalized:
        return [], ""

    remainder = normalized
    hits: list[str] = []
    for token in _PLACEHOLDER_SIGNAL_TOKENS:
        if token in remainder:
            hits.append(token)
            remainder = remainder.replace(token, "")
    remainder = re.sub(r"原始长度\d+(?:字符|字节)?", "", remainder)
    remainder = re.sub(r"\d+", "", remainder)
    return hits, remainder


def _html_body_root_from_soup(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    body = soup.body or soup.find("body")
    if isinstance(body, Tag):
        return body
    if isinstance(soup.html, Tag):
        return soup.html
    return soup


def _detect_corrupted_slide_html(
    path: Path,
    html_text: str,
    snapshot_payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    Detect placeholder-only slide HTML that has lost its editable body structure.

    We intentionally require both placeholder signals and an empty target set so
    legitimate slides that merely mention the marker text are not rejected.
    """
    normalized_html = str(html_text or "")
    payload = snapshot_payload or _build_slide_snapshot_payload(path, normalized_html)
    meaningful_targets = [
        target
        for target in payload.get("targets", [])
        if str(target.get("kind", "") or "") != "slide_canvas"
    ]
    if meaningful_targets:
        return None

    soup = BeautifulSoup(normalized_html, "lxml")
    body = _html_body_root_from_soup(soup)
    body_text = body.get_text(" ", strip=True) if hasattr(body, "get_text") else ""
    direct_children = [
        node for node in getattr(body, "children", [])
        if isinstance(node, Tag)
    ]
    detected_marker = _find_truncated_html_marker(body_text) or _find_truncated_html_marker(normalized_html)
    placeholder_hits, remainder = _placeholder_signal_hits(body_text or normalized_html)
    if not (detected_marker or placeholder_hits):
        return None
    if len(remainder) > 20:
        return None

    return {
        "detected_marker": detected_marker or placeholder_hits[0],
        "body_text_preview": _text_preview(body_text or normalized_html, limit=160),
        "direct_child_tags": [tag.name.lower() for tag in direct_children[:6]],
        "placeholder_hits": placeholder_hits,
    }


def _collect_canonical_slides(slides_dir: Path) -> list[tuple[int, Path]]:
    """Collect canonical slides `slide_<num>.html` sorted by numeric index."""
    indexed: list[tuple[int, Path]] = []
    for f in slides_dir.glob("slide_*.html"):
        num = _canonical_slide_number_from_name(f.name)
        if num is not None:
            indexed.append((num, f))
    indexed.sort(key=lambda x: x[0])
    return indexed


# ── Slide change tracker (file hash → detect modifications) ──────────
_slide_hashes: dict[str, str] = {}  # path → md5 hex
_slide_unchanged_count: dict[str, int] = {}  # path → consecutive unchanged inspect count
_slide_render_failure_count: dict[str, int] = {}  # path → consecutive render/export failures
_slide_observed_hashes: dict[str, str] = {}  # path → latest on-disk hash read/written by tools
_slide_snapshot_registry: dict[str, dict[str, Any]] = {}
_slide_retired_snapshot_registry: dict[str, dict[str, Any]] = {}
_slide_validation_registry: dict[str, dict[str, Any]] = {}
_newly_created_slide_paths: set[str] = set()
_modified_slide_paths: set[str] = set()
_SLIDE_SNAPSHOT_TARGET_LIMIT = 24
_SEMANTIC_EDIT_SCOPES = {
    "text_color",
    "typography",
    "background_style",
    "layout_spacing",
    "border_shadow",
    "image_asset",
    "visibility",
}
_SEMANTIC_PROPERTY_GROUPS = _SEMANTIC_EDIT_SCOPES | {"content_text", "unknown"}
_SEMANTIC_SCOPE_ALLOWED_GROUPS: dict[str, set[str]] = {
    "text_color": {"text_color"},
    "typography": {"typography"},
    "background_style": {"background_style"},
    "layout_spacing": {"layout_spacing"},
    "border_shadow": {"border_shadow"},
    "image_asset": {"image_asset"},
    "visibility": {"visibility"},
}
_SEMANTIC_COMPATIBLE_GROUP_COMBOS: tuple[frozenset[str], ...] = (
    frozenset({"background_style", "border_shadow"}),
    frozenset({"text_color", "typography"}),
)
_REPAIR_INTENTS = {
    "bottom_safe_zone",
    "clipped_canvas",
    "text_overflow",
    "overlap",
    "readability",
    "image_fit",
    "none",
}
_LAYOUT_REPAIR_INTENTS = {
    "bottom_safe_zone",
    "clipped_canvas",
    "text_overflow",
    "overlap",
}
_TYPOGRAPHY_LAYOUT_REPAIR_INTENTS = {
    "bottom_safe_zone",
    "clipped_canvas",
    "text_overflow",
    "readability",
}
_SEMANTIC_REPAIR_COMPATIBLE_GROUP_COMBOS: tuple[tuple[frozenset[str], frozenset[str]], ...] = (
    (frozenset({"typography", "layout_spacing"}), frozenset(_TYPOGRAPHY_LAYOUT_REPAIR_INTENTS)),
)
_REPAIR_INTENT_BY_DIAGNOSTIC_CODE = {
    "bottom_safe_zone_violation": "bottom_safe_zone",
    "clipped_outside_canvas": "clipped_canvas",
    "text_overflow": "text_overflow",
    "off_canvas_absolute": "clipped_canvas",
    "image_overflow": "image_fit",
}
_RECOMMENDED_PATCH_STRATEGIES: dict[str, list[str]] = {
    "bottom_safe_zone": ["nudge_ancestor_up", "tighten_vertical_gap", "reduce_container_height"],
    "clipped_canvas": ["shrink_or_reposition_container", "reduce_outside_offset"],
    "text_overflow": ["tighten_block_spacing", "rebalance_columns", "last_resort_typography"],
    "overlap": ["separate_overlapping_blocks", "nudge_ancestor_up"],
    "readability": ["add_local_backplate", "increase_local_contrast"],
    "image_fit": ["tighten_image_frame", "reduce_outside_offset"],
    "none": [],
}
_SEMANTIC_GROUP_DISPLAY = {
    "text_color": "text color",
    "typography": "typography",
    "background_style": "background/surface",
    "layout_spacing": "layout/spacing",
    "border_shadow": "border/shadow",
    "image_asset": "image asset",
    "visibility": "visibility",
    "content_text": "text/html content",
    "unknown": "unknown",
}

_TITLE_TARGET_CLASSES = {
    "title",
    "h1",
    "hero",
    "heading",
    "card-title",
    "panel-title",
    "title-wrap",
}
_FOOTER_TARGET_CLASSES = {"footer", "cap", "caption", "page", "src"}
_CAPTION_TARGET_CLASSES = {
    "caption",
    "figure-caption",
    "table-caption",
    "image-caption",
    "fig-caption",
    "figcaption",
}
_PILL_TARGET_CLASSES = {"badge", "pill", "pill-tag", "tag", "chip", "label"}
_LEGEND_TARGET_CLASSES = {"legend", "legend-label"}
_CALLOUT_TARGET_CLASSES = {"callout", "annotation"}
_FOOTNOTE_TARGET_CLASSES = {"footnote", "foot-note", "source-note", "note"}
_CHART_AXIS_LABEL_CLASSES = {"axis-label", "chart-axis-label"}
_CHART_DATA_LABEL_CLASSES = {"data-label", "chart-data-label"}
_BODY_TARGET_CLASSES = {
    "body",
    "content",
    "bullets",
    "copy",
    "text",
    "lead",
    "subtitle",
    "summary",
    "meta",
    "subhead",
    "description",
}
_BOTTOM_SAFE_ZONE_MIN_PX = 48.0
_LAYOUT_REPAIR_HINT_CLASSES = {
    "body",
    "col",
    "column",
    "container",
    "content",
    "copy",
    "frame",
    "grid",
    "list",
    "main",
    "panel",
    "stack",
    "text",
    "wrap",
}
_DECORATIVE_LAYOUT_CLASSES = {"accent", "badge", "divider", "dot", "line", "ornament", "rule", "shape"}
_LAYOUT_GUARD_STYLE_ID = "memslides-layout-guard"
_LAYOUT_GUARD_CONTAINER_CLASSES = {
    "body",
    "card",
    "col",
    "column",
    "container",
    "content",
    "content-card",
    "copy",
    "frame",
    "grid",
    "main",
    "panel",
    "scroller",
    "stack",
    "table-wrap",
    "table-wrapper",
    "table-scroller",
    "text",
    "wrap",
}


def _normalize_focus_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_focus_kind(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not raw:
        return ""
    aliases = {
        "new_caption": "caption",
        "new_caption_only": "caption",
        "figure_caption": "caption",
        "figcaption": "caption",
        "image_caption": "caption",
        "table_caption": "caption",
        "caption_only": "caption",
        "新增图注": "caption",
        "图注": "caption",
        "图片说明": "caption",
        "pill": "pill_tag",
        "pill_only": "pill_tag",
        "badge": "pill_tag",
        "chip": "pill_tag",
        "tag": "pill_tag",
        "标签": "pill_tag",
        "legend": "legend_label",
        "legend_only": "legend_label",
        "图例": "legend_label",
        "callout_box": "callout",
        "annotation": "callout",
        "注释": "callout",
        "foot_note": "footnote",
        "脚注": "footnote",
        "inline": "inline_text",
        "text_span": "inline_text",
        "any": "any_text",
    }
    return aliases.get(raw, raw)


def _compute_file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _is_slide_html_path(path: Path) -> bool:
    """Return whether the path looks like a slide HTML file."""
    return path.suffix.lower() == ".html" and _SLIDE_ANY_HTML_RE.fullmatch(path.name) is not None


def _remember_slide_observation(path: Path) -> None:
    """Record the current on-disk hash for slide overwrite protection."""
    if not (path.exists() and path.is_file() and _is_slide_html_path(path)):
        return
    _slide_observed_hashes[str(path)] = _compute_file_hash(path)


def _record_slide_validation_result(
    path: Path,
    *,
    success: bool,
    message: str = "",
    aspect_ratio: str = "16:9",
    diagnostics: list[dict[str, Any]] | None = None,
) -> None:
    """Persist the latest inspect_slide verdict for the current on-disk HTML."""
    if not (path.exists() and path.is_file() and _is_slide_html_path(path)):
        return
    _slide_validation_registry[str(path.resolve())] = {
        "content_hash": _compute_file_hash(path),
        "success": bool(success),
        "message": str(message or "").strip(),
        "aspect_ratio": str(aspect_ratio or "16:9"),
        "diagnostics": list(diagnostics or []),
        "updated_at": _utc_now_iso(),
    }


def _record_slide_render_failure(path: Path) -> int:
    """Return consecutive render/export failure count for a slide path."""
    key = str(path)
    _slide_render_failure_count[key] = _slide_render_failure_count.get(key, 0) + 1
    return _slide_render_failure_count[key]


def _reset_slide_render_failure_count(path: Path) -> None:
    _slide_render_failure_count.pop(str(path), None)


def _repeated_render_failure_rewrite_note(
    path: Path,
    *,
    html_file: str,
    failure_count: int,
) -> str:
    try:
        current_hash = _compute_text_hash(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        current_hash = ""
    protocol = (
        "\n\n🔁 Repeated layout failure detected. Patch repair is not converging for this slide."
        "\n- Stop using `apply_slide_patch` for this render/export error."
        "\n- Rewrite the full page so the dense content is rebalanced, shortened, or split into a safer grid."
    )
    if current_hash:
        protocol += (
            "\n- Full rewrite protocol: call "
            f"`write_html_file(file_path=\"{html_file}\", content=<complete replacement HTML>, "
            f"force_regenerate=true, expected_hash=\"{current_hash}\")`, then call `inspect_slide` again."
        )
    else:
        protocol += (
            "\n- Full rewrite protocol: call "
            "`read_slide_snapshot` to get the latest content hash, then "
            "`write_html_file(..., force_regenerate=true, expected_hash=...)` and inspect again."
        )
    protocol += f"\n- Consecutive render failures for this page: {failure_count}."
    return protocol


def _hex_to_rgb(value: str) -> tuple[int, int, int] | None:
    text = (value or "").strip()
    if not text.startswith("#"):
        return None
    text = text[1:]
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        return None
    try:
        return tuple(int(text[idx : idx + 2], 16) for idx in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return None


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    def _channel(value: int) -> float:
        c = value / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (_channel(v) for v in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> float:
    a = _relative_luminance(fg)
    b = _relative_luminance(bg)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


def _first_css_hex(content: str, pattern: str) -> str:
    match = re.search(pattern, content or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def _first_css_value(content: str, pattern: str) -> str:
    match = re.search(pattern, content or "", flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _css_custom_properties(content: str) -> dict[str, str]:
    variables: dict[str, str] = {}
    for name, value in re.findall(
        r"(--[A-Za-z0-9_-]+)\s*:\s*(#[0-9a-fA-F]{3,6})\b",
        content or "",
        flags=re.IGNORECASE,
    ):
        variables[name.strip()] = value.strip()
    return variables


def _resolve_css_hex(value: str, variables: dict[str, str]) -> str:
    text = (value or "").strip()
    var_match = re.search(r"var\(\s*(--[A-Za-z0-9_-]+)\s*\)", text)
    if var_match:
        resolved = variables.get(var_match.group(1), "")
        if resolved:
            return resolved
    hex_match = re.search(r"#[0-9a-fA-F]{3,6}\b", text)
    return hex_match.group(0) if hex_match else ""


def _css_declarations(block: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for prop, value in re.findall(r"([A-Za-z-]+)\s*:\s*([^;{}]+)", block or ""):
        declarations[prop.strip().lower()] = value.strip()
    return declarations


def _iter_css_color_pairs(content: str) -> list[tuple[str, str, str, float]]:
    variables = _css_custom_properties(content)
    pairs: list[tuple[str, str, str, float]] = []
    for style_block in re.findall(r"<style\b[^>]*>(.*?)</style>", content or "", flags=re.IGNORECASE | re.DOTALL):
        for selector, block in re.findall(r"([^{}]+)\{([^{}]*)\}", style_block, flags=re.DOTALL):
            declarations = _css_declarations(block)
            color_hex = _resolve_css_hex(declarations.get("color", ""), variables)
            bg_value = declarations.get("background-color", "") or declarations.get("background", "")
            bg_hex = _resolve_css_hex(bg_value, variables)
            fg = _hex_to_rgb(color_hex)
            bg = _hex_to_rgb(bg_hex)
            if not (fg and bg):
                continue
            pairs.append((re.sub(r"\s+", " ", selector).strip(), color_hex, bg_hex, _contrast_ratio(fg, bg)))
    return pairs


def _css_style_color_hex(props: dict[str, str], variables: dict[str, str]) -> str:
    return _resolve_css_hex(str(props.get("color", "") or ""), variables)


def _css_style_background_hex(props: dict[str, str], variables: dict[str, str]) -> str:
    for key in ("background-color", "background"):
        value = str(props.get(key, "") or "").strip()
        if not value:
            continue
        if re.fullmatch(r"(?:transparent|none|inherit|initial|unset)", value, flags=re.IGNORECASE):
            continue
        resolved = _resolve_css_hex(value, variables)
        if resolved:
            return resolved
    return ""


def _css_style_has_unresolved_visual_background(props: dict[str, str], variables: dict[str, str]) -> bool:
    for key in ("background-image", "background"):
        value = str(props.get(key, "") or "").strip()
        if not value:
            continue
        if re.fullmatch(r"(?:transparent|none|inherit|initial|unset)", value, flags=re.IGNORECASE):
            continue
        if _resolve_css_hex(value, variables):
            continue
        if re.search(r"url\(|gradient\(|image\(|var\(", value, flags=re.IGNORECASE):
            return True
    return False


def _find_primary_title_tag(soup: BeautifulSoup) -> Tag | None:
    body = soup.body if isinstance(soup.body, Tag) else soup
    for tag in body.find_all(["h1", "h2"]):
        if isinstance(tag, Tag) and _tag_text_preview(tag):
            return tag
    for tag in body.find_all(True):
        if isinstance(tag, Tag) and _tag_has_any_class(tag, _TITLE_TARGET_CLASSES) and _tag_text_preview(tag):
            return tag
    return None


def _title_effective_contrast_metrics(content: str, soup: BeautifulSoup) -> dict[str, Any]:
    """Estimate title contrast against its nearest visible CSS background.

    The old guard compared the title color against the body background, which
    falsely rejects common title bars such as white h1 text inside a deep-blue
    banner on a white slide. This helper keeps the lightweight static check but
    follows the DOM ancestry to find the actual local title surface first.
    """
    variables = _css_custom_properties(content)
    css_rules = _parse_css_rules(_extract_all_style_text(content))
    title_tag = _find_primary_title_tag(soup)
    if not isinstance(title_tag, Tag):
        return {}

    title_props = _collect_tag_style_props(css_rules, title_tag)
    title_color_hex = _css_style_color_hex(title_props, variables)
    color_source = _selector_hint_for_tag(title_tag)
    if not title_color_hex:
        parent = title_tag.parent
        while isinstance(parent, Tag):
            parent_props = _collect_tag_style_props(css_rules, parent)
            title_color_hex = _css_style_color_hex(parent_props, variables)
            if title_color_hex:
                color_source = _selector_hint_for_tag(parent)
                break
            parent = parent.parent
    if not title_color_hex:
        return {}

    body = soup.body if isinstance(soup.body, Tag) else None
    body_bg_hex = ""
    if isinstance(body, Tag):
        body_bg_hex = _css_style_background_hex(_collect_tag_style_props(css_rules, body), variables)

    title_bg_hex = ""
    bg_source = ""
    unresolved_bg_source = ""
    current: Tag | None = title_tag
    while isinstance(current, Tag):
        props = _collect_tag_style_props(css_rules, current)
        title_bg_hex = _css_style_background_hex(props, variables)
        if title_bg_hex:
            bg_source = _selector_hint_for_tag(current)
            break
        if not unresolved_bg_source and _css_style_has_unresolved_visual_background(props, variables):
            unresolved_bg_source = _selector_hint_for_tag(current)
        if current is body:
            break
        current = current.parent if isinstance(current.parent, Tag) else None

    if not title_bg_hex:
        title_bg_hex = body_bg_hex
        bg_source = "body" if body_bg_hex else ""

    result: dict[str, Any] = {
        "title_selector": _selector_hint_for_tag(title_tag),
        "title_foreground": title_color_hex,
        "title_foreground_source": color_source,
    }
    if body_bg_hex:
        fg = _hex_to_rgb(title_color_hex)
        body_bg = _hex_to_rgb(body_bg_hex)
        if fg and body_bg:
            result["title_body_background"] = body_bg_hex
            result["title_body_contrast"] = round(_contrast_ratio(fg, body_bg), 2)
    if title_bg_hex:
        fg = _hex_to_rgb(title_color_hex)
        bg = _hex_to_rgb(title_bg_hex)
        if fg and bg:
            result.update(
                {
                    "title_background": title_bg_hex,
                    "title_background_source": bg_source,
                    "title_effective_contrast": round(_contrast_ratio(fg, bg), 2),
                    "title_background_resolution": (
                        "body_background" if bg_source == "body" else "local_or_ancestor_background"
                    ),
                }
            )
    elif unresolved_bg_source:
        result["title_background_resolution"] = "unresolved_visual_background"
        result["title_background_source"] = unresolved_bg_source
    return result


def _visible_text_length(content: str) -> int:
    text = re.sub(r"<!--.*?-->", " ", content or "", flags=re.DOTALL)
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return len(text)


def _record_visual_quality_report(
    html_path: Path,
    *,
    diagnostics: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    workspace = _resolve_workspace_root(anchor=html_path)
    report_path = workspace / "visual_quality_report.json"
    payload: dict[str, Any] = {"slides": {}, "updated_at": _utc_now_iso()}
    if report_path.exists():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
                payload.setdefault("slides", {})
        except Exception:
            pass
    rel = html_path.name
    try:
        rel = html_path.resolve().relative_to(workspace.resolve()).as_posix()
    except Exception:
        pass
    payload.setdefault("slides", {})[rel] = {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "updated_at": _utc_now_iso(),
    }
    payload["updated_at"] = _utc_now_iso()
    try:
        report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _collect_visual_quality_diagnostics(html_path: Path, content: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    text_len = _visible_text_length(content)
    img_count = len(re.findall(r"<img\b", content or "", flags=re.IGNORECASE))
    metrics: dict[str, Any] = {
        "visible_text_chars": text_len,
        "image_count": img_count,
        "template_shell_reference": "template_shell/assets/" in (content or ""),
        "has_template_reference_comment": "template-reference-use:" in (content or ""),
    }
    if "template_shell/assets/" in (content or "") or "template-shell-use:" in (content or ""):
        diagnostics.append(
            {
                "code": "template_shell_reference",
                "severity": "error",
                "message": "Reference-only template mode must not use template shell assets or template-shell-use comments.",
                "source": "visual_quality",
            }
        )

    visible_text = _visible_text_from_html(content)
    banned_visible_patterns: tuple[tuple[str, str, str], ...] = (
        ("visible_layout_repair_note", r"Layout repaired deterministically|export canvas", "error"),
        ("visible_safe_zone_note", r"保持底部|底部\s*[≥>=]|安全区|safe zone", "warning"),
    )
    for code, pattern, severity in banned_visible_patterns:
        if re.search(pattern, visible_text, flags=re.IGNORECASE):
            diagnostics.append(
                {
                    "code": code,
                    "severity": severity,
                    "message": (
                        "Safe-zone guidance text is visible on the slide; prefer moving this into presenter notes or removing it."
                        if code == "visible_safe_zone_note"
                        else "Implementation/debug repair text is visible on the slide."
                    ),
                    "source": "visual_quality",
                }
            )

    anchor_contracts = _active_anchored_element_contracts_for_path(html_path, content)
    if anchor_contracts:
        marker_count = _count_temp_pref_marker_occurrences(content)
        metrics["temp_pref_visible_count"] = marker_count
        metrics["anchored_element_contracts"] = [
            str(contract.get("text", "") or "") for contract in anchor_contracts
        ]
        diagnostics.extend(_validate_anchored_element_contracts(content, anchor_contracts, path=html_path))

    caption_negative_offsets: list[str] = []
    soup = BeautifulSoup(str(content or ""), "lxml")
    for tag in soup.find_all(["figcaption", "p", "span", "div"]):
        if tag.name != "figcaption" and not _tag_has_any_class(tag, {"caption", "figure-caption", "image-caption", "fig-caption"}):
            continue
        style_props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
        for prop in ("top", "margin-top", "transform"):
            value = style_props.get(prop, "")
            if re.search(r"-\s*(?:[1-9]\d*)px", value):
                caption_negative_offsets.append(_tag_text_preview(tag)[:80] or str(tag.name))
                break
    if caption_negative_offsets:
        diagnostics.append(
            {
                "code": "caption_negative_overlap_risk",
                "severity": "error",
                "message": "Caption uses negative positioning/margin and may overlap the figure.",
                "source": "visual_quality",
                "examples": caption_negative_offsets[:3],
            }
        )

    if text_len < 18 and img_count == 0:
        diagnostics.append(
            {
                "code": "near_blank_slide",
                "severity": "error",
                "message": "Slide appears nearly blank: too little visible text and no image assets.",
                "source": "visual_quality",
            }
        )

    title_contrast_metrics = _title_effective_contrast_metrics(content, soup)
    if title_contrast_metrics:
        metrics.update(title_contrast_metrics)
        contrast_value = title_contrast_metrics.get("title_effective_contrast")
        if isinstance(contrast_value, (int, float)) and float(contrast_value) < 3.0:
            diagnostics.append(
                {
                    "code": "low_title_contrast",
                    "severity": "error",
                    "message": (
                        "Title contrast against its effective background is too low "
                        f"({float(contrast_value):.2f}:1)."
                    ),
                    "source": "visual_quality",
                    "foreground": title_contrast_metrics.get("title_foreground", ""),
                    "background": title_contrast_metrics.get("title_background", ""),
                    "target_selector_hint": title_contrast_metrics.get("title_selector", ""),
                    "background_selector_hint": title_contrast_metrics.get("title_background_source", ""),
                }
            )
        elif title_contrast_metrics.get("title_background_resolution") == "unresolved_visual_background":
            diagnostics.append(
                {
                    "code": "title_contrast_unresolved_background",
                    "severity": "warning",
                    "message": "Title background uses an image/gradient/variable that static QA could not resolve; verify readability visually.",
                    "source": "visual_quality",
                    "target_selector_hint": title_contrast_metrics.get("title_selector", ""),
                    "background_selector_hint": title_contrast_metrics.get("title_background_source", ""),
                }
            )

    low_contrast_rules: list[dict[str, Any]] = []
    for selector, fg_hex, bg_hex, contrast in _iter_css_color_pairs(content):
        threshold = 3.0 if re.search(r"\bh1\b|title", selector, flags=re.IGNORECASE) else 4.5
        if contrast < threshold:
            low_contrast_rules.append(
                {
                    "selector": selector,
                    "foreground": fg_hex,
                    "background": bg_hex,
                    "contrast": round(contrast, 2),
                    "threshold": threshold,
                }
            )
    if low_contrast_rules:
        metrics["low_contrast_rules"] = low_contrast_rules[:8]
        worst = min(low_contrast_rules, key=lambda item: float(item.get("contrast", 99.0)))
        diagnostics.append(
            {
                "code": "low_text_surface_contrast",
                "severity": "warning",
                "message": (
                    "A CSS text/background pair has insufficient contrast: "
                    f"`{worst['selector']}` uses {worst['foreground']} on {worst['background']} "
                    f"({worst['contrast']}:1)."
                ),
                "source": "visual_quality",
            }
        )

    try:
        from memslides.runtime.deck_execution_state import load_deck_execution_state

        state = load_deck_execution_state()
        page = _canonical_slide_number_from_name(html_path.name)
        slide = (state or {}).get("slides", {}).get(str(page), {}) if page else {}
        if str(slide.get("visual_requirement", "none") or "none") == "required":
            bound_path = str(slide.get("bound_asset_path", "") or "")
            if bound_path:
                basename = Path(bound_path).name
                metrics["required_asset"] = basename
                if basename and basename not in (content or ""):
                    diagnostics.append(
                        {
                            "code": "required_asset_missing",
                            "severity": "error",
                            "message": f"Required bound asset `{basename}` is not referenced in this slide HTML.",
                            "source": "visual_quality",
                        }
                    )
    except Exception:
        pass

    if "template-reference-use:" not in (content or "") and "template-layout:" in (content or ""):
        diagnostics.append(
            {
                "code": "missing_reference_audit",
                "severity": "warning",
                "message": "Template slide is missing the template-reference-use audit comment.",
                "source": "visual_quality",
            }
        )

    soft_contract = _soft_visual_preference_contract_from_context(None, content)
    soft_diagnostics = validate_soft_visual_preference_static(content, soft_contract)
    if soft_diagnostics:
        metrics["soft_visual_preference_targets"] = list((soft_contract.get("targets") or {}).keys())
        diagnostics.extend(soft_diagnostics)

    controlled_context = _active_controlled_rewrite_context_for_path(html_path)
    if controlled_context:
        rewrite_diagnostics, rewrite_metrics = _collect_controlled_rewrite_visual_diagnostics(
            html_path,
            content,
            controlled_context,
        )
        if rewrite_metrics:
            metrics.update(rewrite_metrics)
        diagnostics.extend(rewrite_diagnostics)

    _record_visual_quality_report(html_path, diagnostics=diagnostics, metrics=metrics)
    return diagnostics, metrics


def _color_distance(a: str, b: str) -> float | None:
    rgb_a = _hex_to_rgb(a)
    rgb_b = _hex_to_rgb(b)
    if not (rgb_a and rgb_b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(rgb_a, rgb_b)))


def _is_color_near(value: str, target: str, *, threshold: float = 76.0) -> bool:
    distance = _color_distance(value, target)
    return bool(distance is not None and distance <= threshold)


def _soft_visual_preference_contract_from_context(
    context: dict[str, Any] | None = None,
    *extra_texts: Any,
) -> dict[str, Any]:
    context = context if isinstance(context, dict) else _read_current_modify_context()
    text = "\n".join(
        chunk
        for chunk in [_context_text_for_visual_preferences(context), *(str(item or "") for item in extra_texts)]
        if str(chunk or "").strip()
    )
    lowered = text.lower()
    contract: dict[str, Any] = {"active": False, "targets": {}}
    targets: dict[str, dict[str, Any]] = {}
    if any(token in text for token in ("深蓝标题", "深蓝色标题")) or "deep blue title" in lowered:
        targets["title_color"] = {
            "description": "deep blue title",
            "target_hex": "#1D3557",
        }
    if any(token in text for token in ("青绿色强调", "青绿强调", "青绿色作为强调", "青绿色")) or "teal accent" in lowered:
        targets["accent_color"] = {
            "description": "teal accent",
            "target_hex": "#0D9488",
        }
    if any(token in text for token in ("浅灰证据卡片", "浅灰色证据卡片", "浅灰卡片")) or (
        "light gray" in lowered and ("card" in lowered or "evidence" in lowered)
    ):
        targets["evidence_card_surface"] = {
            "description": "light gray evidence cards",
            "target_hex": "#F3F4F6",
        }
    hex_colors = re.findall(r"#[0-9a-fA-F]{3,6}\b", text)
    if hex_colors:
        contract["mentioned_colors"] = _unique_preserve_order(hex_colors)
    contract["targets"] = targets
    contract["active"] = bool(targets)
    return contract


def _iter_css_rules_with_props(content: str) -> list[tuple[str, dict[str, str]]]:
    rules: list[tuple[str, dict[str, str]]] = []
    for style_block in re.findall(r"<style\b[^>]*>(.*?)</style>", content or "", flags=re.IGNORECASE | re.DOTALL):
        for selector, block in re.findall(r"([^{}]+)\{([^{}]*)\}", style_block, flags=re.DOTALL):
            if selector.strip().startswith("@"):
                continue
            props = _css_declarations(block)
            if props:
                rules.append((re.sub(r"\s+", " ", selector).strip(), props))
    return rules


def validate_soft_visual_preference_static(
    html: str,
    contract: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Diagnose soft visual preference drift without mutating the slide."""
    contract = contract or _soft_visual_preference_contract_from_context()
    if not contract.get("active"):
        return []
    targets = contract.get("targets") if isinstance(contract.get("targets"), dict) else {}
    variables = _css_custom_properties(html)
    rules = _iter_css_rules_with_props(html)
    diagnostics: list[dict[str, Any]] = []

    if "title_color" in targets:
        expected = str(targets["title_color"].get("target_hex", "#1D3557") or "#1D3557")
        title_colors: list[str] = []
        for selector, props in rules:
            if re.search(r"\bh1\b|title|heading", selector, flags=re.IGNORECASE):
                color = _resolve_css_hex(props.get("color", ""), variables)
                if color:
                    title_colors.append(color)
        if title_colors and not any(_is_color_near(color, expected) for color in title_colors):
            diagnostics.append(
                {
                    "code": "soft_title_color_drift",
                    "severity": "warning",
                    "message": (
                        "Active visual preference asks for deep-blue titles, but detected title colors "
                        f"look different ({', '.join(_unique_preserve_order(title_colors)[:3])}). "
                        "As the designer, decide how to adjust title styling while preserving the page design."
                    ),
                    "source": "soft_visual_preference",
                    "target": "title_color",
                    "expected": expected,
                    "observed": _unique_preserve_order(title_colors)[:6],
                }
            )

    if "accent_color" in targets:
        expected = str(targets["accent_color"].get("target_hex", "#0D9488") or "#0D9488")
        accent_selectors: list[dict[str, str]] = []
        blueish_hits: list[dict[str, str]] = []
        for selector, props in rules:
            if not re.search(r"accent|underline|rule|highlight|metric|badge|chip|pill|border|arrow|node|bar|callout", selector, flags=re.IGNORECASE):
                continue
            for prop in ("color", "background", "background-color", "border-color", "border", "stroke", "fill"):
                color = _resolve_css_hex(props.get(prop, ""), variables)
                if not color:
                    continue
                accent_selectors.append({"selector": selector, "property": prop, "color": color})
                rgb = _hex_to_rgb(color)
                if rgb and rgb[2] > rgb[1] and rgb[2] > rgb[0] and not _is_color_near(color, expected):
                    blueish_hits.append({"selector": selector, "property": prop, "color": color})
        if blueish_hits:
            diagnostics.append(
                {
                    "code": "soft_accent_color_drift",
                    "severity": "warning",
                    "message": (
                        "Active visual preference asks for teal-green accents, but several accent-like CSS rules "
                        "still read as blue. Please decide which emphasis components should become teal while keeping the design coherent."
                    ),
                    "source": "soft_visual_preference",
                    "target": "accent_color",
                    "expected": expected,
                    "observed": blueish_hits[:8],
                    "repair_strategy": "Model-side design repair only: adjust accent components deliberately; the tool will not rewrite colors for you.",
                }
            )
        elif accent_selectors and not any(_is_color_near(item["color"], expected) for item in accent_selectors):
            diagnostics.append(
                {
                    "code": "soft_accent_color_unclear",
                    "severity": "warning",
                    "message": (
                        "Active visual preference asks for teal-green accents, but no accent-like rule is close to that target. "
                        "Review emphasis lines, badges, arrows, and metric highlights."
                    ),
                    "source": "soft_visual_preference",
                    "target": "accent_color",
                    "expected": expected,
                    "observed": accent_selectors[:8],
                }
            )

    if "evidence_card_surface" in targets:
        expected = str(targets["evidence_card_surface"].get("target_hex", "#F3F4F6") or "#F3F4F6")
        card_surfaces: list[str] = []
        for selector, props in rules:
            if not re.search(r"card|evidence|panel|content-box|content-card", selector, flags=re.IGNORECASE):
                continue
            color = _resolve_css_hex(props.get("background-color", "") or props.get("background", ""), variables)
            if color:
                card_surfaces.append(color)
        if card_surfaces and not any(_is_color_near(color, expected, threshold=58.0) for color in card_surfaces):
            diagnostics.append(
                {
                    "code": "soft_evidence_card_surface_drift",
                    "severity": "warning",
                    "message": (
                        "Active visual preference asks for light-gray evidence cards, but detected card surfaces differ. "
                        "Adjust evidence/card surfaces if that fits the slide's design."
                    ),
                    "source": "soft_visual_preference",
                    "target": "evidence_card_surface",
                    "expected": expected,
                    "observed": _unique_preserve_order(card_surfaces)[:6],
                }
            )
    return diagnostics


def _strict_visual_preference_eval_enabled(context: dict[str, Any] | None = None) -> bool:
    context = context if isinstance(context, dict) else _read_current_modify_context()
    env_value = str(os.environ.get("MEMSLIDES_STRICT_VISUAL_PREFERENCE_EVAL", "") or "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    return bool(context.get("strict_visual_preference_eval", False))


def _current_slide_validation_state(path: Path) -> dict[str, Any] | None:
    if not (path.exists() and path.is_file() and _is_slide_html_path(path)):
        return None
    validation = _slide_validation_registry.get(str(path.resolve()))
    if not isinstance(validation, dict):
        return None
    current_hash = _compute_file_hash(path)
    if validation.get("content_hash") != current_hash:
        return None
    return validation


def _invalidate_slide_validation_for_path(path: Path) -> None:
    """Drop inspect_slide validation state after the HTML changes."""
    try:
        _slide_validation_registry.pop(str(path.resolve()), None)
    except Exception:
        pass


def _is_canonical_slide_path(path: Path) -> bool:
    return path.suffix.lower() == ".html" and _canonical_slide_number_from_name(path.name) is not None


def _can_render_slide_preview() -> bool:
    expected = _get_expected_playwright_binary()
    return expected.exists() or should_auto_install_playwright()


def _mark_new_slide_created(path: Path) -> None:
    if _is_canonical_slide_path(path):
        _newly_created_slide_paths.add(str(path))


def _mark_slide_modified(path: Path) -> None:
    """Track slide HTML files that were actually changed in this tool session."""
    if _is_slide_html_path(path):
        _modified_slide_paths.add(str(path.resolve()))


def _iter_modified_slides_within(root: Path) -> list[Path]:
    """Return modified slides that live under the given workspace directory."""
    root_resolved = root.resolve()
    results: list[Path] = []
    for raw_path in sorted(_modified_slide_paths):
        try:
            candidate = Path(raw_path).resolve()
        except Exception:
            continue
        if not candidate.exists():
            continue
        try:
            candidate.relative_to(root_resolved)
        except ValueError:
            continue
        if _is_slide_html_path(candidate):
            results.append(candidate)
    return results


def _invalidate_slide_snapshots_for_path(path: Path) -> list[str]:
    target = str(path.resolve())
    stale_ids = [
        snapshot_id
        for snapshot_id, payload in _slide_snapshot_registry.items()
        if str(payload.get("resolved_path", "")) == target
    ]
    for snapshot_id in stale_ids:
        _slide_snapshot_registry.pop(snapshot_id, None)
    return stale_ids


def _deck_progress_defers_persona_repair(
    deck_progress: dict[str, Any],
    html_file: str,
) -> tuple[bool, str]:
    """Keep persona repair behind missing or uninspected structural work."""
    if not isinstance(deck_progress, dict) or not deck_progress.get("active"):
        return False, ""
    if deck_progress.get("complete"):
        return False, ""
    next_page = deck_progress.get("next_page", {}) or {}
    next_file = str(next_page.get("file", "") or "").strip()
    if not next_file:
        return False, ""
    current_name = Path(str(html_file or "")).name
    next_name = Path(next_file).name
    if current_name and next_name == current_name:
        return False, ""
    next_action = str(deck_progress.get("next_action", "") or "").strip()
    if next_action in {"write_html", "inspect_or_fix"}:
        return True, next_file
    return False, ""


def _current_slide_persona_retry_required(path: Path) -> bool:
    page = _canonical_slide_number_from_name(path.name)
    if page is None:
        return False
    try:
        from memslides.runtime.deck_execution_state import load_deck_execution_state

        state = load_deck_execution_state()
    except Exception:
        return False
    slide = (state or {}).get("slides", {}).get(str(page), {})
    persona_status = str(slide.get("persona_status", "") or "").strip().lower()
    slide_status = str(slide.get("status", "") or "").strip().lower()
    return persona_status == "retry_required" or slide_status == "persona_retry_required"


def _current_slide_state_meta(path: Path) -> dict[str, Any]:
    page = _canonical_slide_number_from_name(path.name)
    if page is None:
        return {}
    try:
        from memslides.runtime.deck_execution_state import load_deck_execution_state

        state = load_deck_execution_state()
    except Exception:
        return {}
    slide = (state or {}).get("slides", {}).get(str(page), {})
    return slide if isinstance(slide, dict) else {}


def _persona_repair_backup_path(path: Path, content_hash: str) -> Path:
    workspace = _resolve_workspace_root(anchor=path)
    safe_hash = re.sub(r"[^a-fA-F0-9]", "", str(content_hash or ""))[:16] or "unknown"
    return workspace / ".persona_repair_backups" / f"{path.stem}_{safe_hash}.html"


def _maybe_record_persona_repair_backup(path: Path) -> None:
    """Snapshot the last inspect-passed HTML before a soft persona rewrite.

    Persona repair is allowed to fail open; structural validity is not. Keeping
    a tiny on-disk backup lets us restore an already validated page if a later
    persona-focused rewrite causes overflow near the end of a generation run.
    """
    if not (path.exists() and _is_canonical_slide_path(path)):
        return
    if not _current_slide_persona_retry_required(path):
        return
    validation = _current_slide_validation_state(path)
    if not (validation and bool(validation.get("success"))):
        return
    try:
        current_html = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    content_hash = _compute_text_hash(current_html)
    backup_path = _persona_repair_backup_path(path, content_hash)
    try:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            backup_path.write_text(current_html, encoding="utf-8")
        from memslides.runtime.deck_execution_state import record_persona_repair_backup

        record_persona_repair_backup(
            path,
            backup_path=backup_path.as_posix(),
            backup_hash=content_hash,
        )
    except Exception:
        return


def _restore_persona_repair_backup_after_failed_repair(
    path: Path,
    *,
    aspect_ratio: str,
    failure_message: str,
    diagnostics: list[dict[str, Any]] | None = None,
) -> str | None:
    """Restore the last inspect-passed page when soft persona repair breaks render."""
    if not (path.exists() and _is_canonical_slide_path(path)):
        return None
    slide_meta = _current_slide_state_meta(path)
    if not slide_meta:
        return None
    persona_status = str(slide_meta.get("persona_status", "") or "").strip().lower()
    slide_status = str(slide_meta.get("status", "") or "").strip().lower()
    if persona_status != "retry_required" and slide_status != "persona_retry_required":
        return None
    backup = slide_meta.get("persona_repair_backup")
    if not isinstance(backup, dict):
        return None
    backup_path_text = str(backup.get("path", "") or "").strip()
    if not backup_path_text:
        return None
    backup_path = Path(backup_path_text)
    if not backup_path.is_absolute():
        backup_path = _resolve_workspace_root(anchor=path) / backup_path
    if not (backup_path.exists() and backup_path.is_file()):
        return None
    try:
        backup_html = backup_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    backup_hash = _compute_text_hash(backup_html)
    recorded_hash = str(backup.get("content_hash", "") or "").strip()
    if recorded_hash and recorded_hash != backup_hash:
        return None
    try:
        if _compute_text_hash(path.read_text(encoding="utf-8", errors="replace")) == backup_hash:
            return None
    except Exception:
        pass

    try:
        path.write_text(backup_html, encoding="utf-8")
        _invalidate_slide_snapshots_for_path(path)
        _invalidate_slide_validation_for_path(path)
        _reset_slide_render_failure_count(path)
        _remember_slide_observation(path)
        _record_slide_validation_result(
            path,
            success=True,
            message="restored inspect-passed backup after failed persona repair",
            aspect_ratio=aspect_ratio,
            diagnostics=diagnostics or [],
        )
        from memslides.runtime.deck_execution_state import (
            record_persona_verdict,
            record_slide_inspected,
        )

        record_slide_inspected(path, success=True)
        repair_focus = str(slide_meta.get("persona_repair_focus", "") or "").strip()
        record_persona_verdict(
            path,
            verdict="unresolved_release",
            repair_focus=repair_focus,
            details={
                "final_verdict": "unresolved_release",
                "verdict_source": "persona_repair_rollback",
                "restored_backup_path": backup_path.as_posix(),
                "restored_backup_hash": backup_hash,
                "failed_repair_message": str(failure_message or "")[:500],
            },
        )
    except Exception:
        return None
    return (
        "⚠️ Persona repair broke slide rendering, so the runtime restored the last "
        "inspect-passed version of this slide and marked the page-level persona gap "
        "as `unresolved_release`.\n"
        f"- Restored backup: {backup_path.as_posix()}\n"
        "- Structural inspect status is preserved; continue with the next pending slide or finalize when all slides are accepted."
    )


def _persona_retry_patch_block_error(
    path: Path,
    *,
    snapshot_id: str,
    snapshot: dict[str, Any],
    current_hash: str,
    patch_ops: list[dict[str, Any]],
) -> str | None:
    """Force persona contract retries through full-page replacement.

    Persona repairs often require rebalancing content, layout, and evidence as a
    single page. Letting the model append structural fragments with a local
    patch can make a structurally valid slide overflow, so the tool layer
    enforces the same rewrite protocol that inspect_slide reports.
    """
    if not _current_slide_persona_retry_required(path):
        return None
    op_names = [
        str(op.get("op", "") or "").strip()
        for op in patch_ops
        if isinstance(op, dict)
    ]
    return _structured_patch_error(
        "PERSONA_RETRY_REQUIRES_FULL_REWRITE",
        "This slide is in a page-level persona contract retry state. "
        "Do not use apply_slide_patch for persona repair, because local "
        "fragment insertion can break canvas fit. Rewrite the whole page with "
        "`write_html_file(file_path=..., content=<complete replacement HTML>, "
        "force_regenerate=true, expected_hash=<current_content_hash>)`, then "
        "run inspect_slide again.",
        snapshot_id=snapshot_id,
        slide_path=str(snapshot.get("slide_path", "") or path.as_posix()),
        current_content_hash=current_hash,
        blocked_ops=op_names,
    )


def _canonical_slide_requires_patch_error(
    path: Path,
    *,
    force_regenerate: bool,
    expected_hash: str,
) -> str | None:
    """Route existing canonical slide edits through snapshot+patch by default."""
    if not (_is_canonical_slide_path(path) and path.exists()):
        return None
    if str(path) in _newly_created_slide_paths:
        return None

    current_html = path.read_text(encoding="utf-8", errors="replace")
    current_hash = _compute_text_hash(current_html)
    corruption = _detect_corrupted_slide_html(path, current_html)
    validation = _current_slide_validation_state(path)
    failed_validation = bool(validation) and not bool(validation.get("success"))
    persona_retry_required = _current_slide_persona_retry_required(path)
    modify_context = _read_current_modify_context()
    diagram_rewrite_allowed = (
        str(modify_context.get("operation_kind", "") or "").strip() == "diagram_layout"
        and _path_matches_modify_target(path, modify_context.get("target_slide_paths") or [])
    )
    controlled_rewrite_allowed = (
        str(modify_context.get("operation_kind", "") or "").strip() == "controlled_rewrite"
        and _path_matches_modify_target(path, modify_context.get("target_slide_paths") or [])
    )

    if not force_regenerate:
        if corruption:
            marker = str(corruption.get("detected_marker", "") or "placeholder HTML")
            return (
                "Error: Existing canonical slide appears corrupted or placeholder-based "
                f"(detected marker: {marker}). Patch editing is unavailable for this slide. "
                "Call `read_slide_snapshot` to confirm the corruption and obtain the current "
                "`content_hash`, then recover it with "
                "`write_html_file(force_regenerate=true, expected_hash=...)` using full HTML."
            )
        if failed_validation:
            return (
                "Error: Existing canonical slides must be edited via "
                "`read_slide_snapshot` + `apply_slide_patch` by default. "
                "However, this slide's current on-disk HTML most recently failed "
                "`inspect_slide`, so full-slide regenerate is also allowed here. "
                "Call `read_slide_snapshot` to obtain the latest `content_hash`, "
                "then retry with `write_html_file(force_regenerate=true, expected_hash=...)` "
                "using complete regenerated HTML if patch repair is too limited."
            )
        if persona_retry_required:
            return (
                "Error: This slide passed structural inspection but failed its page-level "
                "persona contract. Regenerate the whole page rather than layering local "
                "patches: call `write_html_file(force_regenerate=true, expected_hash=...)` "
                "with complete replacement HTML that satisfies the repair focus."
            )
        if diagram_rewrite_allowed:
            return (
                "Error: This turn is a diagram_layout rewrite for this target slide. "
                "Call `read_slide_snapshot` first, then retry with "
                "`write_html_file(force_regenerate=true, expected_hash=content_hash)` "
                "using complete replacement HTML for the whole diagram page."
            )
        if controlled_rewrite_allowed:
            return (
                "Error: This turn is a controlled_rewrite for this target slide. "
                "Call `read_slide_snapshot` first, then retry with "
                "`write_html_file(force_regenerate=true, expected_hash=content_hash)` "
                "using complete replacement HTML for the same slide. Keep the slide count unchanged."
            )
        return (
            "Error: Existing canonical slides must be edited via "
            "`read_slide_snapshot` + `apply_slide_patch`. `write_html_file` is "
            "reserved for new slides unless you explicitly pass "
            "`force_regenerate=true` with a matching `expected_hash`."
        )

    if (
        corruption is None
        and not failed_validation
        and not persona_retry_required
        and not diagram_rewrite_allowed
        and not controlled_rewrite_allowed
    ):
        return (
            "Error: Refusing full-regenerate overwrite for existing canonical slide "
            "because the current slide is not corrupted. It also has no current failed "
            "`inspect_slide` or page-level persona retry state. Routine existing-slide "
            "edits must use `read_slide_snapshot` + `apply_slide_patch`; "
            "`force_regenerate=true` is reserved for controlled recovery only "
            "(corrupted-slide, failed-inspect, persona-retry full rewrite, diagram_layout target rewrite, or controlled_rewrite target rewrite)."
        )

    if not expected_hash or expected_hash != current_hash:
        return (
            "Error: Refusing full-regenerate overwrite for existing canonical "
            "slide because `expected_hash` does not match the current on-disk "
            "HTML. Re-run `read_slide_snapshot` to fetch the latest hash, then "
            "retry controlled full-regenerate with "
            "`write_html_file(force_regenerate=true, expected_hash=...)`."
        )

    return None


def _overwrite_requires_fresh_read_error(path: Path) -> str | None:
    """Require existing slide HTML to be read from disk before overwrite."""
    if not _is_slide_html_path(path):
        return None
    if not path.exists():
        return None

    key = str(path)
    current_hash = _compute_file_hash(path)
    observed_hash = _slide_observed_hashes.get(key)
    if observed_hash == current_hash:
        return None

    if observed_hash is None:
        return (
            "Error: Refusing to overwrite existing slide without reading the latest "
            "on-disk HTML first. Call `read_file` on this slide, then retry "
            "`write_html_file`. This freshness guard commonly applies to newly "
            "inserted slides or non-canonical helper HTML; routine edits on "
            "pre-existing canonical slides should use `read_slide_snapshot` + "
            "`apply_slide_patch`."
        )

    return (
        "Error: Refusing to overwrite slide because the on-disk HTML changed since "
        "it was last read or written in this tool session. Call `read_file` on "
        "the current slide file, then retry `write_html_file`. If you actually "
        "need to edit a pre-existing canonical slide instead of a newly inserted "
        "one, switch to `read_slide_snapshot` + `apply_slide_patch`."
    )


def _check_slide_changed(path: Path) -> tuple[bool, int, int]:
    """Return (changed, old_size, new_size). First inspect always counts as changed."""
    key = str(path)
    new_hash = _compute_file_hash(path)
    new_size = path.stat().st_size
    old_hash = _slide_hashes.get(key)
    old_size = 0
    changed = old_hash is None or old_hash != new_hash
    if old_hash is not None and old_hash != new_hash:
        # Estimate old size from previous read (not perfect but useful)
        old_size = getattr(_check_slide_changed, "_sizes", {}).get(key, 0)
    if not hasattr(_check_slide_changed, "_sizes"):
        _check_slide_changed._sizes = {}
    _check_slide_changed._sizes[key] = new_size
    _slide_hashes[key] = new_hash
    return changed, old_size, new_size


def _drop_slide_tracking(path: Path) -> None:
    """Delete slide hash/change tracking for a removed file."""
    key = str(path)
    resolved_key = str(path.resolve())
    _slide_hashes.pop(key, None)
    _slide_unchanged_count.pop(key, None)
    _slide_render_failure_count.pop(key, None)
    _slide_observed_hashes.pop(key, None)
    _newly_created_slide_paths.discard(key)
    _modified_slide_paths.discard(resolved_key)
    _invalidate_slide_validation_for_path(path)
    _invalidate_slide_snapshots_for_path(path)
    sizes = getattr(_check_slide_changed, "_sizes", None)
    if isinstance(sizes, dict):
        sizes.pop(key, None)


def _move_slide_tracking(old_path: Path, new_path: Path) -> None:
    """Move slide hash/change tracking when a file is renamed."""
    old_key = str(old_path)
    new_key = str(new_path)
    old_resolved = str(old_path.resolve())
    new_resolved = str(new_path.resolve())
    if old_key in _slide_hashes:
        _slide_hashes[new_key] = _slide_hashes.pop(old_key)
    if old_key in _slide_unchanged_count:
        _slide_unchanged_count[new_key] = _slide_unchanged_count.pop(old_key)
    if old_key in _slide_render_failure_count:
        _slide_render_failure_count[new_key] = _slide_render_failure_count.pop(old_key)
    if old_key in _slide_observed_hashes:
        _slide_observed_hashes[new_key] = _slide_observed_hashes.pop(old_key)
    if old_key in _newly_created_slide_paths:
        _newly_created_slide_paths.discard(old_key)
        _newly_created_slide_paths.add(new_key)
    if old_resolved in _modified_slide_paths:
        _modified_slide_paths.discard(old_resolved)
        _modified_slide_paths.add(new_resolved)
    old_validation = _slide_validation_registry.pop(str(old_path.resolve()), None)
    if old_validation is not None:
        _slide_validation_registry[str(new_path.resolve())] = old_validation
    _invalidate_slide_snapshots_for_path(old_path)
    _invalidate_slide_snapshots_for_path(new_path)
    sizes = getattr(_check_slide_changed, "_sizes", None)
    if isinstance(sizes, dict) and old_key in sizes:
        sizes[new_key] = sizes.pop(old_key)


_PAGE_COUNTER_PATTERNS: tuple[tuple[re.Pattern[str], Callable[[int, int], str]], ...] = (
    (
        re.compile(r"第\s*\d+\s*/\s*\d+\s*页"),
        lambda current, total: f"第{current}/{total}页",
    ),
    (
        re.compile(r"(?i)\bpage\s*\d+\s*/\s*\d+\b"),
        lambda current, total: f"Page {current}/{total}",
    ),
)

_STALE_TOTAL_PAGE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?P<prefix>[（(]\s*)\d+\s*页(?P<suffix>\s*[）)])"),
    re.compile(r"(?P<prefix>[·\-:：]\s*)\d+\s*页\b"),
    re.compile(r"(?i)(?P<prefix>\bdeck\s+of\s*)\d+\s*(?P<suffix>slides?\b)"),
)

_TEMP_PREF_MARKER_TEXT = "TEMP-PREF-C1"
_TEMP_PREF_MARKER_RE = re.compile(r"TEMP-PREF-[A-Za-z0-9_-]+")
_ANCHOR_LAYER_CLASS = "memslides-anchor-layer"
_ANCHOR_ELEMENT_CLASS = "memslides-anchored-element"
_ANCHOR_STYLE_ID = "memslides-anchor-style"
_ANCHOR_DEFAULT_MARGIN_PX = 24
_ANCHOR_ROLES = {"marker", "badge", "source_tag", "watermark", "footer_note"}
_ANCHOR_CLASS_HINTS = {
    _ANCHOR_LAYER_CLASS,
    _ANCHOR_ELEMENT_CLASS,
    "deck-tag",
    "source-mark",
    "source-tag",
    "watermark",
    "footer-marker",
    "footer-note",
    "draft-badge",
    "temp-pref-c1",
    "temp-pref-c2",
}
_ANCHOR_SAFE_STYLE_PROPS = {
    "background",
    "background-color",
    "border",
    "border-color",
    "border-radius",
    "box-shadow",
    "color",
    "font-family",
    "font-size",
    "font-weight",
    "opacity",
    "padding",
    "text-transform",
}
_TEMP_PREF_STYLE_RE = re.compile(
    r"(?is)(?!\s*@)(?:body\s*::?\s*after|\.temp-pref-c\d+|\.source-mark|\.source-tag|\.deck-tag|\.watermark|\.footer-marker|\.footer-note|\.draft-badge|\.memslides-anchor-layer|\.memslides-anchored-element)[^{}]*\{[^{}]*(?:TEMP-PREF-[A-Za-z0-9_-]+|anchor|fixed|bottom|right)[^{}]*\}"
)


def _ensure_html_head_tag(soup: BeautifulSoup) -> Tag:
    head = soup.head or soup.find("head")
    if isinstance(head, Tag):
        return head
    html_tag = soup.find("html")
    head = soup.new_tag("head")
    if isinstance(html_tag, Tag):
        html_tag.insert(0, head)
    else:
        soup.insert(0, head)
    return head


def _ensure_html_body_tag(soup: BeautifulSoup) -> Tag:
    body = soup.body or soup.find("body")
    if isinstance(body, Tag):
        return body
    html_tag = soup.find("html")
    body = soup.new_tag("body")
    if isinstance(html_tag, Tag):
        html_tag.append(body)
    else:
        soup.append(body)
    return body


def _anchored_element_css() -> str:
    margin = _ANCHOR_DEFAULT_MARGIN_PX
    return f"""
.slide {{ position: relative; }}
.{_ANCHOR_LAYER_CLASS} {{
  position: absolute;
  inset: 0;
  overflow: visible;
  pointer-events: none;
  z-index: 9999;
}}
.{_ANCHOR_ELEMENT_CLASS} {{
  position: absolute;
  right: {margin}px;
  bottom: {margin}px;
  max-width: 190px;
  min-height: 22px;
  padding: 6px 10px;
  border-radius: 6px;
  box-sizing: border-box;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-family: Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif;
  font-size: 12px;
  line-height: 1;
  font-weight: 700;
  color: var(--memslides-anchor-color, var(--color-inverse-text, #ffffff));
  background: var(--memslides-anchor-bg, var(--color-title, #1D3557));
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.14);
  opacity: 0.92;
  user-select: none;
}}
"""


def _temp_pref_marker_css(marker_text: str = _TEMP_PREF_MARKER_TEXT) -> str:
    safe_marker = str(marker_text or _TEMP_PREF_MARKER_TEXT).replace("\\", "\\\\").replace('"', '\\"')
    return _anchored_element_css() + f'\nbody .{_ANCHOR_ELEMENT_CLASS}[data-anchor-text="{safe_marker}"] {{ }}'


_TEMP_PREF_MARKER_CSS = _temp_pref_marker_css(_TEMP_PREF_MARKER_TEXT)


def _upsert_anchored_element_style(soup: BeautifulSoup) -> bool:
    css = _anchored_element_css().strip()
    style = soup.find("style", id=_ANCHOR_STYLE_ID)
    if isinstance(style, Tag):
        if str(style.string or "").strip() == css:
            return False
        style.clear()
        style.append("\n" + css + "\n")
        return True
    style = soup.new_tag("style", id=_ANCHOR_STYLE_ID)
    style.append("\n" + css + "\n")
    _ensure_html_head_tag(soup).append(style)
    return True


def _anchor_role_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    if "watermark" in lowered or "水印" in lowered:
        return "watermark"
    if "source" in lowered or "来源" in lowered:
        return "source_tag"
    if "footer" in lowered or "脚注" in lowered:
        return "footer_note"
    if "badge" in lowered or "draft" in lowered or "标签" in lowered:
        return "badge"
    return "marker"


def _anchored_contract(text: str, *, role: str = "", anchor: str = "bottom_right") -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    return {
        "text": cleaned,
        "role": role if role in _ANCHOR_ROLES else _anchor_role_from_text(cleaned),
        "anchor": anchor or "bottom_right",
        "visibility": "exactly_once_visible",
        "scope": "current_deck_with_future_rewrites",
    }


def _anchored_contracts_from_texts(*texts: Any) -> list[dict[str, Any]]:
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    combined = "\n".join(str(text or "") for text in texts if str(text or "").strip())
    for match in _TEMP_PREF_MARKER_RE.finditer(combined):
        marker = match.group(0)
        if marker not in seen:
            seen.add(marker)
            contracts.append(_anchored_contract(marker, role="marker"))

    generic_patterns = (
        r"(?:右下角|bottom[-_\s]?right|watermark|source\s*mark|source\s*tag|badge|水印|来源标记)[^。\n]{0,80}?[\"“']([^\"”'\n]{2,48})[\"”']",
        r"[\"“']([^\"”'\n]{2,48})[\"”'][^。\n]{0,80}?(?:右下角|bottom[-_\s]?right|watermark|source\s*mark|source\s*tag|badge|标签|水印|来源标记)",
    )
    for pattern in generic_patterns:
        for match in re.finditer(pattern, combined, flags=re.IGNORECASE):
            text = re.sub(r"\s+", " ", match.group(1)).strip()
            if not text or text in seen:
                continue
            if len(text) > 48 or len(text.split()) > 8:
                continue
            seen.add(text)
            contracts.append(_anchored_contract(text))
    for match in re.finditer(
        r"(?:右下角|bottom[-_\s]?right|watermark|source\s*mark|source\s*tag|badge|标签|水印|来源标记)[^。\n]{0,60}?(?:叫|为|文字是|named|text|label)?\s*([A-Z][A-Z0-9_-]{2,48})\b",
        combined,
        flags=re.IGNORECASE,
    ):
        text = match.group(1).strip()
        if text and text not in seen:
            seen.add(text)
            contracts.append(_anchored_contract(text))
    return contracts


def _anchored_contracts_from_html_text(*texts: Any) -> list[dict[str, Any]]:
    """Extract only durable internal anchor markers from HTML source.

    Generic anchor phrasing belongs to user/context text. Running those patterns
    over HTML makes generated attributes like `data-anchor-role` feed themselves
    back into new blocking contracts.
    """
    contracts: list[dict[str, Any]] = []
    seen: set[str] = set()
    combined = "\n".join(str(text or "") for text in texts if str(text or "").strip())
    for match in _TEMP_PREF_MARKER_RE.finditer(combined):
        marker = match.group(0)
        if marker not in seen:
            seen.add(marker)
            contracts.append(_anchored_contract(marker, role="marker"))
    return contracts


def _context_text_for_visual_preferences(context: dict[str, Any] | None = None) -> str:
    context = context if isinstance(context, dict) else _read_current_modify_context()
    chunks: list[str] = [str(context.get("raw_user_message", "") or "")]
    for key in ("user_preference_rule_specs", "user_preference_colors", "diagram_contract"):
        value = context.get(key)
        if value:
            try:
                chunks.append(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
            except Exception:
                chunks.append(str(value))
    return "\n".join(chunk for chunk in chunks if chunk)


def _modify_context_applies_to_path(path: Path | None, context: dict[str, Any] | None = None) -> bool:
    context = context if isinstance(context, dict) else _read_current_modify_context()
    if not context:
        return False
    targets = context.get("target_slide_paths") or []
    if path is not None and isinstance(targets, list) and targets:
        if _path_matches_modify_target(path, targets):
            return True
        if _anchored_contracts_from_texts(_context_text_for_visual_preferences(context)):
            return get_current_agent() == "RevisionEditor" or bool(
                context.get("raw_user_message") or context.get("user_preference_rule_specs")
            )
        return False
    return get_current_agent() == "RevisionEditor"


def _active_anchored_element_contracts_for_path(
    path: Path | None = None,
    html_text: str = "",
    *,
    include_context: bool = True,
) -> list[dict[str, Any]]:
    contracts = _anchored_contracts_from_html_text(html_text)
    seen = {str(item.get("text", "") or "") for item in contracts}
    if include_context:
        context = _read_current_modify_context_for_path(path)
        if _modify_context_applies_to_path(path, context):
            for contract in _anchored_contracts_from_texts(_context_text_for_visual_preferences(context)):
                text = str(contract.get("text", "") or "").strip()
                if text and text not in seen:
                    seen.add(text)
                    contracts.append(contract)
    return contracts


def _has_anchor_class_hint(tag: Tag) -> bool:
    classes = {token.lower() for token in _tag_classes(tag)}
    return bool(classes & _ANCHOR_CLASS_HINTS)


def _is_hidden_anchor_candidate(tag: Tag) -> bool:
    props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
    return (
        str(props.get("display", "")).lower() == "none"
        or str(props.get("visibility", "")).lower() == "hidden"
        or str(props.get("opacity", "")).strip() == "0"
        or tag.has_attr("hidden")
        or str(tag.get("aria-hidden", "") or "").lower() == "true"
    )


def _is_anchor_candidate_for_text(tag: Tag, contract_text: str) -> bool:
    text = tag.get_text(" ", strip=True)
    aria = str(tag.get("aria-label", "") or "")
    props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
    class_hit = _has_anchor_class_hint(tag)
    if _ANCHOR_LAYER_CLASS in {token.lower() for token in _tag_classes(tag)}:
        return True
    if contract_text in aria:
        return True
    if class_hit:
        return True
    if text == contract_text:
        return True
    if contract_text in text and props.get("position", "").lower() in {"absolute", "fixed"} and (
        "bottom" in props or "right" in props
    ):
        return True
    return False


def _collect_existing_anchor_style(body: Tag, contract_text: str) -> dict[str, str]:
    for tag in body.find_all(True):
        if tag.parent is None:
            continue
        if not _is_anchor_candidate_for_text(tag, contract_text):
            continue
        props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
        safe: dict[str, str] = {}
        for prop, value in props.items():
            if prop not in _ANCHOR_SAFE_STYLE_PROPS:
                continue
            if prop == "font-size":
                px_value = _parse_css_pixel_value(value)
                if px_value is not None and px_value > 14:
                    continue
            safe[prop] = value
        return safe
    return {}


def _remove_anchor_css_rules(html_text: str, marker_texts: list[str]) -> tuple[str, int]:
    normalized, removed = _TEMP_PREF_STYLE_RE.subn("", html_text)
    for pattern in (
        r"(?is)\s*<style\b[^>]*\bid=[\"']memslides-anchor-style[\"'][^>]*>.*?</style>\s*",
        r"(?is)(?:\.memslides-anchor-layer|\.memslides-anchored-element|\.deck-tag|\.source-mark|\.source-tag|\.watermark|\.footer-marker|\.footer-note|\.draft-badge)\s*\{[^{}]*\}",
    ):
        normalized, count = re.subn(pattern, "", normalized)
        removed += count
    for marker in marker_texts:
        if not marker:
            continue
        normalized, count = re.subn(
            r"(?is)(?:body\s*::?\s*after)\s*\{[^{}]*content\s*:\s*(['\"])"
            + re.escape(marker)
            + r"\1[^{}]*\}",
            "",
            normalized,
        )
        removed += count
    return normalized, removed


def _anchor_layer_parent(body: Tag) -> Tag:
    roots = _top_level_slide_roots(body)
    if roots:
        return roots[0]
    return body


def _normalize_anchored_elements_html(
    html_text: str,
    *,
    path: Path | None = None,
    contracts: list[dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    original = str(html_text or "")
    active_contracts = (
        list(contracts)
        if contracts is not None
        else list(_active_anchored_element_contracts_for_path(path, original))
    )
    active_contracts = [contract for contract in active_contracts if str(contract.get("text", "") or "").strip()]
    if not active_contracts:
        return original, {
            "changed": False,
            "removed_nodes": 0,
            "removed_css_rules": 0,
            "marker_present": False,
            "contracts": [],
        }

    marker_texts = [str(contract.get("text", "") or "").strip() for contract in active_contracts]
    normalized, removed_css_rules = _remove_anchor_css_rules(original, marker_texts)
    soup = BeautifulSoup(normalized, "lxml")
    body = _ensure_html_body_tag(soup)
    preserved_styles = {text: _collect_existing_anchor_style(body, text) for text in marker_texts}
    removed_css_rules += _strip_anchor_related_css_from_soup(soup, marker_texts)
    removed_nodes = 0

    for tag in list(body.find_all(True)):
        if tag.parent is None:
            continue
        text = tag.get_text(" ", strip=True)
        aria = str(tag.get("aria-label", "") or "")
        class_hit = _has_anchor_class_hint(tag)
        text_hit = any(marker and (marker in text or marker in aria) for marker in marker_texts)
        empty_or_metadata_only = not text.strip() and bool(aria.strip())
        generated_anchor = _ANCHOR_LAYER_CLASS in {token.lower() for token in _tag_classes(tag)}
        anchor_candidate = any(_is_anchor_candidate_for_text(tag, marker) for marker in marker_texts)
        if generated_anchor or (text_hit and anchor_candidate) or (class_hit and (empty_or_metadata_only or _is_hidden_anchor_candidate(tag))):
            tag.decompose()
            removed_nodes += 1

    _upsert_anchored_element_style(soup)
    body = _ensure_html_body_tag(soup)
    parent = _anchor_layer_parent(body)
    parent_classes = _tag_classes(parent)
    if parent.name.lower() not in {"body", "html"} and "slide" not in {token.lower() for token in parent_classes}:
        parent["class"] = _unique_preserve_order([*parent_classes, "slide"])

    layer = soup.new_tag("div")
    layer["class"] = _ANCHOR_LAYER_CLASS
    layer["aria-hidden"] = "false"
    layer["data-anchor-scope"] = "current_deck_with_future_rewrites"
    for contract in active_contracts:
        text = str(contract.get("text", "") or "").strip()
        element = soup.new_tag("div")
        element["class"] = _ANCHOR_ELEMENT_CLASS
        element["data-anchor-role"] = str(contract.get("role", "marker") or "marker")
        element["data-anchor"] = str(contract.get("anchor", "bottom_right") or "bottom_right")
        element["data-anchor-text"] = text
        element["aria-label"] = text
        style_props = preserved_styles.get(text, {})
        if style_props:
            element["style"] = _format_css_declarations(style_props)
        element.string = text
        layer.append(element)
    parent.append(layer)

    final_html = str(soup)
    return final_html, {
        "changed": final_html != original,
        "removed_nodes": removed_nodes,
        "removed_css_rules": removed_css_rules,
        "marker_present": True,
        "marker_text": marker_texts[0] if marker_texts else "",
        "contracts": active_contracts,
        "anchored_element_count": len(active_contracts),
    }


def _sync_canonical_page_counter_text(slides_dir: Path) -> list[str]:
    """Update obvious page-counter footer text after canonical add/delete renumbering."""
    canonical_slides = _collect_canonical_slides(slides_dir)
    if not canonical_slides:
        return []

    total = len(canonical_slides)
    updated_slides: list[str] = []
    for current_num, slide_path in canonical_slides:
        html = slide_path.read_text(encoding="utf-8")
        updated_html = html
        replacements = 0

        for pattern, formatter in _PAGE_COUNTER_PATTERNS:
            updated_html, count = pattern.subn(formatter(current_num, total), updated_html)
            replacements += count

        for pattern in _STALE_TOTAL_PAGE_PATTERNS:
            def _replace_total(match: re.Match[str]) -> str:
                prefix = match.groupdict().get("prefix") or ""
                suffix = match.groupdict().get("suffix")
                if suffix is None:
                    return f"{prefix}{total}页"
                if suffix.lower().startswith("slide"):
                    return f"{prefix}{total} {suffix}"
                return f"{prefix}{total}页{suffix}"

            updated_html, count = pattern.subn(_replace_total, updated_html)
            replacements += count

        if updated_html == html:
            continue

        slide_path.write_text(updated_html, encoding="utf-8")
        _remember_slide_observation(slide_path)
        _invalidate_slide_snapshots_for_path(slide_path)
        if replacements > 1:
            updated_slides.append(f"{slide_path.name} ({replacements} counters)")
        else:
            updated_slides.append(slide_path.name)

    return updated_slides


def _normalize_temp_pref_marker_html(html_text: str) -> tuple[str, dict[str, Any]]:
    """Canonicalize TEMP-PREF-* into one visible anchored element inside the slide canvas."""
    contracts = _anchored_contracts_from_html_text(html_text)
    return _normalize_anchored_elements_html(html_text, contracts=contracts)


def _normalize_temp_pref_marker_file(path: Path) -> dict[str, Any]:
    """Normalize TEMP-PREF marker in a slide file and keep validation state coherent."""
    if not (path.exists() and path.is_file() and path.suffix.lower() == ".html"):
        return {"changed": False, "marker_present": False}
    try:
        original = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"changed": False, "marker_present": False}
    normalized, report = _normalize_anchored_elements_html(original, path=path)
    if not report.get("changed") or normalized == original:
        return report
    path.write_text(normalized, encoding="utf-8")
    _invalidate_slide_validation_for_path(path)
    _mark_slide_modified(path)
    _remember_slide_observation(path)
    return report


def _visible_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(str(html_text or ""), "lxml")
    for tag in soup.find_all(["script", "style", "template"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def _count_temp_pref_marker_occurrences(html_text: str) -> int:
    visible = _visible_text_from_html(html_text)
    explicit_visible = len(_TEMP_PREF_MARKER_RE.findall(visible))
    pseudo_count = len(re.findall(r"content\s*:\s*(['\"])TEMP-PREF-[A-Za-z0-9_-]+\1", str(html_text or ""), flags=re.IGNORECASE))
    return explicit_visible + pseudo_count


def _active_controlled_rewrite_context_for_path(html_path: Path) -> dict[str, Any]:
    context = _read_current_modify_context_for_path(html_path)
    if str(context.get("operation_kind", "") or "").strip() != "controlled_rewrite":
        return {}
    targets = context.get("target_slide_paths") or []
    matches_target = _path_matches_modify_target(html_path, targets)
    if not matches_target:
        try:
            resolved = html_path.resolve()
        except Exception:
            resolved = html_path
        bases = [html_path.parent, *list(html_path.parents)[:6]]
        for raw in targets:
            text = str(raw or "").strip()
            if not text:
                continue
            candidate = Path(text)
            if candidate.is_absolute():
                continue
            for base in bases:
                try:
                    if (base / candidate).resolve() == resolved:
                        matches_target = True
                        break
                except Exception:
                    continue
            if matches_target:
                break
    if not matches_target:
        return {}
    return context


def _css_props_for_tag(tag: Tag, css_rules: dict[str, dict[str, str]]) -> dict[str, str]:
    props: dict[str, str] = {}
    tag_name = tag.name.lower() if tag.name else ""
    if tag_name in css_rules:
        props.update(css_rules[tag_name])
    for class_name in _tag_classes(tag):
        selector = f".{class_name}"
        if selector in css_rules:
            props.update(css_rules[selector])
    tag_id = str(tag.get("id", "") or "").strip()
    if tag_id and f"#{tag_id}" in css_rules:
        props.update(css_rules[f"#{tag_id}"])
    props.update(_parse_css_declarations_text(str(tag.get("style", "") or "")))
    return props


def _tag_selector_style_text(tag: Tag, css_rules: dict[str, dict[str, str]]) -> str:
    props = _css_props_for_tag(tag, css_rules)
    return " ".join(f"{key}:{value}" for key, value in props.items()).lower()


def _has_layout_constraint_hint(tag: Tag, css_rules: dict[str, dict[str, str]]) -> bool:
    if tag.name and tag.name.lower() in {"body", "html"}:
        return True
    style_text = _tag_selector_style_text(tag, css_rules)
    if re.search(r"\b(width|max-width|height|max-height|overflow|table-layout)\s*:", style_text):
        return True
    classes = " ".join(_tag_classes(tag)).lower()
    return bool(re.search(r"(slide|canvas|content|table-wrap|table-wrapper|matrix|checklist|action-list)", classes))


def _has_constrained_ancestor(tag: Tag, css_rules: dict[str, dict[str, str]]) -> bool:
    current: Tag | None = tag
    depth = 0
    while current is not None and isinstance(current, Tag) and depth < 6:
        if _has_layout_constraint_hint(current, css_rules):
            return True
        parent = current.parent
        current = parent if isinstance(parent, Tag) else None
        depth += 1
    return False


def _structured_table_metrics(table: Tag, css_rules: dict[str, dict[str, str]]) -> dict[str, Any]:
    rows = [row for row in table.find_all("tr") if row.find_all(["td", "th"])]
    if not rows:
        return {
            "structured": False,
            "row_count": 0,
            "data_row_count": 0,
            "column_count": 0,
            "has_header": False,
            "visible_text_chars": 0,
            "layout_constrained": False,
        }
    header_rows = [row for row in rows if row.find("th") is not None]
    has_thead = table.find("thead") is not None
    first_cells = rows[0].find_all(["td", "th"])
    has_first_row_header = bool(first_cells) and all(cell.name and cell.name.lower() == "th" for cell in first_cells)
    has_header = bool(has_thead or header_rows or has_first_row_header)
    col_counts = [len(row.find_all(["td", "th"])) for row in rows]
    column_count = max(col_counts) if col_counts else 0
    data_row_count = max(0, len(rows) - (1 if has_header else 0))
    visible_text_chars = len(table.get_text(" ", strip=True))
    table_style = _tag_selector_style_text(table, css_rules)
    wrapper_constrained = _has_constrained_ancestor(table, css_rules)
    layout_constrained = bool(
        wrapper_constrained
        or "table-layout:fixed" in table_style.replace(" ", "")
        or re.search(r"\b(width|max-width|overflow)\s*:", table_style)
    )
    structured = bool(
        has_header
        and data_row_count >= 3
        and 2 <= column_count <= 7
        and visible_text_chars >= 80
        and layout_constrained
    )
    return {
        "structured": structured,
        "row_count": len(rows),
        "data_row_count": data_row_count,
        "column_count": column_count,
        "has_header": has_header,
        "visible_text_chars": visible_text_chars,
        "layout_constrained": layout_constrained,
    }


def _is_structured_action_list(tag: Tag, css_rules: dict[str, dict[str, str]]) -> bool:
    classes = " ".join(_tag_classes(tag)).lower()
    if not re.search(r"(action-list|checklist|task-list|matrix|timeline|priority)", classes):
        return False
    items = [
        item for item in tag.find_all(["li", "section", "article", "div"], recursive=True)
        if len(item.get_text(" ", strip=True)) >= 12
    ]
    visible_text_chars = len(tag.get_text(" ", strip=True))
    return bool(len(items) >= 3 and visible_text_chars >= 80 and _has_constrained_ancestor(tag, css_rules))


def _collect_controlled_rewrite_visual_diagnostics(
    html_path: Path,
    content: str,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    soup = BeautifulSoup(str(content or ""), "lxml")
    css_rules = _parse_css_rules(_extract_all_style_text(content))
    raw_text = str(context.get("raw_user_message", "") or "")
    rewrite_decision = context.get("rewrite_decision") if isinstance(context.get("rewrite_decision"), dict) else {}
    request_text = f"{raw_text} {json.dumps(rewrite_decision or {}, ensure_ascii=False)}".lower()

    strong_structure_class_tokens = {
        "card",
        "cards",
        "brief-card",
        "db-card",
        "decision-card",
        "section",
        "section-block",
        "panel",
        "column",
        "columns",
        "grid",
        "brief-grid",
        "summary-grid",
        "three-col",
        "three-column",
        "action-list",
        "checklist",
        "task-list",
        "matrix",
        "comparison-matrix",
        "action-table",
    }
    structure_nodes: list[Tag] = []
    grid_or_column_containers: list[Tag] = []
    structured_tables: list[dict[str, Any]] = []
    sparse_tables: list[dict[str, Any]] = []
    structured_lists: list[Tag] = []
    for tag in soup.find_all(True):
        classes = {token.lower() for token in _tag_classes(tag)}
        class_text = " ".join(classes)
        props = _css_props_for_tag(tag, css_rules)
        has_surface_style = bool(
            props.get("border")
            or props.get("box-shadow")
            or props.get("background")
            or props.get("background-color")
        ) and bool(props.get("padding") or props.get("border-radius") or props.get("min-height"))
        if classes & strong_structure_class_tokens or (
            re.search(r"(card|panel|section|column|grid)", class_text) and has_surface_style
        ):
            text_preview = tag.get_text(" ", strip=True)
            if len(text_preview) >= 6:
                structure_nodes.append(tag)
        display = props.get("display", "").lower()
        grid_template = props.get("grid-template-columns", "").lower()
        flex_direction = props.get("flex-direction", "").lower()
        column_count = props.get("column-count", "").lower()
        if (
            "grid" in display
            or (display == "flex" and flex_direction != "column")
            or "repeat(3" in grid_template
            or re.search(r"(?:^|\s)3(?:\s|$)", column_count)
        ):
            grid_or_column_containers.append(tag)
        if tag.name and tag.name.lower() == "table":
            table_metrics = _structured_table_metrics(tag, css_rules)
            if table_metrics.get("structured"):
                structured_tables.append(table_metrics)
            else:
                sparse_tables.append(table_metrics)
        elif _is_structured_action_list(tag, css_rules):
            structured_lists.append(tag)

    visible_text = _visible_text_from_html(content)
    semantic_labels = (
        "研究问题",
        "核心贡献",
        "为什么继续读",
        "为什么值得继续读",
        "判断点",
        "research question",
        "core contribution",
        "why keep reading",
        "why read on",
    )
    semantic_block_count = sum(1 for label in semantic_labels if label.lower() in visible_text.lower())
    marker_count = _count_temp_pref_marker_occurrences(content)
    metrics.update(
        {
            "controlled_rewrite_structure_nodes": len(structure_nodes),
            "controlled_rewrite_grid_or_column_containers": len(grid_or_column_containers),
            "controlled_rewrite_structured_table_count": len(structured_tables),
            "controlled_rewrite_structured_list_count": len(structured_lists),
            "controlled_rewrite_sparse_table_count": len(sparse_tables),
            "controlled_rewrite_semantic_label_count": semantic_block_count,
            "controlled_rewrite_temp_pref_count": marker_count,
        }
    )

    needs_decision_brief = bool(
        re.search(
            r"(decision brief|brief|三块|三列|三卡片|只保留|研究问题|核心贡献|为什么.*读|扫到重点|开场页|首页|封面)",
            request_text,
            flags=re.IGNORECASE,
        )
    )
    wants_structured_table = bool(
        re.search(
            r"(行动清单|清单|表格|三线表|矩阵|对照|负责人|负责角色|优先级|action list|checklist|table|matrix|comparison)",
            request_text,
            flags=re.IGNORECASE,
        )
    )
    has_card_grid_structure = len(structure_nodes) >= 3 or bool(grid_or_column_containers)
    has_table_or_list_structure = bool(structured_tables or structured_lists)
    has_page_structure = has_card_grid_structure or has_table_or_list_structure
    if not has_page_structure:
        sparse_table_hint = ""
        if sparse_tables:
            best_sparse = max(sparse_tables, key=lambda item: int(item.get("visible_text_chars", 0) or 0))
            sparse_table_hint = (
                " A table is present but does not yet meet structured-table criteria "
                f"(header={best_sparse.get('has_header')}, rows={best_sparse.get('data_row_count')}, "
                f"cols={best_sparse.get('column_count')}, constrained={best_sparse.get('layout_constrained')})."
            )
        diagnostics.append(
            {
                "code": "weak_controlled_rewrite_layout",
                "severity": "error",
                "message": (
                    "Controlled rewrite target lacks clear page-level structure. "
                    "Use visible cards/columns/grid/section blocks, or a constrained structured table/action list "
                    "with a header, at least three rows, reasonable columns, and enough visible text."
                    + sparse_table_hint
                ),
                "source": "visual_quality",
            }
        )
    if needs_decision_brief and not wants_structured_table and (semantic_block_count < 3 or not has_card_grid_structure):
        diagnostics.append(
            {
                "code": "decision_brief_not_visually_structured",
                "severity": "error",
                "message": (
                    "Decision brief rewrite should preserve three semantic blocks and render them "
                    "as distinct cards/columns/sections."
                ),
                "source": "visual_quality",
                "semantic_block_count": semantic_block_count,
            }
        )
    if marker_count > 1:
        diagnostics.append(
            {
                "code": "temp_pref_marker_repeated",
                "severity": "error",
                "message": "TEMP-PREF marker appears more than once on a controlled rewrite target.",
                "source": "visual_quality",
                "marker_count": marker_count,
            }
        )
    return diagnostics, metrics


def _validate_anchored_element_contracts(
    html: str,
    contracts: list[dict[str, Any]] | None = None,
    *,
    path: Path | None = None,
) -> list[dict[str, Any]]:
    active_contracts = (
        list(contracts)
        if contracts is not None
        else list(_active_anchored_element_contracts_for_path(path, html))
    )
    diagnostics: list[dict[str, Any]] = []
    if not active_contracts:
        return diagnostics

    soup = BeautifulSoup(str(html or ""), "lxml")
    visible_text = _visible_text_from_html(html)
    for contract in active_contracts:
        text = str(contract.get("text", "") or "").strip()
        if not text:
            continue
        visible_count = len(re.findall(re.escape(text), visible_text))
        pseudo_count = len(re.findall(r"content\s*:\s*(['\"])" + re.escape(text) + r"\1", str(html or ""), flags=re.IGNORECASE))
        total_count = visible_count + pseudo_count
        if total_count < 1:
            diagnostics.append(
                {
                    "code": "anchored_element_missing",
                    "severity": "error",
                    "message": f"Anchored visual element `{text}` must be visible exactly once; found 0.",
                    "source": "anchored_element",
                    "anchor": contract.get("anchor", "bottom_right"),
                    "text": text,
                    "repair_strategy": "Normalize the requested marker/badge/source tag into `.memslides-anchor-layer` inside the slide canvas.",
                }
            )
            continue
        if total_count > 1:
            diagnostics.append(
                {
                    "code": "anchored_element_duplicate",
                    "severity": "error",
                    "message": f"Anchored visual element `{text}` must be visible exactly once; found {total_count}.",
                    "source": "anchored_element",
                    "anchor": contract.get("anchor", "bottom_right"),
                    "text": text,
                    "count": total_count,
                    "repair_strategy": "Remove duplicate marker/badge nodes and keep one `.memslides-anchored-element`.",
                }
            )

        anchor_tags: list[Tag] = []
        for tag in soup.find_all(True):
            if tag.name.lower() in {"html", "body"}:
                continue
            if _ANCHOR_LAYER_CLASS in {token.lower() for token in _tag_classes(tag)}:
                continue
            if text in tag.get_text(" ", strip=True) or text in str(tag.get("aria-label", "") or ""):
                anchor_tags.append(tag)
        visible_anchor_tags = [
            tag
            for tag in anchor_tags
            if text in tag.get_text(" ", strip=True) and not _is_hidden_anchor_candidate(tag)
        ]
        if not visible_anchor_tags and total_count > 0:
            diagnostics.append(
                {
                    "code": "anchored_element_empty",
                    "severity": "error",
                    "message": f"Anchored visual element `{text}` is present only as hidden/empty metadata, not visible slide text.",
                    "source": "anchored_element",
                    "anchor": contract.get("anchor", "bottom_right"),
                    "text": text,
                    "repair_strategy": "Put the marker text in a visible `.memslides-anchored-element`; aria-label alone is not enough.",
                }
            )
        for tag in visible_anchor_tags[:3]:
            props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
            class_hint = _has_anchor_class_hint(tag)
            if props.get("position", "").lower() == "fixed" or (
                class_hint and _parse_css_pixel_value(props.get("bottom", "")) is not None and (_parse_css_pixel_value(props.get("bottom", "")) or 0) < _ANCHOR_DEFAULT_MARGIN_PX
            ):
                diagnostics.append(
                    {
                        "code": "anchored_element_off_canvas_risk",
                        "severity": "error",
                        "message": f"Anchored visual element `{text}` uses fixed/edge positioning and may be clipped by export.",
                        "source": "anchored_element",
                        "anchor": contract.get("anchor", "bottom_right"),
                        "text": text,
                        "target_selector_hint": _selector_hint_for_tag(tag),
                        "target_dom_path": _build_dom_path(tag),
                        "repair_strategy": "Move the marker into `.memslides-anchor-layer` with right/bottom safe margins inside `.slide`.",
                    }
                )
                break
    return diagnostics


def _active_diagram_contract_for_path(path: Path) -> dict[str, Any]:
    context = _read_current_modify_context()
    if str(context.get("operation_kind", "") or "").strip() != "diagram_layout":
        return {}
    if not _path_matches_modify_target(path, context.get("target_slide_paths") or []):
        return {}
    contract = context.get("diagram_contract")
    if not isinstance(contract, dict):
        return {}
    normalized = dict(contract)
    if not str(normalized.get("marker_text", "") or "").strip():
        try:
            html_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            html_text = ""
        marker_match = _TEMP_PREF_MARKER_RE.search(html_text)
        if marker_match:
            normalized["marker_text"] = marker_match.group(0)
    return normalized


def _normalize_diagram_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("／", "/").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    aliases = {
        "输入token": "input token",
        "输入 token": "input token",
        "输入 tokens": "input token",
        "input": "input token",
        "token": "input token",
        "tokens": "input token",
        "wq": "w_q",
        "wk": "w_k",
        "wv": "w_v",
        "attention scores": "attention score",
        "注意力分数": "attention score",
        "加权求和": "weighted sum",
        "输出": "output",
    }
    return aliases.get(text, text)


def _diagram_node_present(text: str, node: str) -> bool:
    normalized_text = _normalize_diagram_token(text)
    normalized_node = _normalize_diagram_token(node)
    if not normalized_node:
        return True
    if normalized_node in {"q", "k", "v"}:
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(normalized_node)}(?![a-z0-9_])", normalized_text))
    if normalized_node in normalized_text:
        return True
    if normalized_node == "input token":
        return (
            ("input" in normalized_text and "token" in normalized_text)
            or ("输入" in normalized_text and "token" in normalized_text)
        )
    if normalized_node == "attention score":
        return "attention" in normalized_text and ("score" in normalized_text or "分数" in normalized_text)
    if normalized_node == "weighted sum":
        return ("weighted" in normalized_text and "sum" in normalized_text) or "加权" in normalized_text
    return False


def validate_diagram_layout_static(
    html: str,
    contract: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Check whether a target slide actually satisfies a diagram rewrite contract."""
    contract = contract or {}
    if not contract:
        return []
    soup = BeautifulSoup(str(html or ""), "lxml")
    visible_text = _visible_text_from_html(str(html or ""))
    lowered_html = str(html or "").lower()
    diagnostics: list[dict[str, Any]] = []

    required_nodes = [
        str(node)
        for node in (contract.get("required_nodes") or [])
        if str(node or "").strip()
    ]
    missing_nodes = [node for node in required_nodes if not _diagram_node_present(visible_text, node)]
    if missing_nodes:
        diagnostics.append(
            {
                "code": "diagram_required_nodes_missing",
                "severity": "error",
                "message": "Diagram is missing required nodes: " + ", ".join(missing_nodes[:8]),
                "missing_nodes": missing_nodes,
                "source": "diagram_layout",
            }
        )

    title_hint = str(contract.get("title_hint", "") or "").strip()
    title_text = ""
    title_tag = soup.find(["h1", "h2", "title"])
    if isinstance(title_tag, Tag):
        title_text = title_tag.get_text(" ", strip=True)
    if title_hint:
        hint_tokens = [token for token in re.split(r"[^a-zA-Z0-9\u4e00-\u9fff]+", title_hint.lower()) if len(token) >= 4]
        title_lower = title_text.lower()
        if hint_tokens and not any(token in title_lower for token in hint_tokens):
            diagnostics.append(
                {
                    "code": "diagram_title_mismatch",
                    "severity": "error",
                    "message": f"Slide title `{title_text or '<empty>'}` does not reflect diagram task `{title_hint}`.",
                    "title": title_text,
                    "expected_title_hint": title_hint,
                    "source": "diagram_layout",
                }
            )

    arrow_count = len(re.findall(r"(?:→|⇒|➜|➔|-->|->)", str(html or "")))
    arrow_count += len(re.findall(r"<(?:line|path)\b[^>]*(?:marker-end|arrowhead)", str(html or ""), flags=re.IGNORECASE))
    expected_edges = contract.get("required_edges") or []
    min_arrows = min(3, max(1, len(expected_edges) // 2)) if expected_edges else 2
    if arrow_count < min_arrows:
        diagnostics.append(
            {
                "code": "diagram_edges_missing",
                "severity": "error",
                "message": f"Diagram has too few visible/structured arrows ({arrow_count}); expected at least {min_arrows}.",
                "arrow_count": arrow_count,
                "source": "diagram_layout",
            }
        )

    if bool(contract.get("remove_conflicting_assets", False)):
        image_tags = soup.find_all("img")
        conflicting: list[str] = []
        for img in image_tags:
            class_text = " ".join(_tag_classes(img)).lower()
            src_text = str(img.get("src", "") or "").lower()
            alt_text = str(img.get("alt", "") or "").lower()
            if "flowchart" in class_text or "flowchart" in src_text or "pipeline" in src_text:
                continue
            if "generated_visuals" in src_text and ("flowchart" in src_text or "diagram" in src_text):
                continue
            if "transformer" in src_text or "architecture" in alt_text or "架构" in alt_text:
                conflicting.append(str(img.get("src", "") or img.get("alt", ""))[:120])
        if conflicting:
            diagnostics.append(
                {
                    "code": "diagram_conflicting_legacy_asset",
                    "severity": "error",
                    "message": "Old/conflicting figure assets remain on a diagram rewrite slide.",
                    "assets": conflicting[:4],
                    "source": "diagram_layout",
                }
            )
        empty_cards = 0
        for tag in soup.find_all(["div", "section", "article"]):
            classes = {cls.lower() for cls in _tag_classes(tag)}
            if not classes & {"card", "content-row", "figure-box", "bullet"}:
                continue
            text = tag.get_text(" ", strip=True)
            has_media = bool(tag.find(["img", "svg", "canvas", "table"]))
            if len(text) < 2 and not has_media:
                empty_cards += 1
        if empty_cards:
            diagnostics.append(
                {
                    "code": "diagram_empty_legacy_container",
                    "severity": "error",
                    "message": f"Diagram rewrite left {empty_cards} empty legacy container(s).",
                    "count": empty_cards,
                    "source": "diagram_layout",
                }
            )

    marker_text = str(contract.get("marker_text", "") or "").strip()
    if marker_text:
        marker_visible = marker_text in visible_text or bool(
            re.search(r"content\s*:\s*(['\"])" + re.escape(marker_text) + r"\1", str(html or ""), flags=re.IGNORECASE)
        )
        if not marker_visible:
            diagnostics.append(
                {
                    "code": "diagram_marker_missing",
                    "severity": "error",
                    "message": f"Required visual preference marker `{marker_text}` is not visible/non-empty.",
                    "marker_text": marker_text,
                    "source": "diagram_layout",
                }
            )

    if not any(token in lowered_html for token in ("flowchart", "pipeline", "diagram", "<svg", "marker-end", "arrowhead", "→")):
        diagnostics.append(
            {
                "code": "diagram_visual_structure_missing",
                "severity": "error",
                "message": "Slide does not expose a clear diagram/flowchart visual structure.",
                "source": "diagram_layout",
            }
        )
    return diagnostics


def _diagram_style_context_from_html(html: str, contract: dict[str, Any] | None = None) -> dict[str, str]:
    variables = _css_custom_properties(html)
    contract = contract or {}
    marker_text = str(contract.get("marker_text", "") or "").strip()
    return {
        "title": _resolve_css_hex("var(--color-title)", variables) or _resolve_css_hex("var(--color-primary)", variables) or "#1D3557",
        "accent": _resolve_css_hex("var(--color-accent)", variables) or "#0D9488",
        "surface": _resolve_css_hex("var(--color-evidence-bg)", variables) or _resolve_css_hex("var(--color-surface)", variables) or "#F3F4F6",
        "border": _resolve_css_hex("var(--color-evidence-border)", variables) or _resolve_css_hex("var(--color-border)", variables) or "#CBD5E1",
        "text": _resolve_css_hex("var(--color-primary-text)", variables) or "#111827",
        "marker_text": marker_text,
    }


def _fallback_style_context_from_html(html: str) -> dict[str, str]:
    variables = _css_custom_properties(html)
    style = {
        "background": _resolve_css_hex("var(--color-background)", variables) or "#F8FAFC",
        "title": _resolve_css_hex("var(--color-title)", variables) or _resolve_css_hex("var(--color-primary)", variables) or "#1D3557",
        "accent": _resolve_css_hex("var(--color-accent)", variables) or "#0D9488",
        "surface": _resolve_css_hex("var(--color-surface)", variables) or "#FFFFFF",
        "evidence_bg": _resolve_css_hex("var(--color-evidence-bg)", variables) or "#F3F4F6",
        "border": _resolve_css_hex("var(--color-evidence-border)", variables) or _resolve_css_hex("var(--color-border)", variables) or "#CBD5E1",
        "text": _resolve_css_hex("var(--color-primary-text)", variables) or "#111827",
        "muted": _resolve_css_hex("var(--color-muted-text)", variables) or "#475569",
    }
    rules = _iter_css_rules_with_props(html)
    for selector, props in rules:
        selector_l = selector.lower()
        if re.search(r"\bh1\b|title|heading", selector_l):
            color = _resolve_css_hex(props.get("color", ""), variables)
            if color:
                style["title"] = color
                break
    for selector, props in rules:
        selector_l = selector.lower()
        if re.search(r"card|evidence|panel|content-card|text-panel", selector_l):
            bg = _resolve_css_hex(props.get("background-color", "") or props.get("background", ""), variables)
            border = _resolve_css_hex(props.get("border-color", "") or props.get("border", ""), variables)
            if bg:
                style["evidence_bg"] = bg
            if border:
                style["border"] = border
            break
    return style


def build_safe_fallback_diagram_slide(
    html: str,
    diagnostics: list[dict[str, Any]] | None,
    contract: dict[str, Any],
    aspect_ratio: str = "16:9",
) -> str:
    """Build a conservative diagram page that satisfies common flowchart contracts."""
    style = _diagram_style_context_from_html(html, contract)
    nodes = [
        str(node)
        for node in (contract.get("required_nodes") or [])
        if str(node or "").strip()
    ]
    if not nodes:
        nodes = ["Input", "Process", "Output"]
    title = str(contract.get("title_hint", "") or "Pipeline Diagram").strip()
    node_html: list[str] = []
    branch_like = any(_normalize_diagram_token(node) in {"w_q", "w_k", "w_v", "q", "k", "v"} for node in nodes)
    if branch_like:
        rows = [
            ["输入 token", "W_Q", "Q"],
            ["输入 token", "W_K", "K", "attention score", "softmax"],
            ["输入 token", "W_V", "V", "weighted sum", "output"],
        ]
        seen: set[str] = set()
        for row in rows:
            pieces = []
            for item in row:
                if item not in seen or item == "输入 token":
                    label = item
                    seen.add(item)
                    pieces.append(f'<div class="node"><p>{html_escape(label)}</p></div>')
                else:
                    pieces.append(f'<div class="node ghost"><p>{html_escape(item)}</p></div>')
                if item != row[-1]:
                    pieces.append('<div class="arrow"><p>→</p></div>')
            node_html.append('<div class="flow-row">' + "".join(pieces) + "</div>")
    else:
        pieces = []
        for index, node in enumerate(nodes, start=1):
            pieces.append(f'<div class="node"><span>{index}</span><p>{html_escape(node)}</p></div>')
            if index < len(nodes):
                pieces.append('<div class="arrow"><p>→</p></div>')
        node_html.append('<div class="flow-row wrap">' + "".join(pieces) + "</div>")
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>{html_escape(title)}</title>
<style>
body {{ width:1280px; height:720px; margin:0; background:#ffffff; color:{style["text"]}; font-family: Arial, 'Noto Sans CJK SC', 'Microsoft YaHei', sans-serif; }}
.slide {{ width:1280px; height:720px; box-sizing:border-box; padding:56px 64px 64px; position:relative; overflow:hidden; }}
h1 {{ margin:0; font-size:38px; line-height:1.15; color:{style["title"]}; font-weight:800; }}
.rule {{ width:96px; height:4px; background:{style["accent"]}; margin:16px 0 34px; }}
.diagram {{ width:100%; min-height:470px; display:flex; flex-direction:column; justify-content:center; gap:22px; background:{style["surface"]}; border:1px solid {style["border"]}; border-radius:8px; padding:30px; box-sizing:border-box; }}
.flow-row {{ display:flex; align-items:center; justify-content:center; gap:12px; }}
.flow-row.wrap {{ flex-wrap:wrap; row-gap:18px; }}
.node {{ min-width:118px; max-width:170px; min-height:64px; padding:12px 16px; background:#fff; border:2px solid {style["accent"]}; border-radius:8px; box-sizing:border-box; display:flex; align-items:center; justify-content:center; gap:8px; text-align:center; }}
.node.ghost {{ border-style:dashed; opacity:.92; }}
.node span {{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:{style["accent"]}; color:#fff; font-size:12px; font-weight:800; flex:0 0 auto; }}
.node p, .arrow p {{ margin:0; }}
.node p {{ font-size:17px; line-height:1.2; font-weight:700; }}
.arrow {{ color:{style["accent"]}; font-size:28px; font-weight:900; flex:0 0 auto; }}
</style>
</head>
<body>
<div class="slide">
<h1>{html_escape(title)}</h1>
<div class="rule"></div>
<div class="diagram flowchart pipeline" data-diagram-kind="{html_escape(str(contract.get("diagram_kind", "pipeline") or "pipeline"))}">
{''.join(node_html)}
</div>
</div>
</body>
</html>"""
    contracts = []
    marker_text = style.get("marker_text", "")
    if marker_text:
        contracts.append(_anchored_contract(marker_text, role="marker"))
    normalized, _report = _normalize_anchored_elements_html(html_text, contracts=contracts)
    return normalized


# ── HTML slide property extractor ────────────────────────────────────
class _SlideHTMLParser(HTMLParser):
    """Lightweight parser that extracts key CSS properties from slide HTML."""

    def __init__(self):
        super().__init__()
        self._in_style = False
        self._style_text = ""
        self._current_tag = ""
        self._current_class = ""
        self._text_chunks: list[tuple[str, str]] = []  # (class, text)
        self._image_srcs: list[str] = []

    def handle_starttag(self, tag, attrs):
        attr_dict = dict(attrs)
        self._current_tag = tag
        self._current_class = attr_dict.get("class", "")
        if tag == "style":
            self._in_style = True
            self._style_text = ""
        if tag == "img":
            src = attr_dict.get("src", "")
            if src:
                self._image_srcs.append(src)
        # Also capture background-image in inline style
        inline_style = attr_dict.get("style", "")
        if "url(" in inline_style:
            m = re.search(r"url\(['\"]?([^'\")\s]+)['\"]?\)", inline_style)
            if m:
                self._image_srcs.append(m.group(1))

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False

    def handle_data(self, data):
        if self._in_style:
            self._style_text += data
        else:
            text = data.strip()
            if text:
                self._text_chunks.append((self._current_class, text))


def _parse_css_rules(style_text: str) -> dict[str, dict[str, str]]:
    """Parse CSS text into {selector: {property: value}} dict."""
    rules: dict[str, dict[str, str]] = {}
    # Remove comments
    style_text = re.sub(r"/\*.*?\*/", "", style_text, flags=re.DOTALL)
    for m in re.finditer(r"([^{}]+)\{([^{}]+)\}", style_text):
        selector = m.group(1).strip()
        if selector.startswith("@"):
            continue
        props = {}
        for decl in m.group(2).split(";"):
            decl = decl.strip()
            if ":" in decl:
                k, v = decl.split(":", 1)
                props[k.strip().lower()] = v.strip()
        if props:
            rules[selector] = props
    return rules


def _strip_anchor_related_css_from_soup(soup: BeautifulSoup, marker_texts: list[str]) -> int:
    removed = 0
    selector_tokens = (
        "body::after",
        "body:after",
        "footer::after",
        "footer:after",
        ".footer::after",
        ".footer:after",
        ".source-badge",
        ".source-mark",
        ".source-tag",
        ".deck-tag",
        ".watermark",
        ".footer-marker",
        ".footer-note",
        ".draft-badge",
        ".memslides-anchor-layer",
        ".memslides-anchored-element",
    )
    marker_tokens = [token for token in marker_texts if token]
    for style_tag in list(soup.find_all("style")):
        css = str(style_tag.string or style_tag.get_text() or "")
        if not css:
            continue
        updated = css
        for match in list(re.finditer(r"(?P<selector>[^{}]+)\{(?P<body>[^{}]*)\}", css, flags=re.DOTALL)):
            selector = match.group("selector").strip()
            body = match.group("body")
            selector_l = selector.lower()
            body_l = body.lower()
            mentions_marker = any(token in body or token in selector for token in marker_tokens)
            mentions_anchor = any(token in selector_l for token in selector_tokens)
            if mentions_marker or (mentions_anchor and any(prop in body_l for prop in ("content", "position", "bottom", "right"))):
                updated = updated.replace(match.group(0), "")
                removed += 1
        if updated.strip():
            style_tag.string = updated
        else:
            style_tag.decompose()
    return removed


def _parse_css_declarations_text(declarations: str) -> dict[str, str]:
    """Parse a CSS declaration block into an ordered property/value mapping."""
    props: dict[str, str] = {}
    for decl in str(declarations or "").split(";"):
        decl = decl.strip()
        if ":" not in decl:
            continue
        key, value = decl.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key and value:
            props[key] = value
    return props


_LAYOUT_CONTEXT_PROP_KEYS = (
    "display",
    "position",
    "top",
    "left",
    "right",
    "bottom",
    "width",
    "height",
    "max-width",
    "max-height",
    "padding",
    "padding-top",
    "padding-right",
    "padding-bottom",
    "padding-left",
    "margin",
    "margin-top",
    "margin-right",
    "margin-bottom",
    "margin-left",
    "box-sizing",
    "overflow",
    "overflow-x",
    "overflow-y",
    "font-size",
    "line-height",
)
_NODE_ATTR_MUTATION_WHITELIST = {"class", "style", "src", "alt", "width", "height"}
_RULE_ALLOWED_OPS = ["merge_css_rule", "replace_css_rule"]
_EXPECTED_LAYOUT_SIZES: dict[str, tuple[int, int]] = {
    "16:9": (1280, 720),
    "4:3": (960, 720),
    "A1": (2244, 3178),
    "A2": (1587, 2244),
    "A3": (1122, 1587),
    "A4": (794, 1123),
}


_DYNAMIC_LAYOUT_UNIT_RE = re.compile(
    r"(?i)(?:\d+(?:\.\d+)?\s*(?:vw|vh|vmin|vmax|%)|calc\s*\(|min\s*\(|max\s*\()"
)


def _body_layout_risk_notes(html_text: str) -> list[str]:
    """Return deterministic warnings for slide HTML that relies on fragile responsive sizing."""
    parser = _SlideHTMLParser()
    parser.feed(html_text)
    css_rules = _parse_css_rules(parser._style_text)

    body_props: dict[str, str] = {}
    for selector, props in css_rules.items():
        selectors = [part.strip().lower() for part in selector.split(",")]
        if "body" in selectors:
            body_props.update(props)

    inline_body_props: dict[str, str] = {}
    try:
        soup = BeautifulSoup(html_text, "lxml")
        if isinstance(soup.body, Tag):
            inline_body_props = _parse_css_declarations_text(soup.body.get("style", ""))
    except Exception:
        inline_body_props = {}

    combined_body_props = {**body_props, **inline_body_props}
    width = combined_body_props.get("width", "")
    height = combined_body_props.get("height", "")

    notes: list[str] = []
    if not width or not height:
        notes.append(
            "`body` 缺少显式 `width/height` 像素尺寸；请直接写成固定画布尺寸，例如 16:9 使用 `1280px × 720px`。"
        )
    else:
        if not width.lower().endswith("px") or _DYNAMIC_LAYOUT_UNIT_RE.search(width):
            notes.append(f"`body width` 使用了非固定单位（`{width}`）；请改为精确像素值。")
        if not height.lower().endswith("px") or _DYNAMIC_LAYOUT_UNIT_RE.search(height):
            notes.append(f"`body height` 使用了非固定单位（`{height}`）；请改为精确像素值。")

    if re.search(r"\b\d+(?:\.\d+)?(?:vw|vh|vmin|vmax)\b", html_text, re.IGNORECASE) and re.search(
        r"aspect-ratio\s*:\s*16\s*/\s*9",
        html_text,
        re.IGNORECASE,
    ):
        notes.append(
            "检测到 `vw/vh` 与 `aspect-ratio` 组合；这类响应式主画布在 PPT 渲染时很容易把顶部或底部内容裁出画布。"
        )

    if (
        re.search(r"align-items\s*:\s*center", html_text, re.IGNORECASE)
        and re.search(r"justify-content\s*:\s*center", html_text, re.IGNORECASE)
        and re.search(r"aspect-ratio\s*:\s*16\s*/\s*9", html_text, re.IGNORECASE)
    ):
        notes.append(
            "检测到居中 flex 外壳包裹固定长宽比内容区；当内容区高度超出 viewport 时，标题和首屏内容会被整体上推裁切。"
        )

    deduped: list[str] = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return deduped


def _format_css_declarations(props: dict[str, str]) -> str:
    """Serialize CSS declarations using a compact but readable format."""
    return "; ".join(f"{key}: {value}" for key, value in props.items()) + ";"


def _extract_all_style_text(html_text: str) -> str:
    """Concatenate all inline <style> block bodies from one HTML document."""
    chunks = [
        match.group("body")
        for match in re.finditer(
            r"<style\b[^>]*>(?P<body>.*?)</style>",
            html_text,
            re.IGNORECASE | re.DOTALL,
        )
    ]
    return "\n".join(chunks)


def _style_context_subset(props: dict[str, str]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in props.items()
        if key in _LAYOUT_CONTEXT_PROP_KEYS and str(value or "").strip()
    }


def _normalized_style_signature(style_text: str) -> str:
    props = _parse_css_declarations_text(style_text)
    if not props:
        return ""
    ordered = {key: props[key] for key in sorted(props)}
    return _format_css_declarations(ordered)


def _normalized_tag_attr_value(attr_name: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        normalized = " ".join(str(item).strip() for item in value if str(item).strip())
    else:
        normalized = str(value or "").strip()
    if attr_name.lower() == "style":
        return _normalized_style_signature(normalized)
    return normalized


def _normalized_tag_signature(tag: Tag) -> str:
    attrs = {
        str(key).lower(): _normalized_tag_attr_value(str(key), value)
        for key, value in sorted(tag.attrs.items(), key=lambda item: str(item[0]).lower())
        if _normalized_tag_attr_value(str(key), value)
    }
    child_tags = [
        child.name.lower()
        for child in tag.children
        if isinstance(child, Tag)
    ]
    text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
    return json.dumps(
        {
            "name": tag.name.lower(),
            "attrs": attrs,
            "child_tags": child_tags[:12],
            "text_preview": text[:240],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _compute_target_hash(tag: Tag) -> str:
    return hashlib.sha256(_normalized_tag_signature(tag).encode("utf-8")).hexdigest()


def _compute_rule_hash(selector: str, declarations: dict[str, str]) -> str:
    ordered = {key: declarations[key] for key in sorted(declarations)}
    payload = json.dumps(
        {
            "selector": str(selector or "").strip().lower(),
            "declarations": ordered,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _extract_css_rule_entries(html_text: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    style_re = re.compile(
        r"<style\b[^>]*>(?P<style_body>.*?)</style>",
        re.IGNORECASE | re.DOTALL,
    )
    rule_re = re.compile(
        r"(?P<selector>[^{}]+)\{(?P<declarations>[^{}]+)\}",
        re.DOTALL,
    )
    for style_block_index, style_match in enumerate(style_re.finditer(html_text), start=1):
        style_body = style_match.group("style_body")
        block_offset = style_match.start("style_body")
        rule_index = 0
        for rule_match in rule_re.finditer(style_body):
            selector = str(rule_match.group("selector") or "").strip()
            if not selector or selector.lstrip().startswith("@"):
                continue
            declarations = _parse_css_declarations_text(rule_match.group("declarations"))
            if not declarations:
                continue
            rule_index += 1
            declarations_preview = _format_css_declarations(
                {key: declarations[key] for key in sorted(declarations)}
            )
            entries.append(
                {
                    "rule_id": f"rule_{len(entries) + 1:02d}",
                    "selector": selector,
                    "declarations": declarations,
                    "declarations_preview": declarations_preview,
                    "rule_hash": _compute_rule_hash(selector, declarations),
                    "style_block_index": style_block_index,
                    "rule_index": rule_index,
                    "whole_span": (
                        block_offset + rule_match.start(),
                        block_offset + rule_match.end(),
                    ),
                    "declarations_span": (
                        block_offset + rule_match.start("declarations"),
                        block_offset + rule_match.end("declarations"),
                    ),
                }
            )
    return entries


def _selector_mentions_body_or_slide(selector: str) -> bool:
    normalized = str(selector or "").strip().lower()
    return bool(
        normalized
        and (
            re.search(r"(^|[^a-z0-9_-])body([^a-z0-9_-]|$)", normalized)
            or re.search(r"(^|[^a-z0-9_-])\.slide([^a-z0-9_-]|$)", normalized)
        )
    )


def _css_selectors_containing_class(style_text: str, class_name: str) -> list[str]:
    """Return CSS selectors that target a given class name."""
    pattern = re.compile(rf"(^|[^a-zA-Z0-9_-])\.{re.escape(class_name)}(?![a-zA-Z0-9_-])")
    matches: list[str] = []
    for selector in _parse_css_rules(style_text).keys():
        if pattern.search(selector):
            matches.append(selector.strip())
    matches.sort(key=lambda item: (item.count("."), item.count(" "), len(item)), reverse=True)
    return matches


def _html_has_class(html_text: str, class_name: str) -> bool:
    """Return whether the HTML contains a given class token."""
    target = str(class_name or "").strip().lower()
    if not target:
        return False
    for match in re.finditer(
        r'class\s*=\s*(["\'])(?P<value>.*?)\1',
        html_text,
        re.IGNORECASE | re.DOTALL,
    ):
        tokens = {
            token.strip().lower()
            for token in str(match.group("value") or "").split()
            if token.strip()
        }
        if target in tokens:
            return True
    return False


def _unique_preserve_order(values: list[str]) -> list[str]:
    """De-duplicate while preserving order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _unique_tags(tags: list[Tag]) -> list[Tag]:
    seen: set[int] = set()
    ordered: list[Tag] = []
    for tag in tags:
        marker = id(tag)
        if marker in seen:
            continue
        seen.add(marker)
        ordered.append(tag)
    return ordered


def _infer_semantic_selectors(
    html_text: str,
    target_kind: str,
) -> list[str]:
    """Infer the most likely CSS selectors for a semantic slide target."""
    style_text = _extract_all_style_text(html_text)
    selectors: list[str] = []

    if target_kind == "slide_title":
        if re.search(r"<h1\b", html_text, re.IGNORECASE):
            selectors.append("h1")
        for class_name in ("title", "h1", "hero", "heading", "card-title", "panel-title", "title-wrap"):
            if _html_has_class(html_text, class_name):
                existing = _css_selectors_containing_class(style_text, class_name)
                if existing:
                    selectors.extend(existing)
                else:
                    selectors.append(f".{class_name}")
        if not selectors and re.search(r"<h2\b", html_text, re.IGNORECASE):
            selectors.append("h2")
        return _unique_preserve_order(selectors)[:3]

    if target_kind == "footer":
        for class_name in ("footer", "cap", "caption", "page", "src"):
            if _html_has_class(html_text, class_name):
                existing = _css_selectors_containing_class(style_text, class_name)
                if existing:
                    selectors.extend(existing)
                else:
                    selectors.append(f".{class_name}")
        if re.search(r"<footer\b", html_text, re.IGNORECASE):
            selectors.append("footer")
        return _unique_preserve_order(selectors)[:4]

    if target_kind == "caption":
        for class_name in ("caption", "figure-caption", "table-caption", "image-caption", "fig-caption", "note"):
            if _html_has_class(html_text, class_name):
                existing = _css_selectors_containing_class(style_text, class_name)
                if existing:
                    selectors.extend(existing)
                else:
                    selectors.append(f".{class_name}")
        if re.search(r"<figcaption\b", html_text, re.IGNORECASE):
            selectors.append("figcaption")
        return _unique_preserve_order(selectors)[:5]

    if target_kind == "pill_tag":
        for class_name in ("badge", "pill", "pill-tag", "tag", "chip", "label"):
            if _html_has_class(html_text, class_name):
                existing = _css_selectors_containing_class(style_text, class_name)
                if existing:
                    selectors.extend(existing)
                else:
                    selectors.append(f".{class_name}")
        return _unique_preserve_order(selectors)[:5]

    if target_kind in {"table_cell", "chart_axis_label", "chart_data_label", "legend_label", "callout", "any_text"}:
        mapping = {
            "table_cell": ["td", "th", ".table-cell"],
            "chart_axis_label": [".axis-label", ".chart-axis-label"],
            "chart_data_label": [".data-label", ".chart-data-label"],
            "legend_label": [".legend", ".legend-label"],
            "callout": [".callout", ".annotation"],
            "any_text": ["h1", "h2", "p", "li", "span"],
        }
        for selector in mapping[target_kind]:
            if selector.startswith("."):
                if _html_has_class(html_text, selector[1:]):
                    selectors.append(selector)
            elif re.search(rf"<{re.escape(selector)}\b", html_text, re.IGNORECASE):
                selectors.append(selector)
        return _unique_preserve_order(selectors)[:6]

    # body_text
    for selector in ("p", "li"):
        if re.search(rf"<{selector}\b", html_text, re.IGNORECASE):
            selectors.append(selector)
    for class_name in ("body", "content", "bullets", "copy", "text", "lead"):
        if _html_has_class(html_text, class_name):
            existing = _css_selectors_containing_class(style_text, class_name)
            if existing:
                selectors.extend(existing)
            else:
                selectors.append(f".{class_name}")
    return _unique_preserve_order(selectors)[:5]


def _extract_class_tokens_from_attrs(attrs_text: str) -> list[str]:
    """Extract class tokens from an opening tag attribute string."""
    match = re.search(r'class\s*=\s*(["\'])(?P<value>.*?)\1', attrs_text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return [token for token in match.group("value").split() if token]


def _merge_inline_style_text(
    existing_style: str,
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"] = "merge",
) -> tuple[str, str]:
    """Merge or replace one inline style declaration string."""
    existing_props = _parse_css_declarations_text(existing_style)
    if mode == "replace":
        merged_props = dict(desired_props)
    else:
        merged_props = dict(existing_props)
        merged_props.update(desired_props)
    if existing_props == merged_props:
        return existing_style, "already_compliant"
    return _format_css_declarations(merged_props), "updated"


def _rewrite_opening_tag_with_style(
    opening_tag: str,
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"] = "merge",
) -> tuple[str, str]:
    """Rewrite or append the inline style attribute for one opening tag."""
    style_attr_re = re.compile(
        r'(?P<prefix>\sstyle\s*=\s*)(?P<quote>["\'])(?P<value>.*?)(?P=quote)',
        re.IGNORECASE | re.DOTALL,
    )
    match = style_attr_re.search(opening_tag)
    if match:
        updated_style, status = _merge_inline_style_text(
            match.group("value"),
            desired_props,
            mode=mode,
        )
        if status == "already_compliant":
            return opening_tag, status
        replacement = f'{match.group("prefix")}{match.group("quote")}{updated_style}{match.group("quote")}'
        return opening_tag[: match.start()] + replacement + opening_tag[match.end() :], status

    closing = "/>" if opening_tag.rstrip().endswith("/>") else ">"
    insert_at = len(opening_tag) - len(closing)
    patched = (
        opening_tag[:insert_at]
        + f' style="{_format_css_declarations(desired_props)}"'
        + opening_tag[insert_at:]
    )
    return patched, "added_inline_style"


def _semantic_opening_tag_predicates(
    target_kind: str,
) -> list[callable]:
    """Return ordered predicates for matching semantic target opening tags."""
    title_classes = {"title", "h1", "hero", "heading", "card-title", "panel-title", "title-wrap"}
    footer_classes = {"footer", "cap", "caption", "page", "src"}
    body_classes = {"body", "content", "bullets", "copy", "text", "lead"}

    def _tag_is(tag_name: str):
        return lambda tag, classes: tag.lower() == tag_name

    def _has_any(classes_needed: set[str]):
        return lambda tag, classes: bool(classes_needed & {item.lower() for item in classes})

    if target_kind == "slide_title":
        return [_tag_is("h1"), _has_any(title_classes), _tag_is("h2")]
    if target_kind == "footer":
        return [_has_any(footer_classes), _tag_is("footer")]
    if target_kind == "caption":
        return [_has_any({"caption", "figure-caption", "table-caption", "image-caption", "fig-caption", "note"}), _tag_is("figcaption")]
    if target_kind == "pill_tag":
        return [_has_any({"badge", "pill", "pill-tag", "tag", "chip", "label"})]
    if target_kind == "table_cell":
        return [_tag_is("td"), _tag_is("th"), _has_any({"table-cell"})]
    if target_kind == "legend_label":
        return [_has_any({"legend", "legend-label"})]
    if target_kind == "callout":
        return [_has_any({"callout", "annotation"})]
    if target_kind == "chart_axis_label":
        return [_has_any({"axis-label", "chart-axis-label"})]
    if target_kind == "chart_data_label":
        return [_has_any({"data-label", "chart-data-label"})]
    if target_kind == "any_text":
        return [_tag_is("h1"), _tag_is("h2"), _tag_is("p"), _tag_is("li"), _tag_is("span")]
    return [_has_any(body_classes), _tag_is("ul"), _tag_is("ol"), _tag_is("li"), _tag_is("p")]


def _find_semantic_opening_tag_matches(
    html_text: str,
    target_kind: str,
) -> list[dict[str, object]]:
    """Find opening tags that likely correspond to one semantic target."""
    tag_re = re.compile(r"<(?P<tag>[a-zA-Z][a-zA-Z0-9-]*)(?P<attrs>[^<>]*?)>", re.IGNORECASE | re.DOTALL)
    predicates = _semantic_opening_tag_predicates(target_kind)
    matches: list[dict[str, object]] = []
    occupied_spans: set[tuple[int, int]] = set()

    for predicate in predicates:
        for match in tag_re.finditer(html_text):
            tag_name = str(match.group("tag") or "")
            if tag_name.lower() in {"html", "head", "style", "script", "meta", "link", "img"}:
                continue
            attrs_text = match.group("attrs") or ""
            classes = _extract_class_tokens_from_attrs(attrs_text)
            if not predicate(tag_name, classes):
                continue
            span = (match.start(), match.end())
            if span in occupied_spans:
                continue
            occupied_spans.add(span)
            matches.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "tag": tag_name,
                    "classes": classes,
                    "opening_tag": match.group(0),
                }
            )
        if matches:
            break
    return matches


def _upsert_selector_rule(
    style_text: str,
    selector: str,
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"] = "merge",
) -> tuple[str, str]:
    """Insert or update one selector rule inside a <style> block."""
    selector = str(selector or "").strip()
    if not selector:
        return style_text, "skipped"

    rule_re = re.compile(
        rf"(^|}})\s*({re.escape(selector)})\s*\{{(?P<body>[^{{}}]*)\}}",
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = rule_re.search(style_text)
    if match:
        existing_props = _parse_css_declarations_text(match.group("body"))
        if mode == "replace":
            merged_props = dict(desired_props)
        else:
            merged_props = dict(existing_props)
            merged_props.update(desired_props)
        if existing_props == merged_props:
            return style_text, "already_compliant"

        replacement = (
            f"{match.group(1)}\n{selector} {{ {_format_css_declarations(merged_props)} }}"
        )
        updated = style_text[: match.start()] + replacement + style_text[match.end() :]
        return updated, "updated"

    suffix = "" if not style_text.rstrip() else "\n"
    appended = (
        style_text.rstrip()
        + f"{suffix}\n{selector} {{ {_format_css_declarations(desired_props)} }}\n"
    )
    return appended, "created_rule"


def _selector_presence_hint(html_text: str, selector: str) -> bool | None:
    """Best-effort selector presence check for simple tag/class/id selectors."""
    normalized = str(selector or "").strip()
    if not normalized:
        return None
    if re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", normalized):
        return re.search(rf"<{re.escape(normalized)}\b", html_text, re.IGNORECASE) is not None
    if normalized.startswith(".") and len(normalized) > 1:
        class_name = re.escape(normalized[1:])
        return (
            re.search(
                rf'class=["\'][^"\']*(?:^|\s){class_name}(?:\s|$)[^"\']*["\']',
                html_text,
                re.IGNORECASE,
            )
            is not None
        )
    if normalized.startswith("#") and len(normalized) > 1:
        elem_id = re.escape(normalized[1:])
        return re.search(rf'id=["\']{elem_id}["\']', html_text, re.IGNORECASE) is not None
    return None


def _apply_css_rule_to_html(
    html_text: str,
    selector: str,
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"] = "merge",
) -> tuple[str, str]:
    """Apply one CSS selector rule to the first <style> block, or create one."""
    style_re = re.compile(r"(<style\b[^>]*>)(?P<body>.*?)(</style>)", re.IGNORECASE | re.DOTALL)
    match = style_re.search(html_text)
    if match:
        updated_style, status = _upsert_selector_rule(
            match.group("body"),
            selector,
            desired_props,
            mode=mode,
        )
        if status == "already_compliant":
            return html_text, status
        updated_html = (
            html_text[: match.start("body")]
            + updated_style
            + html_text[match.end("body") :]
        )
        return updated_html, status

    new_block = f"<style>\n{selector} {{ {_format_css_declarations(desired_props)} }}\n</style>\n"
    head_close = re.search(r"</head\s*>", html_text, re.IGNORECASE)
    if head_close:
        updated_html = html_text[: head_close.start()] + new_block + html_text[head_close.start() :]
    else:
        updated_html = new_block + html_text
    return updated_html, "created_style_block"


def _patch_existing_semantic_inline_styles(
    html_text: str,
    target_kind: str,
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"] = "merge",
) -> tuple[str, list[dict[str, object]]]:
    """Patch existing inline styles for semantic targets to avoid CSS-vs-inline conflicts."""
    matches = _find_semantic_opening_tag_matches(html_text, target_kind)
    if not matches:
        return html_text, []

    updated_html = html_text
    patched_entries: list[dict[str, object]] = []
    for match in sorted(matches, key=lambda item: int(item["start"]), reverse=True):
        opening_tag = str(match["opening_tag"])
        if not re.search(r"\sstyle\s*=", opening_tag, re.IGNORECASE):
            continue
        patched_tag, status = _rewrite_opening_tag_with_style(
            opening_tag,
            desired_props,
            mode=mode,
        )
        if patched_tag != opening_tag:
            updated_html = (
                updated_html[: int(match["start"])]
                + patched_tag
                + updated_html[int(match["end"]) :]
            )
        patched_entries.append(
            {
                "tag": match["tag"],
                "classes": match["classes"],
                "status": status,
            }
        )
    return updated_html, list(reversed(patched_entries))


def _tag_classes(tag: Tag) -> list[str]:
    classes = tag.get("class", [])
    if isinstance(classes, str):
        return [token for token in classes.split() if token]
    return [str(token) for token in classes if str(token).strip()]


def _tag_has_any_class(tag: Tag, class_tokens: set[str]) -> bool:
    return bool(class_tokens & {token.lower() for token in _tag_classes(tag)})


def _focus_selector_matches_tag(tag: Tag, focus_selector: str) -> bool:
    selector = str(focus_selector or "").strip()
    if not selector:
        return False
    if _selector_targets_tag(tag, selector):
        return True
    selector_l = selector.lower()
    hint_l = _selector_hint_for_tag(tag).lower()
    return bool(selector_l and selector_l == hint_l)


def _focus_text_matches_tag(tag: Tag, focus_text: str) -> bool:
    normalized_focus = _normalize_focus_text(focus_text)
    if not normalized_focus:
        return False
    normalized_tag_text = _normalize_focus_text(tag.get_text(" ", strip=True))
    return bool(normalized_tag_text and normalized_focus in normalized_tag_text)


def _focus_kind_matches_tag(tag: Tag, focus_kind: str) -> bool:
    kind = _normalize_focus_kind(focus_kind)
    if not kind:
        return False
    if kind == "caption":
        return (
            tag.name.lower() == "figcaption"
            or str(tag.get("data-role", "") or "").strip().lower() == "caption"
            or _tag_has_any_class(tag, _CAPTION_TARGET_CLASSES)
        )
    if kind == "pill_tag":
        return _tag_has_any_class(tag, _PILL_TARGET_CLASSES)
    if kind == "table_cell":
        return tag.name.lower() in {"td", "th"} or _tag_has_any_class(tag, {"table-cell"})
    if kind == "legend_label":
        return _tag_has_any_class(tag, _LEGEND_TARGET_CLASSES)
    if kind == "callout":
        return _tag_has_any_class(tag, _CALLOUT_TARGET_CLASSES)
    if kind == "footnote":
        return _tag_has_any_class(tag, _FOOTNOTE_TARGET_CLASSES)
    if kind in {"inline_text", "any_text", "body_text"}:
        return _is_semantic_text_tag(tag)
    return False


def _tag_matches_focus(
    tag: Tag,
    *,
    focus_text: str = "",
    focus_selector: str = "",
    focus_kind: str = "",
) -> bool:
    normalized_text = _normalize_focus_text(focus_text)
    normalized_selector = str(focus_selector or "").strip()
    normalized_kind = _normalize_focus_kind(focus_kind)
    if normalized_text and not _focus_text_matches_tag(tag, normalized_text):
        return False
    if normalized_selector and not _focus_selector_matches_tag(tag, normalized_selector):
        return False
    if normalized_kind and not _focus_kind_matches_tag(tag, normalized_kind):
        return False
    return bool(normalized_text or normalized_selector or normalized_kind)


def _semantic_target_tags(body: Tag, target_kind: str) -> list[Tag]:
    if target_kind == "slide_title":
        h1_tags = body.find_all("h1")
        if h1_tags:
            return h1_tags
        class_tags = [tag for tag in body.find_all(True) if _tag_has_any_class(tag, _TITLE_TARGET_CLASSES)]
        if class_tags:
            return class_tags
        h2_tags = body.find_all("h2")
        if h2_tags:
            return h2_tags
        return []

    if target_kind == "footer":
        class_tags = [tag for tag in body.find_all(True) if _tag_has_any_class(tag, _FOOTER_TARGET_CLASSES)]
        if class_tags:
            return class_tags
        return body.find_all("footer")

    if target_kind in {"caption", "figure_caption"}:
        class_tags = [
            tag for tag in body.find_all(True)
            if (
                _tag_has_any_class(tag, _CAPTION_TARGET_CLASSES)
                or str(tag.get("data-role", "") or "").strip().lower() == "caption"
            )
        ]
        if class_tags:
            return class_tags
        return body.find_all("figcaption")

    if target_kind == "pill_tag":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _PILL_TARGET_CLASSES)
        ]

    if target_kind == "table_cell":
        return body.find_all(["td", "th"])

    if target_kind == "legend_label":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _LEGEND_TARGET_CLASSES)
        ]

    if target_kind == "callout":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _CALLOUT_TARGET_CLASSES)
        ]

    if target_kind == "footnote":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _FOOTNOTE_TARGET_CLASSES)
        ]

    if target_kind == "chart_axis_label":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _CHART_AXIS_LABEL_CLASSES)
        ]

    if target_kind == "chart_data_label":
        return [
            tag for tag in body.find_all(True)
            if _tag_has_any_class(tag, _CHART_DATA_LABEL_CLASSES)
        ]

    if target_kind in {"any_text", "inline_text"}:
        return [tag for tag in body.find_all(True) if _is_semantic_text_tag(tag)]

    class_tags = [
        tag
        for tag in body.find_all(True)
        if _tag_has_any_class(tag, _BODY_TARGET_CLASSES) and _is_leaf_text_block_tag(tag)
    ]
    if class_tags:
        return class_tags
    candidates: list[Tag] = []
    for tag_name in ("ul", "ol", "li", "p", "figcaption", "span", "blockquote"):
        candidates.extend(body.find_all(tag_name))
    return _unique_tags(candidates)


def _is_leaf_text_block_tag(tag: Tag) -> bool:
    tag_name = tag.name.lower()
    if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol", "footer", "span", "figcaption", "blockquote"}:
        return True
    inline_children = {"a", "b", "br", "code", "em", "i", "mark", "small", "span", "strong", "sub", "sup", "u"}
    child_tags = [child for child in tag.children if isinstance(child, Tag)]
    return bool(_tag_text_length(tag)) and bool(child_tags) and all(
        child.name.lower() in inline_children for child in child_tags
    )


def _is_semantic_text_tag(tag: Tag) -> bool:
    tag_name = tag.name.lower()
    if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "ul", "ol", "footer", "span", "figcaption"}:
        return True
    if _tag_has_any_class(tag, _TITLE_TARGET_CLASSES | _FOOTER_TARGET_CLASSES):
        return True
    return _tag_has_any_class(tag, _BODY_TARGET_CLASSES) and _is_leaf_text_block_tag(tag)


def _tag_text_length(tag: Tag) -> int:
    return len(re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip())


def _tag_depth(tag: Tag) -> int:
    return sum(1 for parent in tag.parents if isinstance(parent, Tag))


def _build_dom_path(tag: Tag) -> str:
    steps: list[str] = []
    current: Tag | None = tag
    while isinstance(current, Tag) and current.name:
        parent = current.parent
        if parent is None:
            break
        siblings = [child for child in parent.children if isinstance(child, Tag)]
        try:
            index = siblings.index(current)
        except ValueError:
            break
        steps.append(f"{current.name}[{index}]")
        if not isinstance(parent, Tag):
            break
        current = parent
    return "/".join(reversed(steps))


def _resolve_dom_path(root: BeautifulSoup, dom_path: str) -> Tag | None:
    current: Tag | BeautifulSoup = root
    for raw_step in str(dom_path or "").split("/"):
        match = re.fullmatch(r"([a-zA-Z][a-zA-Z0-9-]*)\[(\d+)\]", raw_step.strip())
        if not match:
            return None
        tag_name = match.group(1).lower()
        index = int(match.group(2))
        children = [child for child in current.children if isinstance(child, Tag)]
        if index < 0 or index >= len(children):
            return None
        child = children[index]
        if child.name.lower() != tag_name:
            return None
        current = child
    return current if isinstance(current, Tag) else None


def _text_preview(text: str, limit: int = 120) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= limit:
        return normalized
    if limit <= 1:
        return normalized[:limit]
    return normalized[: limit - 1].rstrip() + "…"


def _tag_text_preview(tag: Tag) -> str:
    if tag.name.lower() == "img":
        alt_text = _text_preview(str(tag.get("alt", "") or ""))
        if alt_text:
            return alt_text
        src_name = Path(str(tag.get("src", "") or "")).name
        if src_name:
            return f"[image] {src_name}"
        return "[image]"
    text = _text_preview(tag.get_text(" ", strip=True))
    if text:
        return text
    if tag.find("img"):
        return "[image block]"
    return f"<{tag.name}> block"


def _selector_hint_for_tag(tag: Tag) -> str:
    tag_name = tag.name.lower()
    elem_id = str(tag.get("id", "") or "").strip()
    if elem_id:
        return f"{tag_name}#{elem_id}"
    classes = _tag_classes(tag)
    if classes:
        normalized = ".".join(
            re.sub(r"[^a-zA-Z0-9_-]", "-", cls).strip("-")
            for cls in classes[:2]
            if re.sub(r"[^a-zA-Z0-9_-]", "-", cls).strip("-")
        )
        if normalized:
            return f"{tag_name}.{normalized}"
    return tag_name


def _selector_targets_tag(tag: Tag, selector: str) -> bool:
    """Best-effort selector matching for simple tag/class/id diagnostics."""
    normalized = str(selector or "").strip().lower()
    if not normalized:
        return False

    tag_name = tag.name.lower()
    tag_id = str(tag.get("id", "") or "").strip().lower()
    tag_classes = {token.lower() for token in _tag_classes(tag)}

    for raw_part in normalized.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part == tag_name:
            return True
        if tag_id and part == f"#{tag_id}":
            return True
        if any(part == f".{cls}" or part.endswith(f".{cls}") for cls in tag_classes):
            return True
        if not tag_classes and not tag_id:
            if re.search(rf"(^|[^a-zA-Z0-9_-]){re.escape(tag_name)}([.#:\s]|$)", part):
                return True
    return False


def _collect_tag_style_props(css_rules: dict[str, dict[str, str]], tag: Tag) -> dict[str, str]:
    """Approximate the effective style for one tag using matching CSS rules plus inline style."""
    props: dict[str, str] = {}
    for selector, rule_props in css_rules.items():
        if _selector_targets_tag(tag, selector):
            props.update(rule_props)
    inline_style = _parse_css_declarations_text(tag.get("style", ""))
    if inline_style:
        props.update(inline_style)
    return props


def _normalize_semantic_edit_scope(value: Any) -> str:
    """Normalize planner/apply semantic scope names and common natural aliases."""
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "font_color": "text_color",
        "文字颜色": "text_color",
        "字体颜色": "text_color",
        "文本颜色": "text_color",
        "标题颜色": "text_color",
        "正文颜色": "text_color",
        "font": "typography",
        "font_size": "typography",
        "type": "typography",
        "background": "background_style",
        "bg": "background_style",
        "surface": "background_style",
        "背景": "background_style",
        "底色": "background_style",
        "layout": "layout_spacing",
        "spacing": "layout_spacing",
        "position": "layout_spacing",
        "间距": "layout_spacing",
        "布局": "layout_spacing",
        "border": "border_shadow",
        "shadow": "border_shadow",
        "边框": "border_shadow",
        "阴影": "border_shadow",
        "image": "image_asset",
        "figure": "image_asset",
        "asset": "image_asset",
        "图片": "image_asset",
        "图像": "image_asset",
        "visible": "visibility",
        "visibility": "visibility",
        "opacity": "visibility",
        "显示": "visibility",
        "隐藏": "visibility",
    }
    if normalized in aliases:
        return aliases[normalized]
    if raw in aliases:
        return aliases[raw]
    return normalized if normalized in _SEMANTIC_EDIT_SCOPES else normalized


def _semantic_property_group_for_prop(prop_name: str) -> str:
    prop = str(prop_name or "").strip().lower()
    if not prop:
        return "unknown"
    if prop in {"color", "-webkit-text-fill-color", "text-decoration-color", "text-emphasis-color", "caret-color"}:
        return "text_color"
    if prop.startswith("font") or prop in {
        "line-height",
        "letter-spacing",
        "word-spacing",
        "text-align",
        "text-transform",
        "text-indent",
        "white-space",
        "text-wrap",
        "text-overflow",
        "list-style",
        "list-style-type",
        "list-style-position",
    }:
        return "typography"
    if prop.startswith("background") or prop in {
        "background",
        "background-color",
        "background-image",
        "background-size",
        "background-position",
        "background-repeat",
        "background-clip",
        "background-origin",
        "background-blend-mode",
    }:
        return "background_style"
    if (
        prop.startswith("border")
        or prop.startswith("outline")
        or prop in {"box-shadow", "text-shadow", "filter", "backdrop-filter"}
    ):
        return "border_shadow"
    if prop in {"object-fit", "object-position", "image-rendering", "aspect-ratio"}:
        return "image_asset"
    if prop in {"opacity", "visibility", "pointer-events"}:
        return "visibility"
    if prop in {
        "display",
        "position",
        "top",
        "left",
        "right",
        "bottom",
        "inset",
        "inset-inline",
        "inset-block",
        "width",
        "height",
        "min-width",
        "min-height",
        "max-width",
        "max-height",
        "padding",
        "padding-top",
        "padding-right",
        "padding-bottom",
        "padding-left",
        "margin",
        "margin-top",
        "margin-right",
        "margin-bottom",
        "margin-left",
        "gap",
        "row-gap",
        "column-gap",
        "grid",
        "grid-template",
        "grid-template-columns",
        "grid-template-rows",
        "grid-column",
        "grid-row",
        "flex",
        "flex-direction",
        "flex-wrap",
        "flex-basis",
        "flex-grow",
        "flex-shrink",
        "align-items",
        "align-content",
        "align-self",
        "justify-content",
        "justify-items",
        "justify-self",
        "place-items",
        "place-content",
        "overflow",
        "overflow-x",
        "overflow-y",
        "box-sizing",
        "transform",
        "transform-origin",
        "z-index",
        "float",
        "clear",
    }:
        return "layout_spacing"
    return "unknown"


def _semantic_property_groups_for_props(props: dict[str, str]) -> list[str]:
    return _unique_preserve_order(
        [
            group
            for prop in props.keys()
            if (group := _semantic_property_group_for_prop(prop)) != "unknown"
        ]
    )


def _extend_allowed_groups_for_compatible_combo(
    allowed: set[str],
    changed_groups: list[str] | set[str],
    *,
    repair_intents: list[str] | set[str] | None = None,
) -> set[str]:
    """Allow tightly-coupled visual tweaks without disabling scope audit."""
    if not allowed:
        return allowed
    changed = {str(group or "").strip() for group in changed_groups if str(group or "").strip()}
    if not changed:
        return allowed
    expanded = set(allowed)
    for combo in _SEMANTIC_COMPATIBLE_GROUP_COMBOS:
        if changed <= combo and (expanded & combo):
            expanded.update(combo)
    active_repairs = {
        str(intent or "").strip()
        for intent in (repair_intents or [])
        if str(intent or "").strip()
    }
    if active_repairs:
        for combo, supported_repairs in _SEMANTIC_REPAIR_COMPATIBLE_GROUP_COMBOS:
            if changed <= combo and (expanded & combo) and (active_repairs & supported_repairs):
                expanded.update(combo)
    return expanded


def _semantic_attr_group(attr_name: str) -> str:
    attr = str(attr_name or "").strip().lower()
    if attr == "style":
        return "unknown"
    if attr in {"src", "alt"}:
        return "image_asset"
    if attr in {"width", "height"}:
        return "layout_spacing"
    if attr == "class":
        return "unknown"
    return "unknown"


def _semantic_style_subset(props: dict[str, str], *, limit: int = 28) -> dict[str, str]:
    subset: dict[str, str] = {}
    for key, value in props.items():
        if not str(value or "").strip():
            continue
        if _semantic_property_group_for_prop(key) != "unknown" or key in _LAYOUT_CONTEXT_PROP_KEYS:
            subset[str(key)] = str(value)
        if len(subset) >= limit:
            break
    return subset


def _semantic_roles_for_target(
    tag: Tag,
    kind: str,
    effective_style: dict[str, str] | None = None,
) -> list[str]:
    roles: list[str] = []
    kind = str(kind or "").strip()
    if kind == "slide_canvas":
        roles.extend(["canvas", "background_surface", "layout_root"])
    elif kind == "slide_title":
        roles.extend(["text", "title"])
    elif kind in {"text_block", "body_text"}:
        roles.extend(["text", "body"])
    elif kind == "footer":
        roles.extend(["text", "footer"])
    elif kind in {"caption", "figure_caption"}:
        roles.extend(["text", "caption"])
    elif kind == "pill_tag":
        roles.extend(["text", "pill_tag"])
    elif kind == "table_cell":
        roles.extend(["text", "table_cell"])
    elif kind == "legend_label":
        roles.extend(["text", "legend_label"])
    elif kind == "callout":
        roles.extend(["text", "callout", "surface"])
    elif kind == "footnote":
        roles.extend(["text", "footnote"])
    elif kind in {"inline_text", "any_text"}:
        roles.extend(["text", "body"])
    elif kind in {"image", "figure"}:
        roles.extend(["media", kind])
    elif kind == "layout_container":
        roles.extend(["layout_container", "surface"])
    else:
        roles.append(kind or "body_block")

    props = effective_style or {}
    if any(_semantic_property_group_for_prop(prop) == "background_style" for prop in props):
        roles.append("surface")
    if _is_semantic_text_tag(tag) and "text" not in roles:
        roles.append("text")
    return _unique_preserve_order(roles)


def _editable_property_groups_for_roles(roles: list[str], kind: str = "") -> list[str]:
    role_set = {str(role or "").strip().lower() for role in roles}
    groups: list[str] = []
    if (
        "text" in role_set
        or "title" in role_set
        or "body" in role_set
        or "footer" in role_set
        or "caption" in role_set
        or "pill_tag" in role_set
        or "table_cell" in role_set
        or "legend_label" in role_set
        or "callout" in role_set
        or "footnote" in role_set
    ):
        groups.extend(["text_color", "typography", "visibility"])
    if "canvas" in role_set or "background_surface" in role_set or "surface" in role_set:
        groups.extend(["background_style", "border_shadow", "visibility"])
    if "layout_container" in role_set or "layout_root" in role_set or str(kind) in {"layout_container", "body_block", "slide_canvas"}:
        groups.extend(["layout_spacing"])
    if "media" in role_set or "image" in role_set or "figure" in role_set or str(kind) in {"image", "figure"}:
        groups.extend(["image_asset", "layout_spacing", "visibility"])
    if not groups:
        groups.extend(["layout_spacing", "visibility"])
    return _unique_preserve_order(groups)


def _semantic_roles_for_rule(
    selector: str,
    declarations: dict[str, str],
    used_targets: list[dict[str, Any]],
) -> list[str]:
    roles: list[str] = []
    for target in used_targets:
        roles.extend([str(role) for role in target.get("semantic_roles", []) or []])
    selector_l = str(selector or "").lower()
    if _selector_mentions_body_or_slide(selector_l):
        roles.extend(["canvas", "background_surface", "layout_root"])
    if re.search(r"(^|[^a-z0-9_-])h[1-6]([^a-z0-9_-]|$)", selector_l):
        roles.extend(["text", "title"])
    if any(token in selector_l for token in (".title", ".heading", ".hero")):
        roles.extend(["text", "title"])
    if any(token in selector_l for token in (".body", ".content", ".copy", ".text", " p", " li")):
        roles.extend(["text", "body"])
    if any(_semantic_property_group_for_prop(prop) == "background_style" for prop in declarations):
        roles.append("surface")
    return _unique_preserve_order(roles)


def _groups_for_patch_op(op: dict[str, Any]) -> list[str]:
    op_name = str(op.get("op", "") or "").strip()
    if op_name in {"merge_style", "merge_css_rule", "replace_css_rule", "wrap_text_span"}:
        return _semantic_property_groups_for_props(
            _parse_css_declarations_text(str(op.get("declarations", "") or ""))
        )
    if op.get("declarations") and op_name.startswith("batch_"):
        return _semantic_property_groups_for_props(
            _parse_css_declarations_text(str(op.get("declarations", "") or ""))
        )
    if op_name == "set_attr":
        attr_name = str(op.get("attr_name", "") or "").strip().lower()
        if attr_name == "style":
            return _semantic_property_groups_for_props(
                _parse_css_declarations_text(str(op.get("value", "") or ""))
            )
        group = _semantic_attr_group(attr_name)
        return [] if group == "unknown" else [group]
    if op_name == "remove_attr":
        group = _semantic_attr_group(str(op.get("attr_name", "") or ""))
        return [] if group == "unknown" else [group]
    if op_name in {"replace_text", "replace_html", "insert_html", "remove_node"}:
        return ["content_text"]
    return []


def _semantic_context_for_op(op: dict[str, Any]) -> dict[str, Any]:
    edit_scope = _normalize_semantic_edit_scope(op.get("edit_scope"))
    property_group = _normalize_semantic_edit_scope(op.get("property_group"))
    intent_id = str(op.get("intent_id", "") or "").strip()
    repair_intent = str(op.get("repair_intent", "") or "").strip()
    payload: dict[str, Any] = {}
    if edit_scope:
        payload["edit_scope"] = edit_scope
    if property_group:
        payload["property_group"] = property_group
    if intent_id:
        payload["intent_id"] = intent_id
    if repair_intent:
        payload["repair_intent"] = repair_intent
    return payload


def _validate_semantic_scope_for_ops(
    patch_ops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings_out: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for index, op in enumerate(patch_ops, start=1):
        groups = [group for group in _groups_for_patch_op(op) if group not in {"content_text", "unknown"}]
        if not groups:
            continue
        scope = _normalize_semantic_edit_scope(op.get("edit_scope"))
        explicit_group = _normalize_semantic_edit_scope(op.get("property_group"))
        allowed: set[str] = set()
        if scope in _SEMANTIC_SCOPE_ALLOWED_GROUPS:
            allowed.update(_SEMANTIC_SCOPE_ALLOWED_GROUPS[scope])
        if explicit_group in _SEMANTIC_SCOPE_ALLOWED_GROUPS:
            allowed.update(_SEMANTIC_SCOPE_ALLOWED_GROUPS[explicit_group])
        allowed = _extend_allowed_groups_for_compatible_combo(
            allowed,
            groups,
            repair_intents=[op.get("repair_intent")],
        )

        if allowed:
            unexpected = [group for group in groups if group not in allowed]
            if unexpected:
                entry = {
                    "op_index": index,
                    "op": str(op.get("op", "") or ""),
                    "edit_scope": scope,
                    "property_group": explicit_group,
                    "changed_property_groups": groups,
                    "allowed_property_groups": sorted(allowed),
                    "unexpected_property_groups": unexpected,
                    "message": (
                        "Scoped patch changes property groups outside the declared semantic edit scope. "
                        "Use a matching edit_scope/property_group or set allow_scope_override=true for this op."
                    ),
                }
                if bool(op.get("allow_scope_override", False)):
                    warnings_out.append(entry)
                else:
                    conflicts.append(entry)
            continue

        if len(set(groups)) > 1:
            warnings_out.append(
                {
                    "op_index": index,
                    "op": str(op.get("op", "") or ""),
                    "changed_property_groups": groups,
                    "message": (
                        "Unscoped patch changes multiple semantic property groups; "
                        "consider using plan_slide_patch and explicit edit_scope for local edits."
                    ),
                }
            )
    return warnings_out, conflicts


def _is_figure_like_tag(tag: Tag) -> bool:
    if tag.name.lower() == "figure":
        return True
    return _tag_has_any_class(tag, {"figure", "fig", "image-card", "visual", "illustration"})


def _is_layout_container_tag(tag: Tag, props: dict[str, str]) -> bool:
    if tag.name.lower() in {"html", "head", "body", "style", "script", "img"}:
        return False
    if _is_semantic_text_tag(tag):
        return False
    display = str(props.get("display", "") or "").strip().lower()
    position = str(props.get("position", "") or "").strip().lower()
    if display in {"flex", "inline-flex", "grid", "inline-grid"}:
        return True
    if position in {"absolute", "relative", "fixed", "sticky"}:
        return True
    if tag.name.lower() in {"div", "section", "article", "main", "aside", "header", "footer", "figure"}:
        return any(
            key in props
            for key in (
                "width",
                "height",
                "max-width",
                "max-height",
                "overflow",
                "overflow-x",
                "overflow-y",
                "padding",
                "padding-top",
                "padding-right",
                "padding-bottom",
                "padding-left",
                "margin",
                "margin-top",
                "margin-right",
                "margin-bottom",
                "margin-left",
            )
        )
    return False


def _layout_container_tags(
    body: Tag,
    css_rules: dict[str, dict[str, str]],
) -> list[Tag]:
    tags: list[Tag] = []
    for tag in body.find_all(True):
        props = _collect_tag_style_props(css_rules, tag)
        if _is_layout_container_tag(tag, props):
            tags.append(tag)
    return tags


def _preferred_overflow_repair_tags(
    body: Tag,
    css_rules: dict[str, dict[str, str]],
) -> list[Tag]:
    slide_roots = _top_level_slide_roots(body)
    ranked: list[tuple[tuple[int, int, int, int, int], Tag]] = []
    for tag in _layout_container_tags(body, css_rules):
        props = _collect_tag_style_props(css_rules, tag)
        text_len = _tag_text_length(tag)
        if text_len <= 0:
            continue
        class_tokens = {token.lower() for token in _tag_classes(tag)}
        selector_hint = _selector_hint_for_tag(tag).lower()
        has_text_descendants = any(
            _is_semantic_text_tag(descendant) and _tag_text_length(descendant) > 0
            for descendant in tag.find_all(True)
        )
        has_container_hint = bool(class_tokens & _LAYOUT_REPAIR_HINT_CLASSES) or any(
            f".{token}" in selector_hint for token in _LAYOUT_REPAIR_HINT_CLASSES
        )
        has_size_constraints = any(
            str(props.get(key, "") or "").strip()
            for key in ("display", "height", "max-height", "overflow", "overflow-y", "padding", "padding-bottom")
        )
        is_slide_root = tag in slide_roots or _tag_has_any_class(tag, {"slide"})
        is_decorative = bool(class_tokens & _DECORATIVE_LAYOUT_CLASSES) and text_len < 12
        priority = (
            1 if is_decorative else 0,
            0 if has_text_descendants or text_len >= 24 else 1,
            0 if has_container_hint else 1,
            0 if has_size_constraints else 1,
            1 if is_slide_root else 0,
        )
        ranked.append((priority, tag))
    return [tag for _, tag in sorted(ranked, key=lambda item: (item[0], -_tag_depth(item[1])))]


def _target_selection_rank(kind: str) -> int:
    return {
        "slide_canvas": 0,
        "slide_title": 10,
        "caption": 18,
        "figure_caption": 18,
        "pill_tag": 19,
        "text_block": 20,
        "inline_text": 22,
        "any_text": 22,
        "table_cell": 24,
        "legend_label": 24,
        "callout": 24,
        "footnote": 28,
        "footer": 30,
        "layout_container": 40,
        "figure": 50,
        "image": 60,
        "body_block": 70,
        "body_text": 20,
    }.get(kind, 90)


def _diagnostic_priority(code: str) -> int:
    return {
        "canvas_dimension_mismatch": 0,
        "body_padding_without_border_box": 1,
        "suspicious_slide_geometry": 2,
        "off_canvas_absolute": 3,
        "image_overflow": 4,
        "clipped_outside_canvas": 5,
        "bottom_safe_zone_violation": 6,
        "text_overflow": 7,
        "nested_slide_root": 8,
    }.get(str(code or ""), 99)


def _target_allowed_ops(kind: str) -> list[str]:
    if kind == "image":
        return ["replace_html", "merge_style", "remove_node", "insert_html", "set_attr", "remove_attr"]
    if kind == "slide_canvas":
        return ["merge_style", "set_attr", "remove_attr"]
    return [
        "replace_text",
        "replace_html",
        "merge_style",
        "wrap_text_span",
        "remove_node",
        "insert_html",
        "set_attr",
        "remove_attr",
    ]


def _build_snapshot_target(
    tag: Tag,
    kind: str,
    target_id: str,
    *,
    allowed_ops: list[str] | None = None,
    box_context: dict[str, str] | None = None,
    governing_rule_ids: list[str] | None = None,
    effective_style: dict[str, str] | None = None,
) -> dict[str, Any]:
    dom_path = _build_dom_path(tag)
    normalized_box_context = dict(box_context or {})
    effective_style_props = dict(effective_style or {})
    semantic_roles = _semantic_roles_for_target(tag, kind, effective_style_props)
    return {
        "target_id": target_id,
        "kind": kind,
        "selector_hint": _selector_hint_for_tag(tag),
        "dom_path": dom_path,
        "text_preview": _tag_text_preview(tag),
        "allowed_ops": allowed_ops or _target_allowed_ops(kind),
        "target_hash": _compute_target_hash(tag),
        "layout_role": (
            "canvas"
            if kind == "slide_canvas"
            else "layout_container"
            if kind == "layout_container"
            else "title"
            if kind == "slide_title"
            else "text_block"
            if kind in {"text_block", "body_text"}
            else kind
        ),
        "box_context": normalized_box_context,
        "governing_rule_ids": list(governing_rule_ids or []),
        "semantic_roles": semantic_roles,
        "editable_property_groups": _editable_property_groups_for_roles(semantic_roles, kind),
        "effective_style_subset": _semantic_style_subset(effective_style_props),
    }


def _build_slide_summary(
    slide_path: Path,
    targets: list[dict[str, Any]],
    has_more_targets: bool,
    *,
    rules: list[dict[str, Any]] | None = None,
    risk_flags: list[str] | None = None,
) -> str:
    counts = {
        "slide_title": 0,
        "caption": 0,
        "pill_tag": 0,
        "text_block": 0,
        "footer": 0,
        "body_block": 0,
        "image": 0,
        "layout_container": 0,
        "figure": 0,
    }
    previews: dict[str, list[str]] = {key: [] for key in counts}
    for target in targets:
        kind = str(target.get("kind", "body_block"))
        if kind in counts:
            counts[kind] += 1
            preview = str(target.get("text_preview", "") or "")
            if preview and len(previews[kind]) < 2:
                previews[kind].append(preview)

    parts = [
        f"titles={counts['slide_title']}",
        f"captions={counts['caption']}",
        f"text_blocks={counts['text_block']}",
        f"footer={counts['footer']}",
        f"body_blocks={counts['body_block']}",
        f"images={counts['image']}",
        f"layout_containers={counts['layout_container']}",
        f"figures={counts['figure']}",
    ]
    highlight = []
    if previews["slide_title"]:
        highlight.append(f"title: {previews['slide_title'][0]}")
    if previews["caption"]:
        highlight.append(f"caption: {previews['caption'][0]}")
    if previews["text_block"]:
        highlight.append(f"body: {previews['text_block'][0]}")
    if previews["footer"]:
        highlight.append(f"footer: {previews['footer'][0]}")
    if previews["image"]:
        highlight.append(f"image: {previews['image'][0]}")
    summary = f"{slide_path.name}: detected {', '.join(parts)}."
    if highlight:
        summary += " Key previews: " + "; ".join(highlight) + "."
    if rules:
        summary += f" Rules={len(rules)}."
    if risk_flags:
        summary += " Risk flags: " + ", ".join(str(flag) for flag in risk_flags[:4]) + "."
    if has_more_targets:
        summary += " More targets exist; refine with the returned target list only."
    return summary


def _append_unique_metadata_value(payload: dict[str, Any], key: str, value: str) -> None:
    normalized = str(value or "").strip()
    if not normalized:
        return
    bucket = payload.setdefault(key, [])
    if normalized not in bucket:
        bucket.append(normalized)


def _target_supports_layout_repair(target: dict[str, Any]) -> bool:
    roles = {str(role or "").strip().lower() for role in target.get("semantic_roles", []) or []}
    kind = str(target.get("kind", "") or "")
    return bool(roles & {"layout_container", "layout_root", "canvas"}) or kind in {
        "layout_container",
        "body_block",
        "slide_canvas",
    }


def _diagnostic_text_matches_target(diagnostic: dict[str, Any], target: dict[str, Any]) -> bool:
    diagnostic_text = _normalized_preview_match_text(diagnostic.get("text_preview", ""))
    target_text = _normalized_preview_match_text(target.get("text_preview", ""))
    if not diagnostic_text or not target_text:
        return False
    return diagnostic_text in target_text or target_text in diagnostic_text


def _resolve_diagnostic_target_matches(
    diagnostic: dict[str, Any],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    wanted_paths = _unique_preserve_order(
        [
            str(diagnostic.get("target_dom_path", "") or "").strip(),
            *[
                str(dom_path or "").strip()
                for dom_path in diagnostic.get("repair_dom_paths", []) or []
                if str(dom_path or "").strip()
            ],
        ]
    )
    selector_hint = str(diagnostic.get("target_selector_hint", "") or "").strip().lower()
    matches: list[tuple[int, dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for target in targets:
        target_id = str(target.get("target_id", "") or "")
        if not target_id or target_id in seen_ids:
            continue
        score = 0
        target_path = str(target.get("dom_path", "") or "").strip()
        if target_path and target_path in wanted_paths:
            score += 120
        if selector_hint and selector_hint == str(target.get("selector_hint", "") or "").strip().lower():
            score += 35
        if _diagnostic_text_matches_target(diagnostic, target):
            score += 75
        if score <= 0:
            continue
        matches.append((score, target))
        seen_ids.add(target_id)

    sorted_matches = [
        target
        for _, target in sorted(
            matches,
            key=lambda item: (
                -item[0],
                _target_selection_rank(str(item[1].get("kind", "") or "")),
                str(item[1].get("target_id", "") or ""),
            ),
        )
    ]
    if not sorted_matches:
        return []

    expanded: list[dict[str, Any]] = []
    for target in sorted_matches:
        target_id = str(target.get("target_id", "") or "")
        if target_id and target_id not in {str(item.get("target_id", "") or "") for item in expanded}:
            expanded.append(target)
        if not _diagnostic_text_matches_target(diagnostic, target):
            continue
        for child_id in target.get("child_target_ids", []) or []:
            child = next(
                (
                    item
                    for item in targets
                    if str(item.get("target_id", "") or "") == str(child_id or "")
                    and _diagnostic_text_matches_target(diagnostic, item)
                ),
                None,
            )
            if not child:
                continue
            child_target_id = str(child.get("target_id", "") or "")
            if child_target_id and child_target_id not in {str(item.get("target_id", "") or "") for item in expanded}:
                expanded.append(child)
    return expanded[:4]


def _resolve_repair_ancestor_target_ids(
    offender_targets: list[dict[str, Any]],
    target_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    ancestor_ids: list[str] = []
    for offender in offender_targets:
        parent_id = str(offender.get("parent_target_id", "") or "")
        while parent_id:
            parent = target_lookup.get(parent_id)
            if not parent:
                break
            if _target_supports_layout_repair(parent):
                if parent_id not in ancestor_ids:
                    ancestor_ids.append(parent_id)
                if len(ancestor_ids) >= 2:
                    return ancestor_ids
            parent_id = str(parent.get("parent_target_id", "") or "")
    return ancestor_ids


def _resolve_governing_rule_ids(
    offender_targets: list[dict[str, Any]],
    ancestor_target_ids: list[str],
    target_lookup: dict[str, dict[str, Any]],
) -> list[str]:
    rule_ids: list[str] = []
    for target in offender_targets:
        for rule_id in target.get("governing_rule_ids", []) or []:
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(str(rule_id))
    for ancestor_id in ancestor_target_ids:
        ancestor = target_lookup.get(str(ancestor_id or ""))
        if not ancestor:
            continue
        for rule_id in ancestor.get("governing_rule_ids", []) or []:
            if rule_id and rule_id not in rule_ids:
                rule_ids.append(str(rule_id))
    return rule_ids[:4]


def _compact_offending_target(target: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": target.get("target_id", ""),
        "kind": target.get("kind", ""),
        "selector_hint": target.get("selector_hint", ""),
        "text_preview": target.get("text_preview", ""),
        "allowed_ops": target.get("allowed_ops", []),
        "semantic_roles": target.get("semantic_roles", []),
        "editable_property_groups": target.get("editable_property_groups", []),
        "diagnostic_codes": target.get("diagnostic_codes", []),
        "repair_intents": target.get("repair_intents", []),
        "repair_roles": target.get("repair_roles", []),
        "governing_rule_ids": target.get("governing_rule_ids", []),
        "parent_target_id": target.get("parent_target_id", ""),
        "child_target_ids": target.get("child_target_ids", []),
    }


def _build_repair_surface(
    targets: list[dict[str, Any]],
    rules: list[dict[str, Any]],
    validation_diagnostics: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if not validation_diagnostics:
        return [], [], {"diagnostics": []}, validation_diagnostics

    target_lookup = {
        str(target.get("target_id", "") or ""): target
        for target in targets
        if str(target.get("target_id", "") or "").strip()
    }
    rule_lookup = {
        str(rule.get("rule_id", "") or ""): rule
        for rule in rules
        if str(rule.get("rule_id", "") or "").strip()
    }
    repair_candidates: list[dict[str, Any]] = []
    offending_targets: list[dict[str, Any]] = []
    enriched_diagnostics: list[dict[str, Any]] = []
    seen_candidate_keys: set[tuple[str, str, str]] = set()

    for diagnostic in sorted(
        validation_diagnostics,
        key=lambda item: (
            _diagnostic_priority(str(item.get("code", "") or "")),
            _diagnostic_source_rank(str(item.get("source", "") or "")),
            _diagnostic_severity_rank(str(item.get("severity", "") or "")),
        ),
    ):
        enriched = dict(diagnostic)
        repair_intent = _repair_intent_for_diagnostic_code(str(enriched.get("code", "") or ""))
        matched_targets = _resolve_diagnostic_target_matches(enriched, targets)
        ancestor_ids = _resolve_repair_ancestor_target_ids(matched_targets, target_lookup)
        governing_rule_ids = _resolve_governing_rule_ids(matched_targets, ancestor_ids, target_lookup)

        enriched["repair_intent"] = repair_intent
        enriched["offender_target_ids"] = [str(target.get("target_id", "") or "") for target in matched_targets]
        enriched["ancestor_target_ids"] = ancestor_ids
        enriched["governing_rule_ids"] = governing_rule_ids
        enriched["recommended_patch_strategy"] = _RECOMMENDED_PATCH_STRATEGIES.get(repair_intent, [])
        if not enriched.get("target_id") and matched_targets:
            enriched["target_id"] = str(matched_targets[0].get("target_id", "") or "")
        enriched_diagnostics.append(enriched)

        for target in matched_targets:
            _append_unique_metadata_value(target, "repair_roles", "offender")
            _append_unique_metadata_value(target, "diagnostic_codes", str(enriched.get("code", "") or ""))
            _append_unique_metadata_value(target, "repair_intents", repair_intent)
            compact = _compact_offending_target(target)
            if compact["target_id"] and compact["target_id"] not in {item.get("target_id", "") for item in offending_targets}:
                offending_targets.append(compact)
            candidate_key = ("target", compact["target_id"], str(enriched.get("code", "") or ""))
            if candidate_key not in seen_candidate_keys:
                seen_candidate_keys.add(candidate_key)
                repair_candidates.append(
                    {
                        "target_id": compact["target_id"],
                        "kind": compact["kind"],
                        "selector_hint": compact["selector_hint"],
                        "allowed_ops": compact.get("allowed_ops", []),
                        "repair_role": "offender",
                        "reason_code": enriched.get("code", ""),
                        "reason": enriched.get("message", ""),
                        "repair_intent": repair_intent,
                    }
                )

        for ancestor_id in ancestor_ids:
            ancestor = target_lookup.get(ancestor_id)
            if not ancestor:
                continue
            _append_unique_metadata_value(ancestor, "repair_roles", "ancestor_container")
            _append_unique_metadata_value(ancestor, "diagnostic_codes", str(enriched.get("code", "") or ""))
            _append_unique_metadata_value(ancestor, "repair_intents", repair_intent)
            candidate_key = ("target", ancestor_id, str(enriched.get("code", "") or "ancestor"))
            if candidate_key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(candidate_key)
            repair_candidates.append(
                    {
                        "target_id": ancestor_id,
                        "kind": ancestor.get("kind", ""),
                        "selector_hint": ancestor.get("selector_hint", ""),
                        "allowed_ops": ancestor.get("allowed_ops", []),
                        "repair_role": "ancestor_container",
                        "reason_code": enriched.get("code", ""),
                        "reason": enriched.get("message", ""),
                        "repair_intent": repair_intent,
                    }
            )

        for rule_id in governing_rule_ids:
            rule = rule_lookup.get(rule_id)
            if not rule:
                continue
            _append_unique_metadata_value(rule, "repair_roles", "governing_rule")
            _append_unique_metadata_value(rule, "diagnostic_codes", str(enriched.get("code", "") or ""))
            _append_unique_metadata_value(rule, "repair_intents", repair_intent)
            candidate_key = ("rule", rule_id, str(enriched.get("code", "") or "rule"))
            if candidate_key in seen_candidate_keys:
                continue
            seen_candidate_keys.add(candidate_key)
            repair_candidates.append(
                    {
                        "rule_id": rule_id,
                        "kind": "css_rule",
                        "selector_hint": rule.get("selector", ""),
                        "allowed_ops": rule.get("allowed_ops", list(_RULE_ALLOWED_OPS)),
                        "repair_role": "governing_rule",
                        "reason_code": enriched.get("code", ""),
                        "reason": enriched.get("message", ""),
                        "repair_intent": repair_intent,
                    }
            )

    repair_context = {"diagnostics": enriched_diagnostics}
    return repair_candidates, offending_targets, repair_context, enriched_diagnostics


def _build_slide_snapshot_payload(
    path: Path,
    html_text: str,
    *,
    focus_text: str = "",
    focus_selector: str = "",
    focus_kind: str = "",
) -> dict[str, Any]:
    soup = BeautifulSoup(html_text, "lxml")
    body = soup.body or soup.find("body")
    if not isinstance(body, Tag):
        body = soup.html if isinstance(soup.html, Tag) else soup

    css_rules = _parse_css_rules(_extract_all_style_text(html_text))
    validation_diagnostics = _validation_diagnostics_for_slide(path, html_text)
    risk_flags = _unique_preserve_order(
        [str(item.get("code", "") or "") for item in validation_diagnostics if str(item.get("code", "") or "")]
    )

    order_map = {
        id(tag): index
        for index, tag in enumerate(body.find_all(True))
    }
    candidate_map: dict[str, dict[str, Any]] = {}
    repair_dom_paths: list[str] = []
    for diagnostic in sorted(
        validation_diagnostics,
        key=lambda item: _diagnostic_priority(str(item.get("code", "") or "")),
    ):
        for dom_path in diagnostic.get("repair_dom_paths", []) or []:
            normalized = str(dom_path or "").strip()
            if normalized and normalized not in repair_dom_paths:
                repair_dom_paths.append(normalized)

    normalized_focus_text = _normalize_focus_text(focus_text)
    normalized_focus_selector = str(focus_selector or "").strip()
    normalized_focus_kind = _normalize_focus_kind(focus_kind)

    def _infer_focus_target_kind(tag: Tag) -> str:
        if normalized_focus_kind and _focus_kind_matches_tag(tag, normalized_focus_kind):
            if normalized_focus_kind in {"body_text", "any_text"}:
                return "inline_text"
            return normalized_focus_kind
        if _focus_kind_matches_tag(tag, "caption"):
            return "caption"
        if _focus_kind_matches_tag(tag, "pill_tag"):
            return "pill_tag"
        if _focus_kind_matches_tag(tag, "table_cell"):
            return "table_cell"
        if _focus_kind_matches_tag(tag, "legend_label"):
            return "legend_label"
        if _focus_kind_matches_tag(tag, "callout"):
            return "callout"
        if _focus_kind_matches_tag(tag, "footnote"):
            return "footnote"
        if tag.name.lower() == "img":
            return "image"
        if _is_semantic_text_tag(tag):
            return "inline_text"
        if _is_figure_like_tag(tag):
            return "figure"
        return "layout_container"

    def _register_candidate(
        tag: Tag,
        kind: str,
        *,
        source: str = "semantic",
        matched_focus: bool = False,
    ) -> None:
        dom_path = _build_dom_path(tag)
        if not dom_path:
            return
        props = _collect_tag_style_props(css_rules, tag)
        is_repair = dom_path in repair_dom_paths
        is_focus = bool(matched_focus) or _tag_matches_focus(
            tag,
            focus_text=normalized_focus_text,
            focus_selector=normalized_focus_selector,
            focus_kind=normalized_focus_kind,
        )
        rank = _target_selection_rank(kind)
        if is_focus:
            rank -= 35
        elif source == "focus_parent":
            rank -= 20
        if is_repair:
            rank -= 15
        entry = candidate_map.get(dom_path)
        if entry and int(entry.get("rank", 999)) <= rank:
            if is_focus:
                entry["matched_focus"] = True
                entry["source"] = "focus"
            return
        candidate_map[dom_path] = {
            "tag": tag,
            "kind": kind,
            "rank": rank,
            "order": order_map.get(id(tag), len(order_map)),
            "box_context": _style_context_subset(props),
            "effective_style": props,
            "is_repair": is_repair,
            "source": "focus" if is_focus else source,
            "matched_focus": is_focus,
        }

    if isinstance(body, Tag) and body.name and body.name.lower() == "body":
        _register_candidate(body, "slide_canvas")
    for kind in ("slide_title", "footer"):
        for tag in _semantic_target_tags(body, kind):
            _register_candidate(tag, kind)

    for kind in (
        "caption",
        "pill_tag",
        "table_cell",
        "legend_label",
        "callout",
        "footnote",
    ):
        for tag in _semantic_target_tags(body, kind):
            _register_candidate(tag, kind)

    for tag in _semantic_target_tags(body, "body_text"):
        _register_candidate(tag, "text_block")

    for tag in _layout_container_tags(body, css_rules):
        _register_candidate(tag, "layout_container")

    for tag in body.find_all(True):
        if _is_figure_like_tag(tag):
            _register_candidate(tag, "figure")
    for tag in body.find_all("img"):
        _register_candidate(tag, "image")
    for child in [node for node in body.children if isinstance(node, Tag)]:
        _register_candidate(child, "body_block")

    if normalized_focus_text or normalized_focus_selector or normalized_focus_kind:
        for tag in body.find_all(True):
            if not _tag_matches_focus(
                tag,
                focus_text=normalized_focus_text,
                focus_selector=normalized_focus_selector,
                focus_kind=normalized_focus_kind,
            ):
                continue
            _register_candidate(
                tag,
                _infer_focus_target_kind(tag),
                source="focus",
                matched_focus=True,
            )
            parent = tag.parent
            if isinstance(parent, Tag) and parent is not body and parent.name.lower() not in {"html", "head"}:
                parent_kind = "layout_container"
                if _is_semantic_text_tag(parent) and not _is_layout_container_tag(parent, _collect_tag_style_props(css_rules, parent)):
                    parent_kind = "inline_text"
                _register_candidate(
                    parent,
                    parent_kind,
                    source="focus_parent",
                )

    ordered_candidates = sorted(
        candidate_map.values(),
        key=lambda item: (
            0 if str(item.get("kind")) == "slide_canvas" else 1,
            int(item.get("rank", 999)),
            int(item.get("order", 9999)),
        )
    )

    selected_candidates: list[dict[str, Any]] = []
    if ordered_candidates and str(ordered_candidates[0].get("kind")) == "slide_canvas":
        selected_candidates.append(ordered_candidates[0])

    focus_selected = 0
    for candidate in ordered_candidates:
        if candidate in selected_candidates:
            continue
        if not candidate.get("matched_focus") and candidate.get("source") != "focus_parent":
            continue
        if focus_selected >= 8:
            break
        selected_candidates.append(candidate)
        focus_selected += 1

    repair_selected = 0
    for candidate in ordered_candidates:
        if candidate in selected_candidates:
            continue
        if not candidate.get("is_repair"):
            continue
        if repair_selected >= 4:
            break
        selected_candidates.append(candidate)
        repair_selected += 1

    for candidate in ordered_candidates:
        if candidate in selected_candidates:
            continue
        selected_candidates.append(candidate)

    has_more_targets = len(selected_candidates) > _SLIDE_SNAPSHOT_TARGET_LIMIT
    visible_candidates = selected_candidates[:_SLIDE_SNAPSHOT_TARGET_LIMIT]

    targets: list[dict[str, Any]] = []
    target_tags_by_id: dict[str, Tag] = {}
    for candidate in visible_candidates:
        tag = candidate["tag"]
        kind = str(candidate["kind"])
        target_payload = _build_snapshot_target(
            tag,
            kind,
            f"target_{len(targets) + 1:02d}",
            box_context=dict(candidate.get("box_context", {}) or {}),
            effective_style=dict(candidate.get("effective_style", {}) or {}),
        )
        target_payload["source"] = str(candidate.get("source", "") or "semantic")
        target_payload["matched_focus"] = bool(candidate.get("matched_focus", False))
        targets.append(target_payload)
        target_tags_by_id[target_payload["target_id"]] = tag

    target_ids_by_dom_path = {
        str(target.get("dom_path", "") or ""): str(target.get("target_id", "") or "")
        for target in targets
    }
    child_ids_by_parent_id: dict[str, list[str]] = {}
    for target in targets:
        target_id = str(target.get("target_id", "") or "")
        tag = target_tags_by_id.get(target_id)
        parent_id = ""
        parent = tag.parent if isinstance(tag, Tag) else None
        while isinstance(parent, Tag):
            parent_dom_path = _build_dom_path(parent)
            parent_id = target_ids_by_dom_path.get(parent_dom_path, "")
            if parent_id and parent_id != target_id:
                break
            parent = parent.parent
        target["parent_target_id"] = parent_id
        if parent_id:
            child_ids_by_parent_id.setdefault(parent_id, []).append(target_id)
    for target in targets:
        target["child_target_ids"] = child_ids_by_parent_id.get(str(target.get("target_id", "") or ""), [])

    rule_entries = _extract_css_rule_entries(html_text)
    rules: list[dict[str, Any]] = []
    rule_ids_by_target_id: dict[str, list[str]] = {
        str(target.get("target_id", "")): []
        for target in targets
    }
    for rule_entry in rule_entries:
        used_by_target_ids = [
            target_id
            for target_id, tag in target_tags_by_id.items()
            if _selector_targets_tag(tag, str(rule_entry.get("selector", "") or ""))
        ]
        if not used_by_target_ids and not _selector_mentions_body_or_slide(str(rule_entry.get("selector", "") or "")):
            continue
        used_targets = [
            target
            for target in targets
            if str(target.get("target_id", "") or "") in set(used_by_target_ids)
        ]
        declarations = dict(rule_entry.get("declarations", {}) or {})
        semantic_roles = _semantic_roles_for_rule(
            str(rule_entry.get("selector", "") or ""),
            declarations,
            used_targets,
        )
        property_groups_present = _semantic_property_groups_for_props(declarations)
        rule_payload = {
            "rule_id": str(rule_entry.get("rule_id", "")),
            "selector": str(rule_entry.get("selector", "")),
            "declarations_preview": str(rule_entry.get("declarations_preview", "")),
            "declarations": declarations,
            "used_by_target_ids": used_by_target_ids,
            "allowed_ops": list(_RULE_ALLOWED_OPS),
            "rule_hash": str(rule_entry.get("rule_hash", "")),
            "semantic_roles": semantic_roles,
            "editable_property_groups": _editable_property_groups_for_roles(semantic_roles),
            "property_groups_present": property_groups_present,
            "effective_style_subset": _semantic_style_subset(declarations),
        }
        rules.append(rule_payload)
        for target_id in used_by_target_ids:
            rule_ids_by_target_id.setdefault(target_id, []).append(rule_payload["rule_id"])

    for target in targets:
        target["governing_rule_ids"] = rule_ids_by_target_id.get(str(target.get("target_id", "")), [])

    repair_candidates, offending_targets, repair_context, validation_diagnostics = _build_repair_surface(
        targets,
        rules,
        validation_diagnostics,
    )

    return {
        "targets": targets,
        "has_more_targets": has_more_targets,
        "slide_summary": _build_slide_summary(
            path,
            targets,
            has_more_targets,
            rules=rules,
            risk_flags=risk_flags,
        ),
        "rules": rules,
        "risk_flags": risk_flags,
        "repair_candidates": repair_candidates,
        "offending_targets": offending_targets,
        "repair_context": repair_context,
        "validation_diagnostics": validation_diagnostics,
    }


def _structured_patch_error(
    error_code: str,
    message: str,
    **extra: Any,
) -> str:
    payload: dict[str, Any] = {
        "success": False,
        "error_code": error_code,
        "error": message,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _register_slide_snapshot(
    path: Path,
    html_text: str,
    *,
    slide_path: str | None = None,
    snapshot_payload: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """Create and register a fresh snapshot for the current on-disk slide HTML."""
    content_hash = _compute_text_hash(html_text)
    payload = snapshot_payload or _build_slide_snapshot_payload(path, html_text)
    snapshot_id = uuid.uuid4().hex[:12]
    _slide_snapshot_registry[snapshot_id] = {
        "slide_path": slide_path or str(path),
        "resolved_path": str(path.resolve()),
        "content_hash": content_hash,
        "created_at": _utc_now_iso(),
        "targets": payload["targets"],
        "rules": payload.get("rules", []),
        "repair_candidates": payload.get("repair_candidates", []),
        "offending_targets": payload.get("offending_targets", []),
        "repair_context": payload.get("repair_context", {}),
        "risk_flags": payload.get("risk_flags", []),
        "validation_diagnostics": payload.get("validation_diagnostics", []),
        "has_more_targets": bool(payload.get("has_more_targets", False)),
        "slide_summary": str(payload.get("slide_summary", "") or ""),
    }
    return snapshot_id, content_hash, payload


def _retire_slide_snapshot_ids(
    retired_ids: list[str],
    *,
    slide_path: str,
    resolved_path: str,
    current_content_hash: str,
    next_snapshot_id: str = "",
    targets: list[dict[str, Any]] | None = None,
    rules: list[dict[str, Any]] | None = None,
    repair_candidates: list[dict[str, Any]] | None = None,
    offending_targets: list[dict[str, Any]] | None = None,
    repair_context: dict[str, Any] | None = None,
    risk_flags: list[str] | None = None,
    validation_diagnostics: list[dict[str, Any]] | None = None,
    slide_summary: str = "",
    has_more_targets: bool = False,
) -> None:
    if not retired_ids:
        return

    retired_payload = {
        "slide_path": slide_path,
        "resolved_path": resolved_path,
        "current_content_hash": current_content_hash,
        "next_snapshot_id": next_snapshot_id,
        "targets": list(targets or []),
        "rules": list(rules or []),
        "repair_candidates": list(repair_candidates or []),
        "offending_targets": list(offending_targets or []),
        "repair_context": dict(repair_context or {}),
        "risk_flags": list(risk_flags or []),
        "validation_diagnostics": list(validation_diagnostics or []),
        "slide_summary": slide_summary,
        "has_more_targets": bool(has_more_targets),
        "retired_at": _utc_now_iso(),
    }
    for retired_id in retired_ids:
        _slide_retired_snapshot_registry[retired_id] = dict(retired_payload)


def _parse_css_pixel_value(value: str) -> float | None:
    match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)px\s*", str(value or ""), re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _format_px_value(value: float | None) -> str:
    if value is None:
        return "?"
    if float(value).is_integer():
        return f"{int(value)}px"
    return f"{value:.1f}px"


def _format_px_pair(width: float | None, height: float | None) -> str:
    width_text = f"{width:.1f}px" if width is not None else "?"
    height_text = f"{height:.1f}px" if height is not None else "?"
    return f"{width_text} × {height_text}"


def _parse_css_box_edges(props: dict[str, str], prefix: str) -> dict[str, str]:
    edges = {"top": "", "right": "", "bottom": "", "left": ""}
    shorthand = str(props.get(prefix, "") or "").strip()
    if shorthand:
        tokens = [token for token in re.split(r"\s+", shorthand) if token]
        if len(tokens) == 1:
            edges = {side: tokens[0] for side in edges}
        elif len(tokens) == 2:
            edges = {
                "top": tokens[0],
                "right": tokens[1],
                "bottom": tokens[0],
                "left": tokens[1],
            }
        elif len(tokens) == 3:
            edges = {
                "top": tokens[0],
                "right": tokens[1],
                "bottom": tokens[2],
                "left": tokens[1],
            }
        elif len(tokens) >= 4:
            edges = {
                "top": tokens[0],
                "right": tokens[1],
                "bottom": tokens[2],
                "left": tokens[3],
            }
    for side in ("top", "right", "bottom", "left"):
        explicit = str(props.get(f"{prefix}-{side}", "") or "").strip()
        if explicit:
            edges[side] = explicit
    return edges


def _pixel_like_value_from_tag(tag: Tag, attr_name: str) -> float | None:
    raw_value = str(tag.get(attr_name, "") or "").strip()
    if not raw_value:
        return None
    if raw_value.lower().endswith("px"):
        return _parse_css_pixel_value(raw_value)
    try:
        return float(raw_value)
    except Exception:
        return None


def _top_level_slide_roots(body: Tag | BeautifulSoup) -> list[Tag]:
    slide_roots: list[Tag] = []
    for tag in body.find_all(True):
        if not _tag_has_any_class(tag, {"slide"}):
            continue
        has_slide_ancestor = any(
            isinstance(parent, Tag) and _tag_has_any_class(parent, {"slide"})
            for parent in tag.parents
        )
        if not has_slide_ancestor:
            slide_roots.append(tag)
    return slide_roots


def _bottom_safe_zone_violations(
    body: Tag,
    css_rules: dict[str, dict[str, str]],
    *,
    min_bottom_px: float = _BOTTOM_SAFE_ZONE_MIN_PX,
) -> list[tuple[Tag, float]]:
    violations: list[tuple[Tag, float]] = []
    for tag in body.find_all(True):
        if tag.name.lower() in {"html", "head", "body", "style", "script", "img"}:
            continue
        if _tag_has_any_class(tag, {_ANCHOR_LAYER_CLASS, _ANCHOR_ELEMENT_CLASS}):
            continue
        if _tag_text_length(tag) <= 0:
            continue
        props = _collect_tag_style_props(css_rules, tag)
        bottom = _parse_css_pixel_value(props.get("bottom", ""))
        if bottom is None or not (0 < bottom < min_bottom_px):
            continue
        font_size = _parse_css_pixel_value(props.get("font-size", ""))
        if (
            font_size is not None
            and font_size <= 16
            and not (_tag_has_any_class(tag, _FOOTER_TARGET_CLASSES) or tag.name.lower() == "footer")
        ):
            continue
        violations.append((tag, bottom))
    return sorted(violations, key=lambda item: (item[1], -_tag_depth(item[0]), _selector_hint_for_tag(item[0])))


def _diagnostic_payload(
    code: str,
    message: str,
    *,
    source: str,
    severity: str = "warning",
    target_tag: Tag | None = None,
    repair_tags: list[Tag] | None = None,
    metadata: dict[str, Any] | None = None,
    repair_strategy: str = "",
) -> dict[str, Any]:
    target_dom_path = _build_dom_path(target_tag) if isinstance(target_tag, Tag) else ""
    target_selector_hint = _selector_hint_for_tag(target_tag) if isinstance(target_tag, Tag) else ""
    repair_dom_paths = [
        _build_dom_path(tag)
        for tag in (repair_tags or [])
        if isinstance(tag, Tag) and _build_dom_path(tag)
    ]
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "source": source,
        "severity": severity,
        "target_dom_path": target_dom_path,
        "target_selector_hint": target_selector_hint,
        "repair_dom_paths": repair_dom_paths,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    if repair_strategy:
        payload["repair_strategy"] = repair_strategy
    return payload


def _diagnostic_signature(diagnostic: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(diagnostic.get("code", "") or ""),
        str(diagnostic.get("target_dom_path", "") or ""),
        str(diagnostic.get("text_preview", "") or ""),
    )


def _diagnostic_source_rank(source: str) -> int:
    normalized = str(source or "").strip().lower()
    if normalized == "export_validation":
        return 0
    if normalized == "inspect":
        return 1
    if normalized == "visual_quality":
        return 2
    if normalized == "static":
        return 3
    return 9


def _diagnostic_severity_rank(severity: str) -> int:
    normalized = str(severity or "").strip().lower()
    if normalized == "error":
        return 0
    if normalized == "warning":
        return 1
    return 9


def _merge_validation_diagnostics(*diagnostic_lists: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    chosen: dict[tuple[str, str, str], dict[str, Any]] = {}
    for diagnostic_list in diagnostic_lists:
        for diagnostic in diagnostic_list or []:
            if not isinstance(diagnostic, dict):
                continue
            entry = dict(diagnostic)
            signature = _diagnostic_signature(entry)
            current = chosen.get(signature)
            if current is None:
                chosen[signature] = entry
                continue
            current_rank = (
                _diagnostic_source_rank(str(current.get("source", "") or "")),
                _diagnostic_severity_rank(str(current.get("severity", "") or "")),
            )
            entry_rank = (
                _diagnostic_source_rank(str(entry.get("source", "") or "")),
                _diagnostic_severity_rank(str(entry.get("severity", "") or "")),
            )
            if entry_rank < current_rank:
                chosen[signature] = entry
    return sorted(
        chosen.values(),
        key=lambda item: (
            _diagnostic_priority(str(item.get("code", "") or "")),
            _diagnostic_source_rank(str(item.get("source", "") or "")),
            _diagnostic_severity_rank(str(item.get("severity", "") or "")),
            str(item.get("target_dom_path", "") or ""),
        ),
    )


def _repair_intent_for_diagnostic_code(code: str) -> str:
    return _REPAIR_INTENT_BY_DIAGNOSTIC_CODE.get(str(code or ""), "none")


def _normalized_preview_match_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _build_static_slide_diagnostics(
    path: Path,
    html_text: str,
    *,
    aspect_ratio: str = "16:9",
    validation_message: str = "",
    source: str = "static",
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        return diagnostics

    body = _html_body_root_from_soup(soup)
    if not isinstance(body, Tag):
        return diagnostics

    css_rules = _parse_css_rules(_extract_all_style_text(html_text))
    body_props = _collect_tag_style_props(css_rules, body)
    body_width = _parse_css_pixel_value(body_props.get("width", ""))
    body_height = _parse_css_pixel_value(body_props.get("height", ""))
    expected_size = _EXPECTED_LAYOUT_SIZES.get(str(aspect_ratio or "16:9"), _EXPECTED_LAYOUT_SIZES["16:9"])
    expected_width, expected_height = expected_size

    if body_width is None or body_height is None:
        diagnostics.append(
            _diagnostic_payload(
                "canvas_dimension_missing",
                (
                    "`body` 缺少可静态解析的固定像素 `width/height`，"
                    f"{aspect_ratio} 应使用 {expected_width}px × {expected_height}px。"
                ),
                source=source,
                severity="warning",
                target_tag=body,
                repair_tags=[body],
                metadata={
                    "expected_width_px": expected_width,
                    "expected_height_px": expected_height,
                    "body_width": body_props.get("width", ""),
                    "body_height": body_props.get("height", ""),
                },
                repair_strategy="Set `body` to the exact slide canvas size with margin/padding 0 and border-box sizing.",
            )
        )

    if (
        body_width is not None
        and body_height is not None
        and (abs(body_width - expected_width) > 0.5 or abs(body_height - expected_height) > 0.5)
    ):
        diagnostics.append(
            _diagnostic_payload(
                "canvas_dimension_mismatch",
                (
                    f"`body` 画布尺寸为 {_format_px_pair(body_width, body_height)}，"
                    f"与 {aspect_ratio} 期望的 {expected_width}px × {expected_height}px 不一致。"
                ),
                source=source,
                severity="error",
                target_tag=body,
                repair_tags=[body],
                metadata={
                    "expected_width_px": expected_width,
                    "expected_height_px": expected_height,
                    "actual_width_px": body_width,
                    "actual_height_px": body_height,
                },
                repair_strategy="Set `body` to the exact slide canvas size before export.",
            )
        )

    body_padding = _parse_css_box_edges(body_props, "padding")
    has_padding = any(
        (parsed := _parse_css_pixel_value(value)) is not None and parsed > 0
        for value in body_padding.values()
    )
    if has_padding and str(body_props.get("box-sizing", "") or "").strip().lower() != "border-box":
        diagnostics.append(
            _diagnostic_payload(
                "body_padding_without_border_box",
                "`body` 使用了 padding 但未设置 `box-sizing: border-box`，容易把内容实际画布挤出边界。",
                source=source,
                severity="warning",
                target_tag=body,
                repair_tags=[body],
                repair_strategy="Use `box-sizing: border-box` or move padding into the inner `.slide` container.",
            )
        )

    slide_roots = _top_level_slide_roots(body)
    for slide_root in slide_roots[:2]:
        slide_props = _collect_tag_style_props(css_rules, slide_root)
        slide_width = _parse_css_pixel_value(slide_props.get("width", ""))
        slide_height = _parse_css_pixel_value(slide_props.get("height", ""))
        if (
            slide_width is not None
            and slide_height is not None
            and (
                slide_width > expected_width + 0.5
                or slide_height > expected_height + 0.5
                or slide_width < expected_width * 0.75
                or slide_height < expected_height * 0.75
            )
        ):
            diagnostics.append(
                _diagnostic_payload(
                    "slide_root_dimension_risk",
                    (
                        f"`{_selector_hint_for_tag(slide_root)}` 尺寸为 {_format_px_pair(slide_width, slide_height)}，"
                        f"与导出画布 {expected_width}px × {expected_height}px 不一致，可能导致 canvas 裁切。"
                    ),
                    source=source,
                    severity="warning",
                    target_tag=slide_root,
                    repair_tags=[slide_root],
                    metadata={
                        "expected_width_px": expected_width,
                        "expected_height_px": expected_height,
                        "actual_width_px": slide_width,
                        "actual_height_px": slide_height,
                    },
                    repair_strategy="Constrain the top-level `.slide` root to the exact canvas size and hide overflow.",
                )
            )

    bottom_violations = _bottom_safe_zone_violations(body, css_rules)
    if bottom_violations:
        violation_tags = [tag for tag, _ in bottom_violations[:4]]
        violation_values = ", ".join(
            _format_px_value(bottom)
            for _, bottom in bottom_violations[:3]
        )
        diagnostics.append(
            _diagnostic_payload(
                "bottom_safe_zone_violation",
                (
                    f"检测到 {len(bottom_violations)} 个文本元素的 `bottom` 小于 "
                    f"{int(_BOTTOM_SAFE_ZONE_MIN_PX)}px（例如 {violation_values}），"
                    "导出时容易触发 `too close to bottom edge`；这是静态预警，优先以 inspect/export 诊断为准。"
                ),
                source=source,
                severity="warning",
                target_tag=violation_tags[0],
                repair_tags=violation_tags,
                metadata={
                    "min_bottom_px": _BOTTOM_SAFE_ZONE_MIN_PX,
                    "bottom_values_px": [bottom for _, bottom in bottom_violations[:6]],
                },
                repair_strategy="Raise positioned text to `bottom: 48px` or move it into a constrained content container.",
            )
        )

    geometry_issue = _find_suspicious_slide_geometry_issue(html_text)
    if geometry_issue and slide_roots:
        diagnostics.append(
            _diagnostic_payload(
                "suspicious_slide_geometry",
                geometry_issue,
                source=source,
                severity="warning",
                target_tag=slide_roots[0],
                repair_tags=[slide_roots[0]],
                repair_strategy="Use a single fixed-size slide root matching the body canvas.",
            )
        )

    nested_issue = _find_nested_slide_root_issue(body)
    if nested_issue:
        diagnostics.append(
            _diagnostic_payload(
                "nested_slide_root",
                nested_issue,
                source=source,
                severity="error",
                target_tag=slide_roots[0] if slide_roots else None,
                repair_tags=slide_roots[:1],
                repair_strategy="Remove nested `.slide` roots and keep only one top-level slide canvas.",
            )
        )

    for tag in _layout_container_tags(body, css_rules):
        props = _collect_tag_style_props(css_rules, tag)
        explicit_height = _parse_css_pixel_value(props.get("height", "")) or _parse_css_pixel_value(props.get("max-height", ""))
        padding_edges = _parse_css_box_edges(props, "padding")
        padding_px = [
            parsed
            for value in padding_edges.values()
            if (parsed := _parse_css_pixel_value(value)) is not None and parsed > 0
        ]
        if (
            explicit_height is not None
            and padding_px
            and str(props.get("box-sizing", "") or "").strip().lower() != "border-box"
        ):
            diagnostics.append(
                _diagnostic_payload(
                    "container_padding_without_border_box",
                    (
                        f"`{_selector_hint_for_tag(tag)}` 同时设置固定高度和 padding，"
                        "但未使用 `box-sizing: border-box`，实际外框高度可能超过预期并触发导出裁切。"
                    ),
                    source=source,
                    severity="warning",
                    target_tag=tag,
                    repair_tags=[tag],
                    metadata={
                        "height_px": explicit_height,
                        "padding": padding_edges,
                    },
                    repair_strategy="Set `box-sizing: border-box`, cap `max-height`, and hide overflow on the container.",
                )
            )
            break

    for tag in body.find_all(True):
        props = _collect_tag_style_props(css_rules, tag)
        position = str(props.get("position", "") or "").strip().lower()
        if position not in {"absolute", "fixed"}:
            continue
        top = _parse_css_pixel_value(props.get("top", ""))
        left = _parse_css_pixel_value(props.get("left", ""))
        right = _parse_css_pixel_value(props.get("right", ""))
        bottom = _parse_css_pixel_value(props.get("bottom", ""))
        off_canvas = (
            (top is not None and top < 0)
            or (left is not None and left < 0)
            or (right is not None and right < 0)
            or (bottom is not None and bottom < 0)
            or (body_width is not None and left is not None and left > body_width)
            or (body_height is not None and top is not None and top > body_height)
        )
        if not off_canvas:
            width = _parse_css_pixel_value(props.get("width", ""))
            height = _parse_css_pixel_value(props.get("height", ""))
            bottom_edge = None if top is None or height is None else top + height
            right_edge = None if left is None or width is None else left + width
            off_canvas = (
                body_height is not None
                and bottom_edge is not None
                and bottom_edge > body_height - 4
            ) or (
                body_width is not None
                and right_edge is not None
                and right_edge > body_width - 4
            )
        if not off_canvas:
            continue
        diagnostics.append(
            _diagnostic_payload(
                "off_canvas_absolute",
                (
                    f"`{_selector_hint_for_tag(tag)}` 使用了超出画布边界的绝对定位，"
                    "可能导致导出时元素被裁出 slide canvas。"
                ),
                source=source,
                severity="warning",
                target_tag=tag,
                repair_tags=[tag],
                metadata={
                    "position": position,
                    "top": props.get("top", ""),
                    "left": props.get("left", ""),
                    "right": props.get("right", ""),
                    "bottom": props.get("bottom", ""),
                    "width": props.get("width", ""),
                    "height": props.get("height", ""),
                },
                repair_strategy="Move the absolute/fixed element inside the canvas or reduce its width/height.",
            )
        )
        break

    for tag in body.find_all("img"):
        props = _collect_tag_style_props(css_rules, tag)
        width = (
            _parse_css_pixel_value(props.get("width", ""))
            or _parse_css_pixel_value(props.get("max-width", ""))
            or _pixel_like_value_from_tag(tag, "width")
        )
        height = (
            _parse_css_pixel_value(props.get("height", ""))
            or _parse_css_pixel_value(props.get("max-height", ""))
            or _pixel_like_value_from_tag(tag, "height")
        )
        image_too_wide = body_width is not None and width is not None and width > body_width
        image_too_tall = body_height is not None and height is not None and height > body_height
        if not (image_too_wide or image_too_tall):
            continue
        diagnostics.append(
            _diagnostic_payload(
                "image_overflow",
                (
                    f"`{_selector_hint_for_tag(tag)}` 的尺寸为 {_format_px_pair(width, height)}，"
                    f"超过了 `body` 画布 {_format_px_pair(body_width, body_height)}。"
                ),
                source=source,
                severity="warning",
                target_tag=tag,
                repair_tags=[tag],
                metadata={
                    "width_px": width,
                    "height_px": height,
                    "body_width_px": body_width,
                    "body_height_px": body_height,
                },
                repair_strategy="Constrain the media element with max-width/max-height and object-fit: contain.",
            )
        )
        break

    for table in body.find_all("table"):
        rows = table.find_all("tr")
        props = _collect_tag_style_props(css_rules, table)
        table_height = _parse_css_pixel_value(props.get("height", "")) or _parse_css_pixel_value(props.get("max-height", ""))
        parent = table.parent if isinstance(table.parent, Tag) else None
        parent_props = _collect_tag_style_props(css_rules, parent) if isinstance(parent, Tag) else {}
        parent_has_clip = bool(
            str(parent_props.get(key, "") or "").strip()
            for key in ("max-height", "height", "overflow", "overflow-y")
        )
        if (
            len(rows) >= 12
            and not parent_has_clip
        ) or (
            table_height is not None
            and body_height is not None
            and table_height > body_height * 0.7
        ):
            repair_tag = parent if isinstance(parent, Tag) and parent.name.lower() not in {"body", "html"} else table
            diagnostics.append(
                _diagnostic_payload(
                    "table_height_risk",
                    (
                        f"`{_selector_hint_for_tag(table)}` 表格较高或行数较多（rows={len(rows)}），"
                        "建议放入固定高度 wrapper，避免导出时底部裁切。"
                    ),
                    source=source,
                    severity="warning",
                    target_tag=repair_tag,
                    repair_tags=[repair_tag],
                    metadata={
                        "row_count": len(rows),
                        "table_height_px": table_height,
                    },
                    repair_strategy="Wrap or constrain the table with `max-height` and `overflow: hidden`.",
                )
            )
            break

    text_overflow_markers = (
        "overflows body",
        "HTML content overflows body",
    )
    bottom_edge_markers = ("too close to bottom edge",)
    clipped_markers = ("clipped outside the slide canvas",)
    preferred_layout_targets = _preferred_overflow_repair_tags(body, css_rules)
    text_targets = _semantic_target_tags(body, "body_text")

    if any(marker in validation_message for marker in bottom_edge_markers):
        repair_tags = (
            [tag for tag, _ in bottom_violations[:4]]
            or text_targets[:2]
            or preferred_layout_targets[:1]
            or slide_roots[:1]
            or [body]
        )
        diagnostics.append(
            _diagnostic_payload(
                "bottom_safe_zone_violation",
                (
                    "当前页存在底部安全区风险，文本元素过于贴近 slide 底边，"
                    f"请将相关文本或容器的 `bottom` 提高到至少 {int(_BOTTOM_SAFE_ZONE_MIN_PX)}px。"
                ),
                source="inspect",
                severity="error",
                target_tag=repair_tags[0] if repair_tags else None,
                repair_tags=repair_tags,
                repair_strategy="Raise the affected text/container above the bottom safe zone or rebalance into columns.",
            )
        )

    if any(marker in validation_message for marker in clipped_markers):
        image_targets = body.find_all("img")
        repair_tags = image_targets[:1] or preferred_layout_targets[:2] or slide_roots[:1] or [body]
        diagnostics.append(
            _diagnostic_payload(
                "clipped_outside_canvas",
                "当前页存在元素被裁出 slide canvas 的信号，优先收紧图片或外层容器尺寸，并检查绝对定位偏移。",
                source="inspect",
                severity="error",
                target_tag=repair_tags[0] if repair_tags else None,
                repair_tags=repair_tags,
                metadata={"overflow_direction": "unknown"},
                repair_strategy="Constrain the reported element/container to the canvas; for dense pages use a two-column fallback layout.",
            )
        )

    if any(marker in validation_message for marker in text_overflow_markers):
        repair_tags = preferred_layout_targets[:2] or text_targets[:2] or slide_roots[:1] or [body]
        diagnostics.append(
            _diagnostic_payload(
                "text_overflow",
                "当前页存在文本或正文容器溢出信号，优先收紧正文容器尺寸、行高、字体或改为分栏布局。",
                source="inspect",
                severity="error",
                target_tag=repair_tags[0] if repair_tags else None,
                repair_tags=repair_tags,
                repair_strategy="Reduce container density with max-height/overflow constraints or rebalance content into columns.",
            )
        )

    deduped: list[dict[str, Any]] = []
    seen_codes: set[tuple[str, str]] = set()
    for diagnostic in diagnostics:
        signature = (
            str(diagnostic.get("code", "")),
            str(diagnostic.get("target_dom_path", "")),
        )
        if signature in seen_codes:
            continue
        seen_codes.add(signature)
        deduped.append(diagnostic)
    return deduped


def _validation_diagnostics_for_slide(
    path: Path,
    html_text: str,
    *,
    validation_state: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    state = validation_state or _current_slide_validation_state(path)
    if isinstance(state, dict):
        cached = state.get("diagnostics")
        if isinstance(cached, list) and cached:
            return [item for item in cached if isinstance(item, dict)]
        aspect_ratio = str(state.get("aspect_ratio", "") or "16:9")
        message = str(state.get("message", "") or "")
        source = "inspect" if message or not bool(state.get("success", True)) else "inspect"
        diagnostics = _build_static_slide_diagnostics(
            path,
            html_text,
            aspect_ratio=aspect_ratio,
            validation_message=message,
            source=source,
        )
        return diagnostics
    return _build_static_slide_diagnostics(path, html_text, source="static")


def _expected_layout_size(aspect_ratio: str = "16:9") -> tuple[int, int]:
    return _EXPECTED_LAYOUT_SIZES.get(str(aspect_ratio or "16:9"), _EXPECTED_LAYOUT_SIZES["16:9"])


def _infer_static_layout_aspect_ratio(html_text: str, default: str = "16:9") -> str:
    try:
        soup = BeautifulSoup(html_text, "lxml")
        body = _html_body_root_from_soup(soup)
        css_rules = _parse_css_rules(_extract_all_style_text(html_text))
        props = _collect_tag_style_props(css_rules, body) if isinstance(body, Tag) else {}
        width = _parse_css_pixel_value(props.get("width", ""))
        height = _parse_css_pixel_value(props.get("height", ""))
    except Exception:
        return str(default or "16:9")
    if width is None or height is None:
        return str(default or "16:9")
    for aspect, (expected_width, expected_height) in _EXPECTED_LAYOUT_SIZES.items():
        if abs(width - expected_width) <= 1 and abs(height - expected_height) <= 1:
            return aspect
    return str(default or "16:9")


def validate_slide_layout_static(
    html: str,
    aspect_ratio: str = "16:9",
    path: Path | str | None = None,
    *,
    validation_message: str = "",
    source: str = "static",
) -> list[dict[str, Any]]:
    """Return deterministic static layout diagnostics for one slide HTML string."""
    slide_path = Path(path) if path is not None else Path("__memslides_static_slide__.html")
    return _build_static_slide_diagnostics(
        slide_path,
        str(html or ""),
        aspect_ratio=aspect_ratio,
        validation_message=validation_message,
        source=source,
    )


def _set_tag_inline_style_props(
    tag: Tag,
    desired_props: dict[str, str],
    *,
    report: list[dict[str, Any]],
    reason: str,
) -> bool:
    existing = str(tag.get("style", "") or "")
    merged, status = _merge_inline_style_text(existing, desired_props, mode="merge")
    if status == "already_compliant":
        return False
    tag["style"] = merged
    report.append(
        {
            "type": "inline_style",
            "selector": _selector_hint_for_tag(tag),
            "dom_path": _build_dom_path(tag),
            "status": status,
            "reason": reason,
            "declarations": dict(desired_props),
        }
    )
    return True


def _layout_guard_css(width: int, height: int) -> str:
    content_cap = max(120, int(height - _BOTTOM_SAFE_ZONE_MIN_PX - min(96, height * 0.14)))
    media_cap = max(120, int(height - _BOTTOM_SAFE_ZONE_MIN_PX - min(128, height * 0.18)))
    container_selector = ", ".join(
        f".{class_name}" for class_name in sorted(_LAYOUT_GUARD_CONTAINER_CLASSES)
    )
    return (
        "html, body { "
        f"width: {width}px; height: {height}px; margin: 0; padding: 0; "
        "box-sizing: border-box; overflow: hidden; }\n"
        "*, *::before, *::after { box-sizing: border-box; }\n"
        f".{_ANCHOR_LAYER_CLASS}, .{_ANCHOR_ELEMENT_CLASS} {{ box-sizing: border-box; }}\n"
        ".slide { "
        f"width: {width}px; height: {height}px; max-width: {width}px; max-height: {height}px; "
        "box-sizing: border-box; overflow: hidden; }\n"
        f"{container_selector} {{ max-height: {content_cap}px; box-sizing: border-box; }}\n"
        ".table-scroller, .table-wrapper, .table-wrap, .scroller { overflow: hidden; }\n"
        f"img, svg, canvas, table {{ max-width: 100%; max-height: {media_cap}px; }}\n"
        "img { object-fit: contain; }\n"
        "table { table-layout: fixed; border-collapse: collapse; }"
    )


def _upsert_layout_guard_style(soup: BeautifulSoup, width: int, height: int) -> bool:
    css = _layout_guard_css(width, height)
    existing = soup.find("style", id=_LAYOUT_GUARD_STYLE_ID)
    if isinstance(existing, Tag):
        if str(existing.string or "").strip() == css.strip():
            return False
        existing.clear()
        existing.append(css)
        return True

    style_tag = soup.new_tag("style", id=_LAYOUT_GUARD_STYLE_ID)
    style_tag.append(css)
    head = soup.head
    if not isinstance(head, Tag):
        html_root = soup.html
        if isinstance(html_root, Tag):
            head = soup.new_tag("head")
            html_root.insert(0, head)
        else:
            head = None
    if isinstance(head, Tag):
        head.append(style_tag)
    else:
        soup.insert(0, style_tag)
    return True


def _tag_has_layout_guard_class(tag: Tag) -> bool:
    classes = {token.lower() for token in _tag_classes(tag)}
    if classes & _LAYOUT_GUARD_CONTAINER_CLASSES:
        return True
    selector = _selector_hint_for_tag(tag).lower()
    return any(f".{class_name}" in selector for class_name in _LAYOUT_GUARD_CONTAINER_CLASSES)


def _safe_container_max_height_for_tag(
    tag: Tag,
    props: dict[str, str],
    *,
    canvas_height: int,
) -> int:
    top = _parse_css_pixel_value(props.get("top", ""))
    if top is not None and top >= 0:
        return max(96, int(canvas_height - top - _BOTTOM_SAFE_ZONE_MIN_PX - 12))
    return max(120, int(canvas_height - _BOTTOM_SAFE_ZONE_MIN_PX - min(96, canvas_height * 0.14)))


def _layout_static_blockers(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocker_codes = {
        "canvas_dimension_mismatch",
        "nested_slide_root",
        "clipped_outside_canvas",
        "text_overflow",
    }
    blockers: list[dict[str, Any]] = []
    for diagnostic in diagnostics:
        if str(diagnostic.get("severity", "") or "").lower() != "error":
            continue
        code = str(diagnostic.get("code", "") or "")
        if code in blocker_codes:
            blockers.append(diagnostic)
    return blockers


def repair_slide_layout_static(
    html: str,
    diagnostics: list[dict[str, Any]] | None = None,
    aspect_ratio: str = "16:9",
) -> tuple[str, list[dict[str, Any]]]:
    """Apply conservative deterministic layout repairs that are safe before render."""
    html_text = str(html or "")
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        return html_text, []

    body = _html_body_root_from_soup(soup)
    if not isinstance(body, Tag):
        return html_text, []

    expected_width, expected_height = _expected_layout_size(aspect_ratio)
    repair_report: list[dict[str, Any]] = []

    if _upsert_layout_guard_style(soup, expected_width, expected_height):
        repair_report.append(
            {
                "type": "css_guard",
                "selector": f"style#{_LAYOUT_GUARD_STYLE_ID}",
                "reason": "install fixed canvas/layout guard CSS",
                "declarations": {
                    "body.width": f"{expected_width}px",
                    "body.height": f"{expected_height}px",
                    ".slide.overflow": "hidden",
                },
            }
        )

    _set_tag_inline_style_props(
        body,
        {
            "width": f"{expected_width}px",
            "height": f"{expected_height}px",
            "margin": "0",
            "padding": "0",
            "box-sizing": "border-box",
            "overflow": "hidden",
        },
        report=repair_report,
        reason="normalize body canvas size",
    )

    css_rules = _parse_css_rules(_extract_all_style_text(str(soup)))
    slide_roots = _top_level_slide_roots(body)
    if not slide_roots:
        direct_children = [node for node in body.children if isinstance(node, Tag)]
        if len(direct_children) == 1:
            direct_children[0]["class"] = _unique_preserve_order([*_tag_classes(direct_children[0]), "slide"])
            slide_roots = [direct_children[0]]
            repair_report.append(
                {
                    "type": "class",
                    "selector": _selector_hint_for_tag(direct_children[0]),
                    "reason": "promote sole body child to slide root",
                    "class": "slide",
                }
            )

    for slide_root in slide_roots[:2]:
        _set_tag_inline_style_props(
            slide_root,
            {
                "width": f"{expected_width}px",
                "height": f"{expected_height}px",
                "max-width": f"{expected_width}px",
                "max-height": f"{expected_height}px",
                "box-sizing": "border-box",
                "overflow": "hidden",
            },
            report=repair_report,
            reason="normalize top-level slide root",
        )

    css_rules = _parse_css_rules(_extract_all_style_text(str(soup)))
    for tag, bottom in _bottom_safe_zone_violations(body, css_rules):
        _set_tag_inline_style_props(
            tag,
            {"bottom": f"{int(_BOTTOM_SAFE_ZONE_MIN_PX)}px"},
            report=repair_report,
            reason=f"raise positioned text from bottom:{_format_px_value(bottom)}",
        )

    for tag in _layout_container_tags(body, css_rules):
        props = _collect_tag_style_props(css_rules, tag)
        desired: dict[str, str] = {}
        if _tag_has_layout_guard_class(tag):
            desired["box-sizing"] = "border-box"
            if (
                _parse_css_pixel_value(props.get("height", "")) is not None
                or _parse_css_pixel_value(props.get("max-height", "")) is not None
                or str(props.get("display", "") or "").lower() in {"flex", "grid", "inline-flex", "inline-grid"}
            ):
                desired["max-height"] = f"{_safe_container_max_height_for_tag(tag, props, canvas_height=expected_height)}px"
                if tag.find(["table", "ul", "ol"]) or "overflow" in props or _tag_text_length(tag) > 280:
                    desired["overflow"] = "hidden"
        position = str(props.get("position", "") or "").strip().lower()
        top = _parse_css_pixel_value(props.get("top", ""))
        height = _parse_css_pixel_value(props.get("height", ""))
        if position in {"absolute", "fixed"} and top is not None and height is not None:
            safe_height = max(64, int(expected_height - top - _BOTTOM_SAFE_ZONE_MIN_PX - 8))
            if height > safe_height:
                desired["height"] = f"{safe_height}px"
                desired["max-height"] = f"{safe_height}px"
                desired["overflow"] = "hidden"
                desired["box-sizing"] = "border-box"
        if desired:
            _set_tag_inline_style_props(
                tag,
                desired,
                report=repair_report,
                reason="constrain layout container inside slide canvas",
            )

    for table in body.find_all("table"):
        parent = table.parent if isinstance(table.parent, Tag) else None
        repair_tag = parent if isinstance(parent, Tag) and parent.name.lower() not in {"body", "html"} else table
        rows = table.find_all("tr")
        if len(rows) >= 10 or any(str(code.get("code", "")) == "table_height_risk" for code in diagnostics or []):
            _set_tag_inline_style_props(
                repair_tag,
                {
                    "max-height": f"{max(120, expected_height - 180)}px",
                    "overflow": "hidden",
                    "box-sizing": "border-box",
                },
                report=repair_report,
                reason="constrain tall table region",
            )
            break

    return str(soup), repair_report


def _extract_fallback_title(body: Tag, soup: BeautifulSoup) -> str:
    for tag in body.find_all(["h1", "h2"]):
        text = _text_preview(tag.get_text(" ", strip=True), limit=110)
        if text:
            return text
    if isinstance(soup.title, Tag):
        text = _text_preview(soup.title.get_text(" ", strip=True), limit=110)
        if text:
            return text
    for tag in body.find_all(True):
        if tag.name.lower() in {"style", "script"}:
            continue
        text = _text_preview(tag.get_text(" ", strip=True), limit=110)
        if text:
            return text
    return "Slide Summary"


def _extract_fallback_text_items(body: Tag, title: str, *, limit: int = 8) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for tag in body.find_all(["h2", "h3", "p", "li", "blockquote", "figcaption"]):
        text = _text_preview(tag.get_text(" ", strip=True), limit=170)
        normalized = _normalized_preview_match_text(text)
        if not normalized or normalized == _normalized_preview_match_text(title) or normalized in seen:
            continue
        seen.add(normalized)
        items.append(text)
        if len(items) >= limit:
            break
    if items:
        return items
    full_text = _text_preview(body.get_text(" ", strip=True), limit=900)
    for chunk in re.split(r"(?<=[.!?。！？；;])\s+", full_text):
        text = _text_preview(chunk, limit=160)
        normalized = _normalized_preview_match_text(text)
        if normalized and normalized not in seen and normalized != _normalized_preview_match_text(title):
            seen.add(normalized)
            items.append(text)
        if len(items) >= limit:
            break
    return items


def _fallback_image_html(body: Tag) -> str:
    img = body.find("img")
    if not isinstance(img, Tag):
        return ""
    src = str(img.get("src", "") or "").strip()
    if not src:
        return ""
    alt = str(img.get("alt", "") or "").strip()
    return (
        '<figure class="fallback-figure">'
        f'<img src="{html_escape(src, quote=True)}" alt="{html_escape(alt, quote=True)}">'
        + (f"<figcaption>{html_escape(_text_preview(alt, limit=120))}</figcaption>" if alt else "")
        + "</figure>"
    )


def _fallback_table_html(body: Tag) -> str:
    table = body.find("table")
    if not isinstance(table, Tag):
        return ""
    table_copy = BeautifulSoup(str(table), "lxml").find("table")
    if not isinstance(table_copy, Tag):
        return ""
    for tag in table_copy.find_all(True):
        if tag.name.lower() in {"script", "style"}:
            tag.decompose()
            continue
        if tag.has_attr("style"):
            del tag["style"]
    return f'<div class="fallback-table">{str(table_copy)}</div>'


def build_safe_fallback_slide(
    html: str,
    diagnostics: list[dict[str, Any]] | None = None,
    aspect_ratio: str = "16:9",
) -> str:
    """Build a deterministic low-risk slide preserving title, text, image, and table."""
    expected_width, expected_height = _expected_layout_size(aspect_ratio)
    try:
        soup = BeautifulSoup(str(html or ""), "lxml")
        body = _html_body_root_from_soup(soup)
    except Exception:
        body = None
        soup = BeautifulSoup("", "lxml")
    if not isinstance(body, Tag):
        body = soup.new_tag("body")
    style = _fallback_style_context_from_html(str(html or ""))
    title = _extract_fallback_title(body, soup)
    text_items = _extract_fallback_text_items(body, title)
    image_html = _fallback_image_html(body)
    table_html = _fallback_table_html(body)
    has_media = bool(image_html or table_html)
    bullets = "\n".join(
        f"        <li>{html_escape(item)}</li>" for item in text_items[:8]
    ) or "        <li>Key content preserved from the original slide.</li>"
    diagnostics_codes = ", ".join(
        _unique_preserve_order([
            str(item.get("code", "") or "")
            for item in diagnostics or []
            if str(item.get("code", "") or "")
        ])[:4]
    )
    grid_columns = "1.05fr 0.95fr" if has_media else "1fr"
    media_block = ""
    if has_media:
        media_block = (
            '      <aside class="media-panel">\n'
            f"{image_html}\n"
            f"{table_html}\n"
            "      </aside>"
        )

    fallback_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      width: {expected_width}px;
      height: {expected_height}px;
      margin: 0;
      padding: 0;
      overflow: hidden;
      box-sizing: border-box;
      font-family: Arial, "Noto Sans", "Microsoft YaHei", sans-serif;
      background: {style["background"]};
      color: {style["text"]};
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    .slide {{
      width: {expected_width}px;
      height: {expected_height}px;
      padding: 44px 56px 64px 56px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      gap: 22px;
      background: {style["background"]};
    }}
    h1 {{
      margin: 0;
      font-size: 36px;
      line-height: 1.16;
      letter-spacing: 0;
      color: {style["title"]};
    }}
    .fallback-rule {{
      width: 96px;
      height: 4px;
      flex: 0 0 auto;
      background: {style["accent"]};
      margin-top: -8px;
    }}
    .content-grid {{
      min-height: 0;
      max-height: {max(160, expected_height - 150)}px;
      display: grid;
      grid-template-columns: {grid_columns};
      gap: 24px;
      overflow: hidden;
    }}
    .text-panel, .media-panel {{
      min-height: 0;
      overflow: hidden;
      border: 1px solid {style["border"]};
      background: {style["evidence_bg"]};
      padding: 20px 22px;
    }}
    ul {{
      margin: 0;
      padding-left: 22px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    li {{
      font-size: 21px;
      line-height: 1.32;
    }}
    .media-panel {{
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .fallback-figure {{
      margin: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .fallback-figure img {{
      display: block;
      width: 100%;
      max-height: {max(120, int(expected_height * 0.34))}px;
      object-fit: contain;
    }}
    figcaption {{
      font-size: 13px;
      line-height: 1.25;
      color: {style["muted"]};
    }}
    .fallback-table {{
      max-height: {max(120, int(expected_height * 0.34))}px;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 13px;
      line-height: 1.22;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid {style["border"]};
      vertical-align: top;
    }}
    th {{
      font-weight: 700;
      color: {style["title"]};
      background: {style["evidence_bg"]};
    }}
  </style>
</head>
<body>
  <!-- memslides-deterministic-layout-fallback: {html_escape(diagnostics_codes)} -->
  <div class="slide">
    <h1>{html_escape(title)}</h1>
    <div class="fallback-rule"></div>
    <div class="content-grid">
      <main class="text-panel">
        <ul>
{bullets}
        </ul>
      </main>
{media_block}
    </div>
  </div>
</body>
</html>
"""
    normalized, _report = _normalize_anchored_elements_html(
        fallback_html,
        contracts=_anchored_contracts_from_html_text(html),
    )
    return normalized


async def _try_deterministic_fallback_inspect_repair(
    html_path: Path,
    *,
    html_file: str,
    current_html: str,
    diagnostics: list[dict[str, Any]],
    aspect_ratio: str,
    original_error: str,
) -> tuple[bool, str, list[dict[str, Any]]]:
    fallback_html = build_safe_fallback_slide(
        current_html,
        diagnostics,
        aspect_ratio=aspect_ratio,
    )
    fallback_html, _anchor_report = _normalize_anchored_elements_html(fallback_html, path=html_path)
    fallback_diagnostics = validate_slide_layout_static(
        fallback_html,
        aspect_ratio=aspect_ratio,
        path=html_path,
        source="fallback",
    )
    fallback_html, auto_repairs = repair_slide_layout_static(
        fallback_html,
        fallback_diagnostics,
        aspect_ratio=aspect_ratio,
    )
    fallback_diagnostics = validate_slide_layout_static(
        fallback_html,
        aspect_ratio=aspect_ratio,
        path=html_path,
        source="fallback",
    )
    fallback_visual_diagnostics, _fallback_visual_metrics = _collect_visual_quality_diagnostics(html_path, fallback_html)
    fallback_diagnostics = _merge_validation_diagnostics(fallback_diagnostics, fallback_visual_diagnostics)
    fallback_blockers = _layout_static_blockers(fallback_diagnostics)
    fallback_blockers.extend([item for item in fallback_visual_diagnostics if item.get("severity") == "error"])
    if fallback_blockers:
        return (
            False,
            "deterministic fallback skipped because static blockers remain",
            fallback_diagnostics,
        )

    try:
        html_path.write_text(fallback_html, encoding="utf-8")
        _invalidate_slide_snapshots_for_path(html_path)
        _invalidate_slide_validation_for_path(html_path)
        _mark_slide_modified(html_path)
        _remember_slide_observation(html_path)
        await convert_html_to_pptx_with_retry(
            html_path,
            aspect_ratio=aspect_ratio,
            allow_skip_layout_validation_fallback=False,
            preserve_source_html=True,
        )
    except Exception as fallback_error:
        fallback_error_str = str(fallback_error)
        fallback_export_diagnostics = [
            item
            for item in getattr(fallback_error, "pptx_export_diagnostics", []) or []
            if isinstance(item, dict)
        ]
        merged = _merge_validation_diagnostics(fallback_diagnostics, fallback_export_diagnostics)
        _record_slide_validation_result(
            html_path,
            success=False,
            message=fallback_error_str.splitlines()[0][:240],
            aspect_ratio=aspect_ratio,
            diagnostics=merged,
        )
        return (
            False,
            f"deterministic fallback failed: {fallback_error}",
            merged,
        )

    _reset_slide_render_failure_count(html_path)
    _record_slide_validation_result(
        html_path,
        success=True,
        message="deterministic fallback layout repair passed pptx_export",
        aspect_ratio=aspect_ratio,
        diagnostics=fallback_diagnostics,
    )
    try:
        from memslides.runtime.deck_execution_state import record_slide_inspected

        record_slide_inspected(html_path, success=True)
    except Exception:
        pass

    note = (
        "✅ Slide rendered successfully after deterministic fallback repair (HTML→PPTX OK).\n"
        f"- Original render error: {original_error.splitlines()[0][:220]}\n"
        "- The tool rewrote this page into a safe fixed-canvas layout preserving title, core text, image/table content when present.\n"
        f"- File updated: {html_file}\n"
        "- Run `inspect_slide` once more if you need the visual snapshot summary."
    )
    if auto_repairs:
        note += "\nauto_repairs=" + json.dumps(auto_repairs[:6], ensure_ascii=False)
    return True, note, fallback_diagnostics


def _find_nested_slide_root_issue(root: Tag | BeautifulSoup) -> str | None:
    slide_tags = [
        tag for tag in root.find_all(True)
        if _tag_has_any_class(tag, {"slide"})
    ]
    for tag in slide_tags:
        nested = tag.find(
            lambda child: (
                isinstance(child, Tag)
                and child is not tag
                and _tag_has_any_class(child, {"slide"})
            )
        )
        if nested is not None:
            return (
                "检测到嵌套 `.slide` 容器（nested `.slide` root）："
                f"`{_selector_hint_for_tag(tag)}` 内又包含 `{_selector_hint_for_tag(nested)}`。"
            )
    return None


def _find_invalid_list_structure_issue(root: Tag | BeautifulSoup) -> str | None:
    for list_tag in root.find_all(["ul", "ol"]):
        invalid_children = [
            child
            for child in list_tag.children
            if isinstance(child, Tag) and child.name.lower() != "li"
        ]
        if invalid_children:
            child_labels = ", ".join(
                f"<{child.name.lower()}>"
                for child in invalid_children[:3]
            )
            return (
                "检测到非法列表结构（invalid list structure）："
                f"`{_selector_hint_for_tag(list_tag)}` 的直接子节点包含非 `<li>` 元素（{child_labels}）。"
            )
    return None


def _find_suspicious_slide_geometry_issue(html_text: str) -> str | None:
    soup = BeautifulSoup(html_text, "lxml")
    body = _html_body_root_from_soup(soup)
    if not isinstance(body, Tag):
        return None

    css_rules = _parse_css_rules(_extract_all_style_text(html_text))
    body_props = _collect_tag_style_props(css_rules, body)
    body_width = _parse_css_pixel_value(body_props.get("width", ""))
    body_height = _parse_css_pixel_value(body_props.get("height", ""))
    if body_width is None and body_height is None:
        return None

    slide_roots = []
    for tag in body.find_all(True):
        if not _tag_has_any_class(tag, {"slide"}):
            continue
        has_slide_ancestor = any(
            isinstance(parent, Tag) and _tag_has_any_class(parent, {"slide"})
            for parent in tag.parents
        )
        if not has_slide_ancestor:
            slide_roots.append(tag)

    for tag in slide_roots:
        slide_props = _collect_tag_style_props(css_rules, tag)
        slide_width = _parse_css_pixel_value(
            slide_props.get("width", "") or slide_props.get("max-width", "")
        )
        slide_height = _parse_css_pixel_value(
            slide_props.get("height", "") or slide_props.get("max-height", "")
        )
        width_suspicious = (
            body_width is not None
            and slide_width is not None
            and slide_width > 0
            and slide_width < body_width * 0.5
        )
        height_suspicious = (
            body_height is not None
            and slide_height is not None
            and slide_height > 0
            and slide_height < body_height * 0.5
        )
        if width_suspicious or height_suspicious:
            return (
                "检测到可疑的 `.slide` 画布尺寸（suspicious slide geometry）："
                f"`{_selector_hint_for_tag(tag)}` 为 {_format_px_pair(slide_width, slide_height)}，"
                f"但 `body` 为 {_format_px_pair(body_width, body_height)}。"
                "这通常表示 `.slide` 画布被误缩放了。"
            )
    return None


def _collect_slide_structure_issues(
    html_text: str,
    *,
    include_geometry: bool = True,
) -> list[str]:
    try:
        soup = BeautifulSoup(html_text, "lxml")
    except Exception:
        return []

    body = _html_body_root_from_soup(soup)
    issues: list[str] = []
    nested_issue = _find_nested_slide_root_issue(body)
    if nested_issue:
        issues.append(nested_issue)
    list_issue = _find_invalid_list_structure_issue(body)
    if list_issue:
        issues.append(list_issue)
    if include_geometry:
        geometry_issue = _find_suspicious_slide_geometry_issue(html_text)
        if geometry_issue:
            issues.append(geometry_issue)
    return issues


def _fragment_validation_error(fragment: str) -> str | None:
    lowered = str(fragment or "").lower()
    if re.search(r"<\s*(html|head|body|style)\b", lowered, re.IGNORECASE):
        return "HTML fragments must not include <html>, <head>, <body>, or <style>."
    fragment_wrapper = f"<html><body>{str(fragment or '')}</body></html>"
    fragment_soup = BeautifulSoup(fragment_wrapper, "lxml")
    fragment_body = _html_body_root_from_soup(fragment_soup)
    if isinstance(fragment_body, Tag):
        contains_slide_root = any(
            _tag_has_any_class(tag, {"slide"})
            for tag in fragment_body.find_all(True)
        )
        if contains_slide_root:
            return (
                "HTML fragment must not contain another `.slide` root container. "
                "Patch only the inside of the selected target node."
            )
    fragment_issues = _collect_slide_structure_issues(
        fragment_wrapper,
        include_geometry=False,
    )
    if fragment_issues:
        first_issue = fragment_issues[0]
        if "nested `.slide` root" in first_issue:
            return (
                "HTML fragment must not contain another `.slide` root container. "
                "Patch only the inside of the selected target node."
            )
        return first_issue
    fragment_corruption = _detect_corrupted_slide_html(
        Path("__fragment__.html"),
        fragment_wrapper,
    )
    if fragment_corruption:
        truncated_marker = str(fragment_corruption.get("detected_marker", "") or "placeholder HTML")
        return (
            "HTML fragment appears truncated or placeholder-based "
            f"(found marker: {truncated_marker})."
        )
    return None


def _fragment_nodes(html_fragment: str) -> list[Any]:
    fragment_soup = BeautifulSoup(html_fragment, "lxml")
    container = fragment_soup.body if isinstance(fragment_soup.body, Tag) else fragment_soup
    return list(container.contents)


def _append_fragment_into(tag: Tag, html_fragment: str) -> None:
    for node in _fragment_nodes(html_fragment):
        tag.append(node)


def _insert_fragment_near(anchor: Tag, html_fragment: str, position: str) -> None:
    nodes = _fragment_nodes(html_fragment)
    if position == "before":
        for node in nodes:
            anchor.insert_before(node)
        return
    if position == "after":
        for node in reversed(nodes):
            anchor.insert_after(node)
        return
    if position == "prepend":
        for node in reversed(nodes):
            anchor.insert(0, node)
        return
    for node in nodes:
        anchor.append(node)


def _build_slide_index_entry(html_path: Path, workspace_root: Path) -> dict[str, object]:
    """Build a lightweight structure summary for one slide."""
    content = html_path.read_text(encoding="utf-8", errors="replace")
    parser = _SlideHTMLParser()
    parser.feed(content)

    title = ""
    for cls, text in parser._text_chunks:
        if "title" in cls.lower():
            title = text
            break
    if not title:
        h1_match = re.search(r"<h1\b[^>]*>(.*?)</h1>", content, re.IGNORECASE | re.DOTALL)
        if h1_match:
            title = re.sub(r"<[^>]+>", "", h1_match.group(1)).strip()
    if not title:
        title_match = re.search(r"<title\b[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
    if not title and parser._text_chunks:
        title = parser._text_chunks[0][1]

    title_selector_hint = ""
    if re.search(r"<h1\b", content, re.IGNORECASE):
        title_selector_hint = "h1"
    elif re.search(r'class=["\'][^"\']*(?:^|\s)title(?:\s|$)[^"\']*["\']', content, re.IGNORECASE):
        title_selector_hint = ".title"
    elif re.search(r"<h2\b", content, re.IGNORECASE):
        title_selector_hint = "h2"

    semantic_targets = {
        "slide_title": _infer_semantic_selectors(content, "slide_title"),
        "body_text": _infer_semantic_selectors(content, "body_text"),
        "footer": _infer_semantic_selectors(content, "footer"),
    }

    try:
        rel_path = html_path.resolve().relative_to(workspace_root.resolve())
        file_label = rel_path.as_posix()
    except Exception:
        file_label = html_path.as_posix()

    return {
        "file_path": file_label,
        "title": title[:160],
        "title_selector_hint": title_selector_hint,
        "semantic_targets": semantic_targets,
        "has_style_block": "<style" in content.lower(),
        "h1_count": len(re.findall(r"<h1\b", content, re.IGNORECASE)),
        "image_count": len(parser._image_srcs),
    }


def _extract_slide_properties(html_path: Path) -> str:
    """Extract key visual properties from a slide HTML file.

    Returns a human-readable summary of title, fonts, colors, and referenced images.
    """
    content = html_path.read_text(encoding="utf-8")
    parser = _SlideHTMLParser()
    parser.feed(content)

    css_rules = _parse_css_rules(parser._style_text)

    lines = []

    # 1. Title detection — prefer semantic title tags, then fall back to parser heuristics
    title_text = ""
    title_props: dict[str, str] = {}
    try:
        soup = BeautifulSoup(content, "lxml")
        body = soup.body if isinstance(soup.body, Tag) else soup
        semantic_title_tags = _semantic_target_tags(body, "slide_title")
        if semantic_title_tags:
            title_tag = semantic_title_tags[0]
            title_text = title_tag.get_text(" ", strip=True)
            title_props = _collect_tag_style_props(css_rules, title_tag)
    except Exception:
        title_text = ""
        title_props = {}

    if not title_text:
        for cls, text in parser._text_chunks:
            if "title" in cls.lower():
                title_text = text
                break
    if not title_text and parser._text_chunks:
        title_text = parser._text_chunks[0][1]

    if not title_props:
        for selector, props in css_rules.items():
            if "title" in selector.lower() or selector.strip() in (".title", "h1", "h2"):
                title_props = props
                break

    if title_text:
        parts = [f'"{title_text[:60]}"']
        if "font-family" in title_props:
            parts.append(f"font: {title_props['font-family']}")
        if "font-size" in title_props:
            parts.append(f"size: {title_props['font-size']}")
        if "color" in title_props:
            parts.append(f"color: {title_props['color']}")
        if "font-weight" in title_props:
            parts.append(f"weight: {title_props['font-weight']}")
        lines.append(f"  Title: {' | '.join(parts)}")

    # 2. Body/content properties
    body_props = css_rules.get("body", {})
    content_props = {}
    for selector, props in css_rules.items():
        if "content" in selector.lower() or "body" in selector.lower():
            content_props.update(props)
    if body_props:
        bg = body_props.get("background", body_props.get("background-color", ""))
        font = body_props.get("font-family", "")
        color = body_props.get("color", "")
        body_parts = []
        if bg:
            body_parts.append(f"background: {bg}")
        if font:
            body_parts.append(f"font: {font}")
        if color:
            body_parts.append(f"color: {color}")
        if body_parts:
            lines.append(f"  Body: {' | '.join(body_parts)}")

    # 3. Content element count
    body_texts = [t for cls, t in parser._text_chunks if "title" not in cls.lower()]
    if body_texts:
        lines.append(f"  Content elements: {len(body_texts)} text blocks")

    # 4. Image references
    if parser._image_srcs:
        found = 0
        missing = []
        for src in parser._image_srcs:
            if _resolve_local_html_asset_path(src, html_path=html_path) is not None:
                found += 1
            else:
                missing.append(src)
        lines.append(f"  Images: {found} found, {len(missing)} missing")
        if missing:
            lines.append(f"  Missing: {', '.join(missing[:3])}")
    else:
        lines.append("  Images: none referenced")

    return "\n".join(lines) if lines else "  (no properties extracted)"


CONFIG = MemSlidesConfig.load_from_file(os.getenv("MEMSLIDES_CONFIG_FILE"))


def _default_model_ref_for_agent(agent_name: str) -> str:
    return {
        "Researcher": "research_agent",
        "RevisionEditor": "modify_agent",
        "DeckDesigner": "design_agent",
        "TemplatePlanner": "design_agent",
    }.get(agent_name, "design_agent")


def _get_current_agent_llm_config() -> tuple[str, object]:
    """Resolve the actual LLM config serving the current agent."""
    agent_name = get_current_agent()
    model_ref = get_current_agent_model_ref() or _default_model_ref_for_agent(
        agent_name
    )
    try:
        return CONFIG.resolve_llm(model_ref)
    except KeyError:
        warning(
            f"Failed to resolve model_ref '{model_ref}' for agent '{agent_name}', "
            "falling back to design_agent"
        )
        return CONFIG.resolve_llm("design_agent")


def _resolve_local_markdown_asset_path(
    raw_path: str,
    *,
    md_dir: Path,
) -> Path | None:
    """Resolve local markdown asset paths against the workspace and manuscript dir.

    LLMs often emit image paths such as `workspace/outputs/foo.png` or
    `/workspace/outputs/foo.png`. We normalize those pseudo-workspace prefixes
    and search both the markdown directory and the actual workspace root.
    """
    text = str(raw_path or "").strip().strip("\"'")
    if not text or re.match(r"^(?:https?://|data:)", text, re.IGNORECASE):
        return None

    workspace_root = _resolve_workspace_root(anchor=md_dir)
    variants: list[str] = [text]

    normalized = text
    if normalized.startswith("/workspace/"):
        normalized = normalized[len("/workspace/") :]
    elif normalized == "/workspace":
        normalized = ""
    elif normalized.startswith("/workspace"):
        normalized = normalized[len("/workspace") :].lstrip("/")
    elif normalized.startswith("workspace/"):
        normalized = normalized[len("workspace/") :]
    elif normalized == "workspace":
        normalized = ""
    if normalized and normalized not in variants:
        variants.append(normalized)

    for variant in variants:
        if not variant:
            continue
        candidate = Path(variant)
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            continue

        for base_dir in (md_dir, workspace_root, Path.cwd()):
            resolved = (base_dir / candidate).resolve()
            if resolved.exists():
                return resolved
    return None


def _rewrite_image_link(match: re.Match[str], md_dir: Path) -> str:
    alt_text = match.group(1)
    target = match.group(2).strip()
    if not target:
        return match.group(0)
    parts = re.match(r"([^\s]+)(.*)", target)
    if not parts:
        return match.group(0)
    local_path = parts.group(1).strip("\"'")
    rest = parts.group(2)
    p = _resolve_local_markdown_asset_path(local_path, md_dir=md_dir)
    if p is None:
        return match.group(0)

    updated_alt = alt_text
    try:
        with Image.open(p) as img:
            width, height = img.size
        if width > 0 and height > 0 and not re.search(r"\b\d+:\d+\b", updated_alt):
            factor = math.gcd(width, height)
            ratio = f"{width // factor}:{height // factor}"
            updated_alt = f"{updated_alt}, {ratio}" if updated_alt else ratio
    except OSError:
        pass

    # ? since slides were placed in an independent folder, we convert image path to absolute path to avoid broken links
    new_path = p.resolve().as_posix()
    return f"![{updated_alt}]({new_path}{rest})"


def _format_design_finalize_validation_error(path: Path, *, agent_name: str = "") -> str | None:
    html_files = sorted(path.glob("*.html"))
    if not html_files:
        return "Outcome path should be a directory containing HTML files"

    if agent_name == "RevisionEditor":
        modified_html_files = _iter_modified_slides_within(path)
        if modified_html_files:
            html_files = modified_html_files
        else:
            return None

    blocking: list[str] = []
    if agent_name == "DeckDesigner":
        try:
            from memslides.runtime.deck_execution_state import deck_progress_summary

            progress = deck_progress_summary()
            if progress.get("active") and progress.get("expected_slide_count"):
                expected_files: list[Path] = []
                for slide in progress.get("slides", []) or []:
                    rel = str(slide.get("file", "") or "").strip()
                    if not rel:
                        continue
                    expected_files.append((Path.cwd() / rel).resolve())
                if expected_files:
                    html_files = sorted(set(html_files) | {p for p in expected_files if p.parent == path.resolve()})
                    for expected in expected_files:
                        if expected.parent != path.resolve():
                            continue
                        if not expected.exists():
                            blocking.append(f"{expected.name} (expected slide file is missing)")
        except Exception:
            pass

    for html_file in html_files:
        try:
            html_text = html_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            html_text = ""
        diagram_contract = _active_diagram_contract_for_path(html_file)
        if diagram_contract:
            diagram_errors = [
                item
                for item in validate_diagram_layout_static(html_text, diagram_contract)
                if item.get("severity") == "error"
            ]
            if diagram_errors:
                blocking.append(
                    f"{html_file.name} (diagram QA failed: "
                    + "; ".join(str(item.get("code", "")) for item in diagram_errors[:3])
                    + ")"
                )
                continue
        qa_diagnostics, _qa_metrics = _collect_visual_quality_diagnostics(html_file, html_text)
        if _strict_visual_preference_eval_enabled():
            for item in qa_diagnostics:
                if item.get("source") == "soft_visual_preference" and item.get("severity") == "warning":
                    item["severity"] = "error"
        qa_errors = [item for item in qa_diagnostics if item.get("severity") == "error"]
        if qa_errors:
            blocking.append(
                f"{html_file.name} (visual QA failed: "
                + "; ".join(str(item.get("code", "")) for item in qa_errors[:3])
                + ")"
            )
            continue
        validation = _current_slide_validation_state(html_file)
        if validation is None:
            blocking.append(
                f"{html_file.name} (never passed `inspect_slide` on the current HTML)"
            )
            continue
        if not bool(validation.get("success")):
            message = str(validation.get("message", "") or "").strip()
            if message:
                blocking.append(f"{html_file.name} (last inspect failed: {message})")
            else:
                blocking.append(f"{html_file.name} (last inspect failed)")

    if not blocking:
        return None

    preview = "; ".join(blocking[:3])
    if len(blocking) > 3:
        preview += f"; and {len(blocking) - 3} more"
    if agent_name == "RevisionEditor":
        return (
            "Error: RevisionEditor outcome cannot be finalized until every modified slide passes "
            "`inspect_slide` on its current on-disk HTML. "
            f"Pending slides: {preview}. Re-run `inspect_slide`, fix any reported issues, "
            "and only then call `finalize`."
        )

    return (
        "Error: DeckDesigner outcome cannot be finalized until every slide passes "
        "`inspect_slide` on its current on-disk HTML. "
        f"Pending slides: {preview}. Re-run `inspect_slide`, fix any reported issues, "
        "and only then call `finalize`."
    )


class Todo(BaseModel):
    id: str
    content: str
    status: Literal["pending", "in_progress", "completed", "skipped"]


LOCAL_TODO_CSV_PATH = Path("todo.csv")
LOCAL_TODO_LOCK_PATH = Path(".todo.csv.lock")


def _load_todos() -> list[Todo]:
    """Load todos from CSV file."""
    if not LOCAL_TODO_CSV_PATH.exists():
        return []

    lock = FileLock(LOCAL_TODO_LOCK_PATH)
    with lock:
        with open(LOCAL_TODO_CSV_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [Todo(**row) for row in reader]


def _save_todos(todos: list[Todo]) -> None:
    """Save todos to CSV file."""
    lock = FileLock(LOCAL_TODO_LOCK_PATH)
    with lock:
        with open(LOCAL_TODO_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            if todos:
                fieldnames = ["id", "content", "status"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for todo in todos:
                    writer.writerow(todo.model_dump())


@mcp.tool()
def todo_create(todo_content: str) -> str:
    """
    Create a new todo item for TASK PLANNING ONLY.

    ⚠️ USAGE RESTRICTIONS:
    - ONLY use to list slides to be processed or track completion progress
    - NEVER store "waiting for user reply" or questions — todos are INVISIBLE to users
    - NEVER use as a substitute for asking the user a question
    - If you need user confirmation, use text response instead

    Args:
        todo_content (str): Round planning content (e.g., "Process slide_01.html")

    Returns:
        str: Confirmation message with the created todo's ID
    """
    todos = _load_todos()
    new_id = str(len(todos))
    new_todo = Todo(id=new_id, content=todo_content, status="pending")
    todos.append(new_todo)
    _save_todos(todos)
    return f"Todo {new_id} created"


@mcp.tool()
def todo_update(
    idx: int,
    todo_content: str = None,
    status: Literal["completed", "in_progress", "skipped"] = None,
) -> str:
    """
    Update an existing todo item's content or status (ROUND PLANNING ONLY).

    ⚠️ USAGE RESTRICTIONS: Same as todo_create — ONLY for tracking round progress,
    NEVER for user interaction or storing questions.

    Args:
        idx (int): The index of the todo item to update
        todo_content (str, optional): New round planning content
        status (Literal["completed", "in_progress", "skipped"], optional): New status

    Returns:
        str: Confirmation message with the updated todo's ID
    """
    todos = _load_todos()
    if idx < 0 or idx >= len(todos):
        return f"Invalid todo index: {idx}"

    if todo_content is not None:
        todos[idx].content = todo_content
    if status is not None:
        todos[idx].status = status
    _save_todos(todos)
    return "Todo updated successfully"


@mcp.tool()
def todo_list() -> str | list[Todo]:
    """
    Get the current todo list or check if all todos are completed.

    Returns:
        str | list[Todo]: Either a completion message if all todos are done/skipped,
                         or the current list of todo items
    """
    todos = _load_todos()
    if not todos or all(todo.status in ["completed", "skipped"] for todo in todos):
        LOCAL_TODO_CSV_PATH.unlink(missing_ok=True)
        return "All todos completed"
    else:
        return todos


# @mcp.tool()
def ask_user(question: str) -> str:
    """
    Ask the user a question when encounters an unclear requirement.
    """
    print(f"User input required: {question}")
    return input("Your answer: ")


# Issue 4: 合法的 workspace 根目录 markdown 文件名（小写匹配）
_ALLOWED_ROOT_MD = {
    "design_plan.md", "manuscript.md", "readme.md", "index.md",
}
# Issue 4: Agent 常自创的状态总结文件关键词
_STATE_CLUTTER_KEYWORDS = {
    "state_summary", "state summary", "continuation", "seamless",
    "canonical", "working_state", "tool_state", "project_state",
    "session_state", "context_state", "next_state",
}
_DESIGN_PLAN_CANDIDATES = (
    "design_plan.md",
    "design-plan.md",
    "outputs/design_plan.md",
    "outputs/design-plan.md",
)


@mcp.tool()
def write_markdown_file(file_path: str, content: str) -> str:
    """
    Write markdown content to a file.

    In Researcher/TemplatePlanner stages, this is commonly used for manuscripts.
    In DeckDesigner stage, markdown writes are reserved for `design_plan.md` only.

    Args:
        file_path: The absolute path where the markdown file should be saved (must end with .md)
        content: The markdown content to write to the file

    Returns:
        Success message with the file path, or error message if failed
    """
    try:
        path = _normalize_path(file_path)

        # Validate file extension
        if not path.suffix == ".md":
            return f"Error: file_path must end with .md extension, got: {path.suffix}"

        design_stage_target_error = _design_stage_markdown_target_error(path)
        if design_stage_target_error:
            return design_stage_target_error

        # Issue 4: 检测 agent 自创的状态总结文件，重定向到 .history/agent_notes/
        _fname_lower = path.name.lower()
        _is_root_level = (path.parent == _normalize_path(".") or
                          path.parent.name in ("", "."))
        if _is_root_level and _fname_lower not in _ALLOWED_ROOT_MD:
            _is_clutter = any(kw in _fname_lower for kw in _STATE_CLUTTER_KEYWORDS)
            if _is_clutter:
                _notes_dir = path.parent / ".history" / "agent_notes"
                _notes_dir.mkdir(parents=True, exist_ok=True)
                path = _notes_dir / path.name
                info(f"Redirected agent state note to {path}")

        if _is_design_plan_target(path):
            validation = _validate_design_plan_markdown(content)
            if not validation.get("valid"):
                return (
                    "Error: `design_plan.md` must be a structured design spec before it can be saved "
                    f"({_format_design_plan_validation_error(validation)})."
                )

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write content to file
        path.write_text(content, encoding="utf-8")
        if _is_design_plan_target(path):
            _record_design_plan_write(path, content)

        # Get file size for confirmation
        file_size = path.stat().st_size
        line_count = content.count('\n') + 1

        info(f"Markdown file written: {path} ({file_size} bytes, {line_count} lines)")
        return f"Successfully wrote markdown file to: {path}\nFile size: {file_size} bytes\nLine count: {line_count}"

    except Exception as e:
        error(f"Failed to write markdown file {file_path}: {e}")
        return f"Error writing file: {str(e)}"


def _infer_layout_from_html(html_content: str) -> str:
    """Stage 7: 从 HTML 注释中推断布局名称"""
    match = re.search(r'<!--\s*template-layout:\s*(.+?)\s*-->', html_content)
    return match.group(1).strip() if match else ""


def _prepare_html_layout_for_write(
    html_content: str,
    path: Path,
    *,
    force_regenerate: bool = False,
    aspect_ratio: str | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ratio = aspect_ratio or _infer_static_layout_aspect_ratio(html_content, default="16:9")
    diagnostics_before = validate_slide_layout_static(
        html_content,
        aspect_ratio=ratio,
        path=path,
        source="write",
    )
    repairable_codes = {
        "canvas_dimension_missing",
        "canvas_dimension_mismatch",
        "slide_root_dimension_risk",
        "body_padding_without_border_box",
        "container_padding_without_border_box",
        "bottom_safe_zone_violation",
        "off_canvas_absolute",
        "image_overflow",
        "table_height_risk",
        "text_overflow",
        "clipped_outside_canvas",
    }
    should_repair = any(str(item.get("code", "") or "") in repairable_codes for item in diagnostics_before)
    repaired_content = html_content
    auto_repairs: list[dict[str, Any]] = []
    if should_repair:
        repaired_content, auto_repairs = repair_slide_layout_static(
            html_content,
            diagnostics_before,
            aspect_ratio=ratio,
        )
    diagnostics_after = validate_slide_layout_static(
        repaired_content,
        aspect_ratio=ratio,
        path=path,
        source="write",
    )
    remaining_blockers = _layout_static_blockers(diagnostics_after)
    if force_regenerate:
        remaining_blockers = []
    return repaired_content, diagnostics_after, auto_repairs, remaining_blockers


def _format_layout_validation_note(
    diagnostics: list[dict[str, Any]],
    auto_repairs: list[dict[str, Any]],
    remaining_blockers: list[dict[str, Any]],
    *,
    max_items: int = 6,
) -> str:
    if not diagnostics and not auto_repairs and not remaining_blockers:
        return ""
    compact_diagnostics = [
        {
            "code": item.get("code", ""),
            "severity": item.get("severity", ""),
            "selector": item.get("target_selector_hint", ""),
            "repair_strategy": item.get("repair_strategy", ""),
        }
        for item in diagnostics[:max_items]
    ]
    compact_repairs = [
        {
            "type": item.get("type", ""),
            "selector": item.get("selector", ""),
            "reason": item.get("reason", ""),
            "declarations": item.get("declarations", {}),
        }
        for item in auto_repairs[:max_items]
    ]
    compact_blockers = [
        {
            "code": item.get("code", ""),
            "selector": item.get("target_selector_hint", ""),
            "message": item.get("message", ""),
            "repair_strategy": item.get("repair_strategy", ""),
        }
        for item in remaining_blockers[:max_items]
    ]
    return (
        "\nlayout_diagnostics="
        + json.dumps(compact_diagnostics, ensure_ascii=False)
        + "\nauto_repairs="
        + json.dumps(compact_repairs, ensure_ascii=False)
        + "\nremaining_blockers="
        + json.dumps(compact_blockers, ensure_ascii=False)
    )


def _format_diagram_validation_note(diagnostics: list[dict[str, Any]], *, max_items: int = 6) -> str:
    if not diagnostics:
        return "\ndiagram_diagnostics=[]"
    compact = [
        {
            "code": item.get("code", ""),
            "severity": item.get("severity", ""),
            "message": item.get("message", ""),
        }
        for item in diagnostics[:max_items]
    ]
    return "\ndiagram_diagnostics=" + json.dumps(compact, ensure_ascii=False)


async def _inspect_diagram_fallback_once(
    html_path: Path,
    *,
    html_file: str,
    current_html: str,
    diagnostics: list[dict[str, Any]],
    contract: dict[str, Any],
    aspect_ratio: str,
) -> tuple[bool, str, list[dict[str, Any]]]:
    fallback_html = build_safe_fallback_diagram_slide(
        current_html,
        diagnostics,
        contract,
        aspect_ratio=aspect_ratio,
    )
    fallback_html, _anchor_report = _normalize_anchored_elements_html(fallback_html, path=html_path)
    fallback_html, layout_diagnostics, auto_repairs, remaining_blockers = _prepare_html_layout_for_write(
        fallback_html,
        html_path,
        force_regenerate=True,
        aspect_ratio=aspect_ratio,
    )
    visual_diagnostics, _visual_metrics = _collect_visual_quality_diagnostics(html_path, fallback_html)
    visual_errors = [item for item in visual_diagnostics if item.get("severity") == "error"]
    if visual_errors:
        return (
            False,
            "deterministic diagram fallback still violates visual quality requirements",
            _merge_validation_diagnostics(layout_diagnostics, visual_diagnostics),
        )
    if remaining_blockers:
        return (
            False,
            "deterministic diagram fallback skipped because static layout blockers remain",
            layout_diagnostics,
        )
    fallback_diagram_diagnostics = validate_diagram_layout_static(fallback_html, contract)
    if fallback_diagram_diagnostics:
        return (
            False,
            "deterministic diagram fallback still violates diagram contract",
            fallback_diagram_diagnostics,
        )
    try:
        html_path.write_text(fallback_html, encoding="utf-8")
        _invalidate_slide_snapshots_for_path(html_path)
        _invalidate_slide_validation_for_path(html_path)
        _mark_slide_modified(html_path)
        await convert_html_to_pptx_with_retry(
            html_path,
            aspect_ratio=aspect_ratio,
            allow_skip_layout_validation_fallback=False,
            preserve_source_html=True,
        )
    except Exception as exc:
        export_diagnostics = [
            item
            for item in getattr(exc, "pptx_export_diagnostics", []) or []
            if isinstance(item, dict)
        ]
        return False, f"deterministic diagram fallback failed export: {exc}", export_diagnostics

    merged_diagnostics = _merge_validation_diagnostics(layout_diagnostics, fallback_diagram_diagnostics, visual_diagnostics)
    _record_slide_validation_result(
        html_path,
        success=True,
        message="deterministic diagram fallback passed",
        aspect_ratio=aspect_ratio,
        diagnostics=merged_diagnostics,
    )
    try:
        from memslides.runtime.deck_execution_state import record_slide_inspected

        record_slide_inspected(html_path, success=True)
    except Exception:
        pass
    _remember_slide_observation(html_path)
    _reset_slide_render_failure_count(html_path)
    note = _format_layout_validation_note(layout_diagnostics, auto_repairs, remaining_blockers)
    return (
        True,
        "✅ Slide rendered successfully after deterministic diagram fallback.\n"
        "⚙️ The previous rewrite did not satisfy the flowchart/pipeline contract, so MemSlides replaced it with a safe diagram layout."
        + note
        + _format_diagram_validation_note([]),
        merged_diagnostics,
    )


@mcp.tool()
def write_html_file(
    file_path: str,
    content: str,
    force_regenerate: bool = False,
    expected_hash: str = "",
) -> str:
    """
    Write HTML content to a file.

    For new slides, this remains the primary write path.
    For existing canonical slides (`slide_<n>.html`), the default editing path is
    `read_slide_snapshot` + `apply_slide_patch`. Full-file overwrite is reserved
    for corrupted-slide recovery or slides whose current on-disk HTML has already
    failed `inspect_slide`, and is only allowed when `force_regenerate=true`
    and `expected_hash` matches the current on-disk slide HTML.

    Args:
        file_path: The absolute path where the HTML file should be saved (must end with .html)
        content: The HTML content to write to the file
        force_regenerate: Allow full overwrite of an existing canonical slide only when
                         paired with a matching `expected_hash`
        expected_hash: Current SHA256 hash of the on-disk HTML for controlled regenerate

    Returns:
        Success message with the file path, or error message if failed
    """
    try:
        path = _normalize_path(file_path)
        existed_before = path.exists()

        # Validate file extension
        if not path.suffix == ".html":
            return f"Error: file_path must end with .html extension, got: {path.suffix}"

        workspace_guard_error = _workspace_contract_error(
            action="write_html_file",
            path=path,
            html_content=content,
        )
        if workspace_guard_error:
            return workspace_guard_error

        # Guardrail: files under slides/ must be canonical slide names.
        # This prevents auxiliary names such as slide_09a_hidden.html from
        # leaking into downstream page counting/deletion logic.
        if path.parent.name.lower() == "slides":
            if _canonical_slide_number_from_name(path.name) is None:
                return (
                    "Error: Invalid slide filename under slides/. "
                    f"Expected `slide_<number>.html`, got: {path.name}"
                )

        precondition_error = _design_html_precondition_error()
        if precondition_error:
            warning("Rejected HTML write for %s because design plan is not ready", path)
            return precondition_error
        page_queue_error = _template_page_execution_plan_precondition_error()
        if page_queue_error:
            warning("Rejected HTML write for %s because page_execution_plan.md was not read", path)
            return page_queue_error

        corrupted_content = _detect_corrupted_slide_html(path, content)
        if corrupted_content:
            truncated_marker = str(corrupted_content.get("detected_marker", "") or "placeholder HTML")
            warning(
                "Rejected truncated/placeholder HTML write for %s (marker: %s)",
                path,
                truncated_marker,
            )
            return (
                "Error: HTML content appears truncated or placeholder-based "
                f"(found marker: {truncated_marker}). "
                "Refusing to overwrite the slide. Please regenerate the full "
                "HTML and call write_html_file again."
            )

        canonical_guard_error = _canonical_slide_requires_patch_error(
            path,
            force_regenerate=force_regenerate,
            expected_hash=expected_hash,
        )
        if canonical_guard_error:
            warning("Rejected canonical slide overwrite for %s due to protocol guard", path)
            return canonical_guard_error

        bypass_fresh_read_guard = (
            _is_canonical_slide_path(path)
            and path.exists()
            and str(path) not in _newly_created_slide_paths
        )
        overwrite_error = None if bypass_fresh_read_guard else _overwrite_requires_fresh_read_error(path)
        if overwrite_error:
            warning("Rejected HTML overwrite for %s because no trusted read was recorded", path)
            return overwrite_error

        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure charset meta tag is present for correct CJK rendering
        if '<meta charset' not in content.lower():
            content = content.replace('<html>', '<html>\n<head><meta charset="utf-8"></head>', 1)

        content = _normalize_html_image_sources(content, html_path=path)
        content, marker_report = _normalize_anchored_elements_html(content, path=path)
        diagram_contract = _active_diagram_contract_for_path(path)
        diagram_diagnostics: list[dict[str, Any]] = []
        if diagram_contract:
            diagram_diagnostics = validate_diagram_layout_static(content, diagram_contract)
            if diagram_diagnostics and force_regenerate:
                content = build_safe_fallback_diagram_slide(
                    content,
                    diagram_diagnostics,
                    diagram_contract,
                    aspect_ratio=_infer_static_layout_aspect_ratio(content, default="16:9"),
                )
                content, marker_report = _normalize_anchored_elements_html(content, path=path)
                diagram_diagnostics = validate_diagram_layout_static(content, diagram_contract)
            if diagram_diagnostics:
                return _structured_patch_error(
                    "DIAGRAM_LAYOUT_VALIDATION_FAILED",
                    "Slide does not satisfy the active flowchart/pipeline diagram contract.",
                    file_path=str(path),
                    diagram_diagnostics=diagram_diagnostics,
                    repair_strategy=(
                        "Rewrite the whole target slide as the requested diagram: update the title, "
                        "remove conflicting old figures/captions/empty cards, include all required nodes, "
                        "and make arrows/edges visible."
                    ),
                )
        content, layout_diagnostics, auto_repairs, remaining_blockers = _prepare_html_layout_for_write(
            content,
            path,
            force_regenerate=force_regenerate,
        )
        if remaining_blockers:
            return _structured_patch_error(
                "LAYOUT_VALIDATION_FAILED",
                "Static slide layout validation found blockers that deterministic repair could not safely clear.",
                file_path=str(path),
                layout_diagnostics=layout_diagnostics,
                auto_repairs=auto_repairs,
                remaining_blockers=remaining_blockers,
                repair_strategy=(
                    "Use a fixed-size body and `.slide`, constrain the main content/card/table wrapper "
                    "with max-height/overflow, or regenerate with a simpler two-column/card layout."
                ),
            )
        controlled_context = _active_controlled_rewrite_context_for_path(path)
        if controlled_context:
            visual_diagnostics, _visual_metrics = _collect_controlled_rewrite_visual_diagnostics(
                path,
                content,
                controlled_context,
            )
            visual_errors = [item for item in visual_diagnostics if item.get("severity") == "error"]
            if visual_errors:
                return _structured_patch_error(
                    "CONTROLLED_REWRITE_QA_FAILED",
                    "Controlled rewrite target does not yet satisfy page-level structure QA.",
                    file_path=str(path),
                    visual_diagnostics=visual_diagnostics,
                    repair_strategy=(
                        "Regenerate the same slide with a clearly structured layout: headline plus "
                        "three cards/columns/section blocks, preserve the requested semantic blocks, "
                        "and keep any TEMP-PREF marker to one unobtrusive occurrence."
                    ),
                )
        missing_assets = _missing_local_html_assets(path, content)
        if missing_assets:
            return _format_missing_asset_error(path, missing_assets)

        _maybe_record_persona_repair_backup(path)

        # Write content to file
        path.write_text(content, encoding="utf-8")
        _reset_slide_render_failure_count(path)
        _invalidate_slide_snapshots_for_path(path)
        _invalidate_slide_validation_for_path(path)
        _mark_slide_modified(path)
        try:
            from memslides.runtime.deck_execution_state import record_html_written

            record_html_written(path)
        except Exception:
            pass
        if not existed_before:
            _mark_new_slide_created(path)
        _remember_slide_observation(path)

        # Stage 7: Template compliance check + auto-fix
        compliance_note = ""
        try:
            from memslides.tools.template_tools import (
                is_layout_queried,
                get_queried_layouts,
            )

            _builder, _skill = _ensure_template_context()
            if _builder and _skill:
                _layout = _infer_layout_from_html(content) or _builder.get_last_queried_layout()

                # 强制查询检查：如果有布局标注但未调用 query_slide_layout，则警告
                if _layout and not is_layout_queried(_layout):
                    queried = get_queried_layouts()
                    if queried:
                        # 已查询过其他布局，但不是当前布局
                        compliance_note += (
                            f"\n⚠️ MUST_QUERY: 文件已保存，但布局 `{_layout}` 未查询。"
                            f"请立即调用 query_slide_layout(\"{_layout}\") 获取元素规范，然后用 write_html_file 重新生成此页。"
                            f"\n   已查询的布局: {', '.join(queried)}"
                        )
                    else:
                        # 完全没查询过任何布局
                        compliance_note += (
                            f"\n⚠️ MUST_QUERY: 文件已保存，但布局 `{_layout}` 未查询。"
                            f"请立即调用 query_slide_layout(\"{_layout}\") 获取元素详情和字符限制，然后用 write_html_file 重新生成此页。"
                        )

                from memslides.memory.compliance.template_checker import TemplateComplianceChecker

                _checker = TemplateComplianceChecker(_builder.skill)
                _issues = _checker.check(content, _layout, file_path=path)
                if _issues:
                    # Auto-fix fixable issues
                    fixable = [i for i in _issues if i.auto_fixable]
                    unfixable = [i for i in _issues if not i.auto_fixable]
                    if fixable:
                        fixed_content, fixed_list = _checker.auto_fix(content, fixable)
                        if fixed_list:
                            content = fixed_content
                            path.write_text(fixed_content, encoding="utf-8")
                            compliance_note += f"\n⚡ Auto-fixed {len(fixed_list)} issue(s): {', '.join(i.type for i in fixed_list)}"
                    if unfixable:
                        feedback_text = _checker.format_issues_for_retry(unfixable)
                        compliance_note += "\n" + feedback_text
                        # 记录合规反馈
                        _builder.record_compliance_feedback(file_path, len(unfixable), feedback_text)
        except Exception:
            pass  # compliance is non-fatal

        compliance_note += _format_layout_validation_note(
            layout_diagnostics,
            auto_repairs,
            remaining_blockers,
        )
        if diagram_contract:
            compliance_note += _format_diagram_validation_note(diagram_diagnostics)
        if marker_report.get("changed"):
            marker_label = str(marker_report.get("marker_text", "") or _TEMP_PREF_MARKER_TEXT)
            compliance_note += (
                f"\n🔖 Normalized {marker_label} marker to one visible bottom-right anchored element inside the slide canvas."
            )

        # A5: body 固定尺寸与高风险响应式版式提示
        try:
            _layout_risk_notes = _body_layout_risk_notes(content)
            if _layout_risk_notes:
                compliance_note += (
                    "\n⚠️ BODY_LAYOUT: "
                    + "\n   - ".join(["检测到高风险版式信号。", *_layout_risk_notes[:3]])
                )
        except Exception:
            pass

        # A3: 记录 query_slide_layout 调用状态
        try:
            from memslides.tools.template_tools import (
                is_layout_queried as _a3_is_queried,
                get_queried_layouts as _a3_get_queried,
            )

            _builder_a3, _skill_a3 = _ensure_template_context()
            if _builder_a3 and _skill_a3:
                _layout_a3 = _infer_layout_from_html(content)
                if _layout_a3 and _a3_is_queried(_layout_a3):
                    compliance_note += f"\n✅ 布局 `{_layout_a3}` 已通过 query_slide_layout 查询。"
        except Exception:
            pass

        # Get file size for confirmation after all deterministic writes.
        file_size = path.stat().st_size
        line_count = content.count('\n') + 1

        info(f"HTML file written: {path} ({file_size} bytes, {line_count} lines)")
        result = f"Successfully wrote HTML file to: {path}\nFile size: {file_size} bytes\nLine count: {line_count}"
        if compliance_note:
            result += compliance_note
        return result

    except Exception as e:
        error(f"Failed to write HTML file {file_path}: {e}")
        return f"Error writing file: {str(e)}"


@mcp.tool()
def write_new_slide_file(
    file_path: str,
    content: str,
) -> str:
    """Write a brand-new slide HTML file.

    Structural modify turns use this instead of the general write_html_file tool.
    The target file must not already exist.
    """
    path = _normalize_path(file_path)
    created_by_insert = str(path) in _newly_created_slide_paths
    if path.exists() and not created_by_insert:
        return _structured_patch_error(
            "WRITE_NEW_SLIDE_EXISTS",
            "write_new_slide_file only accepts brand-new slide files.",
            file_path=str(path),
        )
    if not _is_canonical_slide_path(path):
        return _structured_patch_error(
            "WRITE_NEW_SLIDE_INVALID",
            "write_new_slide_file requires a canonical slide filename like slide_01.html.",
            file_path=str(path),
        )
    result = write_html_file.fn(
        file_path=file_path,
        content=content,
        force_regenerate=False,
        expected_hash="",
    )
    if isinstance(result, str) and result.startswith("Successfully wrote HTML file"):
        _newly_created_slide_paths.discard(str(path))
    return result


@mcp.tool()
def delete_slide(slide_path: str, renumber: bool = True) -> str:
    """
    Delete a slide HTML file and optionally renumber subsequent slides to maintain
    continuous numbering (e.g., after deleting slide_06.html, slide_07→slide_06, etc.).

    Use this when the user requests to remove a slide entirely from the presentation.
    After deletion, the total slide count will decrease by one.

    Args:
        slide_path: Path to the slide HTML file to delete (e.g., "outputs/slide_06.html"
                    or the legacy alias "slides/slide_06.html").
        renumber: If True (default), renumber all subsequent slides to maintain
                  continuous numbering. If False, only delete the file without renumbering.

    Returns:
        Success message with details of deleted and renamed files, or error message.

    Example:
        delete_slide("outputs/slide_06.html")
        # Deletes slide_06.html and renames slide_07→slide_06, slide_08→slide_07, etc.
    """
    try:
        path = _normalize_path(slide_path)

        # Validate file exists and is an HTML slide
        if not path.exists():
            return f"Error: Slide file not found: {slide_path}"
        if not path.is_file():
            return f"Error: Path is not a file: {slide_path}"
        if path.suffix.lower() != ".html":
            return f"Error: Not an HTML file: {slide_path}"

        # Canonical slides (slide_<num>.html) participate in exported ordering.
        # Non-canonical helpers (e.g., slide_09a_hidden.html) can be deleted,
        # but do not trigger renumbering.
        deleted_num = _canonical_slide_number_from_name(path.name)
        if deleted_num is None and not _SLIDE_ANY_HTML_RE.fullmatch(path.name):
            return (
                "Error: File does not follow slide naming convention "
                f"(expected slide_*.html): {path.name}"
            )

        slides_dir = path.parent

        # Delete the target slide
        path.unlink()
        info(f"Deleted slide: {slide_path}")
        _drop_slide_tracking(path)

        result_lines = [f"✅ Deleted: {path.name}"]

        if renumber and deleted_num is not None:
            # Find all subsequent slides and renumber them
            # Collect slides with number > deleted_num
            subsequent_slides = []
            for num, f in _collect_canonical_slides(slides_dir):
                if num is not None and num > deleted_num:
                    subsequent_slides.append((num, f))

            # Sort by number ascending to rename in order
            subsequent_slides.sort(key=lambda x: x[0])

            renamed_count = 0
            for old_num, old_path in subsequent_slides:
                new_num = old_num - 1
                new_name = f"slide_{new_num:02d}.html"
                new_path = slides_dir / new_name

                # Rename file
                old_path.rename(new_path)
                result_lines.append(f"  📝 Renamed: {old_path.name} → {new_name}")
                renamed_count += 1
                _move_slide_tracking(old_path, new_path)

            if renamed_count > 0:
                result_lines.append(f"\n📊 Total: 1 slide deleted, {renamed_count} slides renumbered")
            else:
                result_lines.append("\n📊 Total: 1 slide deleted (was the last slide, no renumbering needed)")
        elif renumber and deleted_num is None:
            result_lines.append(
                "\n📊 Total: 1 non-canonical slide file deleted (renumbering skipped; "
                "exported ordering only tracks slide_<number>.html)"
            )
        else:
            result_lines.append("\n📊 Total: 1 slide deleted (renumbering skipped)")

        page_counter_updates: list[str] = []
        if renumber and deleted_num is not None:
            page_counter_updates = _sync_canonical_page_counter_text(slides_dir)
            if page_counter_updates:
                result_lines.append(
                    "🔢 Updated page counters: " + ", ".join(page_counter_updates)
                )

        # Count remaining slides
        remaining_total = len(list(slides_dir.glob("slide_*.html")))
        remaining_exportable = len(_collect_canonical_slides(slides_dir))
        result_lines.append(f"📁 Remaining exportable slides: {remaining_exportable}")
        if remaining_total != remaining_exportable:
            result_lines.append(
                f"📁 Remaining auxiliary slide files: {remaining_total - remaining_exportable}"
            )

        return "\n".join(result_lines)

    except Exception as e:
        error(f"Failed to delete slide {slide_path}: {e}")
        return f"Error deleting slide: {str(e)}"


@mcp.tool()
def insert_slide(
    target_slide: str,
    position: Literal["before", "after"] = "after",
    content: str = "",
) -> str:
    """
    Insert a new slide before or after a canonical slide and renumber following
    canonical slides in one operation.

    Args:
        target_slide: Existing canonical slide path, e.g. "outputs/slide_06.html".
        position: "before" inserts at target index, "after" inserts at index+1.
        content: Optional HTML content for the new slide. If empty, a placeholder
                 slide will be created and can be updated later with write_new_slide_file.

    Returns:
        Success message with inserted path, rename details, and remaining counts.
    """
    try:
        target_path = _normalize_path(target_slide)

        if position not in ("before", "after"):
            return f"Error: position must be 'before' or 'after', got: {position}"
        if not target_path.exists():
            return f"Error: Target slide not found: {target_slide}"
        if not target_path.is_file():
            return f"Error: Path is not a file: {target_slide}"
        if target_path.suffix.lower() != ".html":
            return f"Error: Not an HTML file: {target_slide}"

        target_num = _canonical_slide_number_from_name(target_path.name)
        if target_num is None:
            return (
                "Error: insert_slide requires a canonical target filename "
                f"(slide_<number>.html), got: {target_path.name}"
            )

        slides_dir = target_path.parent
        insert_num = target_num if position == "before" else target_num + 1

        # Shift canonical slides at/after insert position in descending order.
        shift_candidates = [
            (num, p)
            for num, p in _collect_canonical_slides(slides_dir)
            if num >= insert_num
        ]
        shift_candidates.sort(key=lambda x: x[0], reverse=True)

        result_lines = []
        shifted_count = 0
        for old_num, old_path in shift_candidates:
            new_num = old_num + 1
            new_name = f"slide_{new_num:02d}.html"
            new_path = slides_dir / new_name
            if new_path.exists():
                return (
                    "Error: Cannot insert slide due to naming collision: "
                    f"{new_path.name} already exists"
                )
            old_path.rename(new_path)
            _move_slide_tracking(old_path, new_path)
            result_lines.append(f"  📝 Renamed: {old_path.name} → {new_name}")
            shifted_count += 1

        new_name = f"slide_{insert_num:02d}.html"
        new_path = slides_dir / new_name

        created_placeholder = not content.strip()
        if content.strip():
            new_content = content
            if "<meta charset" not in new_content.lower():
                new_content = new_content.replace(
                    "<html>", '<html>\n<head><meta charset="utf-8"></head>', 1
                )
        else:
            # Placeholder content keeps layout-safe defaults and can be replaced
            # immediately by write_new_slide_file in the same modify flow.
            new_content = (
                "<!DOCTYPE html>\n"
                '<html lang="zh">\n'
                "<head>\n"
                '<meta charset="UTF-8" />\n'
                "<title>New Slide</title>\n"
                "<style>\n"
                "body{width:1280px;height:720px;margin:0;position:relative;"
                "background:#FFFFFF;color:#0F172A;"
                "font-family:Arial,'Noto Sans CJK SC','Microsoft YaHei',sans-serif}\n"
                ".hint{position:absolute;left:64px;right:64px;top:64px;bottom:64px;"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:22px;line-height:1.5;color:#64748B;text-align:center}\n"
                "</style>\n"
                "</head>\n"
                "<body>\n"
                '<div class="hint">新插入页面（占位）。请用 write_new_slide_file 填充最终内容。</div>\n'
                "</body>\n"
                "</html>\n"
            )

        new_path.write_text(new_content, encoding="utf-8")
        _mark_new_slide_created(new_path)
        _remember_slide_observation(new_path)
        info(f"Inserted slide: {new_path} ({position} {target_path.name})")

        out = [f"✅ Inserted: {new_name} ({position} {target_path.name})"]
        out.extend(result_lines)
        out.append(f"\n📊 Total: 1 slide inserted, {shifted_count} slides renumbered")

        page_counter_updates = _sync_canonical_page_counter_text(slides_dir)
        if page_counter_updates:
            out.append("🔢 Updated page counters: " + ", ".join(page_counter_updates))

        remaining_total = len(list(slides_dir.glob("slide_*.html")))
        remaining_exportable = len(_collect_canonical_slides(slides_dir))
        out.append(f"📁 Remaining exportable slides: {remaining_exportable}")
        if remaining_total != remaining_exportable:
            out.append(f"📁 Remaining auxiliary slide files: {remaining_total - remaining_exportable}")
        out.append(f"📄 New slide path: {new_path}")
        if created_placeholder:
            before_anchor = slides_dir / f"slide_{insert_num - 1:02d}.html" if insert_num > 1 else None
            after_anchor = slides_dir / f"slide_{insert_num + 1:02d}.html"
            out.extend(
                [
                    "status: PLACEHOLDER_CREATED",
                    "next_expected_action: write_new_slide_file",
                    f"write_target: {new_path}",
                ]
            )
            if before_anchor is not None and before_anchor.exists():
                out.append(f"anchor_before: {before_anchor}")
            if after_anchor.exists():
                out.append(f"anchor_after: {after_anchor}")
            out.append(
                "completion_note: This inserted file is only a placeholder. "
                "Use write_new_slide_file to write complete slide HTML before finalize; "
                "use local patch tools only for small repairs after the full slide exists."
            )
        else:
            out.extend(
                [
                    "status: CONTENT_INSERTED",
                    "next_expected_action: inspect_slide",
                    f"inspect_target: {new_path}",
                ]
            )
        return "\n".join(out)

    except Exception as e:
        error(f"Failed to insert slide near {target_slide}: {e}")
        return f"Error inserting slide: {str(e)}"


@mcp.tool()
def list_files(directory: str = ".", max_depth: int = 2) -> str:
    """
    List files and directories in the workspace. Useful for discovering available
    images, slides, attachments, and other resources.

    Args:
        directory: Relative path from workspace root (default: current directory).
                   Common paths: "attachments", "slides", ".".
        max_depth: Maximum directory depth to list (default: 2).

    Returns:
        A formatted file listing with sizes, or error message.
    """
    try:
        base = Path(directory)
        if not base.exists():
            return f"Directory not found: {directory}"
        if not base.is_dir():
            return f"Not a directory: {directory}"

        lines = []
        for p in sorted(base.rglob("*")):
            # Respect max_depth
            rel = p.relative_to(base)
            if len(rel.parts) > max_depth:
                continue
            # Skip hidden files and __pycache__
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
                continue
            prefix = "  " * (len(rel.parts) - 1)
            if p.is_file():
                size = p.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / (1024*1024):.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                lines.append(f"{prefix}{rel.name}  ({size_str})")
            else:
                lines.append(f"{prefix}{rel.name}/")

        if not lines:
            return f"Directory '{directory}' is empty"
        return f"Files in '{directory}':\n" + "\n".join(lines)
    except Exception as e:
        return f"Error listing files: {e}"


@mcp.tool()
def read_file(file_path: str, offset: int = 0, limit: int = 500) -> str:
    """
    Read the content of a text file. Useful for reading existing slide HTML,
    markdown manuscripts, design plans, or other text files.

    Args:
        file_path: Path to the file (absolute or relative to workspace).
        offset: Line number to start reading from (0-indexed, default: 0).
        limit: Maximum number of lines to read (default: 500).

    Returns:
        The file content with line numbers, or error message.
    """
    try:
        path = _normalize_path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"
        if not path.is_file():
            return f"Not a file: {file_path}"

        # Only allow text files
        text_exts = {".html", ".htm", ".css", ".js", ".md", ".txt", ".json",
                     ".yaml", ".yml", ".csv", ".xml", ".svg", ".py", ".sh"}
        if path.suffix.lower() not in text_exts:
            return f"Cannot read binary file: {file_path} (extension: {path.suffix})"

        with open(path, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
        _remember_slide_observation(path)
        if _is_design_plan_target(path):
            _record_design_plan_read(path)
        elif _is_page_execution_plan_target(path):
            _record_control_document_read(path)

        total = len(all_lines)
        selected = all_lines[offset:offset + limit]
        numbered = []
        for i, line in enumerate(selected, start=offset + 1):
            numbered.append(f"{i:4d} | {line.rstrip()}")

        header = f"File: {file_path} ({total} lines total)"
        if offset > 0 or offset + limit < total:
            header += f" [showing lines {offset+1}-{min(offset+limit, total)}]"
        return header + "\n" + "\n".join(numbered)
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def read_slide_snapshot(
    slide_path: str,
    focus_text: str = "",
    focus_selector: str = "",
    focus_kind: str = "",
) -> str:
    """
    Read an existing slide as a bounded editable snapshot instead of returning
    the full HTML body to the model.

    The snapshot contains a short-lived `snapshot_id`, the current `content_hash`,
    a compact slide summary, and a capped list of semantic/body-block targets
    that can be edited later via `apply_slide_patch`. Use `focus_text`,
    `focus_selector`, and/or `focus_kind` to expose a newly created or otherwise
    precise element, such as one caption paragraph, as a top-ranked target.
    """
    try:
        path = _normalize_path(slide_path)
        workspace_guard_error = _workspace_contract_error(
            action="read_slide_snapshot",
            path=path,
        )
        if workspace_guard_error:
            return workspace_guard_error
        if not path.exists():
            return _structured_patch_error(
                "SNAPSHOT_NOT_FOUND",
                f"Slide file not found: {slide_path}",
                slide_path=slide_path,
            )
        if not path.is_file() or path.suffix.lower() != ".html":
            return _structured_patch_error(
                "SNAPSHOT_NOT_FOUND",
                f"Not an HTML slide file: {slide_path}",
                slide_path=slide_path,
            )

        html_text = path.read_text(encoding="utf-8", errors="replace")
        workspace_guard_error = _workspace_contract_error(
            action="read_slide_snapshot",
            path=path,
            html_content=html_text,
        )
        if workspace_guard_error:
            return workspace_guard_error
        _remember_slide_observation(path)

        content_hash = _compute_text_hash(html_text)
        payload = _build_slide_snapshot_payload(
            path,
            html_text,
            focus_text=focus_text,
            focus_selector=focus_selector,
            focus_kind=focus_kind,
        )
        corruption = _detect_corrupted_slide_html(path, html_text, payload)
        if corruption:
            return _structured_patch_error(
                "CORRUPTED_SLIDE",
                "Slide HTML is already corrupted or placeholder-only, so patch editing is disabled. "
                "Regenerate the full slide with `write_html_file(force_regenerate=true, expected_hash=content_hash)`.",
                slide_path=str(path),
                content_hash=content_hash,
                detected_marker=corruption.get("detected_marker", ""),
                body_text_preview=corruption.get("body_text_preview", ""),
                recovery_hint=(
                    "Use the returned `content_hash` as `expected_hash` when recovering "
                    "this slide with full regenerated HTML."
                ),
            )

        snapshot_id, content_hash, payload = _register_slide_snapshot(
            path,
            html_text,
            slide_path=str(path),
            snapshot_payload=payload,
        )

        return json.dumps(
            {
                "success": True,
                "slide_path": str(path),
                "snapshot_id": snapshot_id,
                "content_hash": content_hash,
                "slide_summary": payload["slide_summary"],
                "targets": payload["targets"],
                "rules": payload.get("rules", []),
                "repair_candidates": payload.get("repair_candidates", []),
                "offending_targets": payload.get("offending_targets", []),
                "repair_context": payload.get("repair_context", {}),
                "risk_flags": payload.get("risk_flags", []),
                "validation_diagnostics": payload.get("validation_diagnostics", []),
                "has_more_targets": payload["has_more_targets"],
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        error(f"read_slide_snapshot failed for {slide_path}: {e}")
        return _structured_patch_error(
            "PATCH_APPLY_FAILED",
            f"read_slide_snapshot failed: {e}",
            slide_path=slide_path,
        )

class SemanticPatchContext(BaseModel):
    edit_scope: str | None = Field(
        default=None,
        description=(
            "Optional semantic scope from plan_slide_patch, such as text_color, "
            "typography, background_style, layout_spacing, border_shadow, image_asset, or visibility."
        ),
    )
    property_group: str | None = Field(
        default=None,
        description="Optional CSS property group this op is intended to modify.",
    )
    intent_id: str | None = Field(
        default=None,
        description="Optional caller-provided id tying ops back to one semantic edit intent.",
    )
    repair_intent: str | None = Field(
        default=None,
        description=(
            "Optional repair intent such as bottom_safe_zone, clipped_canvas, text_overflow, "
            "overlap, readability, or image_fit."
        ),
    )
    allow_scope_override: bool = Field(
        default=False,
        description=(
            "Set true only when an op intentionally changes properties outside edit_scope/property_group."
        ),
    )


class ReplaceTextPatchOp(SemanticPatchContext):
    op: Literal["replace_text"] = Field(
        description="Replace the plain-text content inside one existing target node.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to edit.",
    )
    text: str = Field(
        description="New plain-text content for the target node.",
    )


class ReplaceHtmlPatchOp(SemanticPatchContext):
    op: Literal["replace_html"] = Field(
        description="Replace the inner HTML of one existing target node.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to edit.",
    )
    html_fragment: str = Field(
        description="Safe HTML fragment to place inside the target node.",
    )


class MergeStylePatchOp(SemanticPatchContext):
    op: Literal["merge_style"] = Field(
        description="Merge inline CSS declarations into one existing target node.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to edit.",
    )
    declarations: str = Field(
        description="CSS declarations such as `color: #1d4ed8; font-size: 40px;`.",
    )


class WrapTextSpanPatchOp(SemanticPatchContext):
    op: Literal["wrap_text_span"] = Field(
        description="Wrap one exact text span inside an existing text target and apply inline CSS to that span.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node containing the text span.",
    )
    text_span: str = Field(
        description="Exact text span to wrap inside one text node.",
    )
    occurrence: int = Field(
        default=1,
        description="1-based occurrence of text_span under the target node.",
    )
    wrapper_tag: Literal["span", "strong", "em", "mark"] = Field(
        default="span",
        description="Safe wrapper tag used around the matched span.",
    )
    declarations: str = Field(
        description="CSS declarations to merge into the wrapper, such as `color: #6D28D9; font-weight: 700;`.",
    )


class RemoveNodePatchOp(SemanticPatchContext):
    op: Literal["remove_node"] = Field(
        description="Delete one existing target node from the slide.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to remove.",
    )


class InsertHtmlPatchOp(SemanticPatchContext):
    op: Literal["insert_html"] = Field(
        description="Insert a new HTML fragment near an existing anchor target.",
    )
    anchor_target_id: str = Field(
        description="Existing target_id from read_slide_snapshot.targets used as the insertion anchor.",
    )
    position: Literal["before", "after", "prepend", "append"] = Field(
        description="Where to insert the fragment relative to the anchor target.",
    )
    html_fragment: str = Field(
        description="Safe HTML fragment to insert near the anchor target.",
    )


class SetAttrPatchOp(SemanticPatchContext):
    op: Literal["set_attr"] = Field(
        description="Set one allowed HTML attribute on an existing target node.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to edit.",
    )
    attr_name: str = Field(
        description="One allowed attribute name: class, style, src, alt, width, height.",
    )
    value: str = Field(
        description="New attribute value.",
    )


class RemoveAttrPatchOp(SemanticPatchContext):
    op: Literal["remove_attr"] = Field(
        description="Remove one allowed HTML attribute from an existing target node.",
    )
    target_id: str = Field(
        description="Exact target_id returned by read_slide_snapshot.targets for the node to edit.",
    )
    attr_name: str = Field(
        description="One allowed attribute name: class, style, src, alt, width, height.",
    )


class MergeCssRulePatchOp(SemanticPatchContext):
    op: Literal["merge_css_rule"] = Field(
        description="Merge CSS declarations into one existing exposed rule.",
    )
    rule_id: str = Field(
        description="Exact rule_id returned by read_slide_snapshot.rules for the rule to edit.",
    )
    declarations: str = Field(
        description="CSS declarations such as `max-width: 560px; gap: 24px;`.",
    )


class ReplaceCssRulePatchOp(SemanticPatchContext):
    op: Literal["replace_css_rule"] = Field(
        description="Replace the declarations of one existing exposed rule.",
    )
    rule_id: str = Field(
        description="Exact rule_id returned by read_slide_snapshot.rules for the rule to edit.",
    )
    declarations: str = Field(
        description="Complete replacement declarations for the rule.",
    )


SlidePatchOp = Annotated[
    ReplaceTextPatchOp
    | ReplaceHtmlPatchOp
    | MergeStylePatchOp
    | WrapTextSpanPatchOp
    | RemoveNodePatchOp
    | InsertHtmlPatchOp
    | SetAttrPatchOp
    | RemoveAttrPatchOp
    | MergeCssRulePatchOp
    | ReplaceCssRulePatchOp,
    Field(discriminator="op"),
]


def _patch_op_to_dict(op: Any) -> dict[str, Any] | None:
    if isinstance(op, BaseModel):
        return op.model_dump(exclude_none=True)
    if isinstance(op, dict):
        return op
    return None


def _snapshot_target_lookup(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("target_id", "") or ""): item
        for item in snapshot.get("targets", [])
        if isinstance(item, dict) and str(item.get("target_id", "") or "").strip()
    }


def _snapshot_rule_lookup(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("rule_id", "") or ""): item
        for item in snapshot.get("rules", [])
        if isinstance(item, dict) and str(item.get("rule_id", "") or "").strip()
    }


def _resolve_bound_target(
    soup: BeautifulSoup,
    binding: dict[str, Any],
) -> Tag | None:
    dom_path = str(binding.get("dom_path", "") or "")
    expected_hash = str(binding.get("target_hash", "") or "")
    selector_hint = str(binding.get("selector_hint", "") or "")

    direct = _resolve_dom_path(soup, dom_path)
    if isinstance(direct, Tag):
        direct_hash = _compute_target_hash(direct)
        if not expected_hash or direct_hash == expected_hash:
            return direct

    if not expected_hash:
        return None

    body = _html_body_root_from_soup(soup)
    matches = [
        tag
        for tag in body.find_all(True)
        if _compute_target_hash(tag) == expected_hash
    ]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if selector_hint:
        hinted = [
            tag for tag in matches
            if _selector_hint_for_tag(tag) == selector_hint
        ]
        if len(hinted) == 1:
            return hinted[0]
    return matches[0]


def _refresh_working_target_bindings(
    soup: BeautifulSoup,
    working_target_bindings: dict[str, dict[str, Any]],
) -> None:
    """Refresh target hashes after earlier ops mutate ancestors in the same batch."""
    for target_id, binding in list(working_target_bindings.items()):
        resolved = _resolve_bound_target(soup, binding)
        if resolved is None:
            direct = _resolve_dom_path(soup, str(binding.get("dom_path", "") or ""))
            if isinstance(direct, Tag):
                expected_hint = str(binding.get("selector_hint", "") or "")
                current_hint = _selector_hint_for_tag(direct)
                if not expected_hint or current_hint == expected_hint:
                    resolved = direct
        if resolved is None:
            continue
        working_target_bindings[target_id] = {
            **binding,
            "dom_path": _build_dom_path(resolved),
            "target_hash": _compute_target_hash(resolved),
            "selector_hint": _selector_hint_for_tag(resolved),
            "text_preview": _tag_text_preview(resolved),
        }


def _resolve_bound_rule(
    html_text: str,
    binding: dict[str, Any],
) -> dict[str, Any] | None:
    expected_hash = str(binding.get("rule_hash", "") or "")
    expected_selector = str(binding.get("selector", "") or "")
    entries = _extract_css_rule_entries(html_text)

    for entry in entries:
        if (
            str(entry.get("selector", "") or "") == expected_selector
            and str(entry.get("rule_hash", "") or "") == expected_hash
        ):
            return entry
    for entry in entries:
        if str(entry.get("rule_hash", "") or "") == expected_hash:
            return entry
    return None


def _build_rebind_hints(
    snapshot: dict[str, Any],
    patch_ops: list[dict[str, Any]],
    fresh_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    target_lookup = _snapshot_target_lookup(snapshot)
    rule_lookup = _snapshot_rule_lookup(snapshot)
    hints: list[dict[str, Any]] = []
    for index, op in enumerate(patch_ops, start=1):
        op_name = str(op.get("op", "") or "")
        if op_name in {"insert_html"}:
            target_id = str(op.get("anchor_target_id", "") or "")
            binding = target_lookup.get(target_id)
            if not binding:
                continue
            suggestions = [
                target
                for target in fresh_payload.get("targets", [])
                if isinstance(target, dict)
                and (
                    str(target.get("kind", "") or "") == str(binding.get("kind", "") or "")
                    or str(target.get("selector_hint", "") or "") == str(binding.get("selector_hint", "") or "")
                )
            ][:3]
            hints.append(
                {
                    "op_index": index,
                    "kind": "target",
                    "missing_id": target_id,
                    "reason": "anchor_changed",
                    "suggested_target_ids": [item.get("target_id", "") for item in suggestions],
                }
            )
            continue
        if op_name in {"merge_css_rule", "replace_css_rule"}:
            rule_id = str(op.get("rule_id", "") or "")
            binding = rule_lookup.get(rule_id)
            if not binding:
                continue
            suggestions = [
                rule
                for rule in fresh_payload.get("rules", [])
                if isinstance(rule, dict)
                and str(rule.get("selector", "") or "") == str(binding.get("selector", "") or "")
            ][:3]
            hints.append(
                {
                    "op_index": index,
                    "kind": "rule",
                    "missing_id": rule_id,
                    "reason": "rule_changed",
                    "suggested_rule_ids": [item.get("rule_id", "") for item in suggestions],
                }
            )
            continue
        target_id = str(op.get("target_id", "") or "")
        binding = target_lookup.get(target_id)
        if not binding:
            continue
        suggestions = [
            target
            for target in fresh_payload.get("targets", [])
            if isinstance(target, dict)
            and (
                str(target.get("kind", "") or "") == str(binding.get("kind", "") or "")
                or str(target.get("selector_hint", "") or "") == str(binding.get("selector_hint", "") or "")
                or str(target.get("text_preview", "") or "") == str(binding.get("text_preview", "") or "")
            )
        ][:3]
        hints.append(
            {
                "op_index": index,
                "kind": "target",
                "missing_id": target_id,
                "reason": "target_changed",
                "suggested_target_ids": [item.get("target_id", "") for item in suggestions],
            }
        )
    return hints


def _stale_snapshot_error_payload(
    *,
    snapshot_id: str,
    slide_path: str,
    current_content_hash: str,
    next_snapshot_id: str,
    payload: dict[str, Any],
    rebind_hints: list[dict[str, Any]] | None = None,
    message: str,
) -> str:
    return _structured_patch_error(
        "STALE_SNAPSHOT",
        message,
        snapshot_id=snapshot_id,
        slide_path=slide_path,
        current_content_hash=current_content_hash,
        next_snapshot_id=next_snapshot_id,
        targets=payload.get("targets", []),
        rules=payload.get("rules", []),
        repair_candidates=payload.get("repair_candidates", []),
        offending_targets=payload.get("offending_targets", []),
        repair_context=payload.get("repair_context", {}),
        risk_flags=payload.get("risk_flags", []),
        validation_diagnostics=payload.get("validation_diagnostics", []),
        slide_summary=str(payload.get("slide_summary", "") or ""),
        has_more_targets=bool(payload.get("has_more_targets", False)),
        rebind_hints=list(rebind_hints or []),
    )


def _intent_requests_readability_repair(
    edit_intent: str,
    *,
    target_scope: str = "",
    requested_properties: list[str] | None = None,
) -> bool:
    text = " ".join(
        part
        for part in (
            str(edit_intent or ""),
            str(target_scope or ""),
            " ".join(str(prop or "") for prop in (requested_properties or [])),
        )
        if part
    ).strip().lower()
    if not text:
        return False
    readability_tokens = (
        "readability",
        "readable",
        "legible",
        "contrast",
        "text-shadow",
        "background chip",
        "可读性",
        "清晰",
        "看清",
        "可见",
        "对比度",
    )
    if any(token in text for token in readability_tokens):
        return True
    caption_tokens = ("caption", "figure-caption", "figcaption", "图注", "图片说明")
    repair_tokens = ("text-shadow", "background", "padding", "radius", "底色", "阴影", "描边", "衬底", "留白")
    return any(token in text for token in caption_tokens) and any(
        token in text
        for token in repair_tokens
    )


def _diagnostic_intent_candidates(validation_diagnostics: list[dict[str, Any]]) -> list[str]:
    intents: list[str] = []
    for diagnostic in validation_diagnostics:
        intent = _repair_intent_for_diagnostic_code(str(diagnostic.get("code", "") or ""))
        if intent != "none" and intent not in intents:
            intents.append(intent)
    return intents


def _infer_repair_intent(
    edit_intent: str,
    *,
    validation_diagnostics: list[dict[str, Any]] | None = None,
    target_scope: str = "",
    requested_properties: list[str] | None = None,
) -> str:
    text = " ".join(
        part
        for part in (
            str(edit_intent or ""),
            str(target_scope or ""),
            " ".join(str(prop or "") for prop in (requested_properties or [])),
        )
        if part
    ).strip().lower()
    diagnostic_intents = _diagnostic_intent_candidates(validation_diagnostics or [])

    if _intent_requests_readability_repair(
        edit_intent,
        target_scope=target_scope,
        requested_properties=requested_properties,
    ):
        return "readability"
    if any(token in text for token in ("object-fit", "object fit", "fit image", "图片适配", "图像适配")):
        return "image_fit"
    if any(token in text for token in ("overlap", "overlapping", "重叠", "碰撞")):
        return "overlap"

    bottom_tokens = (
        "bottom safe",
        "safe zone",
        "too close to bottom",
        "move up",
        "nudge up",
        "lift",
        "raise",
        "底部安全区",
        "底边",
        "往上挪",
        "上移",
        "抬高",
    )
    if any(token in text for token in bottom_tokens):
        return "bottom_safe_zone"
    if any(token in text for token in ("clipped", "clip", "outside canvas", "裁出", "超出画布", "被裁切")):
        return "clipped_canvas"
    if any(token in text for token in ("overflow", "溢出")):
        return "text_overflow"

    property_groups = _semantic_property_groups_for_props(
        {str(prop): "<requested>" for prop in (requested_properties or [])}
    )
    if diagnostic_intents and "layout_spacing" in property_groups:
        return diagnostic_intents[0]
    if diagnostic_intents and any(token in text for token in ("fix", "repair", "adjust", "修改", "修复", "调整", "处理")):
        return diagnostic_intents[0]
    return "none"


def _operation_kind_for_patch_plan(
    edit_intent: str,
    *,
    edit_scope: str,
    repair_intent: str,
) -> str:
    if repair_intent in _LAYOUT_REPAIR_INTENTS:
        return "layout_repair"
    if repair_intent == "image_fit" or edit_scope == "image_asset":
        return "asset_repair"
    if edit_scope in {"text_color", "typography", "background_style", "border_shadow", "visibility"}:
        return "style_edit"
    text = str(edit_intent or "").strip().lower()
    if any(token in text for token in ("replace text", "replace copy", "add paragraph", "修改文案", "替换文案", "补充内容")):
        return "content_edit"
    return "style_edit"


def _infer_edit_scope_from_intent(
    edit_intent: str,
    *,
    target_scope: str = "",
    requested_properties: list[str] | None = None,
) -> str:
    explicit_scope = _normalize_semantic_edit_scope(target_scope)
    if explicit_scope in _SEMANTIC_EDIT_SCOPES:
        return explicit_scope

    property_groups = _semantic_property_groups_for_props(
        {str(prop): "<requested>" for prop in (requested_properties or [])}
    )
    if len(property_groups) == 1 and property_groups[0] in _SEMANTIC_EDIT_SCOPES:
        return property_groups[0]

    text = str(edit_intent or "").strip().lower()
    if not text:
        return property_groups[0] if property_groups else "layout_spacing"
    if _intent_requests_readability_repair(
        text,
        target_scope=target_scope,
        requested_properties=requested_properties,
    ):
        return "visibility"

    has_background = any(token in text for token in (
        "background",
        "bg",
        "surface",
        "canvas",
        "背景",
        "底色",
        "底板",
        "背景色",
        "背景图",
        "壳层",
    ))
    has_text_color = any(token in text for token in (
        "font color",
        "text color",
        "title color",
        "body color",
        "copy color",
        "文字颜色",
        "字体颜色",
        "文本颜色",
        "标题颜色",
        "正文颜色",
        "字色",
    ))
    mentions_text_surface = any(token in text for token in (
        "text",
        "title",
        "body",
        "copy",
        "font",
        "heading",
        "文字",
        "字体",
        "文本",
        "标题",
        "正文",
    ))
    mentions_color_value = bool(
        re.search(r"#[0-9a-f]{3,8}\b", text)
        or any(
            token in text
            for token in (
                "white",
                "black",
                "red",
                "blue",
                "green",
                "yellow",
                "purple",
                "gray",
                "grey",
                "orange",
                "pink",
                "cyan",
                "magenta",
                "白色",
                "黑色",
                "红色",
                "蓝色",
                "绿色",
                "黄色",
                "灰色",
                "橙色",
                "紫色",
            )
        )
    )
    if mentions_text_surface and mentions_color_value and not has_background:
        has_text_color = True
    if has_text_color and not has_background:
        return "text_color"
    if has_background:
        return "background_style"
    if any(token in text for token in (
        "font-size",
        "font size",
        "font family",
        "font-weight",
        "typography",
        "字号",
        "字体大小",
        "字体",
        "加粗",
        "行距",
        "字距",
        "对齐文本",
    )):
        return "typography"
    if any(token in text for token in (
        "spacing",
        "layout",
        "position",
        "align",
        "margin",
        "padding",
        "gap",
        "overflow",
        "overlap",
        "间距",
        "布局",
        "位置",
        "对齐",
        "边距",
        "留白",
        "溢出",
        "重叠",
    )):
        return "layout_spacing"
    if any(token in text for token in ("border", "shadow", "outline", "边框", "描边", "阴影")):
        return "border_shadow"
    if any(token in text for token in ("image", "figure", "photo", "src", "图片", "图像", "图表", "配图", "照片")):
        return "image_asset"
    if any(token in text for token in ("hide", "show", "visible", "opacity", "隐藏", "显示", "透明")):
        return "visibility"
    return property_groups[0] if property_groups else "layout_spacing"


def _target_score_for_edit_scope(
    target: dict[str, Any],
    edit_scope: str,
    *,
    repair_intent: str = "none",
) -> int:
    roles = {str(role or "").strip().lower() for role in target.get("semantic_roles", []) or []}
    groups = {
        str(group or "").strip().lower()
        for group in target.get("editable_property_groups", []) or []
    }
    repair_roles = {
        str(role or "").strip().lower()
        for role in target.get("repair_roles", []) or []
    }
    kind = str(target.get("kind", "") or "")
    score = 0
    if edit_scope in groups:
        score += 20
    if repair_intent != "none":
        if "offender" in repair_roles:
            score += 130
        if "ancestor_container" in repair_roles:
            score += 110
        if repair_intent in _LAYOUT_REPAIR_INTENTS and _target_supports_layout_repair(target):
            score += 25
    if edit_scope == "text_color":
        if "text" in roles:
            score += 40
        if "title" in roles:
            score += 10
        if kind == "slide_canvas":
            score -= 80
    elif edit_scope == "typography":
        if "text" in roles:
            score += 35
        if kind == "slide_canvas":
            score -= 60
    elif edit_scope == "background_style":
        if roles & {"canvas", "background_surface", "surface"}:
            score += 45
        if "text" in roles:
            score -= 25
    elif edit_scope == "layout_spacing":
        if roles & {"layout_container", "layout_root", "canvas"}:
            score += 35
        if kind in {"layout_container", "body_block", "slide_canvas"}:
            score += 12
    elif edit_scope == "border_shadow":
        if roles & {"surface", "layout_container", "text", "media"}:
            score += 25
    elif edit_scope == "image_asset":
        if roles & {"media", "image", "figure"}:
            score += 50
    elif edit_scope == "visibility":
        score += 15
    return score


def _rule_score_for_edit_scope(
    rule: dict[str, Any],
    edit_scope: str,
    target_lookup: dict[str, dict[str, Any]],
    *,
    repair_intent: str = "none",
) -> int:
    groups = {
        str(group or "").strip().lower()
        for group in rule.get("editable_property_groups", []) or []
    }
    roles = {str(role or "").strip().lower() for role in rule.get("semantic_roles", []) or []}
    repair_roles = {
        str(role or "").strip().lower()
        for role in rule.get("repair_roles", []) or []
    }
    score = 0
    if edit_scope in groups:
        score += 20
    for target_id in rule.get("used_by_target_ids", []) or []:
        score = max(
            score,
            _target_score_for_edit_scope(
                target_lookup.get(str(target_id), {}),
                edit_scope,
                repair_intent=repair_intent,
            ),
        )
    if repair_intent != "none":
        if "governing_rule" in repair_roles:
            score += 125
        if repair_intent in _LAYOUT_REPAIR_INTENTS and "layout_spacing" in groups:
            score += 35
    selector_l = str(rule.get("selector", "") or "").lower()
    if edit_scope == "text_color":
        if roles & {"text", "title", "body", "footer"}:
            score += 35
        if _selector_mentions_body_or_slide(selector_l):
            score -= 45
    elif edit_scope == "background_style":
        if roles & {"canvas", "background_surface", "surface"}:
            score += 40
        if roles & {"text", "title", "body", "footer"} and not roles & {"surface"}:
            score -= 20
    elif edit_scope == "image_asset" and roles & {"media", "image", "figure"}:
        score += 40
    return score


def _declaration_template_for_scope(edit_scope: str) -> str:
    return {
        "text_color": "color: <new text color>;",
        "typography": "font-size: <size>; font-weight: <weight>;",
        "background_style": "background-color: <new background color>;",
        "layout_spacing": "padding: <value>; gap: <value>;",
        "border_shadow": "border: <width> solid <color>; box-shadow: <shadow>;",
        "image_asset": "object-fit: contain; object-position: center;",
        "visibility": "opacity: <0-1>; visibility: <visible|hidden>;",
    }.get(edit_scope, "<property>: <value>;")


def _planner_entry_has_repair_role(entry: dict[str, Any], role: str) -> bool:
    wanted = str(role or "").strip().lower()
    if not wanted:
        return False
    return wanted in {
        str(item or "").strip().lower()
        for item in entry.get("repair_roles", []) or []
    }


def _build_patch_ops_template(
    *,
    edit_scope: str,
    operation_kind: str,
    repair_intent: str,
    candidate_targets: list[dict[str, Any]],
    candidate_rules: list[dict[str, Any]],
    recommended_patch_strategy: list[str] | None = None,
    readability_repair: bool = False,
) -> list[dict[str, Any]]:
    declarations = _declaration_template_for_scope(edit_scope)
    semantic_context = {
        "edit_scope": edit_scope,
        "property_group": edit_scope,
        "intent_id": "<planner_intent_id>",
    }
    if repair_intent != "none":
        semantic_context["repair_intent"] = repair_intent
    recommended_patch_strategy = list(recommended_patch_strategy or [])
    if readability_repair and candidate_targets:
        readable_targets = sorted(
            candidate_targets,
            key=lambda target: (
                0 if bool(target.get("matched_focus", False)) else 1,
                0 if str(target.get("kind", "") or "") in {"caption", "figure_caption"} else 1,
                0 if "text" in {str(role or "").strip().lower() for role in target.get("semantic_roles", []) or []} else 1,
                -int(target.get("score", 0) or 0),
            ),
        )
        primary_target = readable_targets[0]
        return [
            {
                "op": "merge_style",
                "target_id": primary_target.get("target_id", ""),
                "declarations": (
                    "text-shadow: 0 1px 0 rgba(0,0,0,0.12); "
                    "background: rgba(0,0,0,0.04); "
                    "padding: 4px 6px; "
                    "border-radius: 6px;"
                ),
                "edit_scope": "visibility",
                "property_group": "visibility",
                "intent_id": "<planner_intent_id>",
                "allow_scope_override": True,
            }
        ]
    if operation_kind == "layout_repair":
        ancestor_target = next(
            (target for target in candidate_targets if _planner_entry_has_repair_role(target, "ancestor_container")),
            None,
        )
        offender_target = next(
            (target for target in candidate_targets if _planner_entry_has_repair_role(target, "offender")),
            None,
        )
        layout_target = ancestor_target or offender_target or (candidate_targets[0] if candidate_targets else None)
        governing_rule = next(
            (rule for rule in candidate_rules if _planner_entry_has_repair_role(rule, "governing_rule")),
            None,
        )
        layout_rule = governing_rule or (candidate_rules[0] if candidate_rules else None)
        templates: list[dict[str, Any]] = []

        if repair_intent == "bottom_safe_zone":
            if layout_target:
                templates.append(
                    {
                        "op": "merge_style",
                        "target_id": layout_target.get("target_id", ""),
                        "declarations": (
                            "box-sizing: border-box; max-height: <safe px>; overflow: hidden; "
                            "padding-bottom: 48px; gap: <tighter px>;"
                        ),
                        **semantic_context,
                    }
                )
            if layout_rule:
                templates.append(
                    {
                        "op": "merge_css_rule",
                        "rule_id": layout_rule.get("rule_id", ""),
                        "declarations": (
                            "box-sizing: border-box; max-height: <safe px>; overflow: hidden; "
                            "padding-bottom: 48px; gap: <tighter px>;"
                        ),
                        **semantic_context,
                    }
                )
            elif offender_target:
                templates.append(
                    {
                        "op": "merge_style",
                        "target_id": offender_target.get("target_id", ""),
                        "declarations": "bottom: 48px; max-height: <safe px>; overflow: hidden;",
                        **semantic_context,
                    }
                )
            return [item for item in templates if item.get("target_id") or item.get("rule_id")]

        if repair_intent == "clipped_canvas":
            if layout_target:
                templates.append(
                    {
                        "op": "merge_style",
                        "target_id": layout_target.get("target_id", ""),
                        "declarations": (
                            "box-sizing: border-box; width: <fit width>; height: <fit height>; "
                            "max-width: <canvas-safe width>; max-height: <canvas-safe height>; overflow: hidden;"
                        ),
                        **semantic_context,
                    }
                )
            if layout_rule:
                templates.append(
                    {
                        "op": "merge_css_rule",
                        "rule_id": layout_rule.get("rule_id", ""),
                        "declarations": "box-sizing: border-box; max-width: <fit width>; max-height: <fit height>; overflow: hidden;",
                        **semantic_context,
                    }
                )
            return [item for item in templates if item.get("target_id") or item.get("rule_id")]

        if repair_intent in {"text_overflow", "overlap"}:
            if layout_rule:
                templates.append(
                    {
                        "op": "merge_css_rule",
                        "rule_id": layout_rule.get("rule_id", ""),
                        "declarations": "box-sizing: border-box; max-height: <fit height>; overflow: hidden; gap: <tighter px>; padding: <tighter px>;",
                        **semantic_context,
                    }
                )
            if layout_target:
                templates.append(
                    {
                        "op": "merge_style",
                        "target_id": layout_target.get("target_id", ""),
                        "declarations": "box-sizing: border-box; height: <fit height>; max-height: <fit height>; overflow: hidden; align-content: start;",
                        **semantic_context,
                    }
                )
            return [item for item in templates if item.get("target_id") or item.get("rule_id")]

        if recommended_patch_strategy and layout_target:
            return [
                {
                    "op": "merge_style",
                    "target_id": layout_target.get("target_id", ""),
                    "declarations": "top: <adjust px>; left: <adjust px>; width: <adjust width>;",
                    **semantic_context,
                }
            ]
    if edit_scope == "image_asset":
        image_targets = [
            target for target in candidate_targets
            if str(target.get("kind", "") or "") == "image"
        ]
        if image_targets:
            return [
                {
                    "op": "set_attr",
                    "target_id": image_targets[0].get("target_id", ""),
                    "attr_name": "src",
                    "value": "<new image path>",
                    **semantic_context,
                }
            ]
    if candidate_rules:
        return [
            {
                "op": "merge_css_rule",
                "rule_id": candidate_rules[0].get("rule_id", ""),
                "declarations": declarations,
                **semantic_context,
            }
        ]
    if candidate_targets:
        primary_target = candidate_targets[0]
        templates = [
            {
                "op": "merge_style",
                "target_id": primary_target.get("target_id", ""),
                "declarations": declarations,
                **semantic_context,
            }
        ]
        text_preview = str(primary_target.get("text_preview", "") or "").strip()
        if edit_scope in {"text_color", "typography"} and text_preview:
            templates.append(
                {
                    "op": "wrap_text_span",
                    "target_id": primary_target.get("target_id", ""),
                    "text_span": "<exact text span>",
                    "occurrence": 1,
                    "wrapper_tag": "span",
                    "declarations": declarations,
                    **semantic_context,
                }
            )
        return templates
    return []


def _compact_planner_target(target: dict[str, Any], score: int) -> dict[str, Any]:
    return {
        "target_id": target.get("target_id", ""),
        "kind": target.get("kind", ""),
        "selector_hint": target.get("selector_hint", ""),
        "text_preview": target.get("text_preview", ""),
        "allowed_ops": target.get("allowed_ops", []),
        "source": target.get("source", ""),
        "matched_focus": bool(target.get("matched_focus", False)),
        "parent_target_id": target.get("parent_target_id", ""),
        "child_target_ids": target.get("child_target_ids", []),
        "semantic_roles": target.get("semantic_roles", []),
        "editable_property_groups": target.get("editable_property_groups", []),
        "effective_style_subset": target.get("effective_style_subset", {}),
        "box_context": target.get("box_context", {}),
        "repair_roles": target.get("repair_roles", []),
        "diagnostic_codes": target.get("diagnostic_codes", []),
        "repair_intents": target.get("repair_intents", []),
        "governing_rule_ids": target.get("governing_rule_ids", []),
        "score": score,
    }


def _compact_planner_rule(rule: dict[str, Any], score: int) -> dict[str, Any]:
    return {
        "rule_id": rule.get("rule_id", ""),
        "selector": rule.get("selector", ""),
        "allowed_ops": rule.get("allowed_ops", list(_RULE_ALLOWED_OPS)),
        "used_by_target_ids": rule.get("used_by_target_ids", []),
        "semantic_roles": rule.get("semantic_roles", []),
        "editable_property_groups": rule.get("editable_property_groups", []),
        "property_groups_present": rule.get("property_groups_present", []),
        "effective_style_subset": rule.get("effective_style_subset", {}),
        "repair_roles": rule.get("repair_roles", []),
        "diagnostic_codes": rule.get("diagnostic_codes", []),
        "repair_intents": rule.get("repair_intents", []),
        "score": score,
    }


@mcp.tool()
def plan_slide_patch(
    slide_path: str,
    edit_intent: str,
    target_scope: str = "",
    requested_properties: list[str] | None = None,
    focus_text: str = "",
    focus_selector: str = "",
    focus_kind: str = "",
) -> str:
    """
    Plan a local slide edit before applying low-level DOM/CSS patch operations.

    This non-destructive planner reads the same bounded snapshot surface as
    `read_slide_snapshot`, infers a semantic edit scope, and returns the targets,
    rules, property groups, risks, and patch skeleton that best match the intent.
    Use this before `apply_slide_patch` for local style edits so the model edits
    the intended semantic surface instead of blindly changing unrelated fields.
    """
    try:
        path = _normalize_path(slide_path)
        workspace_guard_error = _workspace_contract_error(
            action="plan_slide_patch",
            path=path,
        )
        if workspace_guard_error:
            return workspace_guard_error
        if not path.exists() or not path.is_file() or path.suffix.lower() != ".html":
            return _structured_patch_error(
                "SNAPSHOT_NOT_FOUND",
                f"Not an HTML slide file: {slide_path}",
                slide_path=slide_path,
            )

        html_text = path.read_text(encoding="utf-8", errors="replace")
        workspace_guard_error = _workspace_contract_error(
            action="plan_slide_patch",
            path=path,
            html_content=html_text,
        )
        if workspace_guard_error:
            return workspace_guard_error

        normalized_requested_properties = [
            str(prop).strip().lower()
            for prop in (requested_properties or [])
            if str(prop).strip()
        ]
        normalized_focus_kind = _normalize_focus_kind(focus_kind) or _normalize_focus_kind(target_scope)
        payload = _build_slide_snapshot_payload(
            path,
            html_text,
            focus_text=focus_text,
            focus_selector=focus_selector,
            focus_kind=normalized_focus_kind,
        )
        content_hash = _compute_text_hash(html_text)
        corruption = _detect_corrupted_slide_html(path, html_text, payload)
        if corruption:
            return _structured_patch_error(
                "CORRUPTED_SLIDE",
                "Slide HTML is already corrupted or placeholder-only, so local patch planning is disabled.",
                slide_path=str(path),
                content_hash=content_hash,
                detected_marker=corruption.get("detected_marker", ""),
                body_text_preview=corruption.get("body_text_preview", ""),
            )

        snapshot_id, content_hash, payload = _register_slide_snapshot(
            path,
            html_text,
            slide_path=str(path),
            snapshot_payload=payload,
        )

        validation_diagnostics = [
            item
            for item in payload.get("validation_diagnostics", []) or []
            if isinstance(item, dict)
        ]
        repair_intent = _infer_repair_intent(
            edit_intent,
            validation_diagnostics=validation_diagnostics,
            target_scope=target_scope,
            requested_properties=normalized_requested_properties,
        )
        edit_scope = _infer_edit_scope_from_intent(
            edit_intent,
            target_scope=target_scope,
            requested_properties=normalized_requested_properties,
        )
        readability_repair = _intent_requests_readability_repair(
            edit_intent,
            target_scope=target_scope,
            requested_properties=normalized_requested_properties,
        )
        if repair_intent in _LAYOUT_REPAIR_INTENTS:
            edit_scope = "layout_spacing"
        elif repair_intent == "image_fit":
            edit_scope = "image_asset"
        elif repair_intent == "readability" or readability_repair:
            edit_scope = "visibility"
        operation_kind = _operation_kind_for_patch_plan(
            edit_intent,
            edit_scope=edit_scope,
            repair_intent=repair_intent,
        )
        recommended_patch_strategy = _RECOMMENDED_PATCH_STRATEGIES.get(repair_intent, [])
        allowed_groups = sorted(_SEMANTIC_SCOPE_ALLOWED_GROUPS.get(edit_scope, {edit_scope}))

        targets = [target for target in payload.get("targets", []) if isinstance(target, dict)]
        target_lookup = _snapshot_target_lookup({"targets": targets})

        def _planner_target_score(target: dict[str, Any]) -> int:
            score = _target_score_for_edit_scope(target, edit_scope, repair_intent=repair_intent)
            kind = str(target.get("kind", "") or "")
            roles = {str(role or "").strip().lower() for role in target.get("semantic_roles", []) or []}
            if bool(target.get("matched_focus", False)):
                score += 70
            elif str(target.get("source", "") or "") == "focus_parent":
                score += 25
            if normalized_focus_kind:
                if kind == normalized_focus_kind or normalized_focus_kind in roles:
                    score += 35
                elif normalized_focus_kind == "caption" and kind in {"caption", "figure_caption"}:
                    score += 35
            if readability_repair:
                if kind in {"caption", "figure_caption"}:
                    score += 45
                if "text" in roles:
                    score += 15
                if kind == "slide_canvas":
                    score -= 80
            return score

        target_scores = [
            (_planner_target_score(target), target)
            for target in targets
        ]
        ranked_targets = [
            (score, target)
            for score, target in sorted(target_scores, key=lambda item: (-item[0], _target_selection_rank(str(item[1].get("kind", "")))))
            if score > 0
        ]

        rules = [rule for rule in payload.get("rules", []) if isinstance(rule, dict)]
        rule_scores = [
            (_rule_score_for_edit_scope(rule, edit_scope, target_lookup, repair_intent=repair_intent), rule)
            for rule in rules
        ]
        ranked_rules = [
            (score, rule)
            for score, rule in sorted(rule_scores, key=lambda item: (-item[0], str(item[1].get("selector", ""))))
            if score > 0
        ]

        candidate_targets = [
            _compact_planner_target(target, score)
            for score, target in ranked_targets[:8]
        ]
        safe_insert_anchors: list[dict[str, Any]] = []
        seen_insert_anchor_ids: set[str] = set()
        for score, target in [
            *ranked_targets,
            *[(0, target) for target in targets],
        ]:
            target_id = str(target.get("target_id", "") or "")
            if not target_id or target_id in seen_insert_anchor_ids:
                continue
            allowed_ops = {
                str(op or "").strip()
                for op in target.get("allowed_ops", []) or []
            }
            if "insert_html" not in allowed_ops:
                continue
            safe_insert_anchors.append(_compact_planner_target(target, score))
            seen_insert_anchor_ids.add(target_id)
            if len(safe_insert_anchors) >= 6:
                break
        candidate_rules = [
            _compact_planner_rule(rule, score)
            for score, rule in ranked_rules[:8]
        ]
        patch_ops_template = _build_patch_ops_template(
            edit_scope=edit_scope,
            operation_kind=operation_kind,
            repair_intent=repair_intent,
            candidate_targets=candidate_targets,
            candidate_rules=candidate_rules,
            recommended_patch_strategy=recommended_patch_strategy,
            readability_repair=readability_repair,
        )

        risk_warnings: list[dict[str, Any]] = []
        if edit_scope == "text_color":
            risk_warnings.append(
                {
                    "code": "scope_excludes_background",
                    "message": (
                        "This intent is planned as text_color: use CSS `color` on text targets/rules. "
                        "Background/surface properties are not part of the recommended patch template."
                    ),
                }
            )
        if operation_kind == "layout_repair":
            risk_warnings.append(
                {
                    "code": "layout_repair_requires_recheck",
                    "message": (
                        "This is planned as a render-aware layout repair. After applying the patch, run "
                        "`inspect_slide` again to confirm the exporter diagnostics are actually cleared."
                    ),
                }
            )
        if payload.get("risk_flags"):
            risk_warnings.append(
                {
                    "code": "existing_slide_risk_flags",
                    "risk_flags": payload.get("risk_flags", []),
                    "message": "Existing validation risk flags are present; keep local edits narrow unless repairing those risks.",
                }
            )

        planner_notes = [
            "Fill placeholder values in patch_ops_template, then call apply_slide_patch with the returned snapshot_id/content_hash.",
            "Use edit_scope/property_group on scoped style ops so apply_slide_patch can audit semantic drift.",
            "For insert_html, choose anchor_target_id only from safe_insert_anchors or a candidate target whose allowed_ops includes insert_html.",
        ]
        if operation_kind == "layout_repair":
            planner_notes.append(
                "For layout/export failures, prioritize offending_targets, repair_context diagnostics, and the recommended patch strategy instead of generic text or color targets."
            )
        if readability_repair:
            planner_notes.append(
                "Readability repair may intentionally combine text-shadow/background/padding; the template sets allow_scope_override=true for that cross-property local patch."
            )

        return json.dumps(
            {
                "success": True,
                "slide_path": str(path),
                "snapshot_id": snapshot_id,
                "content_hash": content_hash,
                "edit_intent": edit_intent,
                "operation_kind": operation_kind,
                "repair_intent": repair_intent,
                "recommended_patch_strategy": recommended_patch_strategy,
                "recommended_edit_scope": edit_scope,
                "allowed_property_groups": allowed_groups,
                "requested_properties": normalized_requested_properties,
                "focus": {
                    "focus_text": str(focus_text or ""),
                    "focus_selector": str(focus_selector or ""),
                    "focus_kind": normalized_focus_kind,
                },
                "candidate_targets": candidate_targets,
                "safe_insert_anchors": safe_insert_anchors,
                "candidate_rules": candidate_rules,
                "patch_ops_template": patch_ops_template,
                "planner_notes": planner_notes,
                "risk_warnings": risk_warnings,
                "slide_summary": payload.get("slide_summary", ""),
                "repair_candidates": payload.get("repair_candidates", []),
                "offending_targets": payload.get("offending_targets", []),
                "repair_context": payload.get("repair_context", {}),
                "validation_diagnostics": validation_diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        error(f"plan_slide_patch failed for {slide_path}: {e}")
        return _structured_patch_error(
            "PATCH_APPLY_FAILED",
            f"plan_slide_patch failed: {e}",
            slide_path=slide_path,
        )


def _patch_css_rule_html(
    html_text: str,
    binding: dict[str, Any],
    desired_props: dict[str, str],
    *,
    mode: Literal["merge", "replace"],
) -> tuple[str | None, dict[str, Any] | None]:
    rule_entry = _resolve_bound_rule(html_text, binding)
    if rule_entry is None:
        return None, None

    existing_props = dict(rule_entry.get("declarations", {}) or {})
    merged_props = dict(desired_props) if mode == "replace" else {**existing_props, **desired_props}
    if merged_props == existing_props:
        return html_text, {
            "rule_id": str(binding.get("rule_id", "") or ""),
            "selector": str(binding.get("selector", "") or ""),
            "status": "already_compliant",
        }

    decl_start, decl_end = rule_entry["declarations_span"]
    updated_html = (
        html_text[:decl_start]
        + _format_css_declarations(merged_props)
        + html_text[decl_end:]
    )
    return updated_html, {
        "rule_id": str(binding.get("rule_id", "") or ""),
        "selector": str(binding.get("selector", "") or ""),
        "status": "updated",
        "declarations": merged_props,
    }


def _is_text_span_wrapper(tag: Tag, desired_props: dict[str, str]) -> bool:
    if tag.name.lower() not in {"span", "strong", "em", "mark"}:
        return False
    existing_props = _parse_css_declarations_text(str(tag.get("style", "") or ""))
    return all(existing_props.get(prop) == value for prop, value in desired_props.items())


def _wrap_text_span_in_target(
    soup: BeautifulSoup,
    target: Tag,
    *,
    text_span: str,
    occurrence: int,
    wrapper_tag: str,
    desired_props: dict[str, str],
) -> tuple[bool, dict[str, Any]]:
    span = str(text_span or "")
    if not span:
        return False, {"status": "error", "error_code": "TEXT_SPAN_NOT_FOUND_OR_UNSAFE", "error": "`text_span` must not be empty."}
    wrapper = str(wrapper_tag or "span").strip().lower()
    if wrapper not in {"span", "strong", "em", "mark"}:
        return False, {"status": "error", "error_code": "INVALID_WRAPPER_TAG", "error": f"Unsupported wrapper_tag: {wrapper}"}
    desired_occurrence = max(1, int(occurrence or 1))

    seen = 0
    for text_node in list(target.find_all(string=True)):
        if not isinstance(text_node, NavigableString):
            continue
        parent = text_node.parent
        if not isinstance(parent, Tag) or parent.name.lower() in {"script", "style"}:
            continue
        text = str(text_node)
        search_from = 0
        while True:
            found_at = text.find(span, search_from)
            if found_at < 0:
                break
            seen += 1
            if seen != desired_occurrence:
                search_from = found_at + len(span)
                continue

            existing_parent = text_node.parent
            if (
                isinstance(existing_parent, Tag)
                and existing_parent.name.lower() == wrapper
                and str(existing_parent.get_text("", strip=False)) == span
                and _is_text_span_wrapper(existing_parent, desired_props)
            ):
                return False, {"status": "already_compliant", "matched_text": span}

            before = text[:found_at]
            matched = text[found_at : found_at + len(span)]
            after = text[found_at + len(span) :]
            new_tag = soup.new_tag(wrapper)
            new_tag.string = matched
            new_tag["style"] = _format_css_declarations(desired_props)
            if before:
                text_node.insert_before(NavigableString(before))
            text_node.insert_before(new_tag)
            if after:
                text_node.insert_before(NavigableString(after))
            text_node.extract()
            return True, {
                "status": "updated",
                "matched_text": span,
                "occurrence": desired_occurrence,
                "wrapper_tag": wrapper,
                "declarations": desired_props,
            }
    return False, {
        "status": "error",
        "error_code": "TEXT_SPAN_NOT_FOUND_OR_UNSAFE",
        "error": f"Text span `{span}` occurrence {desired_occurrence} was not found inside one text node.",
        "occurrences_found": seen,
    }


def _diff_css_props(
    before: dict[str, str],
    after: dict[str, str],
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for prop in sorted(set(before) | set(after)):
        before_value = str(before.get(prop, "") or "")
        after_value = str(after.get(prop, "") or "")
        if before_value == after_value:
            continue
        group = _semantic_property_group_for_prop(prop)
        changes.append(
            {
                "property": prop,
                "before": before_value,
                "after": after_value,
                "property_group": group,
            }
        )
    return changes


def _collect_inline_style_state(html_text: str) -> dict[str, dict[str, Any]]:
    soup = BeautifulSoup(html_text, "lxml")
    body = _html_body_root_from_soup(soup)
    state: dict[str, dict[str, Any]] = {}
    for tag in body.find_all(True):
        style = str(tag.get("style", "") or "")
        props = _parse_css_declarations_text(style)
        if not props:
            continue
        dom_path = _build_dom_path(tag)
        if not dom_path:
            continue
        state[dom_path] = {
            "selector_hint": _selector_hint_for_tag(tag),
            "text_preview": _tag_text_preview(tag),
            "props": props,
        }
    return state


def _collect_attr_state(html_text: str) -> dict[str, dict[str, Any]]:
    soup = BeautifulSoup(html_text, "lxml")
    body = _html_body_root_from_soup(soup)
    state: dict[str, dict[str, Any]] = {}
    for tag in body.find_all(True):
        attrs: dict[str, str] = {}
        for attr_name in ("src", "alt", "width", "height", "class"):
            if tag.has_attr(attr_name):
                attrs[attr_name] = _normalized_tag_attr_value(attr_name, tag.get(attr_name, ""))
        if not attrs:
            continue
        dom_path = _build_dom_path(tag)
        if not dom_path:
            continue
        state[dom_path] = {
            "selector_hint": _selector_hint_for_tag(tag),
            "text_preview": _tag_text_preview(tag),
            "attrs": attrs,
        }
    return state


def _changed_properties_by_group(changes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        group = str(change.get("property_group", "") or "unknown")
        if group == "unknown":
            continue
        grouped.setdefault(group, []).append(change)
    return grouped


def _explicit_allowed_groups_for_ops(patch_ops: list[dict[str, Any]]) -> set[str]:
    allowed: set[str] = set()
    changed_groups: list[str] = []
    repair_intents: list[str] = []
    for op in patch_ops:
        scope = _normalize_semantic_edit_scope(op.get("edit_scope"))
        group = _normalize_semantic_edit_scope(op.get("property_group"))
        if scope in _SEMANTIC_SCOPE_ALLOWED_GROUPS:
            allowed.update(_SEMANTIC_SCOPE_ALLOWED_GROUPS[scope])
        if group in _SEMANTIC_SCOPE_ALLOWED_GROUPS:
            allowed.update(_SEMANTIC_SCOPE_ALLOWED_GROUPS[group])
        changed_groups.extend(_groups_for_patch_op(op))
        repair_intents.append(str(op.get("repair_intent", "") or ""))
    return _extend_allowed_groups_for_compatible_combo(
        allowed,
        changed_groups,
        repair_intents=repair_intents,
    )


def _build_semantic_patch_audit(
    before_html: str,
    after_html: str,
    patch_ops: list[dict[str, Any]],
) -> dict[str, Any]:
    changes: list[dict[str, Any]] = []

    before_rules = _parse_css_rules(_extract_all_style_text(before_html))
    after_rules = _parse_css_rules(_extract_all_style_text(after_html))
    for selector in sorted(set(before_rules) | set(after_rules)):
        for change in _diff_css_props(before_rules.get(selector, {}), after_rules.get(selector, {})):
            changes.append(
                {
                    "location": "css_rule",
                    "selector": selector,
                    **change,
                }
            )

    before_inline = _collect_inline_style_state(before_html)
    after_inline = _collect_inline_style_state(after_html)
    for dom_path in sorted(set(before_inline) | set(after_inline)):
        before_entry = before_inline.get(dom_path, {})
        after_entry = after_inline.get(dom_path, {})
        for change in _diff_css_props(
            dict(before_entry.get("props", {}) or {}),
            dict(after_entry.get("props", {}) or {}),
        ):
            changes.append(
                {
                    "location": "inline_style",
                    "dom_path": dom_path,
                    "selector_hint": after_entry.get("selector_hint") or before_entry.get("selector_hint", ""),
                    "text_preview": after_entry.get("text_preview") or before_entry.get("text_preview", ""),
                    **change,
                }
            )

    before_attrs = _collect_attr_state(before_html)
    after_attrs = _collect_attr_state(after_html)
    for dom_path in sorted(set(before_attrs) | set(after_attrs)):
        before_entry = before_attrs.get(dom_path, {})
        after_entry = after_attrs.get(dom_path, {})
        before_values = dict(before_entry.get("attrs", {}) or {})
        after_values = dict(after_entry.get("attrs", {}) or {})
        for attr_name in sorted(set(before_values) | set(after_values)):
            before_value = before_values.get(attr_name, "")
            after_value = after_values.get(attr_name, "")
            if before_value == after_value:
                continue
            group = _semantic_attr_group(attr_name)
            if group == "unknown":
                continue
            changes.append(
                {
                    "location": "attribute",
                    "dom_path": dom_path,
                    "selector_hint": after_entry.get("selector_hint") or before_entry.get("selector_hint", ""),
                    "text_preview": after_entry.get("text_preview") or before_entry.get("text_preview", ""),
                    "property": attr_name,
                    "before": before_value,
                    "after": after_value,
                    "property_group": group,
                }
            )

    allowed_groups = _explicit_allowed_groups_for_ops(patch_ops)
    unexpected = [
        change
        for change in changes
        if allowed_groups
        and str(change.get("property_group", "") or "unknown") not in allowed_groups
        and str(change.get("property_group", "") or "unknown") != "unknown"
    ]
    changed_targets = []
    for index, op in enumerate(patch_ops, start=1):
        entry = {
            "op_index": index,
            "op": str(op.get("op", "") or ""),
            **_semantic_context_for_op(op),
        }
        if op.get("target_id"):
            entry["target_id"] = str(op.get("target_id", "") or "")
        if op.get("anchor_target_id"):
            entry["anchor_target_id"] = str(op.get("anchor_target_id", "") or "")
        if op.get("rule_id"):
            entry["rule_id"] = str(op.get("rule_id", "") or "")
        changed_targets.append(entry)

    return {
        "changed_properties_by_group": _changed_properties_by_group(changes[:80]),
        "changed_targets": changed_targets,
        "unexpected_property_changes": unexpected[:40],
        "changed_property_count": len(changes),
    }


def _repair_roles_for_patch_op(
    op: dict[str, Any],
    snapshot: dict[str, Any],
) -> list[str]:
    roles: list[str] = []
    target_lookup = _snapshot_target_lookup(snapshot)
    rule_lookup = _snapshot_rule_lookup(snapshot)
    target_id = str(op.get("target_id", "") or "")
    anchor_target_id = str(op.get("anchor_target_id", "") or "")
    rule_id = str(op.get("rule_id", "") or "")
    for candidate_id in (target_id, anchor_target_id):
        if not candidate_id:
            continue
        for role in target_lookup.get(candidate_id, {}).get("repair_roles", []) or []:
            normalized_role = str(role or "").strip()
            if normalized_role and normalized_role not in roles:
                roles.append(normalized_role)
    if rule_id:
        for role in rule_lookup.get(rule_id, {}).get("repair_roles", []) or []:
            normalized_role = str(role or "").strip()
            if normalized_role and normalized_role not in roles:
                roles.append(normalized_role)
    return roles


def _diagnostic_summary_signature(diagnostic: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(diagnostic.get("code", "") or ""),
        str(diagnostic.get("target_dom_path", "") or ""),
        str(diagnostic.get("text_preview", "") or ""),
    )


def _build_repair_patch_audit(
    *,
    snapshot: dict[str, Any],
    patch_ops: list[dict[str, Any]],
    next_validation_diagnostics: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool, list[dict[str, Any]], list[dict[str, Any]]]:
    previous_diagnostics = [
        item for item in snapshot.get("validation_diagnostics", []) or [] if isinstance(item, dict)
    ]
    previous_signatures = {
        _diagnostic_summary_signature(item): item
        for item in previous_diagnostics
    }
    next_signatures = {
        _diagnostic_summary_signature(item): item
        for item in next_validation_diagnostics
        if isinstance(item, dict)
    }
    resolved_diagnostics = [
        dict(item)
        for signature, item in previous_signatures.items()
        if signature not in next_signatures
    ]
    remaining_diagnostics = [dict(item) for item in next_signatures.values()]

    repair_intents: list[str] = []
    touched_roles: list[str] = []
    touched_entries: list[dict[str, Any]] = []
    for index, op in enumerate(patch_ops, start=1):
        repair_intent = str(op.get("repair_intent", "") or "").strip()
        if repair_intent and repair_intent not in repair_intents:
            repair_intents.append(repair_intent)
        roles = _repair_roles_for_patch_op(op, snapshot)
        for role in roles:
            if role and role not in touched_roles:
                touched_roles.append(role)
        touched_entries.append(
            {
                "op_index": index,
                "op": str(op.get("op", "") or ""),
                "repair_intent": repair_intent,
                "repair_roles": roles,
                "target_id": str(op.get("target_id", "") or ""),
                "anchor_target_id": str(op.get("anchor_target_id", "") or ""),
                "rule_id": str(op.get("rule_id", "") or ""),
            }
        )

    needs_render_recheck = bool(
        set(repair_intents) & _LAYOUT_REPAIR_INTENTS
        or any(
            str(item.get("code", "") or "") in _REPAIR_INTENT_BY_DIAGNOSTIC_CODE
            for item in previous_diagnostics
        )
    )
    audit = {
        "repair_intents": repair_intents,
        "touched_repair_roles": touched_roles,
        "touched_entries": touched_entries,
    }
    return audit, needs_render_recheck, resolved_diagnostics, remaining_diagnostics


@mcp.tool()
def apply_slide_patch(
    snapshot_id: Annotated[
        str,
        Field(
            description="snapshot_id returned by read_slide_snapshot for the current slide state.",
        ),
    ],
    patch_ops: Annotated[
        list[SlidePatchOp],
        Field(
            min_length=1,
            description=(
                "Ordered non-empty list of patch operations. "
                "Never call apply_slide_patch without at least one concrete op."
            ),
        ),
    ],
    expected_hash: Annotated[
        str,
        Field(
            description="content_hash returned by read_slide_snapshot for the same snapshot_id. Must match exactly.",
        ),
    ],
) -> str:
    """
    Apply deterministic DOM/CSS patches to an existing slide snapshot.

    This is the primary editing protocol for existing canonical slides. The
    model provides bounded patch intent while the tool applies changes against
    the current on-disk HTML and rejects stale writes via `expected_hash`.

    Args:
        snapshot_id: Snapshot token returned by `read_slide_snapshot`.
        patch_ops: Non-empty ordered patch list. Each op must include an `op`
            type plus the fields required for that op. Build these ops using
            `target_id` values from `read_slide_snapshot`.
        expected_hash: The exact `content_hash` returned by `read_slide_snapshot`
            for the same slide version.
    """
    snapshot_key = str(snapshot_id or "").strip()
    snapshot = _slide_snapshot_registry.get(snapshot_key)
    if not snapshot:
        retired_snapshot = _slide_retired_snapshot_registry.get(snapshot_key)
        if retired_snapshot:
            return _structured_patch_error(
                "STALE_SNAPSHOT",
                "Snapshot was already consumed by a successful slide update. "
                "Use the returned `next_snapshot_id` or re-run `read_slide_snapshot`.",
                snapshot_id=snapshot_id,
                slide_path=str(retired_snapshot.get("slide_path", "")),
                current_content_hash=str(retired_snapshot.get("current_content_hash", "")),
                next_snapshot_id=str(retired_snapshot.get("next_snapshot_id", "")),
                targets=retired_snapshot.get("targets", []),
                rules=retired_snapshot.get("rules", []),
                repair_candidates=retired_snapshot.get("repair_candidates", []),
                offending_targets=retired_snapshot.get("offending_targets", []),
                repair_context=retired_snapshot.get("repair_context", {}),
                risk_flags=retired_snapshot.get("risk_flags", []),
                validation_diagnostics=retired_snapshot.get("validation_diagnostics", []),
                slide_summary=str(retired_snapshot.get("slide_summary", "")),
                has_more_targets=bool(retired_snapshot.get("has_more_targets", False)),
            )
        return _structured_patch_error(
            "SNAPSHOT_NOT_FOUND",
            f"Snapshot `{snapshot_id}` was not found. Re-run `read_slide_snapshot` first.",
            snapshot_id=snapshot_id,
        )

    resolved_path = str(snapshot.get("resolved_path", "") or "")
    path = Path(resolved_path)
    if not resolved_path or not path.exists():
        return _structured_patch_error(
            "STALE_SNAPSHOT",
            "The slide file referenced by this snapshot no longer exists or moved. "
            "Re-run `read_slide_snapshot` on the current slide path.",
            snapshot_id=snapshot_id,
            slide_path=str(snapshot.get("slide_path", "")),
        )

    try:
        workspace_guard_error = _workspace_contract_error(
            action="apply_slide_patch",
            path=path,
        )
        if workspace_guard_error:
            return workspace_guard_error
        if not isinstance(patch_ops, list):
            return _structured_patch_error(
                "PATCH_APPLY_FAILED",
                "`patch_ops` must be a list of patch operations.",
                snapshot_id=snapshot_id,
            )
        normalized_ops: list[dict[str, Any]] = []
        for index, op in enumerate(patch_ops, start=1):
            op_dict = _patch_op_to_dict(op)
            if not isinstance(op_dict, dict):
                return _structured_patch_error(
                    "PATCH_APPLY_FAILED",
                    f"Patch op #{index} must be an object.",
                    snapshot_id=snapshot_id,
                )
            normalized_ops.append(op_dict)

        current_html = path.read_text(encoding="utf-8", errors="replace")
        current_hash = _compute_text_hash(current_html)
        snapshot_hash = str(snapshot.get("content_hash", "") or "")
        if not expected_hash or expected_hash != snapshot_hash:
            return _structured_patch_error(
                "STALE_SNAPSHOT",
                "The provided `expected_hash` does not match this snapshot. "
                "Retry with the exact `content_hash` returned by `read_slide_snapshot`.",
                snapshot_id=snapshot_id,
                slide_path=str(snapshot.get("slide_path", "")),
                current_content_hash=current_hash,
            )

        persona_retry_error = _persona_retry_patch_block_error(
            path,
            snapshot_id=snapshot_id,
            snapshot=snapshot,
            current_hash=current_hash,
            patch_ops=normalized_ops,
        )
        if persona_retry_error:
            return persona_retry_error

        semantic_warnings, semantic_conflicts = _validate_semantic_scope_for_ops(normalized_ops)
        if semantic_conflicts:
            return _structured_patch_error(
                "PATCH_SCOPE_CONFLICT",
                "Patch operation conflicts with its declared semantic edit scope. "
                "Adjust edit_scope/property_group or set allow_scope_override=true for intentional cross-scope edits.",
                snapshot_id=snapshot_id,
                slide_path=str(snapshot.get("slide_path", "")),
                scope_conflicts=semantic_conflicts,
                semantic_warnings=semantic_warnings,
            )

        target_lookup = _snapshot_target_lookup(snapshot)
        rule_lookup = _snapshot_rule_lookup(snapshot)
        working_target_bindings = {
            key: dict(value)
            for key, value in target_lookup.items()
        }
        working_rule_bindings = {
            key: dict(value)
            for key, value in rule_lookup.items()
        }
        rebound_from_stale_snapshot = False
        stale_retired_ids: list[str] = []
        if current_hash != expected_hash:
            current_payload = _build_slide_snapshot_payload(path, current_html)
            corruption = _detect_corrupted_slide_html(path, current_html, current_payload)
            if corruption:
                retired_ids = _invalidate_slide_snapshots_for_path(path)
                _retire_slide_snapshot_ids(
                    retired_ids,
                    slide_path=str(snapshot.get("slide_path", "")),
                    resolved_path=resolved_path,
                    current_content_hash=current_hash,
                )
                return _structured_patch_error(
                    "CORRUPTED_SLIDE",
                    "Slide HTML is now corrupted or placeholder-only, so patch editing is unavailable. "
                    "Recover the slide with full regenerate instead.",
                    snapshot_id=snapshot_id,
                    slide_path=str(snapshot.get("slide_path", "")),
                    content_hash=current_hash,
                    current_content_hash=current_hash,
                    detected_marker=corruption.get("detected_marker", ""),
                    body_text_preview=corruption.get("body_text_preview", ""),
                )

            current_soup = BeautifulSoup(current_html, "lxml")
            can_rebind = True
            for op_dict in normalized_ops:
                op_name = str(op_dict.get("op", "") or "").strip()
                if op_name == "insert_html":
                    binding = target_lookup.get(str(op_dict.get("anchor_target_id", "") or "").strip())
                    if not binding or _resolve_bound_target(current_soup, binding) is None:
                        can_rebind = False
                        break
                    continue
                if op_name in {"merge_css_rule", "replace_css_rule"}:
                    binding = rule_lookup.get(str(op_dict.get("rule_id", "") or "").strip())
                    if not binding or _resolve_bound_rule(current_html, binding) is None:
                        can_rebind = False
                        break
                    continue
                binding = target_lookup.get(str(op_dict.get("target_id", "") or "").strip())
                if not binding or _resolve_bound_target(current_soup, binding) is None:
                    can_rebind = False
                    break

            if not can_rebind:
                stale_retired_ids = _invalidate_slide_snapshots_for_path(path)
                next_snapshot_id, current_hash, next_payload = _register_slide_snapshot(
                    path,
                    current_html,
                    slide_path=str(snapshot.get("slide_path", "")),
                    snapshot_payload=current_payload,
                )
                rebind_hints = _build_rebind_hints(snapshot, normalized_ops, next_payload)
                _retire_slide_snapshot_ids(
                    stale_retired_ids,
                    slide_path=str(snapshot.get("slide_path", "")),
                    resolved_path=resolved_path,
                    current_content_hash=current_hash,
                    next_snapshot_id=next_snapshot_id,
                    targets=next_payload["targets"],
                    rules=next_payload.get("rules", []),
                    repair_candidates=next_payload.get("repair_candidates", []),
                    offending_targets=next_payload.get("offending_targets", []),
                    repair_context=next_payload.get("repair_context", {}),
                    risk_flags=next_payload.get("risk_flags", []),
                    validation_diagnostics=next_payload.get("validation_diagnostics", []),
                    slide_summary=str(next_payload["slide_summary"] or ""),
                    has_more_targets=bool(next_payload["has_more_targets"]),
                )
                return _stale_snapshot_error_payload(
                    snapshot_id=snapshot_id,
                    slide_path=str(snapshot.get("slide_path", "")),
                    current_content_hash=current_hash,
                    next_snapshot_id=next_snapshot_id,
                    payload=next_payload,
                    rebind_hints=rebind_hints,
                    message=(
                        "Snapshot hash is stale and at least one referenced target or rule changed. "
                        "Use the returned fresh snapshot and `rebind_hints` to continue."
                    ),
                )
            rebound_from_stale_snapshot = True

        corruption = _detect_corrupted_slide_html(path, current_html)
        if corruption:
            return _structured_patch_error(
                "CORRUPTED_SLIDE",
                "Slide HTML is corrupted or placeholder-only, so patch editing is unavailable. "
                "Recover the slide with full regenerate instead.",
                snapshot_id=snapshot_id,
                slide_path=str(snapshot.get("slide_path", "")),
                content_hash=current_hash,
                detected_marker=corruption.get("detected_marker", ""),
                body_text_preview=corruption.get("body_text_preview", ""),
            )

        applied_ops: list[dict[str, Any]] = []
        changed = False
        updated_html = current_html
        layout_diagnostics: list[dict[str, Any]] = []
        auto_repairs: list[dict[str, Any]] = []
        remaining_blockers: list[dict[str, Any]] = []
        for index, op in enumerate(normalized_ops, start=1):
            op_name = str(op.get("op", "") or "").strip()
            if op_name not in {
                "replace_text",
                "replace_html",
                "merge_style",
                "wrap_text_span",
                "remove_node",
                "insert_html",
                "set_attr",
                "remove_attr",
                "merge_css_rule",
                "replace_css_rule",
            }:
                return _structured_patch_error(
                    "UNSUPPORTED_PATCH_OP",
                    f"Unsupported patch op: {op_name or '<empty>'}",
                    snapshot_id=snapshot_id,
                    op=op,
                )

            if op_name in {"merge_css_rule", "replace_css_rule"}:
                rule_id = str(op.get("rule_id", "") or "").strip()
                rule_binding = working_rule_bindings.get(rule_id)
                if not rule_binding:
                    return _structured_patch_error(
                        "RULE_NOT_FOUND",
                        f"Rule `{rule_id}` was not found in the current snapshot.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                if op_name not in set(rule_binding.get("allowed_ops", []) or _RULE_ALLOWED_OPS):
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"Rule `{rule_id}` does not allow op `{op_name}`.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                desired_props = _parse_css_declarations_text(str(op.get("declarations", "") or ""))
                if not desired_props:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"`{op_name}` requires non-empty CSS declarations.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                patched_html, rule_status = _patch_css_rule_html(
                    updated_html,
                    rule_binding,
                    desired_props,
                    mode="merge" if op_name == "merge_css_rule" else "replace",
                )
                if patched_html is None or rule_status is None:
                    return _structured_patch_error(
                        "RULE_NOT_FOUND",
                        f"Rule `{rule_id}` no longer matches the current HTML.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                updated_html = patched_html
                if rule_status.get("status") == "updated":
                    changed = True
                    updated_declarations = dict(rule_status.get("declarations", {}) or {})
                    working_rule_bindings[rule_id] = {
                        **rule_binding,
                        "rule_hash": _compute_rule_hash(
                            str(rule_binding.get("selector", "") or ""),
                            updated_declarations,
                        ),
                        "declarations": updated_declarations,
                        "declarations_preview": _format_css_declarations(updated_declarations),
                    }
                applied_ops.append(
                    {
                        "op": op_name,
                        "rule_id": rule_id,
                        "selector": rule_status.get("selector", ""),
                        "status": rule_status.get("status", ""),
                    }
                )
                continue

            soup = BeautifulSoup(updated_html, "lxml")
            if op_name == "insert_html":
                anchor_id = str(op.get("anchor_target_id", "") or "").strip()
                anchor_binding = working_target_bindings.get(anchor_id)
                if not anchor_binding:
                    return _structured_patch_error(
                        "TARGET_NOT_FOUND",
                        f"Anchor target `{anchor_id}` was not found in the current snapshot.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                if op_name not in set(anchor_binding.get("allowed_ops", []) or _target_allowed_ops(str(anchor_binding.get("kind", "") or ""))):
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"Target `{anchor_id}` does not allow op `{op_name}`.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                target = _resolve_bound_target(soup, anchor_binding)
                if target is None:
                    return _structured_patch_error(
                        "TARGET_NOT_FOUND",
                        f"Anchor target `{anchor_id}` no longer matches the current HTML.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                position = str(op.get("position", "") or "").strip().lower()
                if position not in {"before", "after", "prepend", "append"}:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"Unsupported insert position: {position or '<empty>'}",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                html_fragment = str(op.get("html_fragment", "") or "")
                fragment_error = _fragment_validation_error(html_fragment)
                if fragment_error:
                    return _structured_patch_error(
                        "INVALID_HTML_FRAGMENT",
                        fragment_error,
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                _insert_fragment_near(target, html_fragment, position)
                applied_ops.append(
                    {
                        "op": op_name,
                        "anchor_target_id": anchor_id,
                        "position": position,
                    }
                )
                changed = True
                working_target_bindings[anchor_id] = {
                    **anchor_binding,
                    "dom_path": _build_dom_path(target),
                    "target_hash": _compute_target_hash(target),
                    "selector_hint": _selector_hint_for_tag(target),
                    "text_preview": _tag_text_preview(target),
                }
                updated_html = str(soup)
                _refresh_working_target_bindings(soup, working_target_bindings)
                continue

            target_id = str(op.get("target_id", "") or "").strip()
            target_binding = working_target_bindings.get(target_id)
            if not target_binding:
                return _structured_patch_error(
                    "TARGET_NOT_FOUND",
                    f"Target `{target_id}` was not found in the current snapshot.",
                    snapshot_id=snapshot_id,
                    op=op,
                )
            if op_name not in set(target_binding.get("allowed_ops", []) or _target_allowed_ops(str(target_binding.get("kind", "") or ""))):
                return _structured_patch_error(
                    "PATCH_APPLY_FAILED",
                    f"Target `{target_id}` does not allow op `{op_name}`.",
                    snapshot_id=snapshot_id,
                    op=op,
                )
            target = _resolve_bound_target(soup, target_binding)
            if target is None:
                return _structured_patch_error(
                    "TARGET_NOT_FOUND",
                    f"Target `{target_id}` no longer matches the current HTML.",
                    snapshot_id=snapshot_id,
                    op=op,
                )

            if op_name == "replace_text":
                new_text = str(op.get("text", "") or "")
                current_text = target.get_text("", strip=False)
                if current_text != new_text or any(isinstance(child, Tag) for child in target.children):
                    target.clear()
                    target.append(new_text)
                    changed = True
                    working_target_bindings[target_id] = {
                        **target_binding,
                        "dom_path": _build_dom_path(target),
                        "target_hash": _compute_target_hash(target),
                        "selector_hint": _selector_hint_for_tag(target),
                        "text_preview": _tag_text_preview(target),
                    }
                    updated_html = str(soup)
                    _refresh_working_target_bindings(soup, working_target_bindings)
                applied_ops.append({"op": op_name, "target_id": target_id})
                continue

            if op_name == "replace_html":
                html_fragment = str(op.get("html_fragment", "") or "")
                fragment_error = _fragment_validation_error(html_fragment)
                if fragment_error:
                    return _structured_patch_error(
                        "INVALID_HTML_FRAGMENT",
                        fragment_error,
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                target.clear()
                _append_fragment_into(target, html_fragment)
                changed = True
                working_target_bindings[target_id] = {
                    **target_binding,
                    "dom_path": _build_dom_path(target),
                    "target_hash": _compute_target_hash(target),
                    "selector_hint": _selector_hint_for_tag(target),
                    "text_preview": _tag_text_preview(target),
                }
                updated_html = str(soup)
                _refresh_working_target_bindings(soup, working_target_bindings)
                applied_ops.append({"op": op_name, "target_id": target_id})
                continue

            if op_name == "merge_style":
                desired_props = _parse_css_declarations_text(str(op.get("declarations", "") or ""))
                if not desired_props:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        "`merge_style` requires non-empty CSS declarations.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                merged_style, _ = _merge_inline_style_text(
                    str(target.get("style", "") or ""),
                    desired_props,
                    mode="merge",
                )
                if str(target.get("style", "") or "") != merged_style:
                    target["style"] = merged_style
                    working_target_bindings[target_id] = {
                        **target_binding,
                        "dom_path": _build_dom_path(target),
                        "target_hash": _compute_target_hash(target),
                        "selector_hint": _selector_hint_for_tag(target),
                        "text_preview": _tag_text_preview(target),
                    }
                    updated_html = str(soup)
                    _refresh_working_target_bindings(soup, working_target_bindings)
                    changed = True
                    status = "updated"
                else:
                    status = "already_compliant"
                applied_ops.append(
                    {
                        "op": op_name,
                        "target_id": target_id,
                        "declarations": desired_props,
                        "status": status,
                    }
                )
                continue

            if op_name == "wrap_text_span":
                desired_props = _parse_css_declarations_text(str(op.get("declarations", "") or ""))
                if not desired_props:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        "`wrap_text_span` requires non-empty CSS declarations.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                try:
                    occurrence = int(op.get("occurrence", 1) or 1)
                except (TypeError, ValueError):
                    occurrence = 1
                did_wrap, wrap_status = _wrap_text_span_in_target(
                    soup,
                    target,
                    text_span=str(op.get("text_span", "") or ""),
                    occurrence=occurrence,
                    wrapper_tag=str(op.get("wrapper_tag", "") or "span"),
                    desired_props=desired_props,
                )
                if wrap_status.get("status") == "error":
                    return _structured_patch_error(
                        str(wrap_status.get("error_code", "") or "TEXT_SPAN_NOT_FOUND_OR_UNSAFE"),
                        str(wrap_status.get("error", "") or "Could not safely wrap text span."),
                        snapshot_id=snapshot_id,
                        op=op,
                        occurrences_found=wrap_status.get("occurrences_found", 0),
                    )
                if did_wrap:
                    working_target_bindings[target_id] = {
                        **target_binding,
                        "dom_path": _build_dom_path(target),
                        "target_hash": _compute_target_hash(target),
                        "selector_hint": _selector_hint_for_tag(target),
                        "text_preview": _tag_text_preview(target),
                    }
                    updated_html = str(soup)
                    _refresh_working_target_bindings(soup, working_target_bindings)
                    changed = True
                applied_ops.append(
                    {
                        "op": op_name,
                        "target_id": target_id,
                        "text_span": str(op.get("text_span", "") or ""),
                        "occurrence": occurrence,
                        "wrapper_tag": str(op.get("wrapper_tag", "") or "span"),
                        "declarations": desired_props,
                        "status": wrap_status.get("status", ""),
                    }
                )
                continue

            if op_name == "set_attr":
                attr_name = str(op.get("attr_name", "") or "").strip().lower()
                if attr_name not in _NODE_ATTR_MUTATION_WHITELIST:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"`set_attr` only allows: {', '.join(sorted(_NODE_ATTR_MUTATION_WHITELIST))}.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                raw_value = str(op.get("value", "") or "")
                value = _normalized_style_signature(raw_value) if attr_name == "style" else raw_value
                existing_value = _normalized_tag_attr_value(attr_name, target.get(attr_name, ""))
                if existing_value != value:
                    target[attr_name] = value
                    working_target_bindings[target_id] = {
                        **target_binding,
                        "dom_path": _build_dom_path(target),
                        "target_hash": _compute_target_hash(target),
                        "selector_hint": _selector_hint_for_tag(target),
                        "text_preview": _tag_text_preview(target),
                    }
                    updated_html = str(soup)
                    _refresh_working_target_bindings(soup, working_target_bindings)
                    changed = True
                    status = "updated"
                else:
                    status = "already_compliant"
                applied_ops.append(
                    {
                        "op": op_name,
                        "target_id": target_id,
                        "attr_name": attr_name,
                        "status": status,
                    }
                )
                continue

            if op_name == "remove_attr":
                attr_name = str(op.get("attr_name", "") or "").strip().lower()
                if attr_name not in _NODE_ATTR_MUTATION_WHITELIST:
                    return _structured_patch_error(
                        "PATCH_APPLY_FAILED",
                        f"`remove_attr` only allows: {', '.join(sorted(_NODE_ATTR_MUTATION_WHITELIST))}.",
                        snapshot_id=snapshot_id,
                        op=op,
                    )
                if target.has_attr(attr_name):
                    del target[attr_name]
                    working_target_bindings[target_id] = {
                        **target_binding,
                        "dom_path": _build_dom_path(target),
                        "target_hash": _compute_target_hash(target),
                        "selector_hint": _selector_hint_for_tag(target),
                        "text_preview": _tag_text_preview(target),
                    }
                    updated_html = str(soup)
                    _refresh_working_target_bindings(soup, working_target_bindings)
                    changed = True
                    status = "updated"
                else:
                    status = "already_absent"
                applied_ops.append(
                    {
                        "op": op_name,
                        "target_id": target_id,
                        "attr_name": attr_name,
                        "status": status,
                    }
                )
                continue

            if op_name == "remove_node":
                target.decompose()
                working_target_bindings.pop(target_id, None)
                applied_ops.append({"op": op_name, "target_id": target_id})
                changed = True
                updated_html = str(soup)
                _refresh_working_target_bindings(soup, working_target_bindings)
                continue

        if changed:
            structure_issues = _collect_slide_structure_issues(updated_html)
            if structure_issues:
                return _structured_patch_error(
                    "INVALID_SLIDE_STRUCTURE",
                    "Patched slide would introduce invalid structure. "
                    + structure_issues[0],
                    snapshot_id=snapshot_id,
                    slide_path=str(snapshot.get("slide_path", "")),
                    issues=structure_issues,
                )
            workspace_guard_error = _workspace_contract_error(
                action="apply_slide_patch",
                path=path,
                html_content=updated_html,
            )
            if workspace_guard_error:
                return workspace_guard_error
            updated_html = _normalize_html_image_sources(updated_html, html_path=path)
            updated_html, marker_report = _normalize_anchored_elements_html(updated_html, path=path)
            if marker_report.get("changed"):
                changed = True
            updated_html, layout_diagnostics, auto_repairs, remaining_blockers = _prepare_html_layout_for_write(
                updated_html,
                path,
                force_regenerate=False,
            )
            if remaining_blockers:
                return _structured_patch_error(
                    "LAYOUT_VALIDATION_FAILED",
                    "Patched slide would still violate static layout validation after deterministic repair.",
                    snapshot_id=snapshot_id,
                    slide_path=str(snapshot.get("slide_path", "")),
                    layout_diagnostics=layout_diagnostics,
                    auto_repairs=auto_repairs,
                    remaining_blockers=remaining_blockers,
                    repair_strategy=(
                        "Patch the layout container instead of the text leaf: set box-sizing:border-box, "
                        "cap height/max-height, reduce padding/gap, or rebalance the content into columns."
                    ),
                )
            path.write_text(updated_html, encoding="utf-8")
            _invalidate_slide_validation_for_path(path)
            _mark_slide_modified(path)
            _remember_slide_observation(path)

        semantic_audit = _build_semantic_patch_audit(current_html, updated_html, normalized_ops)
        if semantic_audit.get("unexpected_property_changes"):
            semantic_warnings.append(
                {
                    "code": "unexpected_property_changes",
                    "message": (
                        "The final HTML diff includes property groups outside the explicit edit scope. "
                        "Review semantic_audit.unexpected_property_changes."
                    ),
                    "count": len(semantic_audit.get("unexpected_property_changes", []) or []),
                }
            )

        new_hash = _compute_text_hash(updated_html)
        next_snapshot_id = ""
        next_targets: list[dict[str, Any]] = []
        next_rules: list[dict[str, Any]] = []
        next_repair_candidates: list[dict[str, Any]] = []
        next_offending_targets: list[dict[str, Any]] = list(snapshot.get("offending_targets", []) or [])
        next_repair_context: dict[str, Any] = dict(snapshot.get("repair_context", {}) or {})
        next_risk_flags: list[str] = []
        next_validation_diagnostics: list[dict[str, Any]] = list(snapshot.get("validation_diagnostics", []) or [])
        has_more_targets = False
        slide_summary = ""
        if changed or rebound_from_stale_snapshot:
            retired_ids = _invalidate_slide_snapshots_for_path(path)
            next_payload = _build_slide_snapshot_payload(path, updated_html)
            next_snapshot_id, new_hash, next_payload = _register_slide_snapshot(
                path,
                updated_html,
                slide_path=str(snapshot.get("slide_path", "")),
                snapshot_payload=next_payload,
            )
            next_targets = next_payload["targets"]
            next_rules = next_payload.get("rules", [])
            next_repair_candidates = next_payload.get("repair_candidates", [])
            next_offending_targets = next_payload.get("offending_targets", [])
            next_repair_context = next_payload.get("repair_context", {})
            next_risk_flags = next_payload.get("risk_flags", [])
            next_validation_diagnostics = next_payload.get("validation_diagnostics", [])
            has_more_targets = bool(next_payload["has_more_targets"])
            slide_summary = str(next_payload["slide_summary"] or "")
            _retire_slide_snapshot_ids(
                retired_ids,
                slide_path=str(snapshot.get("slide_path", "")),
                resolved_path=resolved_path,
                current_content_hash=new_hash,
                next_snapshot_id=next_snapshot_id,
                targets=next_targets,
                rules=next_rules,
                repair_candidates=next_repair_candidates,
                offending_targets=next_offending_targets,
                repair_context=next_repair_context,
                risk_flags=next_risk_flags,
                validation_diagnostics=next_validation_diagnostics,
                slide_summary=slide_summary,
                has_more_targets=has_more_targets,
            )
        repair_audit, needs_render_recheck, resolved_diagnostics, remaining_diagnostics = (
            _build_repair_patch_audit(
                snapshot=snapshot,
                patch_ops=normalized_ops,
                next_validation_diagnostics=next_validation_diagnostics,
            )
        )
        return json.dumps(
            {
                "success": True,
                "slide_path": str(snapshot.get("slide_path", "")),
                "snapshot_id": snapshot_id,
                "applied_ops": applied_ops,
                "changed": changed,
                "new_content_hash": new_hash,
                "next_snapshot_id": next_snapshot_id,
                "next_content_hash": new_hash,
                "rebound_from_stale_snapshot": rebound_from_stale_snapshot,
                "targets": next_targets,
                "rules": next_rules,
                "repair_candidates": next_repair_candidates,
                "offending_targets": next_offending_targets,
                "repair_context": next_repair_context,
                "risk_flags": next_risk_flags,
                "validation_diagnostics": next_validation_diagnostics,
                "has_more_targets": has_more_targets,
                "slide_summary": slide_summary,
                "semantic_warnings": semantic_warnings,
                "semantic_audit": semantic_audit,
                "repair_audit": repair_audit,
                "needs_render_recheck": needs_render_recheck,
                "resolved_diagnostics": resolved_diagnostics,
                "remaining_diagnostics": remaining_diagnostics,
                "layout_diagnostics": layout_diagnostics,
                "auto_repairs": auto_repairs,
                "remaining_blockers": remaining_blockers,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        error(f"apply_slide_patch failed for snapshot {snapshot_id}: {e}")
        return _structured_patch_error(
            "PATCH_APPLY_FAILED",
            f"apply_slide_patch failed: {e}",
            snapshot_id=snapshot_id,
        )


@mcp.tool()
def scan_slide_index(directory: str = "") -> str:
    """
    Summarize the current slide deck structure for planning global modifications.

    Returns a JSON summary with slide file paths, detected titles, title selector
    hints, and a few lightweight structure signals. Prefer this over reading every
    slide when you need to plan batch edits across many pages.
    """
    try:
        if directory:
            slide_dir = _normalize_path(directory)
            if slide_dir.suffix:
                slide_dir = slide_dir.parent
        else:
            slide_dir = _get_active_slide_dir()

        if slide_dir is None or not slide_dir.exists():
            return json.dumps(
                {"success": False, "error": "No active slide directory found", "slides": []},
                ensure_ascii=False,
                indent=2,
            )

        slide_paths = _collect_canonical_slides(slide_dir)
        results = [
            _build_slide_index_entry(path, Path.cwd())
            for _, path in slide_paths
        ]
        payload = {
            "success": True,
            "slide_dir": str(slide_dir.resolve()),
            "slide_count": len(results),
            "slides": results,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps(
            {"success": False, "error": f"scan_slide_index failed: {e}", "slides": []},
            ensure_ascii=False,
            indent=2,
        )


@mcp.tool()
def batch_update_css_rule(
    file_paths: list[str],
    selector: str,
    declarations: str,
    mode: Literal["merge", "replace"] = "merge",
) -> str:
    """
    Apply the same CSS rule update across multiple slide HTML files.

    This is useful for global style changes such as making all slide titles blue
    or unifying footer typography, without rewriting each entire HTML file.
    """
    desired_props = _parse_css_declarations_text(declarations)
    if not file_paths:
        return json.dumps(
            {"success": False, "error": "file_paths must not be empty", "results": []},
            ensure_ascii=False,
            indent=2,
        )
    if not selector.strip():
        return json.dumps(
            {"success": False, "error": "selector must not be empty", "results": []},
            ensure_ascii=False,
            indent=2,
        )
    if not desired_props:
        return json.dumps(
            {"success": False, "error": "declarations must contain at least one CSS property", "results": []},
            ensure_ascii=False,
            indent=2,
        )

    results: list[dict[str, object]] = []
    overall_success = True
    for raw_path in file_paths:
        path = _normalize_path(raw_path)
        entry: dict[str, object] = {
            "file_path": str(raw_path),
            "selector": selector,
        }
        try:
            if not path.exists() or path.suffix.lower() != ".html":
                entry["status"] = "error"
                entry["error"] = "HTML file not found"
                overall_success = False
                results.append(entry)
                continue

            html_text = path.read_text(encoding="utf-8")
            selector_present = _selector_presence_hint(html_text, selector)
            updated_html, status = _apply_css_rule_to_html(
                html_text,
                selector,
                desired_props,
                mode=mode,
            )
            if updated_html != html_text:
                updated_html = _normalize_html_image_sources(updated_html, html_path=path)
                updated_html, marker_report = _normalize_anchored_elements_html(updated_html, path=path)
                path.write_text(updated_html, encoding="utf-8")
                _invalidate_slide_validation_for_path(path)
                _mark_slide_modified(path)
                if marker_report.get("changed"):
                    entry["temp_pref_marker_normalized"] = marker_report
            _remember_slide_observation(path)

            entry["status"] = status
            entry["selector_present"] = selector_present
            results.append(entry)
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            overall_success = False
            results.append(entry)

    payload = {
        "success": overall_success,
        "selector": selector,
        "declarations": desired_props,
        "mode": mode,
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def batch_update_semantic_style(
    file_paths: list[str],
    target_kind: Literal[
        "slide_title",
        "body_text",
        "footer",
        "caption",
        "pill_tag",
        "table_cell",
        "chart_axis_label",
        "chart_data_label",
        "legend_label",
        "callout",
        "any_text",
    ],
    declarations: str,
    mode: Literal["merge", "replace"] = "merge",
    property_group: str = "",
    edit_scope: str = "",
    allow_scope_override: bool = False,
) -> str:
    """
    Apply one style update to a semantic slide target across multiple HTML files.

    Unlike `batch_update_css_rule`, this tool infers the most likely selector(s)
    per file based on real HTML/CSS structure, which is helpful when different
    slides use `h1`, `.title`, `.h1`, or `h2` for their main title.
    """
    desired_props = _parse_css_declarations_text(declarations)
    if not file_paths:
        return json.dumps(
            {"success": False, "error": "file_paths must not be empty", "results": []},
            ensure_ascii=False,
            indent=2,
        )
    if not desired_props:
        return json.dumps(
            {"success": False, "error": "declarations must contain at least one CSS property", "results": []},
            ensure_ascii=False,
            indent=2,
        )

    semantic_op = {
        "op": "batch_update_semantic_style",
        "declarations": declarations,
        "property_group": property_group,
        "edit_scope": edit_scope,
        "allow_scope_override": allow_scope_override,
    }
    semantic_warnings, semantic_conflicts = _validate_semantic_scope_for_ops([semantic_op])
    if semantic_conflicts:
        return json.dumps(
            {
                "success": False,
                "error_code": "PATCH_SCOPE_CONFLICT",
                "error": (
                    "Batch style update conflicts with its declared semantic edit scope. "
                    "Adjust edit_scope/property_group or set allow_scope_override=true."
                ),
                "scope_conflicts": semantic_conflicts,
                "semantic_warnings": semantic_warnings,
                "results": [],
            },
            ensure_ascii=False,
            indent=2,
        )

    results: list[dict[str, object]] = []
    overall_success = True
    for raw_path in file_paths:
        path = _normalize_path(raw_path)
        entry: dict[str, object] = {
            "file_path": str(raw_path),
            "target_kind": target_kind,
        }
        try:
            if not path.exists() or path.suffix.lower() != ".html":
                entry["status"] = "error"
                entry["error"] = "HTML file not found"
                overall_success = False
                results.append(entry)
                continue

            html_text = path.read_text(encoding="utf-8")
            selectors = _infer_semantic_selectors(html_text, target_kind)
            if not selectors:
                entry["status"] = "error"
                entry["error"] = f"No selectors inferred for target_kind={target_kind}"
                overall_success = False
                results.append(entry)
                continue

            updated_html = html_text
            selector_results: list[dict[str, str]] = []
            for selector in selectors:
                updated_html, selector_status = _apply_css_rule_to_html(
                    updated_html,
                    selector,
                    desired_props,
                    mode=mode,
                )
                selector_results.append({
                    "selector": selector,
                    "status": selector_status,
                })

            updated_html, inline_patch_results = _patch_existing_semantic_inline_styles(
                updated_html,
                target_kind,
                desired_props,
                mode=mode,
            )

            if updated_html != html_text:
                updated_html = _normalize_html_image_sources(updated_html, html_path=path)
                updated_html, marker_report = _normalize_anchored_elements_html(updated_html, path=path)
                path.write_text(updated_html, encoding="utf-8")
                _invalidate_slide_validation_for_path(path)
                _mark_slide_modified(path)
                if marker_report.get("changed"):
                    entry["temp_pref_marker_normalized"] = marker_report
            _remember_slide_observation(path)

            semantic_audit = _build_semantic_patch_audit(html_text, updated_html, [semantic_op])
            entry["status"] = "updated" if updated_html != html_text else "already_compliant"
            entry["selectors"] = selectors
            entry["selector_results"] = selector_results
            entry["semantic_audit"] = semantic_audit
            if inline_patch_results:
                entry["inline_patch_results"] = inline_patch_results
            results.append(entry)
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)
            overall_success = False
            results.append(entry)

    payload = {
        "success": overall_success,
        "target_kind": target_kind,
        "declarations": desired_props,
        "mode": mode,
        "property_group": _normalize_semantic_edit_scope(property_group),
        "edit_scope": _normalize_semantic_edit_scope(edit_scope),
        "semantic_warnings": semantic_warnings,
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def patch_semantic_inline_style(
    file_path: str,
    target_kind: Literal[
        "slide_title",
        "body_text",
        "footer",
        "caption",
        "pill_tag",
        "table_cell",
        "chart_axis_label",
        "chart_data_label",
        "legend_label",
        "callout",
        "any_text",
    ],
    declarations: str,
    scope: Literal["first", "all"] = "first",
    mode: Literal["merge", "replace"] = "merge",
) -> str:
    """
    Patch inline styles on semantic target elements inside a single slide HTML file.

    Use this for fine-grained exception repair after a deck-level batch edit, when
    rewriting the full HTML would be unnecessarily risky.
    """
    desired_props = _parse_css_declarations_text(declarations)
    if not desired_props:
        return json.dumps(
            {"success": False, "error": "declarations must contain at least one CSS property"},
            ensure_ascii=False,
            indent=2,
        )

    path = _normalize_path(file_path)
    if not path.exists() or path.suffix.lower() != ".html":
        return json.dumps(
            {"success": False, "error": f"HTML file not found: {file_path}"},
            ensure_ascii=False,
            indent=2,
        )

    try:
        html_text = path.read_text(encoding="utf-8")
        matches = _find_semantic_opening_tag_matches(html_text, target_kind)
        if not matches:
            return json.dumps(
                {
                    "success": False,
                    "error": f"No semantic target matches found for target_kind={target_kind}",
                    "file_path": file_path,
                },
                ensure_ascii=False,
                indent=2,
            )

        selected_matches = matches[:1] if scope == "first" else matches
        updated_html = html_text
        patched_entries: list[dict[str, object]] = []
        for match in sorted(selected_matches, key=lambda item: int(item["start"]), reverse=True):
            patched_tag, status = _rewrite_opening_tag_with_style(
                str(match["opening_tag"]),
                desired_props,
                mode=mode,
            )
            if patched_tag != match["opening_tag"]:
                updated_html = (
                    updated_html[: int(match["start"])]
                    + patched_tag
                    + updated_html[int(match["end"]) :]
                )
            patched_entries.append(
                {
                    "tag": match["tag"],
                    "classes": match["classes"],
                    "status": status,
                }
            )

        if updated_html != html_text:
            updated_html = _normalize_html_image_sources(updated_html, html_path=path)
            updated_html, marker_report = _normalize_anchored_elements_html(updated_html, path=path)
            path.write_text(updated_html, encoding="utf-8")
            _invalidate_slide_validation_for_path(path)
            _mark_slide_modified(path)
        else:
            marker_report = {"changed": False}
        _remember_slide_observation(path)

        return json.dumps(
            {
                "success": True,
                "file_path": file_path,
                "target_kind": target_kind,
                "scope": scope,
                "mode": mode,
                "declarations": desired_props,
                "patched": list(reversed(patched_entries)),
                "changed": updated_html != html_text,
                "temp_pref_marker_normalized": marker_report if marker_report.get("changed") else {},
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "file_path": file_path},
            ensure_ascii=False,
            indent=2,
        )


@mcp.tool()
def list_memory_artifacts(category: str = "all") -> str:
    """
    List memory system artifacts for debugging and observability.

    Args:
        category: Category to list. Options:
            - "all": List all artifact directories
            - "rounds": List round artifacts (.memory/rounds/)
            - "injections": List memory injection files (.memory/injections/)
            - "extractions": List legacy extraction artifacts (.memory/extractions/), if present
            - "params": List parameter snapshots (.memory/params/)

    Returns:
        Formatted listing of memory artifacts with sizes and timestamps.
    """
    try:
        memory_dir = Path(".memory")
        if not memory_dir.exists():
            return "No .memory directory found in workspace"

        lines = []

        def list_dir_contents(dir_path: Path, prefix: str = "") -> list[str]:
            """递归列出目录内容"""
            result = []
            if not dir_path.exists():
                return result
            for p in sorted(dir_path.iterdir()):
                if p.is_dir():
                    result.append(f"{prefix}{p.name}/")
                    # 只展开一层子目录
                    for sub in sorted(p.iterdir()):
                        if sub.is_file():
                            size = sub.stat().st_size
                            if size > 1024:
                                size_str = f"{size / 1024:.1f}KB"
                            else:
                                size_str = f"{size}B"
                            result.append(f"{prefix}  {sub.name}  ({size_str})")
                        elif sub.is_dir():
                            result.append(f"{prefix}  {sub.name}/")
                else:
                    size = p.stat().st_size
                    if size > 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    result.append(f"{prefix}{p.name}  ({size_str})")
            return result

        categories = {
            "rounds": memory_dir / "rounds",
            "injections": memory_dir / "injections",
            "extractions": memory_dir / "extractions",
            "params": memory_dir / "params",
        }

        if category == "all":
            lines.append("=== Memory Artifacts ===")
            for cat_name, cat_path in categories.items():
                if cat_path.exists():
                    lines.append(f"\n[{cat_name}]")
                    lines.extend(list_dir_contents(cat_path, "  "))
        elif category in categories:
            cat_path = categories[category]
            if cat_path.exists():
                lines.append(f"=== {category} ===")
                lines.extend(list_dir_contents(cat_path))
            else:
                return f"No {category} directory found"
        else:
            return f"Unknown category: {category}. Options: all, rounds, injections, extractions, params"

        if not lines:
            return "No memory artifacts found"
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing memory artifacts: {e}"


@mcp.tool()
def thinking(thought: str):
    """This tool is for explicitly reasoning about the current round state and next actions."""
    info(f"Thought: {thought}")
    return thought


@mcp.tool()
def finalize(outcome: str) -> str:
    """
    When all tasks are finished, call this function to finalize the loop.
    Args:
        outcome (str): The path to the final outcome file or directory.
    """
    # here we conduct some final checks on agent's outcome
    path = Path(outcome)
    if not path.exists():
        return f"Outcome {outcome} does not exist"
    html_files = sorted(path.glob("*.html")) if path.is_dir() else []
    # Use module-level variable for reliable agent tracking
    agent_name = get_current_agent()
    if agent_name == "Researcher":
        md_dir = path.parent
        if not (path.is_file() and path.suffix == ".md"):
            # 提供更具指导性的错误消息
            if path.is_dir():
                # 尝试在目录中找到markdown文件
                md_files = list(path.glob("*.md"))
                if md_files:
                    return f"Error: You provided a directory path. Please call finalize with the markdown FILE path, not the directory. Found these files: {', '.join(f.name for f in md_files)}. Use the full path like: {md_files[0].resolve()}"
                else:
                    return f"Error: {path} is a directory, but no .md files found. Please provide the complete markdown file path (e.g., /path/to/file.md)"
            else:
                return f"Error: Outcome must be a markdown file with .md extension. You provided: {path}. Please provide the complete path to the .md file."
        with open(path, encoding="utf-8") as f:
            content = f.read()

        # 校验 --- 分隔符：内容多但没分页时拒绝 finalize
        _pages = _split_markdown_slide_pages(content)
        _h2_count = content.count("\n## ")
        if len(_pages) == 1 and _h2_count >= 3:
            return (
                f"Error: Manuscript has {_h2_count} sections but no '---' page separators. "
                f"Each slide's content MUST be separated by a line containing only '---'. "
                f"Please rewrite the manuscript with proper '---' separators between slides, "
                f"then save it again with write_markdown_file and call finalize with the updated file."
            )

        try:
            content = re.sub(
                r"!\[(.*?)\]\((.*?)\)",
                lambda match: _rewrite_image_link(match, md_dir),
                content,
            )
            shutil.copyfile(path, md_dir / ("." + path.name))
            path.write_text(content, encoding="utf-8")
        except Exception as e:
            error(f"Failed to rewrite image links: {e}")

    elif agent_name == "TemplatePlanner":
        if not (path.is_file() and path.suffix == ".pptx"):
            return "Outcome file should be a pptx file"
        prs = Presentation(str(path))
        if len(prs.slides) <= 0:
            return "PPTX file should contain at least one slide"
    elif agent_name == "DeckDesigner":
        if len(html_files) <= 0:
            return "Outcome path should be a directory containing HTML files"
        if not all(f.stem.startswith("slide_") for f in html_files):
            return "All HTML files should start with 'slide_'"
    elif agent_name == "RevisionEditor":
        if path.is_file() and path.suffix == ".html":
            pass  # single slide modification — valid
        elif path.is_dir():
            if len(html_files) <= 0:
                return "Outcome path should contain at least one HTML slide file"
        else:
            warning(f"RevisionEditor agent outcome is not an HTML file or directory: {path}")
    else:
        warning(f"Unverifiable agent: {agent_name}")

    if path.is_dir() and html_files and (
        agent_name in {"DeckDesigner", "RevisionEditor"}
        or any(f.stem.startswith("slide_") for f in html_files)
    ):
        for html_file in html_files:
            _normalize_temp_pref_marker_file(html_file)
        html_files = sorted(path.glob("*.html")) if path.is_dir() else html_files
        validation_error = _format_design_finalize_validation_error(path, agent_name=agent_name)
        if validation_error:
            return validation_error

    if LOCAL_TODO_CSV_PATH.exists():
        LOCAL_TODO_CSV_PATH.unlink()
    if LOCAL_TODO_LOCK_PATH.exists():
        LOCAL_TODO_LOCK_PATH.unlink()

    info(f"Agent {agent_name} finalized the outcome: {outcome}")
    return outcome


def _page_contract_has_high_priority_signal(page_contract: dict[str, Any]) -> bool:
    if not isinstance(page_contract, dict):
        return False
    for list_key in ("hard_requirements", "component_requirements"):
        if any(str(item or "").strip() for item in page_contract.get(list_key, []) or []):
            return True
    style_requirements = page_contract.get("style_requirements")
    if isinstance(style_requirements, dict) and style_requirements.get("css_values"):
        return True
    for key in ("page_role", "persona_signal", "required_component", "selected_layout", "bound_asset_path"):
        if str(page_contract.get(key, "") or "").strip():
            return True
    return False


def _parse_json_object_from_text(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _profile_page_verifier_llm_decision(
    *,
    page_contract: dict[str, Any],
    evaluation: dict[str, Any],
    inspect_digest: str,
) -> dict[str, Any]:
    try:
        try:
            _, llm = CONFIG.resolve_llm("fast_model")
        except Exception:
            _, llm = _get_current_agent_llm_config()
        prompt = (
            "You are a lightweight slide verifier. Decide only whether the generated slide clearly violates "
            "the current page-level contract. Do not fail vague style wishes. If evidence is ambiguous, return pass.\n\n"
            "Return JSON with keys: verdict, reason, repair_focus.\n"
            "Allowed verdicts: pass, retry_required.\n\n"
            f"Page contract:\n{json.dumps(page_contract, ensure_ascii=False, indent=2)[:1800]}\n\n"
            f"Deterministic findings:\n{json.dumps(evaluation, ensure_ascii=False, indent=2)[:1600]}\n\n"
            f"Inspect digest:\n{inspect_digest[:1200]}"
        )
        response = await llm.run(
            messages=[{"role": "user", "content": prompt}],
            retry_times=resolve_llm_retry_times(llm, minimum=1),
            request_kwargs={
                "temperature": 0.0,
                "max_tokens": 220,
                "reasoning_effort": "minimal",
            },
        )
        parsed = _parse_json_object_from_text(extract_response_text(response))
        verdict = str((parsed or {}).get("verdict", "pass") or "pass").strip().lower()
        if verdict not in {"pass", "retry_required"}:
            verdict = "pass"
        return {
            "verdict": verdict,
            "reason": str((parsed or {}).get("reason", "") or "").strip(),
            "repair_focus": str((parsed or {}).get("repair_focus", "") or "").strip(),
            "source": "llm",
        }
    except Exception as exc:
        warning("profile page verifier LLM fallback failed (non-fatal): %s", exc)
        return {
            "verdict": "pass",
            "reason": "",
            "repair_focus": "",
            "source": "llm_failed",
        }


async def _run_profile_page_verifier(
    *,
    html_path: Path,
    current_html: str,
    props_summary: str,
    structure_issues: list[str],
    compliance_text: str,
) -> dict[str, Any] | None:
    if get_current_agent().strip().lower() != "deckdesigner":
        return None
    try:
        from memslides.pipelines.generation_support import evaluate_profile_execution_page
        from memslides.runtime.deck_execution_state import (
            load_deck_execution_state,
            record_persona_verdict,
        )
    except Exception:
        return None

    match = _SLIDE_CANONICAL_RE.fullmatch(html_path.name)
    if not match:
        return None
    page_num = int(match.group(1))
    deck_state = load_deck_execution_state() or {}
    slide_meta = ((deck_state.get("slides") or {}).get(str(page_num)) or {})
    if not isinstance(slide_meta, dict):
        return None

    design_plan_text = ""
    design_plan_path = _find_design_plan_path(_resolve_workspace_root())
    if design_plan_path and design_plan_path.exists():
        try:
            design_plan_text = design_plan_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            design_plan_text = ""

    evaluation = evaluate_profile_execution_page(
        page_contract=slide_meta,
        slide_html=current_html,
        slide_path=html_path,
        design_plan_text=design_plan_text,
        slide_meta=slide_meta,
    )
    if not evaluation.get("applicable"):
        return None

    existing_retry_count = int(slide_meta.get("persona_retry_count", 0) or 0)
    repair_focus = str(evaluation.get("repair_focus", "") or "").strip()
    final_verdict = "pass"
    verdict_source = "deterministic"
    llm_result: dict[str, Any] | None = None

    if evaluation.get("deterministic_verdict") == "retry_required":
        final_verdict = "unresolved_release" if existing_retry_count >= 2 else "retry_required"
    elif evaluation.get("needs_llm_review") and _page_contract_has_high_priority_signal(slide_meta):
        inspect_digest = (
            f"page={page_num}\n"
            f"props:\n{props_summary}\n"
            f"structural_issues={structure_issues[:3]}\n"
            f"compliance={compliance_text[:500]}"
        )
        llm_result = await _profile_page_verifier_llm_decision(
            page_contract=slide_meta,
            evaluation=evaluation,
            inspect_digest=inspect_digest,
        )
        verdict_source = str(llm_result.get("source", "llm") or "llm")
        if llm_result.get("verdict") == "retry_required":
            final_verdict = "unresolved_release" if existing_retry_count >= 2 else "retry_required"
            repair_focus = str(llm_result.get("repair_focus", "") or "").strip() or repair_focus
        else:
            final_verdict = "pass"

    if final_verdict != "retry_required":
        repair_focus = repair_focus if final_verdict == "unresolved_release" else ""

    detail_payload = {
        **evaluation,
        "final_verdict": final_verdict,
        "verdict_source": verdict_source,
    }
    if llm_result:
        detail_payload["llm_result"] = llm_result
    record_persona_verdict(
        html_path,
        verdict=final_verdict,
        repair_focus=repair_focus,
        details=detail_payload,
    )
    return detail_payload


@mcp.tool()
async def inspect_slide(
    html_file: str,
    aspect_ratio: Literal["16:9", "4:3", "A1", "A2", "A3", "A4"] = "16:9",
) -> list | str:
    """
    Render and inspect a slide HTML file. Returns a rendered image (if the model
    supports vision) plus a structured text summary of the slide's CSS properties,
    change status, and referenced resources.

    Use this after writing/modifying a slide to verify your changes took effect.
    Once the summary confirms your modifications, call `finalize` to complete.

    Returns:
        list: [ImageContent, TextContent] when vision is available
        str: Text-only summary when vision is not available, or error message
    """
    html_path = _normalize_path(html_file).absolute()
    if not (html_path.exists() and html_path.suffix == ".html"):
        return f"HTML path {html_path} does not exist or is not an HTML file"

    # ── 1. Change detection (before rendering) ───────────────────────
    changed, old_size, new_size = _check_slide_changed(html_path)

    skip_preview_render = not _can_render_slide_preview()

    current_html = html_path.read_text(encoding="utf-8", errors="replace")
    diagnostics: list[dict[str, Any]] = []
    workspace_guard_error = _workspace_contract_error(
        action="inspect_slide",
        path=html_path,
        html_content=current_html,
    )
    if workspace_guard_error:
        return workspace_guard_error
    marker_report = _normalize_temp_pref_marker_file(html_path)
    if marker_report.get("changed"):
        current_html = html_path.read_text(encoding="utf-8", errors="replace")
        changed = True
        new_size = html_path.stat().st_size
        diagnostics.append(
            {
                "code": "temp_pref_marker_normalized",
                "severity": "info",
                "message": "Anchored TEMP-PREF marker rules/nodes were normalized to one visible in-canvas anchored element.",
                "source": "inspect",
                "marker_text": marker_report.get("marker_text", ""),
                "removed_nodes": marker_report.get("removed_nodes", 0),
                "removed_css_rules": marker_report.get("removed_css_rules", 0),
            }
        )

    diagnostics.extend(
        _build_static_slide_diagnostics(
            html_path,
            current_html,
            aspect_ratio=aspect_ratio,
            source="inspect",
        )
    )
    missing_assets = _missing_local_html_assets(html_path, current_html)
    if missing_assets:
        diagnostics.append(
            {
                "code": "missing_local_asset",
                "severity": "error",
                "message": f"Missing local image/CSS assets: {', '.join(missing_assets[:5])}",
                "source": "inspect",
            }
        )
        error_message = _format_missing_asset_error(html_path, missing_assets)
        _record_slide_validation_result(
            html_path,
            success=False,
            message=error_message,
            aspect_ratio=aspect_ratio,
            diagnostics=diagnostics,
        )
        try:
            from memslides.runtime.deck_execution_state import record_slide_inspected

            record_slide_inspected(html_path, success=False)
        except Exception:
            pass
        return error_message
    if skip_preview_render:
        _record_slide_validation_result(
            html_path,
            success=True,
            message="preview render skipped",
            aspect_ratio=aspect_ratio,
            diagnostics=diagnostics,
        )
        summary_parts = [
            "✅ Slide validated in static mode (preview render skipped).",
        ]
    else:
        # ── 2. HTML→PPTX conversion (validates layout) ───────────────────
        try:
            await convert_html_to_pptx_with_retry(
                html_path,
                aspect_ratio=aspect_ratio,
                allow_skip_layout_validation_fallback=False,
                preserve_source_html=True,
            )
        except Exception as e:
            error_str = str(e)
            render_failure_count = _record_slide_render_failure(html_path)
            export_diagnostics = [
                item
                for item in getattr(e, "pptx_export_diagnostics", []) or []
                if isinstance(item, dict)
            ]
            diagnostics = _merge_validation_diagnostics(diagnostics, export_diagnostics)
            _record_slide_validation_result(
                html_path,
                success=False,
                message=error_str.splitlines()[0][:240],
                aspect_ratio=aspect_ratio,
                diagnostics=diagnostics,
            )
            try:
                from memslides.runtime.deck_execution_state import record_slide_inspected

                record_slide_inspected(html_path, success=False)
            except Exception:
                pass
            # 为尺寸不匹配提供更具体的修复指导
            if "aspect ratio mismatch" in error_str or "don't match presentation layout" in error_str:
                size_map = {"16:9": "1280x720", "4:3": "960x720", "A1": "2244x3178", "A2": "1587x2244", "A3": "1122x1587", "A4": "794x1123"}
                expected = size_map.get(aspect_ratio, "1280x720")
                return (
                    f"❌ Render failed: {e}\n\n"
                    f"🔧 FIX: The <body> CSS dimensions are wrong. "
                    f"For {aspect_ratio} layout, set exactly:\n"
                    f"  body {{ width: {expected.split('x')[0]}px; height: {expected.split('x')[1]}px; }}\n"
                    f"Rewrite the slide with correct body dimensions and call inspect_slide again."
                )
            if (
                "clipped outside the slide canvas" in error_str
                or "HTML content overflows body" in error_str
                or "too close to bottom edge" in error_str
                or "bottom_safe_zone" in error_str
                or "bottom safe" in error_str.lower()
            ):
                restored_persona_backup = _restore_persona_repair_backup_after_failed_repair(
                    html_path,
                    aspect_ratio=aspect_ratio,
                    failure_message=error_str,
                    diagnostics=diagnostics,
                )
                if restored_persona_backup:
                    return restored_persona_backup
                fallback_message = ""
                if render_failure_count >= 2:
                    fallback_success, fallback_message, fallback_diagnostics = (
                        await _try_deterministic_fallback_inspect_repair(
                            html_path,
                            html_file=html_file,
                            current_html=current_html,
                            diagnostics=diagnostics,
                            aspect_ratio=aspect_ratio,
                            original_error=error_str,
                        )
                    )
                    if fallback_success:
                        return fallback_message
                    diagnostics = _merge_validation_diagnostics(diagnostics, fallback_diagnostics)
                repeated_note = (
                    _repeated_render_failure_rewrite_note(
                        html_path,
                        html_file=html_file,
                        failure_count=render_failure_count,
                    )
                    if render_failure_count >= 2
                    else ""
                )
                return (
                    f"❌ Render failed: {e}\n\n"
                    "🔧 FIX: Foreground content is leaving the slide canvas. "
                    "Use an explicit fixed-size `body` (for 16:9: `1280px × 720px`), "
                    "avoid `vw/vh` wrappers that vertically/horizontally center an oversized `.slide`, "
                    "and switch tall single-column image pages to a constrained two-column/card layout before calling inspect_slide again."
                    + ("\n\n⚠️ Deterministic fallback attempted but did not pass: " + fallback_message if render_failure_count >= 2 else "")
                    + repeated_note
                )
            return f"❌ Render failed: {e}"

    # ── 3. Extract HTML properties ───────────────────────────────────
    try:
        props_summary = _extract_slide_properties(html_path)
    except Exception as e:
        props_summary = f"  (property extraction failed: {e})"
    visual_diagnostics, visual_metrics = _collect_visual_quality_diagnostics(html_path, current_html)
    if _strict_visual_preference_eval_enabled():
        for item in visual_diagnostics:
            if item.get("source") == "soft_visual_preference" and item.get("severity") == "warning":
                item["severity"] = "error"
    diagnostics.extend(visual_diagnostics)
    visual_errors = [item for item in visual_diagnostics if item.get("severity") == "error"]
    if visual_errors:
        message = "VISUAL_QUALITY_FAILED: " + "; ".join(
            str(item.get("message", "")) for item in visual_errors[:4]
        )
        _record_slide_validation_result(
            html_path,
            success=False,
            message=message,
            aspect_ratio=aspect_ratio,
            diagnostics=diagnostics,
        )
        try:
            from memslides.runtime.deck_execution_state import record_slide_inspected

            record_slide_inspected(html_path, success=False)
        except Exception:
            pass
        return (
            "❌ Visual quality failed.\n"
            + "\n".join(f"- {item.get('code')}: {item.get('message')}" for item in visual_errors[:6])
            + "\n\nRewrite this slide using readable contrast, required real assets, visible in-canvas anchored elements, and reference-only template styling before calling inspect_slide again."
        )
    if not skip_preview_render:
        _record_slide_validation_result(
            html_path,
            success=True,
            message="",
            aspect_ratio=aspect_ratio,
            diagnostics=diagnostics,
        )
    _reset_slide_render_failure_count(html_path)

    structure_issues = _collect_slide_structure_issues(
        current_html
    )

    # ── 3b. Stage 7: Template compliance check (lightweight) ─────────
    compliance_text = ""
    try:
        _builder, _skill = _ensure_template_context()
        if _builder and _skill:
            from memslides.memory.compliance.template_checker import TemplateComplianceChecker
            from memslides.tools.template_tools import is_layout_queried
            _checker = TemplateComplianceChecker(_builder.skill)
            _layout = _builder.get_last_queried_layout()
            html_text = html_path.read_text(encoding="utf-8")
            # 从 HTML 注释推断布局（优先于 last_queried）
            _inferred = _infer_layout_from_html(html_text)
            if _inferred:
                _layout = _inferred
            # 布局未查询警告
            if _layout and not is_layout_queried(_layout):
                compliance_text += (
                    f"\n⚠️ MUST_QUERY: 当前页布局 `{_layout}` 未通过 query_slide_layout 查询。"
                    f"请调用 query_slide_layout(\"{_layout}\") 获取元素规范后重新生成。\n"
                )
            _issues = _checker.check(html_text, _layout, file_path=html_path)
            if _issues:
                compliance_text += _checker.format_issues_for_retry(_issues)
                # 记录合规反馈
                _builder.record_compliance_feedback(str(html_path), len(_issues), compliance_text)
    except Exception as _e:
        pass  # compliance check is non-fatal

    try:
        from memslides.runtime.deck_execution_state import (
            deck_progress_summary,
            record_slide_inspected,
        )

        record_slide_inspected(html_path, success=True)
        persona_verifier_result = await _run_profile_page_verifier(
            html_path=html_path,
            current_html=current_html,
            props_summary=props_summary,
            structure_issues=structure_issues,
            compliance_text=compliance_text,
        )
        _deck_progress = deck_progress_summary()
    except Exception:
        persona_verifier_result = None
        _deck_progress = {"active": False, "complete": True}

    # ── 4. Build text summary ────────────────────────────────────────
    if skip_preview_render:
        summary_parts = ["✅ Slide validated successfully (static mode, preview render skipped)."]
    else:
        summary_parts = ["✅ Slide rendered successfully (HTML→PPTX OK)."]

    diagram_contract = _active_diagram_contract_for_path(html_path)
    if diagram_contract:
        diagram_diagnostics = validate_diagram_layout_static(current_html, diagram_contract)
        if diagram_diagnostics:
            fallback_success, fallback_message, fallback_diagnostics = await _inspect_diagram_fallback_once(
                html_path,
                html_file=html_file,
                current_html=current_html,
                diagnostics=diagram_diagnostics,
                contract=diagram_contract,
                aspect_ratio=aspect_ratio,
            )
            if fallback_success:
                return fallback_message
            diagnostics = _merge_validation_diagnostics(diagnostics, fallback_diagnostics)
            message = "DIAGRAM_LAYOUT_VALIDATION_FAILED: " + "; ".join(
                str(item.get("code", "")) for item in diagram_diagnostics[:4]
            )
            _record_slide_validation_result(
                html_path,
                success=False,
                message=message,
                aspect_ratio=aspect_ratio,
                diagnostics=_merge_validation_diagnostics(diagnostics, diagram_diagnostics),
            )
            try:
                from memslides.runtime.deck_execution_state import record_slide_inspected

                record_slide_inspected(html_path, success=False)
            except Exception:
                pass
            return (
                "❌ Diagram layout validation failed.\n"
                + "\n".join(f"- {item.get('code')}: {item.get('message')}" for item in diagram_diagnostics[:6])
                + "\n\nRewrite the full target slide as a coherent flowchart/pipeline page, or use `render_flowchart_asset` and replace the old layout."
                + ("\n\n⚠️ Deterministic diagram fallback did not pass: " + fallback_message if fallback_message else "")
            )
        diagnostics = _merge_validation_diagnostics(diagnostics, diagram_diagnostics)
        summary_parts.append("\n✅ Diagram layout contract passed.")

    # Change status
    if changed and old_size > 0:
        summary_parts.append(f"⚡ File CHANGED since last inspect ({old_size}→{new_size} bytes).")
    elif changed:
        summary_parts.append(f"📄 First inspection of this file ({new_size} bytes).")
    else:
        path_key = str(html_path)
        _slide_unchanged_count[path_key] = _slide_unchanged_count.get(path_key, 0) + 1
        _unchanged_n = _slide_unchanged_count[path_key]
        revision_hint = _revision_inspect_completion_hint(html_path)
        if _unchanged_n >= 2:
            if revision_hint:
                summary_parts.append(
                    f"🔁 File UNCHANGED (inspected {_unchanged_n} times without modification). "
                    "The current slide content is stable for this RevisionEditor round."
                )
            elif _deck_progress.get("active") and not _deck_progress.get("complete"):
                summary_parts.append(
                    f"🔁 File UNCHANGED (inspected {_unchanged_n} times without modification). "
                    "The current slide content is stable. Do not finalize yet; continue with the next missing or unchecked slide."
                )
            else:
                summary_parts.append(
                    f"🔁 File UNCHANGED (inspected {_unchanged_n} times without modification). "
                    "The slide content is stable. **Call `finalize` NOW to complete the round.**"
                )
        else:
            summary_parts.append(
                f"⚠️ File UNCHANGED since last inspect ({new_size} bytes). "
                "No new modifications detected since the previous inspect."
            )

    summary_parts.append(f"\n📋 Slide Properties:\n{props_summary}")
    if structure_issues:
        summary_parts.append(
            "\n⚠️ Structural warnings:\n- " + "\n- ".join(structure_issues[:3])
        )
    visual_warnings = [item for item in visual_diagnostics if item.get("severity") == "warning"]
    if visual_warnings:
        summary_parts.append(
            "\n⚠️ Visual quality warnings:\n- "
            + "\n- ".join(str(item.get("message", "")) for item in visual_warnings[:3])
        )
    if compliance_text:
        summary_parts.append(f"\n{compliance_text}")
    persona_final_verdict = str((persona_verifier_result or {}).get("final_verdict", "") or "").strip()
    defer_persona_repair, deferred_next_file = _deck_progress_defers_persona_repair(_deck_progress, html_file)
    if persona_final_verdict == "retry_required":
        repair_focus = str((persona_verifier_result or {}).get("repair_focus", "") or "").strip()
        if defer_persona_repair:
            summary_parts.append(
                "\n⚠️ Page-level contract gap recorded for later persona repair."
                + (
                    f"\n- Deferred repair focus: {repair_focus}"
                    if repair_focus
                    else ""
                )
                + "\n- Stage A priority: finish structural inspection for every expected slide first."
                + f"\n- Continue with `{deferred_next_file}` now; do not rewrite this page until deck progress points back to it."
            )
        else:
            try:
                rewrite_hash = _compute_text_hash(html_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                rewrite_hash = ""
            summary_parts.append(
                "\n⚠️ Page-level contract check requires another full-page rewrite."
                + (
                    f"\n- Repair focus: {repair_focus}"
                    if repair_focus
                    else ""
                )
                + (
                    "\n- Full rewrite protocol: call "
                    f"`write_html_file(file_path=\"{html_file}\", content=<complete replacement HTML>, "
                    f"force_regenerate=true, expected_hash=\"{rewrite_hash}\")`."
                    if rewrite_hash
                    else ""
                )
                + "\n- Do not use `apply_slide_patch` for this persona contract retry; rebalance the whole page so the new component fits."
            )
    elif persona_final_verdict == "unresolved_release":
        repair_focus = str((persona_verifier_result or {}).get("repair_focus", "") or "").strip()
        summary_parts.append(
            "\n⚠️ Page-level contract remains partially unresolved, but retry budget is exhausted."
            + (
                f"\n- Last repair focus: {repair_focus}"
                if repair_focus
                else ""
            )
        )
    elif persona_final_verdict == "pass":
        summary_parts.append("\n✅ Page-level contract check passed.")
    if changed:
        _slide_unchanged_count[str(html_path)] = 0
    if structure_issues:
        summary_parts.append(
            "\n💡 HTML→PPTX 虽然通过，但当前页仍有结构/画布异常信号；请修复后再 `finalize`。"
        )
    elif persona_final_verdict == "retry_required" and not defer_persona_repair:
        summary_parts.append(
            "\n💡 Rewrite this slide with full replacement HTML using `write_html_file(..., force_regenerate=true, expected_hash=...)`, then inspect it again."
        )
    else:
        revision_hint = _revision_inspect_completion_hint(html_path)
        if revision_hint:
            summary_parts.append(revision_hint)
        elif _deck_progress.get("active") and not _deck_progress.get("complete"):
            next_page = _deck_progress.get("next_page", {}) or {}
            if _deck_progress.get("next_action") == "write_html":
                summary_parts.append(
                    "\n💡 Current slide passed. Do not finalize yet. "
                    f"Continue by writing `{next_page.get('file')}`."
                )
            elif _deck_progress.get("next_action") == "inspect_or_fix":
                summary_parts.append(
                    "\n💡 Current slide passed. Do not finalize yet. "
                    f"Continue by inspecting or fixing `{next_page.get('file')}`."
                )
            else:
                summary_parts.append(
                    "\n💡 Current slide passed. Continue until every expected slide exists and passes inspection."
                )
        elif changed:
            summary_parts.append(
                "\n💡 Modifications confirmed. If the slide looks correct, call `finalize` now."
            )
        else:
            summary_parts.append(
                "\n💡 No new changes detected. If the slide is correct, call `finalize` now."
            )
    text_summary = "\n".join(summary_parts)

    # ── 5. Render image (always attempt if multimodal) ───────────────
    # Reuse existing slide preview images from .slide_images-pdf-*/ directories,
    # which are already generated by PlaywrightConverter.convert_to_pdf() during
    # the main HTML→PDF→PPTX pipeline. No extra conversion needed.
    PREVIEW_MAX_WIDTH = 640
    image_content = None
    resolved_model_ref, current_llm_cfg = _get_current_agent_llm_config()
    if getattr(current_llm_cfg, "is_multimodal", False):
        try:
            # Map slide_XX.html → slide_XX.jpg in the nearest .slide_images-pdf-* dir
            slide_stem = html_path.stem  # e.g. "slide_01"
            preview_dirs = sorted(
                Path.cwd().glob(".slide_images-pdf-*"),
                key=lambda d: d.stat().st_mtime, reverse=True,
            )
            preview_path = None
            for pdir in preview_dirs:
                candidate = pdir / f"{slide_stem}.jpg"
                if candidate.exists():
                    preview_path = candidate
                    break
            if preview_path:
                img = Image.open(preview_path)
                if img.width > PREVIEW_MAX_WIDTH:
                    ratio = PREVIEW_MAX_WIDTH / img.width
                    img = img.resize(
                        (PREVIEW_MAX_WIDTH, int(img.height * ratio)),
                        Image.LANCZOS,
                    )
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                base64_data = (
                    f"data:image/jpeg;base64,"
                    f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                )
                image_content = ImageContent(
                    type="image",
                    data=base64_data,
                    mimeType="image/jpeg",
                )
        except Exception as e:
            warning(f"Image rendering failed (non-fatal): {e}")
    else:
        info(
            f"inspect_slide running in text-only mode for agent={get_current_agent()} "
            f"(resolved_model_ref={resolved_model_ref})"
        )

    # ── 6. Return combined result ────────────────────────────────────
    if image_content is not None:
        return [
            image_content,
            TextContent(type="text", text=text_summary),
        ]
    else:
        return text_summary


@mcp.tool()
def inspect_manuscript(md_file: str) -> dict:
    """
    Inspect the markdown manuscript for general statistics and image asset validation.
    Args:
        md_file (str): The path to the markdown file
    """
    md_path = Path(md_file)
    if not md_path.exists():
        return {"error": f"file does not exist: {md_file}"}
    if not md_file.lower().endswith(".md"):
        return {"error": f"file is not a markdown file: {md_file}"}

    with open(md_file, encoding="utf-8") as f:
        markdown = f.read()

    pages = _split_markdown_slide_pages(markdown)
    result = defaultdict(list)
    result["num_pages"] = len(pages)
    result["language"] = _detect_language_label(markdown)
    expected_pages = _expected_num_pages(anchor=md_path)
    if expected_pages is not None:
        result["expected_num_pages"] = expected_pages
        result["page_count_delta"] = len(pages) - expected_pages
        result["page_count_status"] = (
            "match" if len(pages) == expected_pages else "mismatch"
        )
        if len(pages) != expected_pages:
            result["warnings"].append(
                "Page-count mismatch: "
                f"requested {expected_pages} pages but manuscript currently has {len(pages)}."
            )

    # 检查是否缺少 --- 分隔符（内容多但只有 1 页说明没分页）
    h2_count = markdown.count("\n## ")
    if len(pages) == 1 and h2_count >= 3:
        result["warnings"].append(
            f"CRITICAL: Manuscript has {h2_count} sections but only 1 page (no '---' separators found). "
            f"Each slide MUST be separated by a line containing only '---'. "
            f"Please rewrite the manuscript with '---' between each slide's content."
        )

    seen_images = set()
    for match in re.finditer(r"!\[(.*?)\]\((.*?)\)", markdown):
        label, path = match.group(1), match.group(2)
        path = path.split()[0].strip("\"'")

        if path in seen_images:
            continue
        seen_images.add(path)

        if re.match(r"https?://", path):
            result["warnings"].append(
                f"External link detected: {match.group(0)}, consider downloading to local storage."
            )
            continue

        if _resolve_local_markdown_asset_path(path, md_dir=md_path.parent) is None:
            result["warnings"].append(f"Image file does not exist: {path}")

        if not label.strip():
            result["warnings"].append(f"Image {path} is missing alt text.")

        count = markdown.count(path)
        if count > 1:
            result["warnings"].append(
                f"Image {path} used {count} times in the whole presentation manuscript."
            )

    if len(result["warnings"]) == 0:
        result["success"].append(
            "Image asset validation passed: all referenced images exist."
        )

    return result


@mcp.tool()
def explore_workspace_images(
    min_size_kb: int = 10,
    max_results: int = 30,
    directory: str = ".",
) -> dict:
    """Explore all usable images in the workspace, returning file metadata and dimensions.

    Scans the workspace for image files (png, jpg, jpeg, webp), filters out small
    icons and system cache files, and returns a sorted list with path, size, and
    pixel dimensions.  The agent can then call `image_caption` on specific images
    of interest.

    Args:
        min_size_kb: Minimum file size in KB to include (filters out tiny icons). Default 10.
        max_results: Maximum number of images to return. Default 30.
        directory: Directory to scan, relative to workspace root. Default "." (entire workspace).

    Returns:
        dict with image list including paths, sizes, and dimensions.
    """
    search_root = Path(directory).resolve()
    if not search_root.exists():
        return {"success": False, "error": f"Directory not found: {directory}", "images": []}

    # Directories to exclude (system caches, slide screenshots, history)
    _EXCLUDE_PREFIXES = {".slide_images", ".history", "__pycache__", ".git"}
    min_bytes = min_size_kb * 1024
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}

    seen: set[str] = set()
    candidates: list[dict] = []

    for ext in image_exts:
        for img_path in search_root.rglob(f"*{ext}"):
            # Skip excluded directories
            if any(part.startswith(tuple(_EXCLUDE_PREFIXES)) for part in img_path.parts):
                continue
            abs_str = str(img_path.resolve())
            if abs_str in seen:
                continue
            if not img_path.is_file():
                continue
            file_size = img_path.stat().st_size
            if file_size < min_bytes:
                continue
            seen.add(abs_str)

            # Read dimensions
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                w, h = 0, 0

            candidates.append({
                "path": abs_str,
                "filename": img_path.name,
                "size_kb": round(file_size / 1024, 1),
                "width": w,
                "height": h,
            })

    # Sort by size descending (larger images are usually more useful)
    candidates.sort(key=lambda x: x["size_kb"], reverse=True)
    candidates = candidates[:max_results]

    return {
        "success": True,
        "total_found": len(seen),
        "returned": len(candidates),
        "min_size_kb": min_size_kb,
        "images": candidates,
        "hint": "Use image_caption(path) to get a description of any image before deciding to use it.",
    }


# Backward-compatible direct-call handle for tests and internal helpers that
# still use tool.fn(...). FastMCP currently exposes plain functions here.
for _tool_fn in (
    write_html_file,
    write_new_slide_file,
    insert_slide,
    apply_slide_patch,
    batch_update_semantic_style,
    inspect_slide,
):
    if callable(_tool_fn) and not hasattr(_tool_fn, "fn"):
        _tool_fn.fn = _tool_fn


if __name__ == "__main__":
    from memslides.tools.deck_tools import main as _deck_tools_main

    _deck_tools_main(sys.argv[1:])
